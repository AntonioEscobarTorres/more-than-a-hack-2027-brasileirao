"""Possession-change hazard (task F1-08): per-frame evidence that possession is switching team.

The hazard is **symmetric** under a home/away swap: it says *whether* possession is changing
between H and A, never *to whom* (that is ``team_logit``, contract C2). It is meant for the
transitions of the decoder (F2-07), not for the emission.

Raw features (all even under a home/away swap; computed per ``half`` block, never across
``half == 0``; ``W`` = ``window_frames``):

    ball_acc         log1p(min(ball_acc, acc_clip_mps2)); NaN where ball_acc is NaN
    ball_dir_change  ball_dir_change (already in [0, pi]); NaN where undefined
    abs_d_diff       min(|d_diff|, dist_clip_m); NaN where d_diff is NaN
    d_min            min(fmin(d_near_home, d_near_away), dist_clip_m); NaN if both are NaN
    near_flip        1 if the majority sign of d_diff over [t-W, t) differs from the one over
                     (t, t+W] (both non-zero), else 0
    dlogit           |mean(team_logit_rules_v1 over [t-W, t)) - mean(over (t, t+W])|; 0 where a
                     half-window is empty (block edges)
    gap_length       log1p(min(gap_length, gap_clip_frames))
    since_last_obs   log1p(min(since_last_obs, gap_clip_frames)); NaN before the first observation
    until_next_obs   log1p(min(until_next_obs, gap_clip_frames)); NaN after the last observation

Design matrix (:func:`design_matrix`): a feature that can be NaN contributes 0 explicitly
(``np.where(np.isfinite(x), x, 0.0)``) and gets a 0/1 indicator column ``has_<feature>``
(rule 4: imputation only inside the design matrix, always with an indicator).

Output of the ``calibrated`` kind (likelihood ratio, as in F3-04)::

    h = clip(logit_model - logit(pi), -h_clip, +h_clip);  h = 0 where half == 0;  float32

with ``pi`` the fraction of positives in the training set of the selected fold.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..deadball.features import half_blocks
from ..variants import register_hazard
from .control import require_columns
from .team import team_logit_rules_v1

# Raw feature -> True when it can be NaN (gets an indicator column in the design matrix).
RAW_FEATURES = {
    "ball_acc": True,
    "ball_dir_change": True,
    "abs_d_diff": True,
    "d_min": True,
    "near_flip": False,
    "dlogit": False,
    "gap_length": False,
    "since_last_obs": True,
    "until_next_obs": True,
}
REQUIRED_COLUMNS = ("half", "ball_acc", "ball_dir_change", "d_diff", "d_near_home",
                    "d_near_away", "gap_length", "since_last_obs", "until_next_obs")
FEATURE_KEYS = ("acc_clip_mps2", "dist_clip_m", "gap_clip_frames", "h_clip", "team_params")
MODEL_KEYS = ("window_frames", "C", "pi", "intercept", "mean", "std", "weight")


def design_columns(raw_features) -> list[str]:
    """Design-matrix column names for the raw features ``raw_features`` (in that order)."""
    cols = []
    for name in raw_features:
        if name not in RAW_FEATURES:
            raise ValueError("unknown hazard feature {!r}".format(name))
        cols.append(name)
        if RAW_FEATURES[name]:
            cols.append("has_" + name)
    return cols


def _f64(master: pd.DataFrame, name: str) -> np.ndarray:
    return master[name].to_numpy(dtype="float64")


def _log_clip(x: np.ndarray, upper: float) -> np.ndarray:
    """log1p(min(x, upper)) for x >= 0; NaN stays NaN."""
    with np.errstate(invalid="ignore"):
        return np.log1p(np.minimum(np.maximum(x, 0.0), upper))


def _half_window_sums(x: np.ndarray, w: int) -> tuple[np.ndarray, ...]:
    """For one block: sums and counts of ``x`` over [t-w, t) and (t, t+w], truncated at the edges."""
    n = len(x)
    cs = np.r_[0.0, np.cumsum(x, dtype=np.float64)]
    t = np.arange(n)
    lo = np.maximum(t - w, 0)
    hi = np.minimum(t + w + 1, n)
    left_sum, left_n = cs[t] - cs[lo], t - lo
    right_sum, right_n = cs[hi] - cs[t + 1], hi - (t + 1)
    return left_sum, left_n, right_sum, right_n


def _near_flip(d_diff: np.ndarray, w: int) -> np.ndarray:
    sign = np.where(np.isfinite(d_diff), np.sign(np.where(np.isfinite(d_diff), d_diff, 0.0)), 0.0)
    left, _, right, _ = _half_window_sums(sign, w)
    return ((left * right) < 0).astype(np.float64)


def _dlogit(team: np.ndarray, w: int) -> np.ndarray:
    left, n_left, right, n_right = _half_window_sums(team, w)
    ok = (n_left > 0) & (n_right > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        diff = np.abs(left / n_left - right / n_right)
    return np.where(ok, diff, 0.0)


def build_hazard_features(master: pd.DataFrame, fparams: dict) -> pd.DataFrame:
    """Raw hazard features (float64, NaN = no evidence), same index as ``master``.

    Rows with ``half == 0`` are NaN in every column.
    """
    require_columns(master, REQUIRED_COLUMNS)
    w = int(fparams["window_frames"])
    if w < 1:
        raise ValueError("window_frames must be >= 1, got {}".format(w))
    n = len(master)
    half = master["half"].to_numpy()

    d_diff = _f64(master, "d_diff")
    with np.errstate(invalid="ignore"):
        feats = {
            "ball_acc": _log_clip(_f64(master, "ball_acc"), fparams["acc_clip_mps2"]),
            "ball_dir_change": _f64(master, "ball_dir_change"),
            "abs_d_diff": np.minimum(np.abs(d_diff), fparams["dist_clip_m"]),
            "d_min": np.minimum(np.fmin(_f64(master, "d_near_home"), _f64(master, "d_near_away")),
                                fparams["dist_clip_m"]),
            "near_flip": np.full(n, np.nan),
            "dlogit": np.full(n, np.nan),
            "gap_length": _log_clip(_f64(master, "gap_length"), fparams["gap_clip_frames"]),
            "since_last_obs": _log_clip(_f64(master, "since_last_obs"),
                                        fparams["gap_clip_frames"]),
            "until_next_obs": _log_clip(_f64(master, "until_next_obs"),
                                        fparams["gap_clip_frames"]),
        }

    team = team_logit_rules_v1(master, fparams["team_params"]).astype(np.float64)
    for start, stop in half_blocks(half):
        feats["near_flip"][start:stop] = _near_flip(d_diff[start:stop], w)
        feats["dlogit"][start:stop] = _dlogit(team[start:stop], w)

    out = pd.DataFrame(feats, index=master.index)
    out.loc[half == 0, :] = np.nan
    return out


def design_matrix(raw: pd.DataFrame, raw_features) -> np.ndarray:
    """float64 (n, k) design matrix: NaN contributes 0 explicitly, plus ``has_*`` indicators."""
    cols = []
    for name in raw_features:
        x = raw[name].to_numpy(dtype=np.float64)
        finite = np.isfinite(x)
        cols.append(np.where(finite, x, 0.0))
        if RAW_FEATURES[name]:
            cols.append(finite.astype(np.float64))
    return np.column_stack(cols) if cols else np.zeros((len(raw), 0))


def logit(p: float) -> float:
    return float(np.log(p) - np.log1p(-p))


def model_logit(x: np.ndarray, model: dict, columns) -> np.ndarray:
    """intercept + sum_j w_j * (x_j - mean_j) / std_j, written as an explicit float64 sum."""
    mean = np.array([model["mean"][c] for c in columns], dtype=np.float64)
    std = np.array([model["std"][c] for c in columns], dtype=np.float64)
    weight = np.array([model["weight"][c] for c in columns], dtype=np.float64)
    z = (x - mean) / std
    return float(model["intercept"]) + (z * weight).sum(axis=1)


def hazard_from_model(master: pd.DataFrame, calib: dict, model: dict) -> np.ndarray:
    """``h`` (float32, finite) of one fitted model; 0 where ``half == 0``."""
    settings = {**calib["settings"], "window_frames": int(model["window_frames"])}
    raw = build_hazard_features(master, settings)
    columns = design_columns(calib["features"])
    x = design_matrix(raw, calib["features"])
    h_clip = float(settings["h_clip"])  # shared by every model
    h = np.clip(model_logit(x, model, columns) - logit(float(model["pi"])), -h_clip, h_clip)
    use = (master["half"].to_numpy() != 0) & np.isfinite(h)
    return (np.where(use, h, 0.0) + 0.0).astype(np.float32)


def load_hazard_file(path) -> dict:
    """Read ``configs/calib/hazard.yaml`` and check its top-level layout."""
    path = Path(path)
    calib = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(calib, dict):
        raise ValueError("{}: expected a mapping".format(path))
    for key in ("features", "settings", "all", "folds"):
        if key not in calib:
            raise ValueError("{}: missing key {!r}".format(path, key))
    missing = [k for k in FEATURE_KEYS if k not in calib["settings"]]
    if missing:
        raise ValueError("{}: settings missing keys {}".format(path, missing))
    columns = design_columns(calib["features"])
    for name, model in [("all", calib["all"])] + list(calib["folds"].items()):
        missing = [k for k in MODEL_KEYS if k not in model]
        if missing:
            raise ValueError("{}: model {!r} missing keys {}".format(path, name, missing))
        for part in ("mean", "std", "weight"):
            if sorted(model[part]) != sorted(columns):
                raise ValueError("{}: model {!r} {} must cover {}".format(
                    path, name, part, columns))
    return calib


def select_model(calib: dict, holdout: str | None) -> dict:
    """``all`` when ``holdout`` is None, else ``folds[holdout]`` (an error if it is missing)."""
    if holdout is None:
        return calib["all"]
    folds = calib["folds"]
    if holdout not in folds:
        raise ValueError("hazard calibrated_file has no folds[{!r}]".format(holdout))
    return folds[holdout]


@register_hazard("calibrated")
def hazard_calibrated(master: pd.DataFrame, params: dict, *, holdout=None) -> np.ndarray:
    """Calibrated hazard: float32[n], finite, |h| <= h_clip, 0 where half == 0."""
    if "calibrated_file" not in params:
        raise ValueError("hazard kind 'calibrated' needs params['calibrated_file'] (D-011)")
    calib = load_hazard_file(params["calibrated_file"])
    return hazard_from_model(master, calib, select_model(calib, holdout))


__all__ = ["RAW_FEATURES", "build_hazard_features", "design_columns", "design_matrix",
           "hazard_calibrated", "hazard_from_model", "load_hazard_file", "model_logit",
           "select_model"]
