"""GBDT possession-change hazard (F2-09, decision D-029): features, model files and inference.

Same target and output convention as the logistic hazard of F1-08 (:mod:`modeling.evidence.hazard`):
the model ``g(x)`` is a ``HistGradientBoostingClassifier`` trained on eligible IDSSE frames (H or A)
with target "less than ``target_w_frames`` frames from a team change", and the evidence is the
likelihood ratio

    h = clip(g(x) - logit(pi), -h_clip, +h_clip);  h = 0 where half == 0;  float32

with ``pi`` the share of positives in the training rows of that model (``all`` or the fold's).

Every feature is **even** under a home/away swap (products and absolute values of antisymmetric
signals), so ``h`` says *whether* possession is changing, never *to whom*. Features
(:func:`hazard_gbdt_feature_names`, in this order):

- the F1-08 raw features of the D-025 ``hazard_nodir`` fit (no ``ball_dir_change``, which failed the
  BAR-ATH range check): ``near_flip`` and ``dlogit`` at windows 12 and 25 (``near_flip_12`` ...), then
  ``ball_acc``, ``abs_d_diff``, ``d_min``, ``near_flip``, ``dlogit``, ``gap_length``,
  ``since_last_obs``, ``until_next_obs`` at window 50;
- control EMAs: ``c`` is the pass-bridge signal of F1-05 (``f1 * ev_prox_w * ev_ball_conf`` where
  the ball is valid, 0 elsewhere, centred median of ``bridge_median_frames``), smoothed backward
  (causal) and forward (anti-causal) with time constant ``tau``: ``ctl_prod_tau = bwd * fwd``,
  ``ctl_absdiff_tau = |bwd - fwd|``, ``ctl_absb_tau = |bwd|``, ``ctl_absf_tau = |fwd|`` for
  ``tau`` in :data:`CTL_TAUS`. Before a team change the past says one team and the future the
  other: the product turns negative and the difference grows;
- touch EMAs: the same with the controlled-touch feature ``f9`` (NaN -> 0), ``tch_prod_tau`` and
  ``tch_absdiff_tau`` for ``tau`` in :data:`TOUCH_TAUS`;
- evidence mass: backward and forward EMAs of ``|c|`` (``ctlmass_bwd_tau``, ``ctlmass_fwd_tau``)
  for ``tau`` in :data:`MASS_TAUS`;
- ``d_diff`` windows: past and future nanmeans over ``W`` frames, ``dd_prod_W`` and
  ``dd_absdiff_W`` for ``W`` in :data:`DD_WINDOWS`.

Every EMA and window stays inside its ``half`` block. Missing values stay NaN for the model (no
imputation). The inputs are the master columns already checked on BAR-ATH by
``tools/hazard_feature_shift.py`` and ``tools/gbdt_feature_shift.py``; ``tools/stack_feature_shift.py``
checks these derived features as well.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import yaml

from .. import paths
from ..variants import register_hazard
from .collective import _half_runs
from .control import build_control_features
from .hazard import build_hazard_features
from .team import _causal_ema_1d, _centered_median_per_half, build_touch_feature
from .windows import rolling_window_stats

VARIANT = "gbdt"
RAW_W50 = ("ball_acc", "abs_d_diff", "d_min", "near_flip", "dlogit", "gap_length",
           "since_last_obs", "until_next_obs")
SHORT_WINDOWS = (12, 25)
SHORT_RAW = ("near_flip", "dlogit")
CTL_TAUS = (10, 25, 50, 100, 200)
TOUCH_TAUS = (25, 75, 200)
MASS_TAUS = (25, 100)
DD_WINDOWS = (100, 200)
CALIB_KEYS = ("variant", "sklearn_version", "model_file", "fold_dir", "features",
              "hazard_settings", "control_params", "h_clip")


def hazard_gbdt_feature_names() -> list[str]:
    names = []
    for w in SHORT_WINDOWS:
        names.extend("{}_{}".format(c, w) for c in SHORT_RAW)
    names.extend(RAW_W50)
    for tau in CTL_TAUS:
        names.extend(["ctl_prod_{}".format(tau), "ctl_absdiff_{}".format(tau),
                      "ctl_absb_{}".format(tau), "ctl_absf_{}".format(tau)])
    for tau in TOUCH_TAUS:
        names.extend(["tch_prod_{}".format(tau), "tch_absdiff_{}".format(tau)])
    for tau in MASS_TAUS:
        names.extend(["ctlmass_bwd_{}".format(tau), "ctlmass_fwd_{}".format(tau)])
    for w in DD_WINDOWS:
        names.extend(["dd_prod_{}".format(w), "dd_absdiff_{}".format(w)])
    return names


def ema_pair(c: np.ndarray, half: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Backward (causal) and forward (anti-causal) EMAs of ``c`` inside each half block.

    Both are 0 on ``half == 0`` (callers mask them to NaN there).
    """
    bwd = np.zeros(len(c), dtype=np.float64)
    fwd = np.zeros(len(c), dtype=np.float64)
    for start, stop in _half_runs(np.asarray(half)):
        seg = np.asarray(c[start:stop], dtype=np.float64)
        if len(seg) == 0:
            continue
        bwd[start:stop] = _causal_ema_1d(seg, float(tau))
        fwd[start:stop] = _causal_ema_1d(seg[::-1], float(tau))[::-1]
    return bwd, fwd


def control_signal(master: pd.DataFrame, control_params: Mapping) -> tuple[np.ndarray, np.ndarray]:
    """``(c, f9)``: the median-filtered pass-bridge signal and the touch feature (NaN -> 0).

    Both are antisymmetric under a team swap and 0 where the ball is not valid.
    """
    p = dict(control_params)
    half = master["half"].to_numpy()
    ev = build_control_features(master, p)
    valid = ev["ev_ball_valid"].to_numpy() & (half != 0)
    d_diff = master["d_diff"].to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore"):
        f1 = -np.tanh(d_diff / 2.0)
        c_raw = (f1 * ev["ev_prox_w"].to_numpy(dtype=np.float64)
                 * ev["ev_ball_conf"].to_numpy(dtype=np.float64))
    c = np.where(valid & np.isfinite(c_raw), c_raw, 0.0)
    c = _centered_median_per_half(c, half, int(p["bridge_median_frames"]))
    f9 = build_touch_feature(master, p, ev=ev).to_numpy(dtype=np.float64)
    f9 = np.where(np.isfinite(f9), f9, 0.0)
    return c, f9


def build_hazard_gbdt_features(master: pd.DataFrame, hazard_settings: Mapping,
                               control_params: Mapping) -> pd.DataFrame:
    """Feature frame (float64, NaN kept, columns of :func:`hazard_gbdt_feature_names`).

    ``hazard_settings`` are the F1-08 feature settings (``acc_clip_mps2``, ``dist_clip_m``,
    ``gap_clip_frames``, ``h_clip``, ``team_params``); ``control_params`` the team parameters of
    the control and touch features (the team section's ``params`` plus the GBDT feature parameters
    of ``configs/calib/team_gbdt.yaml``).
    """
    half = master["half"].to_numpy()
    cols: dict[str, np.ndarray] = {}
    for w in SHORT_WINDOWS:
        raw = build_hazard_features(master, {**dict(hazard_settings), "window_frames": w})
        for name in SHORT_RAW:
            cols["{}_{}".format(name, w)] = raw[name].to_numpy(dtype=np.float64)
    raw = build_hazard_features(master, {**dict(hazard_settings), "window_frames": 50})
    for name in RAW_W50:
        cols[name] = raw[name].to_numpy(dtype=np.float64)

    c, f9 = control_signal(master, control_params)
    for tau in CTL_TAUS:
        b, f = ema_pair(c, half, tau)
        cols["ctl_prod_{}".format(tau)] = b * f
        cols["ctl_absdiff_{}".format(tau)] = np.abs(b - f)
        cols["ctl_absb_{}".format(tau)] = np.abs(b)
        cols["ctl_absf_{}".format(tau)] = np.abs(f)
    for tau in TOUCH_TAUS:
        b, f = ema_pair(f9, half, tau)
        cols["tch_prod_{}".format(tau)] = b * f
        cols["tch_absdiff_{}".format(tau)] = np.abs(b - f)
    abs_c = np.abs(c)
    for tau in MASS_TAUS:
        b, f = ema_pair(abs_c, half, tau)
        cols["ctlmass_bwd_{}".format(tau)] = b
        cols["ctlmass_fwd_{}".format(tau)] = f
    d_diff = master["d_diff"].to_numpy(dtype=np.float64)
    for w in DD_WINDOWS:
        r = rolling_window_stats(d_diff, half, w, ("past_mean", "fut_mean"))
        cols["dd_prod_{}".format(w)] = r["past_mean"] * r["fut_mean"]
        cols["dd_absdiff_{}".format(w)] = np.abs(r["past_mean"] - r["fut_mean"])

    names = hazard_gbdt_feature_names()
    feats = pd.DataFrame({k: cols[k] for k in names}, index=master.index)
    feats.loc[half == 0, :] = np.nan
    return feats


# -- calibration file and models -----------------------------------------------------------------

def resolve_repo_path(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else paths.repo_root() / p


@lru_cache(maxsize=8)
def _load_yaml(path_str: str, mtime_ns: int) -> dict:
    del mtime_ns
    with open(path_str, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or data.get("variant") != VARIANT:
        raise ValueError("invalid GBDT hazard calibration file: {}".format(path_str))
    missing = [k for k in CALIB_KEYS if k not in data]
    if missing:
        raise ValueError("GBDT hazard calibration file {} misses {}".format(path_str, missing))
    if list(data["features"]) != hazard_gbdt_feature_names():
        raise ValueError("feature list of {} does not match the code".format(path_str))
    return data


def load_hazard_gbdt_calibration(path) -> dict:
    p = resolve_repo_path(path)
    if not p.is_file():
        raise FileNotFoundError("GBDT hazard calibration file not found: {}".format(p))
    return _load_yaml(str(p.resolve()), p.stat().st_mtime_ns)


def fold_model_path(calib: Mapping, holdout: str) -> Path:
    """``<fold_dir>/<holdout>.joblib``, ``fold_dir`` relative to ``ext_dir()`` (not committed)."""
    fold_dir = Path(calib["fold_dir"])
    if not fold_dir.is_absolute():
        fold_dir = paths.ext_dir() / fold_dir
    return fold_dir / "{}.joblib".format(holdout)


@lru_cache(maxsize=16)
def _load_bundle(path_str: str, mtime_ns: int) -> dict:
    del mtime_ns
    import joblib
    return joblib.load(path_str)


def load_model_bundle(path, *, sklearn_version: str, features) -> dict:
    """``{"sklearn_version", "features", "model", "pi"}``; refuses another scikit-learn version."""
    import sklearn

    if sklearn.__version__ != sklearn_version:
        raise RuntimeError("GBDT hazard needs scikit-learn {}, found {}".format(
            sklearn_version, sklearn.__version__))
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            "GBDT hazard model not found: {} (run python -m modeling.calibration.stack_gbdt)".format(p))
    bundle = _load_bundle(str(p.resolve()), p.stat().st_mtime_ns)
    if bundle.get("sklearn_version") != sklearn_version:
        raise RuntimeError("model {} was saved with scikit-learn {}, expected {}".format(
            p, bundle.get("sklearn_version"), sklearn_version))
    if list(bundle.get("features", [])) != list(features):
        raise ValueError("model {} has a different feature list".format(p))
    pi = float(bundle["pi"])
    if not 0.0 < pi < 1.0:
        raise ValueError("model {} has an invalid pi {}".format(p, pi))
    return bundle


def hazard_from_bundle(feats: pd.DataFrame, half: np.ndarray, bundle: Mapping,
                       h_clip: float) -> np.ndarray:
    """``clip(g(x) - logit(pi), +-h_clip)``, 0 on ``half == 0``, float32, single-threaded."""
    from threadpoolctl import threadpool_limits

    X = feats.to_numpy(dtype=np.float64)
    pi = float(bundle["pi"])
    with threadpool_limits(limits=1):
        g = bundle["model"].decision_function(X).astype(np.float64)
    h = np.clip(g - np.log(pi / (1.0 - pi)), -float(h_clip), float(h_clip))
    use = (np.asarray(half) != 0) & np.isfinite(h)
    return (np.where(use, h, 0.0) + 0.0).astype(np.float32)


@register_hazard("gbdt")
def hazard_gbdt(master: pd.DataFrame, params: dict, *, holdout=None) -> np.ndarray:
    """GBDT hazard: float32[n], finite, |h| <= h_clip, 0 where half == 0 (``all`` model when
    ``holdout`` is None, else the leave-one-match-out model of ``holdout``)."""
    if "calibrated_file" not in params:
        raise ValueError("hazard kind 'gbdt' needs params['calibrated_file'] (D-011)")
    calib = load_hazard_gbdt_calibration(params["calibrated_file"])
    path = (resolve_repo_path(calib["model_file"]) if holdout is None
            else fold_model_path(calib, holdout))
    bundle = load_model_bundle(path, sklearn_version=str(calib["sklearn_version"]),
                               features=calib["features"])
    feats = build_hazard_gbdt_features(master, calib["hazard_settings"], calib["control_params"])
    return hazard_from_bundle(feats, master["half"].to_numpy(), bundle, float(calib["h_clip"]))


__all__ = ["CTL_TAUS", "DD_WINDOWS", "MASS_TAUS", "TOUCH_TAUS", "build_hazard_gbdt_features",
           "control_signal", "ema_pair", "fold_model_path", "hazard_from_bundle", "hazard_gbdt",
           "hazard_gbdt_feature_names", "load_hazard_gbdt_calibration", "load_model_bundle"]
