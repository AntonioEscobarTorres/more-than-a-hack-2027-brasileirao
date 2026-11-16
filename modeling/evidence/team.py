"""Team-possession evidence team_logit (contract C2): rules_v1 and calibrated.

Ball term (Appendix A.1, F1-02):

    t = -kappa * tanh(d_diff / sigma_d) * ev_prox_w * ev_rel_w * ev_ball_conf * contest
    contest = contest_factor if ev_contest == 1 else 1
    t = clip(t, -kappa, +kappa);  t = 0 where the ball is not valid, d_diff is NaN or half == 0

Sign convention: d_diff = d_near_home - d_near_away, so home closer -> d_diff < 0 -> t > 0
(favours H). t == 0 means "no evidence".

Calibrated variant (F1-05):
    Uses two antisymmetric logistic models without intercept ("ball" when the ball is valid,
    "gap" when it is not), scaled by uncentred training standard deviations and clipped to
    +-TEAM_LOGIT_MAX (8.0). Inner products use explicit float64 elementwise multiplication and sum.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from preprocessing.params import MAX_GAP_FRAMES

from ..contracts import TEAM_LOGIT_MAX
from ..variants import register_team
from .collective import _half_runs, build_collective_features
from .control import build_control_features, require_columns

COLLECTIVE_WEIGHTS = (("w_adv", "ev_coll_adv"), ("w_spread", "ev_coll_spread"),
                      ("w_speed", "ev_coll_speed"))

BALL_BASE_FEATURES = ("f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8")
GAP_BASE_FEATURES = ("ev_coll_adv", "ev_coll_spread", "ev_coll_speed")

DEFAULT_BRIDGE_MEDIAN_FRAMES = 7
DEFAULT_BRIDGE_TAUS = (15, 35)
DEFAULT_R_TOUCH_M = 2.0
DEFAULT_TOUCH_MIN_FRAMES = 3
DEFAULT_TOUCH_WIN_FRAMES = 9
DEFAULT_TOUCH_DIR_CHANGE_RAD = 0.5
DEFAULT_TOUCH_ACC_MPS2 = 8.0
DEFAULT_TAU_SP_TEAM = 75.0


def _term(weight: float, x: np.ndarray) -> np.ndarray:
    """weight * x, and exactly 0 where x is NaN (rule 4)."""
    return np.where(np.isfinite(x), weight * x, 0.0)


def ball_team_logit(master: pd.DataFrame, params: dict) -> np.ndarray:
    """F1-02 ball term, float64 (before the collective term and the float32 cast)."""
    require_columns(master, ("d_diff", "half"))
    ev = build_control_features(master, params)
    kappa = params["kappa"]

    d_diff = master["d_diff"].to_numpy(dtype="float64")
    contest = np.where(ev["ev_contest"].to_numpy(dtype="float64") == 1,
                       params["contest_factor"], 1.0)
    with np.errstate(invalid="ignore"):
        t = (-kappa * np.tanh(d_diff / params["sigma_d"])
             * ev["ev_prox_w"].to_numpy(dtype="float64")
             * ev["ev_rel_w"].to_numpy(dtype="float64")
             * ev["ev_ball_conf"].to_numpy(dtype="float64") * contest)
    t = np.clip(t, -kappa, kappa)

    use = (ev["ev_ball_valid"].to_numpy() & np.isfinite(d_diff)
           & (master["half"].to_numpy() != 0) & np.isfinite(t))
    return np.where(use, t, 0.0) + 0.0                   # + 0.0 turns -0.0 into +0.0


@register_team("rules_v1")
def team_logit_rules_v1(master: pd.DataFrame, params: dict, *, holdout=None) -> np.ndarray:
    """C2 team_logit, rules_v1: float32[n], finite, |t| <= kappa, 0 where half == 0."""
    t = ball_team_logit(master, params)

    coll = params["collective"]
    if any(coll[w] != 0 for w, _ in COLLECTIVE_WEIGHTS):
        require_columns(master, ("ball_x", "ball_y", "gap_length"))
        feats = build_collective_features(master, params)
        valid = (np.isfinite(master["ball_x"].to_numpy(dtype="float64"))
                 & np.isfinite(master["ball_y"].to_numpy(dtype="float64")))
        with np.errstate(invalid="ignore"):           # gap_length is NaN only where half == 0
            long_gap = master["gap_length"].to_numpy(dtype="float64") > MAX_GAP_FRAMES
        active = (~valid | long_gap) & (master["half"].to_numpy() != 0)

        extra = np.zeros(len(master))
        for weight, column in COLLECTIVE_WEIGHTS:
            extra += _term(coll[weight], feats[column].to_numpy(dtype="float64"))
        kappa = params["kappa"]
        t = np.where(active, np.clip(t + extra, -kappa, kappa), t) + 0.0

    return t.astype(np.float32)


# -- candidate features: pass bridge, touch, and tag prior --------------------------------

def _centered_median_per_half(c: np.ndarray, half: np.ndarray, win_frames: int) -> np.ndarray:
    """Centered moving median of ``c`` inside each half block, zero-padded at half boundaries."""
    win = int(win_frames)
    if win < 1:
        raise ValueError("bridge_median_frames must be >= 1, got {}".format(win))
    if win % 2 == 0:
        win += 1
    if win == 1:
        return np.where(half != 0, c, 0.0)

    pad = win // 2
    out = np.zeros(len(c), dtype="float64")
    for start, stop in _half_runs(half):
        seg = c[start:stop]
        if len(seg) == 0:
            continue
        padded = np.pad(seg, (pad, pad), mode="constant", constant_values=0.0)
        windows = np.lib.stride_tricks.sliding_window_view(padded, win)
        out[start:stop] = np.median(windows, axis=1)
    return out + 0.0


def _causal_ema_1d(seg: np.ndarray, tau: float) -> np.ndarray:
    """Forward causal exponential moving average with time constant ``tau`` frames."""
    alpha = 1.0 / float(tau)
    decay = 1.0 - alpha
    out = np.empty(len(seg), dtype="float64")
    acc = 0.0
    for i in range(len(seg)):
        acc = decay * acc + alpha * float(seg[i])
        out[i] = acc
    return out


def build_pass_bridge_features(
    master: pd.DataFrame,
    params: Mapping,
    *,
    ev: pd.DataFrame | None = None,
    taus: Sequence[int | float] | None = None,
) -> pd.DataFrame:
    """Pass-bridge features ``f10_<tau>`` and ``f11_<tau>`` per half block (F1-05 Update 3).

    ``c(t) = f1(t) * ev_prox_w(t) * ev_ball_conf(t)`` where the ball is valid, and 0 otherwise.
    Filtered with a centered median of ``bridge_median_frames``, then smoothed forward and
    backward with exponential moving averages inside each half block.
    """
    require_columns(master, ("d_diff", "half"))
    if ev is None:
        ev = build_control_features(master, dict(params))

    half = master["half"].to_numpy()
    valid = ev["ev_ball_valid"].to_numpy() & (half != 0)
    d_diff = master["d_diff"].to_numpy(dtype="float64")
    with np.errstate(invalid="ignore"):
        f1 = -np.tanh(d_diff / 2.0)
        c_raw = (f1 * ev["ev_prox_w"].to_numpy(dtype="float64")
                 * ev["ev_ball_conf"].to_numpy(dtype="float64"))
    c = np.where(valid & np.isfinite(c_raw), c_raw, 0.0)

    med_win = int(params.get("bridge_median_frames", DEFAULT_BRIDGE_MEDIAN_FRAMES))
    c_filt = _centered_median_per_half(c, half, med_win)

    if taus is None:
        taus = tuple(params.get("bridge_tau_frames", DEFAULT_BRIDGE_TAUS))

    cols = {}
    for tau in taus:
        tau_label = int(tau) if float(tau).is_integer() else tau
        bwd = np.zeros(len(master), dtype="float64")
        fwd = np.zeros(len(master), dtype="float64")
        for start, stop in _half_runs(half):
            seg = c_filt[start:stop]
            if len(seg) == 0:
                continue
            bwd[start:stop] = _causal_ema_1d(seg, float(tau))
            fwd[start:stop] = _causal_ema_1d(seg[::-1], float(tau))[::-1]
        s = bwd + fwd
        prod = np.maximum(bwd * fwd, 0.0)
        f10 = np.where(half != 0, s, np.nan)
        f11 = np.where(half != 0, np.sign(s) * np.sqrt(prod), np.nan)
        cols["f10_{}".format(tau_label)] = (f10 + 0.0).astype(np.float32)
        cols["f11_{}".format(tau_label)] = (f11 + 0.0).astype(np.float32)

    return pd.DataFrame(cols, index=master.index)


def _window_sum_per_half(mask: np.ndarray, half: np.ndarray, win_frames: int) -> np.ndarray:
    """Centered window count of boolean ``mask`` strictly within each half block."""
    win = int(win_frames)
    if win < 1:
        raise ValueError("touch_win_frames must be >= 1, got {}".format(win))
    if win % 2 == 0:
        win += 1
    pad = win // 2
    out = np.zeros(len(mask), dtype="float64")
    x = mask.astype("float64")
    for start, stop in _half_runs(half):
        seg = x[start:stop]
        if len(seg) == 0:
            continue
        padded = np.pad(seg, (pad, pad), mode="constant", constant_values=0.0)
        csum = np.r_[0.0, np.cumsum(padded)]
        out[start:stop] = csum[win:] - csum[:-win]
    return out


def touch_components(
    master: pd.DataFrame,
    params: Mapping,
    *,
    ev: pd.DataFrame | None = None,
) -> dict[str, np.ndarray]:
    """The two conditions of the touch feature, frame by frame (F1-05 Update 4).

    Returns boolean arrays ``valid`` (ball valid inside a half), ``abrupt`` (an abrupt ball change
    inside the centered window) and ``persist`` (the closer team within ``r_touch_m`` on at least
    ``touch_min_frames`` frames of the window), plus ``f1`` (float64).
    """
    require_columns(master, ("d_diff", "d_near_home", "d_near_away", "half"))
    if ev is None:
        ev = build_control_features(master, dict(params))

    half = master["half"].to_numpy()
    valid = ev["ev_ball_valid"].to_numpy() & (half != 0)
    d_home = master["d_near_home"].to_numpy(dtype="float64")
    d_away = master["d_near_away"].to_numpy(dtype="float64")
    d_diff = master["d_diff"].to_numpy(dtype="float64")

    r_touch = float(params.get("r_touch_m", DEFAULT_R_TOUCH_M))
    min_frames = int(params.get("touch_min_frames", DEFAULT_TOUCH_MIN_FRAMES))
    win_frames = int(params.get("touch_win_frames", DEFAULT_TOUCH_WIN_FRAMES))
    dir_thr = float(params.get("touch_dir_change_rad", DEFAULT_TOUCH_DIR_CHANGE_RAD))
    acc_thr = float(params.get("touch_acc_mps2", DEFAULT_TOUCH_ACC_MPS2))

    if "ball_dir_change" in master.columns:
        dir_ch = master["ball_dir_change"].to_numpy(dtype="float64")
    else:
        dir_ch = np.full(len(master), np.nan)
    if "ball_acc" in master.columns:
        acc = master["ball_acc"].to_numpy(dtype="float64")
    else:
        acc = np.full(len(master), np.nan)

    with np.errstate(invalid="ignore"):
        abrupt_frame = (valid & ((np.isfinite(dir_ch) & (np.abs(dir_ch) >= dir_thr))
                                 | (np.isfinite(acc) & (np.abs(acc) >= acc_thr))))
        home_ctrl = (valid & np.isfinite(d_home) & (d_home <= r_touch)
                     & (~np.isfinite(d_away) | (d_home < d_away)))
        away_ctrl = (valid & np.isfinite(d_away) & (d_away <= r_touch)
                     & (~np.isfinite(d_home) | (d_away < d_home)))
        f1 = -np.tanh(d_diff / 2.0)

    count_home = _window_sum_per_half(home_ctrl, half, win_frames)
    count_away = _window_sum_per_half(away_ctrl, half, win_frames)
    abrupt_win = _window_sum_per_half(abrupt_frame, half, win_frames) >= 1.0

    persist = ((home_ctrl & (count_home >= min_frames))
               | (away_ctrl & (count_away >= min_frames)))
    return {"valid": valid, "abrupt": abrupt_win, "persist": persist, "f1": f1}


def build_touch_feature(
    master: pd.DataFrame,
    params: Mapping,
    *,
    ev: pd.DataFrame | None = None,
) -> pd.Series:
    """Antisymmetric controlled-touch feature ``f9 = f1 * touch`` (F1-05 Update 4).

    ``touch`` is even under team swap and uses the conditions of :func:`touch_components`:
      1. Abrupt ball change: ``|ball_dir_change| >= touch_dir_change_rad`` or
         ``|ball_acc| >= touch_acc_mps2`` inside the centered window. Required only when
         ``touch_require_abrupt`` is true (the default); see decision D-021.
      2. Persistence: the closer team is within ``r_touch_m`` for at least ``touch_min_frames``
         frames of the centered window inside the same half block.
    """
    comp = touch_components(master, params, ev=ev)
    touch = comp["persist"]
    if bool(params.get("touch_require_abrupt", True)):
        touch = touch & comp["abrupt"]
    touch = touch.astype("float64")
    f9 = np.where(comp["valid"], comp["f1"] * touch, np.nan) + 0.0
    return pd.Series(f9.astype(np.float32), index=master.index, name="f9")


def team_tag_prior(master: pd.DataFrame, params: Mapping) -> np.ndarray:
    """Optional pre-registered tag prior (F1-05 Update 8), float64[n].

    Zero on IDSSE (where tag columns are absent/zero) and whenever ``w_tag_phase`` and
    ``w_sp_team`` are zero. Computed per half block and strictly 0 on ``half == 0``.
    """
    n = len(master)
    w_phase = float(params.get("w_tag_phase", 0.0))
    w_sp = float(params.get("w_sp_team", 0.0))
    if w_phase == 0.0 and w_sp == 0.0:
        return np.zeros(n, dtype="float64")

    half = master["half"].to_numpy()
    valid = (np.isfinite(master["ball_x"].to_numpy(dtype="float64"))
             & np.isfinite(master["ball_y"].to_numpy(dtype="float64")))
    out = np.zeros(n, dtype="float64")

    if w_phase != 0.0 and "n_tags_favor_home" in master.columns and "n_tags_favor_away" in master.columns:
        nh = master["n_tags_favor_home"].to_numpy(dtype="float64")
        na = master["n_tags_favor_away"].to_numpy(dtype="float64")
        with np.errstate(invalid="ignore"):
            bal = np.clip(nh - na, -2.0, 2.0)
        use_phase = (~valid) & (half != 0) & np.isfinite(bal)
        out += np.where(use_phase, w_phase * bal, 0.0)

    if w_sp != 0.0 and "is_set_piece" in master.columns and "sp_team" in master.columns:
        tau_sp = float(params.get("tau_sp_team", DEFAULT_TAU_SP_TEAM))
        is_sp = master["is_set_piece"].to_numpy(dtype=bool)
        sp_team_raw = master["sp_team"].astype(str).str.strip().str.upper().to_numpy()
        sp_sign = np.where(sp_team_raw == "H", 1.0, np.where(sp_team_raw == "A", -1.0, 0.0))
        since_sp = (master["frames_since_sp_end"].to_numpy(dtype="float64")
                    if "frames_since_sp_end" in master.columns else np.full(n, np.nan))

        sp_term = np.zeros(n, dtype="float64")
        for start, stop in _half_runs(half):
            last_sign = 0.0
            for idx in range(start, stop):
                if is_sp[idx]:
                    last_sign = sp_sign[idx]
                    sp_term[idx] = last_sign
                elif last_sign != 0.0 and np.isfinite(since_sp[idx]) and since_sp[idx] >= 0.0:
                    sp_term[idx] = last_sign * np.exp(-since_sp[idx] / tau_sp)
        out += w_sp * sp_term

    out = np.where(half != 0, out, 0.0) + 0.0
    return out


def build_team_calibration_features(
    master: pd.DataFrame,
    params: Mapping,
    *,
    include_bridge: bool = True,
    include_touch: bool = True,
    taus: Sequence[int | float] | None = None,
) -> pd.DataFrame:
    """Compute all candidate features for ``ball`` and ``gap`` calibration models in memory."""
    require_columns(master, ("d_diff", "half"))
    ev = build_control_features(master, dict(params))
    coll = build_collective_features(master, dict(params))

    half = master["half"].to_numpy()
    valid = ev["ev_ball_valid"].to_numpy() & (half != 0)
    d_diff = master["d_diff"].to_numpy(dtype="float64")

    if "pressure_home" in master.columns and "pressure_away" in master.columns:
        p_diff = (master["pressure_home"].to_numpy(dtype="float64")
                  - master["pressure_away"].to_numpy(dtype="float64"))
    else:
        d_h = master["d_near_home"].to_numpy(dtype="float64")
        d_a = master["d_near_away"].to_numpy(dtype="float64")
        p_diff = np.exp(-d_h / 3.0) - np.exp(-d_a / 3.0)

    if "n_home_r3" in master.columns and "n_away_r3" in master.columns:
        r3_diff = (master["n_home_r3"].to_numpy(dtype="float64")
                   - master["n_away_r3"].to_numpy(dtype="float64"))
    else:
        r3_diff = (master["n_home_r2"].to_numpy(dtype="float64")
                   - master["n_away_r2"].to_numpy(dtype="float64"))

    with np.errstate(invalid="ignore"):
        f1 = np.where(valid, -np.tanh(d_diff / 2.0), np.nan)
        f2 = np.where(valid, -np.tanh(d_diff / 6.0), np.nan)
        f3 = f1 * ev["ev_prox_w"].to_numpy(dtype="float64")
        f4 = f1 * ev["ev_rel_w"].to_numpy(dtype="float64")
        f5 = f1 * ev["ev_ball_conf"].to_numpy(dtype="float64")
        f6 = f1 * (1.0 - ev["ev_contest"].to_numpy(dtype="float64"))
        f7 = np.where(valid, p_diff, np.nan)
        f8 = np.where(valid, r3_diff, np.nan)

    data = {
        "ev_ball_valid": valid,
        "f1": f1.astype(np.float32),
        "f2": f2.astype(np.float32),
        "f3": f3.astype(np.float32),
        "f4": f4.astype(np.float32),
        "f5": f5.astype(np.float32),
        "f6": f6.astype(np.float32),
        "f7": f7.astype(np.float32),
        "f8": f8.astype(np.float32),
        "ev_coll_adv": coll["ev_coll_adv"].to_numpy(dtype=np.float32),
        "ev_coll_spread": coll["ev_coll_spread"].to_numpy(dtype=np.float32),
        "ev_coll_speed": coll["ev_coll_speed"].to_numpy(dtype=np.float32),
    }
    out = pd.DataFrame(data, index=master.index)

    if include_touch:
        out["f9"] = build_touch_feature(master, params, ev=ev)
    if include_bridge:
        bridge_df = build_pass_bridge_features(master, params, ev=ev, taus=taus)
        for col in bridge_df.columns:
            out[col] = bridge_df[col]

    return out


@lru_cache(maxsize=32)
def _load_calibrated_yaml(path_str: str, mtime_ns: int) -> dict:
    del mtime_ns
    with open(path_str, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or "ball" not in data or "gap" not in data:
        raise ValueError("invalid team calibration file: {}".format(path_str))
    return data


def load_team_calibration(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError("team calibration file not found: {}".format(p))
    return _load_calibrated_yaml(str(p.resolve()), p.stat().st_mtime_ns)


def _select_fold_block(section: Mapping, holdout: str | None) -> Mapping:
    if holdout is None:
        return section["all"]
    folds = section.get("folds", {})
    if holdout not in folds:
        raise KeyError("holdout {!r} not found in calibration folds {}".format(
            holdout, sorted(folds)))
    return folds[holdout]


def _score_linear_section(
    feats_df: pd.DataFrame,
    section: Mapping,
    holdout: str | None,
) -> np.ndarray:
    """Explicit float64 sum of ``w * f / scale`` with NaN -> 0 contribution (rules 4 and 6)."""
    block = _select_fold_block(section, holdout)
    feat_names = list(section["features"])
    scales = block["scales"]
    weights = block["weights"]

    if not feat_names:
        return np.zeros(len(feats_df), dtype="float64")

    cols = []
    eff_w = np.empty(len(feat_names), dtype="float64")
    for idx, name in enumerate(feat_names):
        raw = feats_df[name].to_numpy(dtype="float64")
        cols.append(np.where(np.isfinite(raw), raw, 0.0))
        scale = float(scales[name])
        w = float(weights[name])
        eff_w[idx] = w / scale if scale > 0.0 else 0.0

    X = np.column_stack(cols)
    return (X * eff_w).sum(axis=1)


@register_team("calibrated")
def team_logit_calibrated(master: pd.DataFrame, params: dict, *, holdout=None) -> np.ndarray:
    """C2 team_logit, calibrated on IDSSE: float32[n], finite, |t| <= 8, 0 where half == 0."""
    if "calibrated_file" not in params or not params["calibrated_file"]:
        raise ValueError("calibrated team_logit requires params['calibrated_file']")

    calib = load_team_calibration(params["calibrated_file"])
    f_params = {**params, **calib.get("feature_params", {})}

    needed = set(calib["ball"]["features"]) | set(calib["gap"]["features"])
    need_touch = "f9" in needed
    need_bridge = any(name.startswith(("f10_", "f11_")) for name in needed)
    taus = calib.get("feature_params", {}).get("bridge_tau_frames")

    feats = build_team_calibration_features(
        master, f_params, include_bridge=need_bridge, include_touch=need_touch, taus=taus)

    ball_score = _score_linear_section(feats, calib["ball"], holdout)
    gap_score = _score_linear_section(feats, calib["gap"], holdout)

    half = master["half"].to_numpy()
    valid = feats["ev_ball_valid"].to_numpy()
    t = np.where(valid, ball_score, gap_score)
    t = t + team_tag_prior(master, f_params)
    t = np.clip(t, -TEAM_LOGIT_MAX, TEAM_LOGIT_MAX)
    t = np.where((half != 0) & np.isfinite(t), t, 0.0) + 0.0
    return t.astype(np.float32)


__all__ = [
    "BALL_BASE_FEATURES",
    "GAP_BASE_FEATURES",
    "ball_team_logit",
    "build_pass_bridge_features",
    "build_team_calibration_features",
    "build_touch_feature",
    "load_team_calibration",
    "team_logit_calibrated",
    "team_logit_rules_v1",
    "team_tag_prior",
    "touch_components",
]