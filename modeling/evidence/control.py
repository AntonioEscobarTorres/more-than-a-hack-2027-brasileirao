"""Control features (task F1-01), computed in memory from the master table.

``params`` is ``cfg["team"]["params"]`` of a config read by :func:`modeling.config.load_config`.

NaN means "no evidence" and is never replaced by 0: where a column has a defined value in the
absence of an input (``ev_ball_conf`` and ``ev_rel_w``), that value is set explicitly from the
NaN pattern. Every column is computed frame by frame: no window, so nothing crosses a half.

Columns (same index as the master; float32 unless noted):
    ev_ball_valid  bool; ball_x and ball_y are finite
    ev_ball_conf   1.0 if ball_observed == 1; ball_conf_filled if ball_filled == 1; 0.0 otherwise
    ev_dmin        fmin(d_near_home, d_near_away); NaN only if both are NaN
    ev_prox_w      exp(-max(0, ev_dmin - r_ctrl) / l_prox); NaN if ev_dmin is NaN
    ev_rel_near    speed_rel of the team with the smaller d_near (tie -> home); NaN if unavailable
    ev_rel_w       1 / (1 + (ev_rel_near / v_rel)^2); rel_w_when_nan if ev_rel_near is NaN with a
                   valid ball; NaN without a valid ball
    ev_contest     1.0 if n_home_r2 >= 1 and n_away_r2 >= 1, else 0.0; NaN without a valid ball
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "ball_x", "ball_y", "ball_observed", "ball_filled", "d_near_home", "d_near_away",
    "speed_rel_home", "speed_rel_away", "n_home_r2", "n_away_r2",
)
CONTROL_COLUMNS = ("ev_ball_valid", "ev_ball_conf", "ev_dmin", "ev_prox_w", "ev_rel_near",
                   "ev_rel_w", "ev_contest")


def require_columns(master: pd.DataFrame, columns) -> None:
    """Raise if ``master`` lacks any of ``columns`` (never fall back to a default)."""
    missing = [c for c in columns if c not in master.columns]
    if missing:
        raise ValueError("master is missing columns {}".format(missing))


def _f64(master: pd.DataFrame, name: str) -> np.ndarray:
    return master[name].to_numpy(dtype="float64")


def build_control_features(master: pd.DataFrame, params: dict) -> pd.DataFrame:
    require_columns(master, REQUIRED_COLUMNS)
    r_ctrl, l_prox, v_rel = params["r_ctrl"], params["l_prox"], params["v_rel"]
    conf_filled, rel_w_when_nan = params["ball_conf_filled"], params["rel_w_when_nan"]

    valid = np.isfinite(_f64(master, "ball_x")) & np.isfinite(_f64(master, "ball_y"))

    observed = _f64(master, "ball_observed") == 1
    filled = _f64(master, "ball_filled") == 1
    ball_conf = np.where(observed, 1.0, np.where(filled, conf_filled, 0.0))

    d_home, d_away = _f64(master, "d_near_home"), _f64(master, "d_near_away")
    dmin = np.fmin(d_home, d_away)                                   # NaN only if both are NaN
    prox_w = np.exp(-np.maximum(0.0, dmin - r_ctrl) / l_prox)        # NaN stays NaN

    home_ok, away_ok = np.isfinite(d_home), np.isfinite(d_away)
    home_closer = home_ok & (~away_ok | (d_home <= d_away))          # tie -> home
    rel_near = np.where(home_closer, _f64(master, "speed_rel_home"),
                        np.where(away_ok, _f64(master, "speed_rel_away"), np.nan))

    rel_w = 1.0 / (1.0 + (rel_near / v_rel) ** 2)
    rel_w = np.where(np.isfinite(rel_near), rel_w, rel_w_when_nan)
    rel_w = np.where(valid, rel_w, np.nan)

    contest = (_f64(master, "n_home_r2") >= 1) & (_f64(master, "n_away_r2") >= 1)
    contest = np.where(valid, contest.astype("float64"), np.nan)

    f32 = np.float32
    return pd.DataFrame({
        "ev_ball_valid": valid,
        "ev_ball_conf": ball_conf.astype(f32),
        "ev_dmin": dmin.astype(f32),
        "ev_prox_w": prox_w.astype(f32),
        "ev_rel_near": rel_near.astype(f32),
        "ev_rel_w": rel_w.astype(f32),
        "ev_contest": contest.astype(f32),
    }, index=master.index)


__all__ = ["CONTROL_COLUMNS", "REQUIRED_COLUMNS", "build_control_features", "require_columns"]
