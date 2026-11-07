"""Control features, derived ONLY from the master table.

NaN is never replaced by 0. Where a feature has a documented
neutral value (ev_ball_conf = 0 when there is no ball, ev_rel_w = rel_w_missing when the relative
speed is unavailable, ev_contest = False when the counts are NaN) it is derived explicitly from
the NaN pattern, not with fillna.

Every column is computed frame by frame, or from columns the preprocessing already computed per
half (since_last_obs / until_next_obs): there is no window here, so nothing can cross the boundary
of a half and nothing introduces a delay. All tie-breaks are deterministic (ties go to home).

Columns (index = the master's index):
    ev_ball_conf  1.0 if ball_observed == 1; ball_conf_filled if ball_filled == 1; 0.0 otherwise
    ev_dmin       min(d_near_home, d_near_away); one NaN is ignored, both NaN -> NaN
    ev_prox_w     exp(-max(0, ev_dmin - r_ctrl) / l_prox)
    ev_rel_near   speed_rel of the team with the smaller d_near (ties -> home); NaN if unavailable
    ev_rel_w      1 / (1 + (ev_rel_near / v_rel)^2); rel_w_missing when ev_rel_near is NaN
    ev_contest    True if both teams have >= 1 player within 2 m of the ball (n_*_r2)
    ev_since      min(since_last_obs, until_next_obs); one NaN is ignored, both NaN -> NaN
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .params import ControlParams

REQUIRED_COLUMNS = (
    "d_near_home", "d_near_away", "n_home_r2", "n_away_r2", "speed_rel_home", "speed_rel_away",
    "ball_observed", "ball_filled", "since_last_obs", "until_next_obs",
)
CONTROL_COLUMNS = ("ev_ball_conf", "ev_dmin", "ev_prox_w", "ev_rel_near", "ev_rel_w",
                   "ev_contest", "ev_since")


def _f64(master, name):
    return master[name].to_numpy(dtype="float64")


def build_control_features(master: pd.DataFrame, params: ControlParams) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in master.columns]
    if missing:
        raise ValueError("master is missing columns {}".format(missing))

    d_home, d_away = _f64(master, "d_near_home"), _f64(master, "d_near_away")

    observed = _f64(master, "ball_observed") == 1
    filled = _f64(master, "ball_filled") == 1
    ev_ball_conf = np.where(observed, 1.0, np.where(filled, params.ball_conf_filled, 0.0))

    ev_dmin = np.fmin(d_home, d_away)                          # NaN only if both are NaN
    ev_prox_w = np.exp(-np.maximum(0.0, ev_dmin - params.r_ctrl) / params.l_prox)   # NaN stays NaN

    home_ok, away_ok = ~np.isnan(d_home), ~np.isnan(d_away)
    home_closer = home_ok & (~away_ok | (d_home <= d_away))    # ties go to home
    rel_near = np.where(home_closer, _f64(master, "speed_rel_home"), _f64(master, "speed_rel_away"))
    rel_near = np.where(home_ok | away_ok, rel_near, np.nan)
    ev_rel_w = 1.0 / (1.0 + (rel_near / params.v_rel) ** 2)
    ev_rel_w = np.where(np.isnan(rel_near), params.rel_w_missing, ev_rel_w)

    ev_contest = (_f64(master, "n_home_r2") >= 1) & (_f64(master, "n_away_r2") >= 1)   # NaN -> False

    ev_since = np.fmin(_f64(master, "since_last_obs"), _f64(master, "until_next_obs"))

    return pd.DataFrame({
        "ev_ball_conf": ev_ball_conf, "ev_dmin": ev_dmin, "ev_prox_w": ev_prox_w,
        "ev_rel_near": rel_near, "ev_rel_w": ev_rel_w, "ev_contest": ev_contest,
        "ev_since": ev_since,
    }, index=master.index)
