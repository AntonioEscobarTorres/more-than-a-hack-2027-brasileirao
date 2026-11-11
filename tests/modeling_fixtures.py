"""Synthetic masters and the Appendix A.1 team parameters for the F1 evidence tests.

Nothing here reads the match data: every master is built from literals or a seeded RNG.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from modeling.config import validate_config

PITCH_L = 105.3

# team.params of Appendix A.7, run through the C7 validator so the tests use exactly what a
# config file would give the evidence functions.
_TEAM_PARAMS_RAW = {
    "pitch_l_m": PITCH_L, "pitch_w_m": 68.0, "kappa": 3.0, "sigma_d": 2.0, "r_ctrl": 2.0,
    "l_prox": 3.0, "v_rel": 6.0, "ball_conf_filled": 0.6, "contest_factor": 0.5,
    "rel_w_when_nan": 0.5,
    "collective": {"win_frames": 25, "w_adv": 0.0, "w_spread": 0.0, "w_speed": 0.0},
}


def team_params(**collective) -> dict:
    """Validated team.params; keyword arguments override the collective sub-section."""
    raw = copy.deepcopy(_TEAM_PARAMS_RAW)
    raw["collective"].update(collective)
    cfg = {"name": "t", "pipeline": "hmm", "ball_source": "tactical",
           "team": {"kind": "rules_v1", "params": raw, "calibrated_file": None},
           "dead": _DEAD, "decoder": _DECODER, "postprocess": _POST}
    return validate_config(cfg)["team"]["params"]


_DEAD = {
    "kind": "none",
    "params": {
        "long_gap_frames": 12, "exit_line_m": 3.0, "exit_speed_mps": 0.5, "restart_line_m": 1.5,
        "goal_area_depth_m": 6.0, "goal_area_half_width_m": 9.2, "pitch_w_m": 68.0,
        "air_speed_mps": 8.0, "air_max_frames": 100, "smooth_frames": 25, "rest_speed_mps": 0.3,
        "rest_min_frames": 25, "fast_speed_mps": 6.0, "sp_prox_tau_frames": 250,
        "sp_after_tau_frames": 75, "goal_window_frames": 1500,
        "weights": {"out_of_pitch": 4.0, "gap_exit": 3.0, "gap_air": -1.0, "gap_long": 0.5,
                    "still": 3.0, "speed_high": -2.0, "speed_low": 1.5, "ball_rest": 2.0,
                    "fast_in_pitch": -3.0, "tag_set_piece": 1.0, "tag_sp_prox": 1.0,
                    "tag_sp_after": -1.0, "tag_goal_window": 2.0},
    },
    "calibrated_file": None,
}
_DECODER = {
    "lam_team": 0.2, "lam_dead": 0.2, "mean_duration_s": {"H": 15.0, "A": 15.0, "D": 20.0},
    "exit_split": {"H": {"A": 0.5, "D": 0.5}, "A": {"H": 0.5, "D": 0.5},
                   "D": {"H": 0.5, "A": 0.5}},
    "min_dwell_frames": {"H": 12, "A": 12, "D": 25},
}
_POST = {"midpoint_neutral_switch": False, "boundary_shift_file": None, "d_bridge_frames": 0,
         "min_spell_frames": 0}


def dead_params(**overrides) -> dict:
    """Validated dead.params (Appendix A.2/A.3); keyword arguments override top-level keys."""
    dead = copy.deepcopy(_DEAD)
    dead["params"].update(overrides)
    cfg = {"name": "t", "pipeline": "hmm", "ball_source": "tactical",
           "team": {"kind": "none", "params": copy.deepcopy(_TEAM_PARAMS_RAW),
                    "calibrated_file": None},
           "dead": dead, "decoder": _DECODER, "postprocess": _POST}
    return validate_config(cfg)["dead"]["params"]


def make_master(n: int = 1, **cols) -> pd.DataFrame:
    """Valid observed ball, home 1 m from it and away 5 m, everything inside half 1.

    Any column can be overridden with a scalar or an array. ``d_diff`` follows the distances
    unless given; the ball position is NaN wherever the ball is neither observed nor filled,
    unless ``ball_x``/``ball_y`` are given.
    """
    base = {
        "half": 1, "ball_observed": 1, "ball_filled": 0, "gap_length": 0.0,
        "d_near_home": 1.0, "d_near_away": 5.0, "n_home_r2": 1.0, "n_away_r2": 0.0,
        "speed_rel_home": 1.0, "speed_rel_away": 1.0,
        "centroid_home_x": 40.0, "centroid_away_x": 60.0,
        "spread_home_x": 10.0, "spread_home_y": 8.0, "spread_away_x": 9.0, "spread_away_y": 7.0,
        "mean_speed_home": 3.0, "mean_speed_away": 2.5,
    }
    base.update({k: v for k, v in cols.items() if k not in ("d_diff", "ball_x", "ball_y")})
    df = pd.DataFrame({k: np.broadcast_to(np.asarray(v, dtype="float64"), (n,)).copy()
                       for k, v in base.items()})
    df["half"] = df["half"].astype(np.int8)
    has_ball = (df["ball_observed"] == 1) | (df["ball_filled"] == 1)
    for axis, default in (("ball_x", 50.0), ("ball_y", 30.0)):
        value = cols.get(axis)
        df[axis] = (np.where(has_ball, default, np.nan) if value is None
                    else np.broadcast_to(np.asarray(value, dtype="float64"), (n,)).copy())
    df["d_diff"] = (df["d_near_home"] - df["d_near_away"] if "d_diff" not in cols
                    else np.broadcast_to(np.asarray(cols["d_diff"], dtype="float64"), (n,)).copy())
    return df


def fuzz_master(n: int, seed: int) -> pd.DataFrame:
    """Continuous random distances and speeds (no ties); a mix of observed/filled/no ball."""
    rng = np.random.default_rng(seed)
    kind = rng.integers(0, 3, n)                      # 0 no ball, 1 observed, 2 filled
    m = make_master(
        n, ball_observed=(kind == 1).astype(float), ball_filled=(kind == 2).astype(float),
        d_near_home=rng.uniform(0, 25, n), d_near_away=rng.uniform(0, 25, n),
        n_home_r2=rng.integers(0, 3, n), n_away_r2=rng.integers(0, 3, n),
        speed_rel_home=rng.uniform(0, 25, n), speed_rel_away=rng.uniform(0, 25, n))
    no_ball = kind == 0
    for c in ("d_near_home", "d_near_away", "n_home_r2", "n_away_r2", "speed_rel_home",
              "speed_rel_away", "d_diff"):
        m.loc[no_ball, c] = np.nan                    # SCHEMA.md: NaN without a ball
    return m


def swap_teams(master: pd.DataFrame) -> pd.DataFrame:
    """Exchange home and away: swap every *_home/*_away column, flip d_diff and mirror x.

    Home always attacks towards increasing x, so the swapped match is also mirrored in x.
    """
    def swap(name):
        if "home" in name:
            return name.replace("home", "away")
        if "away" in name:
            return name.replace("away", "home")
        return name

    out = master.rename(columns=swap)
    out["d_diff"] = -out["d_diff"]
    for c in ("ball_x", "centroid_home_x", "centroid_away_x"):
        out[c] = PITCH_L - out[c]
    return out[master.columns]
