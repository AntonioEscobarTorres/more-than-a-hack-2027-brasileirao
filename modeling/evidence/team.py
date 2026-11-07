"""Team-possession evidence ``team_logit`` (contract C2), variant ``rules_v1`` (F1-02, F1-04).

Ball term (Appendix A.1, F1-02):

    t = -kappa * tanh(d_diff / sigma_d) * ev_prox_w * ev_rel_w * ev_ball_conf * contest
    contest = contest_factor if ev_contest == 1 else 1
    t = clip(t, -kappa, +kappa);  t = 0 where the ball is not valid, d_diff is NaN or half == 0

Sign convention: d_diff = d_near_home - d_near_away, so home closer -> d_diff < 0 -> t > 0
(favours H). t == 0 means "no evidence". The ball term is frame by frame (no window, no delay).

Collective term (F1-04, off by default): where the ball is not valid or the ball gap is longer
than the interpolation limit (gap_length > MAX_GAP_FRAMES), add

    w_adv * ev_coll_adv + w_spread * ev_coll_spread + w_speed * ev_coll_speed

with each term 0 where its feature is NaN, then clip to +-kappa again. With the three weights at
0 the output is bit-identical to the ball term alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from preprocessing.params import MAX_GAP_FRAMES

from ..variants import register_team
from .collective import build_collective_features
from .control import build_control_features, require_columns

COLLECTIVE_WEIGHTS = (("w_adv", "ev_coll_adv"), ("w_spread", "ev_coll_spread"),
                      ("w_speed", "ev_coll_speed"))


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


__all__ = ["ball_team_logit", "team_logit_rules_v1"]
