"""team_logit v1, rule based (Appendix A.1 of the execution plan).

    t = -kappa * tanh(d_diff / sigma_d) * ev_prox_w * ev_rel_w * ev_ball_conf * contest
    contest = contest_factor if ev_contest else 1
    t = clip(t, -kappa, +kappa);  t = 0 where the ball is not valid

Sign convention (the most error-prone line of the project): d_diff = d_near_home - d_near_away,
so home closer -> d_diff < 0 -> t > 0 favours H; away closer -> t < 0 favours A. t == 0 means "no
evidence", and the decoder then keeps the previous state (the rule for a loose ball: the last team
that had control keeps possession).

Contract C2: float32[n], finite, |t| <= kappa. t is 0 (never NaN) wherever any input is missing:
no valid ball, or d_diff not available (a team without tracked players). This is evidence, not a
fill of the feature: the master columns stay NaN. Frame by frame: no window, no delay.
Nothing here is tuned; the parameters are the priors of Appendix A.1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .control import build_control_features
from .team_logit_params import TeamLogitParams


def team_logit_v1(master: pd.DataFrame, params: TeamLogitParams) -> np.ndarray:
    if "d_diff" not in master.columns:
        raise ValueError("master is missing column 'd_diff'")
    ev = build_control_features(master, params)
    d_diff = master["d_diff"].to_numpy(dtype="float64")

    contest = np.where(ev["ev_contest"].to_numpy(), params.contest_factor, 1.0)
    with np.errstate(invalid="ignore"):                  # NaN inputs are handled explicitly below
        t = (-params.kappa * np.tanh(d_diff / params.sigma_d) * ev["ev_prox_w"].to_numpy()
             * ev["ev_rel_w"].to_numpy() * ev["ev_ball_conf"].to_numpy() * contest)
    t = np.clip(t, -params.kappa, params.kappa)
    t = np.where(np.isfinite(t) & (ev["ev_ball_conf"].to_numpy() > 0), t, 0.0)
    t = t + 0.0                                          # -0.0 -> +0.0
    return t.astype(np.float32)
