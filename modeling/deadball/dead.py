"""Dead-ball evidence ``dead_logit`` (contract C2), variant ``rules_v1`` (F3-02).

``params`` is ``cfg["dead"]["params"]`` of a config read by :func:`modeling.config.load_config`
(the same dict :func:`modeling.deadball.features.build_deadball_features` takes).

The logit is a log-likelihood ratio, ``log p(obs | D) - log p(obs | alive)``: positive favours a
dead ball, negative favours live play and 0 means "no evidence". It is the sum of the terms of
Appendix A.3, each weighted by ``params["weights"][<term>]``:

    out_of_pitch     weight                                   if out_of_pitch == 1
    gap_exit         weight                                   if db_gap_kind == GAP_EXIT
    gap_air          weight                                   if db_gap_kind == GAP_AIR
    gap_long         weight                                   if db_gap_kind == GAP_LONG
    still            weight * clip((db_still_frac - 0.3) / 0.4, 0, 1)
    speed_high       weight * clip((db_mean_speed - 2.5) / 2.0, 0, 1)
    speed_low        weight * clip((1.5 - db_mean_speed) / 1.0, 0, 1)
    ball_rest        weight                                   if db_ball_rest == 1
    fast_in_pitch    weight                                   if db_fast_in_pitch == 1
    tag_set_piece    weight                                   if is_set_piece
    tag_sp_prox      weight * db_sp_prox
    tag_sp_after     weight * db_sp_after
    tag_goal_window  weight                                   if db_goal_window == 1

A term whose feature is NaN is exactly 0, and every term is 0 where ``half == 0``. The sum is
clipped to +-``DEAD_LOGIT_MAX`` and returned as float32.

The four ``tag_*`` terms are the Smart Tagging part. They are also available on their own as
:func:`dead_tag_prior`, which the calibrated variant (F3-04) adds to its fitted logit: the
external matches have no tags, so the tag weights cannot be fitted and stay at this prior.

The pre-kickoff segment is not part of the logit: it is the ``force_d`` mask (C4).

Variant ``calibrated`` (F3-04): ``d = clip(d_fit + dead_tag_prior, +-DEAD_LOGIT_MAX)``, where
``d_fit`` is the logistic model fitted on the external IDSSE masters
(:mod:`modeling.calibration.dead`, file ``params["calibrated_file"]``). The tag weights stay at the
prior of Appendix A.3 because the external matches have no tags. ``holdout`` selects the model
fitted without that match.

Variant ``gbdt`` (F3-08): the same sum, ``d = clip(d_fit + dead_tag_prior, +-DEAD_LOGIT_MAX)``, with
``d_fit`` from a gradient-boosted tree model fitted on the same data
(:mod:`modeling.deadball.dead_gbdt`, :mod:`modeling.calibration.dead_gbdt`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..calibration.dead import calibrated_d_fit, load_calibration
from ..contracts import DEAD_LOGIT_MAX
from ..variants import register_dead
from .features import GAP_AIR, GAP_EXIT, GAP_LONG, build_deadball_features

# Term names, in the order they are summed. They are the keys of ``params["weights"]``.
KINEMATIC_TERMS = ("out_of_pitch", "gap_exit", "gap_air", "gap_long", "still", "speed_high",
                   "speed_low", "ball_rest", "fast_in_pitch")
TAG_TERMS = ("tag_set_piece", "tag_sp_prox", "tag_sp_after", "tag_goal_window")
RULES_V1_TERMS = KINEMATIC_TERMS + TAG_TERMS

# Shape of the three ramp terms of Appendix A.3, as (start, width): the ramp is 0 at ``start``
# and reaches 1 at ``start + width`` (``start - width`` for the decreasing one). Together with
# the weights they define the rules_v1 variant; the config schema (A.7) carries only the weights.
STILL_RAMP = (0.3, 0.4)            # db_still_frac: 0 up to 0.3, 1 from 0.7
SPEED_HIGH_RAMP = (2.5, 2.0)       # db_mean_speed (m/s): 0 up to 2.5, 1 from 4.5
SPEED_LOW_RAMP = (1.5, 1.0)        # db_mean_speed (m/s), decreasing: 0 from 1.5, 1 up to 0.5

REQUIRED_COLUMNS = ("half", "out_of_pitch", "is_set_piece")


def _ramp_up(x: np.ndarray, start: float, width: float) -> np.ndarray:
    return np.clip((x - start) / width, 0.0, 1.0)          # NaN stays NaN


def _ramp_down(x: np.ndarray, start: float, width: float) -> np.ndarray:
    return np.clip((start - x) / width, 0.0, 1.0)          # NaN stays NaN


def _scaled(weight: float, x: np.ndarray) -> np.ndarray:
    """weight * x, and exactly 0 where x is NaN (rule 4)."""
    return np.where(np.isfinite(x), weight * x, 0.0)


def _flag(weight: float, on: np.ndarray) -> np.ndarray:
    """weight where the bool array ``on`` is True, 0 elsewhere (a NaN feature compares False)."""
    return np.where(on, weight, 0.0)


def dead_logit_terms(master: pd.DataFrame, params: dict) -> dict:
    """Every term of the rules_v1 sum, before clipping: ``{term: float64[n]}``.

    The keys are ``RULES_V1_TERMS``, in that order. Each array is 0 where its feature is NaN and
    where ``half == 0``.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in master.columns]
    if missing:
        raise ValueError("master is missing columns {}".format(missing))
    db = build_deadball_features(master, params)
    w = params["weights"]

    def col(name):
        return db[name].to_numpy(dtype="float64")

    gap_kind = db["db_gap_kind"].to_numpy()
    out_of_pitch = master["out_of_pitch"].to_numpy(dtype="float64")
    set_piece = master["is_set_piece"].to_numpy(dtype="float64", na_value=np.nan)
    still, mean_speed = col("db_still_frac"), col("db_mean_speed")

    terms = {
        "out_of_pitch": _flag(w["out_of_pitch"], out_of_pitch == 1),
        "gap_exit": _flag(w["gap_exit"], gap_kind == GAP_EXIT),
        "gap_air": _flag(w["gap_air"], gap_kind == GAP_AIR),
        "gap_long": _flag(w["gap_long"], gap_kind == GAP_LONG),
        "still": _scaled(w["still"], _ramp_up(still, *STILL_RAMP)),
        "speed_high": _scaled(w["speed_high"], _ramp_up(mean_speed, *SPEED_HIGH_RAMP)),
        "speed_low": _scaled(w["speed_low"], _ramp_down(mean_speed, *SPEED_LOW_RAMP)),
        "ball_rest": _flag(w["ball_rest"], col("db_ball_rest") == 1),
        "fast_in_pitch": _flag(w["fast_in_pitch"], col("db_fast_in_pitch") == 1),
        "tag_set_piece": _flag(w["tag_set_piece"], set_piece == 1),
        "tag_sp_prox": _scaled(w["tag_sp_prox"], col("db_sp_prox")),
        "tag_sp_after": _scaled(w["tag_sp_after"], col("db_sp_after")),
        "tag_goal_window": _flag(w["tag_goal_window"], col("db_goal_window") == 1),
    }
    in_half = master["half"].to_numpy() != 0
    # + 0.0 turns -0.0 (a negative weight times 0) into +0.0.
    return {name: np.where(in_half, terms[name], 0.0) + 0.0 for name in RULES_V1_TERMS}


def _sum_terms(terms: dict, names) -> np.ndarray:
    """Explicit float64 sum, always in the order of ``names`` (determinism)."""
    total = np.zeros(len(next(iter(terms.values()))), dtype="float64")
    for name in names:
        total = total + terms[name]
    return total


def dead_tag_prior(master: pd.DataFrame, params: dict) -> np.ndarray:
    """Sum of the Smart Tagging terms (``TAG_TERMS``), float64[n], not clipped.

    0 where ``half == 0`` and on a master without tags, such as the external IDSSE masters.
    """
    return _sum_terms(dead_logit_terms(master, params), TAG_TERMS)


@register_dead("rules_v1")
def dead_logit_rules_v1(master: pd.DataFrame, params: dict, *, holdout=None) -> np.ndarray:
    """C2 dead_logit, rules_v1: float32[n], finite, |d| <= DEAD_LOGIT_MAX, 0 where half == 0.

    ``holdout`` is ignored: this variant has no fitted parameters.
    """
    terms = dead_logit_terms(master, params)
    d = _sum_terms(terms, KINEMATIC_TERMS) + _sum_terms(terms, TAG_TERMS)
    d = np.clip(d, -DEAD_LOGIT_MAX, DEAD_LOGIT_MAX) + 0.0
    return d.astype(np.float32)


@register_dead("calibrated")
def dead_logit_calibrated(master: pd.DataFrame, params: dict, *, holdout=None) -> np.ndarray:
    """C2 dead_logit, calibrated (F3-04): float32[n], |d| <= DEAD_LOGIT_MAX, 0 where half == 0.

    ``params`` is ``cfg["dead"]["params"]`` plus ``calibrated_file`` (absolute path, injected by
    ``modeling.pipeline.variant_params``). The features are rebuilt with these ``params``, which
    must match the ones of the fit. ``holdout`` selects ``folds[holdout]``; without it the model
    ``all`` is used. The sum is explicit, in float64.
    """
    path = params.get("calibrated_file")
    if path is None:
        raise ValueError("dead kind 'calibrated' needs params['calibrated_file']")
    missing = [c for c in REQUIRED_COLUMNS if c not in master.columns]
    if missing:
        raise ValueError("master is missing columns {}".format(missing))
    doc = load_calibration(path)
    fitted = calibrated_d_fit(master, params, doc, holdout)
    d = fitted + dead_tag_prior(master, params)
    d = np.clip(d, -DEAD_LOGIT_MAX, DEAD_LOGIT_MAX)
    d = np.where(master["half"].to_numpy() != 0, d, 0.0) + 0.0
    return d.astype(np.float32)


@register_dead("gbdt")
def dead_logit_gbdt(master: pd.DataFrame, params: dict, *, holdout=None) -> np.ndarray:
    """C2 dead_logit, GBDT (F3-08): float32[n], |d| <= DEAD_LOGIT_MAX, 0 where half == 0.

    ``params`` is ``cfg["dead"]["params"]`` plus ``calibrated_file`` (``configs/calib/dead_gbdt.yaml``,
    absolute, injected by ``modeling.pipeline.variant_params``). ``holdout`` selects the fold model
    trained without that match; without it the model ``all`` is used. The tags stay at the prior of
    Appendix A.3, as in the ``calibrated`` kind. The sum is explicit, in float64.
    """
    from .dead_gbdt import gbdt_d_fit

    missing = [c for c in REQUIRED_COLUMNS if c not in master.columns]
    if missing:
        raise ValueError("master is missing columns {}".format(missing))
    fitted = gbdt_d_fit(master, params, holdout)
    d = fitted + dead_tag_prior(master, params)
    d = np.clip(d, -DEAD_LOGIT_MAX, DEAD_LOGIT_MAX)
    d = np.where(master["half"].to_numpy() != 0, d, 0.0) + 0.0
    return d.astype(np.float32)


__all__ = ["KINEMATIC_TERMS", "REQUIRED_COLUMNS", "RULES_V1_TERMS", "SPEED_HIGH_RAMP",
           "SPEED_LOW_RAMP", "STILL_RAMP", "TAG_TERMS", "dead_logit_calibrated", "dead_logit_gbdt",
           "dead_logit_rules_v1", "dead_logit_terms", "dead_tag_prior"]
