"""Tests for aggregated error analysis on synthetic data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation import possession_metrics as pm
from modeling.scoring.error_analysis import (
    DURATION_BINS,
    FRAME_CATEGORIES,
    aggregate_error_analysis,
    analyze_match_errors,
    render_error_analysis_report,
)
from tests.modeling_fixtures import make_master, team_params


def _synthetic_error_match() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Build a 450-frame match (25 fps) with known visible/short-gap/long-gap and spell durations.

    Layout (25 fps):
    - Frames 0..49 (2.0 s, 50 frames): D (dead ball)
    - Frames 50..89 (1.6 s, 40 frames, < 2 s): H spell #1 (missed -> predicted as A)
    - Frames 90..214 (5.0 s, 125 frames, 2-10 s): A spell #2 (matched -> predicted as A)
    - Frames 215..224 (10 frames): halftime gap (half == 0, label X)
    - Frames 225..500 (11.0 s, 275 frames, > 10 s): H spell #3 (matched -> predicted as H)
    """
    n = 500
    half = np.ones(n, dtype=np.int8)
    half[215:225] = 0
    half[225:] = 2

    gt = np.full(n, pm.H, dtype=np.int8)
    gt[0:50] = pm.D
    gt[50:90] = pm.H
    gt[90:215] = pm.A
    gt[215:225] = pm.X
    gt[225:500] = pm.H

    # Prediction: misses the short H spell (50..89) by calling it A; gets everything else right
    pred = gt.copy()
    pred[50:90] = pm.A

    ball_obs = np.ones(n, dtype=float)
    gap_len = np.zeros(n, dtype=float)
    # Put a 5-frame short gap at 60..65 and a 20-frame long gap at 70..90 inside the missed spell
    ball_obs[60:65] = 0.0
    gap_len[60:65] = 5.0
    ball_obs[70:90] = 0.0
    gap_len[70:90] = 20.0
    ball_obs[215:225] = 0.0
    gap_len[215:225] = np.nan

    master = make_master(n, half=half, ball_observed=ball_obs, ball_filled=0.0, gap_length=gap_len)

    # Team logit inside the missed H spell (50..90):
    # - 50..70 (20 frames): near zero (0.1)
    # - 70..90 (20 frames): wrong sign (-2.0, favouring A instead of H)
    t_logit = np.where(gt == pm.H, 2.0, np.where(gt == pm.A, -2.0, 0.0)).astype(np.float64)
    t_logit[50:70] = 0.1
    t_logit[70:90] = -2.0
    t_logit[half == 0] = 0.0

    return master, gt, pred, t_logit


def test_frame_categories_and_missed_possession_diagnostics():
    master, gt, pred, t_logit = _synthetic_error_match()
    res = analyze_match_errors(master, gt, pred, t_logit, long_gap_frames=12, fps=25.0)

    assert set(res["frame_categories"]) == set(FRAME_CATEGORIES)
    fc = res["frame_categories"]
    # Total graded frames = 500 - 10 (half==0) = 490; errors = 40 frames (50..89)
    assert fc["all_graded"]["n_frames"] == 490
    assert fc["all_graded"]["n_errors"] == 40

    # Inside the 40 error frames: 15 visible, 5 short_gap, 20 long_gap
    assert fc["short_gap"]["n_frames"] == 5 and fc["short_gap"]["n_errors"] == 5
    assert fc["long_gap"]["n_frames"] == 20 and fc["long_gap"]["n_errors"] == 20
    assert fc["ball_visible"]["n_errors"] == 15

    # Restart is at frame 50 (end of D 0..50); 3 s = 75 frames -> covers 50..125, so all 40 errors are inside
    assert fc["near_restart_3s"]["n_errors"] == 40
    # Switch between H(50..90) and A(90..215) is at frame 90; +-2 s = +-50 frames -> covers 40..140
    assert fc["near_possession_switch_2s"]["n_errors"] == 40

    sb = res["spell_bins"]
    assert set(DURATION_BINS).issubset(set(sb))
    assert sb["lt_2s"]["n_gt"] == 1 and sb["lt_2s"]["n_missed"] == 1
    assert sb["2s_to_10s"]["n_gt"] == 1 and sb["2s_to_10s"]["n_missed"] == 0
    assert sb["gt_10s"]["n_gt"] == 1 and sb["gt_10s"]["n_missed"] == 0

    # Missed spell diagnostics (40 frames: 15 visible, 25 gap; 20 near-zero, 20 wrong-sign)
    assert np.isclose(sb["lt_2s"]["vis_frac"], 15.0 / 40.0)
    assert np.isclose(sb["lt_2s"]["gap_frac"], 25.0 / 40.0)
    assert np.isclose(sb["lt_2s"]["near_zero_frac"], 20.0 / 40.0)
    assert np.isclose(sb["lt_2s"]["wrong_sign_frac"], 20.0 / 40.0)
    assert np.isclose(sb["lt_2s"]["pred_opp_frac"], 1.0)

    # Predicted spells: A(50..215) and H(225..500) -> 2 predicted vs 3 ground-truth
    assert res["metrics"]["n_gt_spells"] == 3
    assert res["metrics"]["n_pred_spells"] == 2
    assert np.isclose(res["metrics"]["spell_count_ratio"], 2.0 / 3.0)


def test_switch_and_restart_windows_never_cross_half():
    master, gt, pred, t_logit = _synthetic_error_match()
    # Put a D segment right at the end of half 1 (frames 200..215)
    gt2 = gt.copy()
    gt2[200:215] = pm.D
    pred2 = gt2.copy()
    # Make an error at the start of half 2 (frames 225..235)
    pred2[225:235] = pm.A

    res = analyze_match_errors(master, gt2, pred2, t_logit, long_gap_frames=12, fps=25.0)
    # The D at the end of half 1 must not trigger a restart window in half 2
    assert res["frame_categories"]["near_restart_3s"]["n_errors"] == 0


def test_aggregate_and_render_markdown():
    master, gt, pred, t_logit = _synthetic_error_match()
    m1 = analyze_match_errors(master, gt, pred, t_logit, long_gap_frames=12, fps=25.0)
    agg = aggregate_error_analysis({"m1": m1, "m2": m1})

    assert agg["summary"]["total_gt_spells"] == 6
    assert agg["summary"]["total_pred_spells"] == 4
    assert np.isclose(agg["summary"]["pooled_spell_count_ratio"], 4.0 / 6.0)

    cfg = {
        "name": "test_v1",
        "pipeline": "hmm",
        "ball_source": "tactical",
        "team": {"kind": "rules_v1", "params": team_params(), "calibrated_file": None},
    }
    md = render_error_analysis_report(agg, cfg, "default", extra_variants={"rules_v1": agg})
    assert "possession count ratio (pred / true)" in md
    assert "near_possession_switch_2s" in md
    assert "near_restart_3s" in md
    assert "< 2 s" in md and "2-10 s" in md and "> 10 s" in md