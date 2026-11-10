"""Synthetic tests for C3 emission and DecoderParams (task F2-01)."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from modeling.decode.emission import DEFAULT_NEG, logits_to_logscores
from modeling.decode.params import DecoderParams, _p_stay_from_mean
from tests.modeling_fixtures import _DECODER


def _decoder_cfg(**overrides) -> dict:
    cfg = copy.deepcopy(_DECODER)
    cfg.update(overrides)
    return cfg


# -- logits_to_logscores ------------------------------------------------------------------------


def test_zero_logits_give_zero_scores():
    n = 5
    t = np.zeros(n, dtype=np.float64)
    d = np.zeros(n, dtype=np.float64)
    force = np.zeros(n, dtype=bool)
    scores = logits_to_logscores(t, d, lam_team=0.2, lam_dead=0.2, force_d=force)
    assert scores.dtype == np.float64
    assert scores.shape == (n, 3)
    np.testing.assert_allclose(scores, 0.0, atol=0.0, rtol=0.0)


def test_negating_team_logit_swaps_h_and_a():
    rng = np.random.default_rng(0)
    t = rng.uniform(-4.0, 4.0, size=20)
    d = rng.uniform(-2.0, 2.0, size=20)
    force = np.zeros(20, dtype=bool)
    lam = 0.35
    s = logits_to_logscores(t, d, lam, 0.1, force)
    s_flip = logits_to_logscores(-t, d, lam, 0.1, force)
    np.testing.assert_allclose(s[:, 0], s_flip[:, 1], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(s[:, 1], s_flip[:, 0], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(s[:, 2], s_flip[:, 2], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(s[:, 0] - s[:, 1], lam * t, rtol=0.0, atol=1e-12)


def test_large_logits_are_finite():
    t = np.array([1e4, -1e4, 1e4], dtype=np.float64)
    d = np.array([1e4, -1e4, 0.0], dtype=np.float64)
    force = np.zeros(3, dtype=bool)
    scores = logits_to_logscores(t, d, 1.0, 1.0, force)
    assert np.isfinite(scores).all()


def test_force_d_sets_alive_to_neg_and_dead_to_zero():
    t = np.array([2.0, -1.0, 0.5], dtype=np.float64)
    d = np.array([1.0, 0.0, -0.5], dtype=np.float64)
    force = np.array([False, True, False])
    neg = -1.0e6
    scores = logits_to_logscores(t, d, 0.2, 0.3, force, neg=neg)
    assert scores[1, 0] == neg
    assert scores[1, 1] == neg
    assert scores[1, 2] == 0.0
    # Unforced frame keeps the C3 formula.
    expected_h = 0.2 * (np.log(2.0) + (-np.logaddexp(0.0, -2.0)))
    np.testing.assert_allclose(scores[0, 0], expected_h, rtol=0.0, atol=1e-12)


def test_strong_team_evidence_caps_alive_push():
    # |t| → ∞ ⇒ logσ(t) → 0 and logσ(−t) → −∞, so S_H → lam·log2, S_A → −∞.
    t = np.array([1e4], dtype=np.float64)
    d = np.zeros(1, dtype=np.float64)
    scores = logits_to_logscores(t, d, 0.5, 0.2, np.array([False]))
    np.testing.assert_allclose(scores[0, 0], 0.5 * np.log(2.0), rtol=0.0, atol=1e-12)
    assert scores[0, 1] < -1e3


# -- DecoderParams ------------------------------------------------------------------------------


def test_from_config_p_stay_closed_form():
    cfg = _decoder_cfg()
    params = DecoderParams.from_config(cfg, fps=25)
    assert params.lam_team == 0.2
    assert params.lam_dead == 0.2
    assert params.neg == DEFAULT_NEG
    for s, name in enumerate(("H", "A", "D")):
        mean_f = cfg["mean_duration_s"][name] * 25.0
        m = cfg["min_dwell_frames"][name]
        expected = _p_stay_from_mean(mean_f, m)
        np.testing.assert_allclose(params.p_stay[s], expected, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            params.mean_duration_frames[s], mean_f, rtol=0.0, atol=0.0)
        assert int(params.min_dwell[s]) == m
    np.testing.assert_allclose(params.exit_split[0, 0], 0.0)
    np.testing.assert_allclose(params.exit_split[0, 1:], 0.5)


def test_from_config_rejects_mean_not_exceeding_dwell():
    cfg = _decoder_cfg()
    cfg["mean_duration_s"] = {"H": 0.1, "A": 15.0, "D": 20.0}  # 2.5 frames < 12
    with pytest.raises(ValueError, match="must exceed"):
        DecoderParams.from_config(cfg, fps=25)


def test_p_stay_monte_carlo_mean_duration():
    """Simulate terminal geometric stays; mean length within 2% of the target."""
    mean_frames = 100.0
    min_dwell = 20
    p = _p_stay_from_mean(mean_frames, min_dwell)
    rng = np.random.default_rng(42)
    n_trials = 20_000
    # Length = (min_dwell - 1) + Geo(1-p) with mean 1/(1-p), support starting at 1.
    extra = rng.geometric(1.0 - p, size=n_trials)
    lengths = (min_dwell - 1) + extra
    np.testing.assert_allclose(lengths.mean(), mean_frames, rtol=0.02, atol=0.0)


def _decoder_cfg(**exit_rows):
    rows = {"H": {"A": 0.5, "D": 0.5}, "A": {"H": 0.5, "D": 0.5}, "D": {"H": 0.5, "A": 0.5}}
    rows.update(exit_rows)
    return {"lam_team": 0.2, "lam_dead": 0.2,
            "mean_duration_s": {"H": 15.0, "A": 15.0, "D": 20.0},
            "exit_split": rows,
            "min_dwell_frames": {"H": 12, "A": 12, "D": 25}}


def test_decoder_params_arrays_are_read_only():
    params = DecoderParams.from_config(_decoder_cfg())
    for name in ("mean_duration_frames", "exit_split", "min_dwell", "p_stay",
                 "log_p_stay", "log_exit"):
        with pytest.raises(ValueError):
            getattr(params, name)[0] = 0


def test_decoder_params_reject_exit_split_outside_unit_interval():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        DecoderParams.from_config(_decoder_cfg(H={"A": 1.2, "D": -0.2}))
