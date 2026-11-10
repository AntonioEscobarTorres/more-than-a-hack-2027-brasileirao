"""Synthetic-only tests for IDSSE degradation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from modeling import paths
from modeling.config import load_config
from modeling.external.degrade import degrade, validate_degrade_config
from modeling.external.files import ExternalMatch, expected_half


@pytest.fixture
def cfg():
    path = Path(__file__).resolve().parents[1] / "configs" / "degrade_default.yaml"
    config = load_config(path)
    config["player_visible_mean_s"] = 5.0
    return config


@pytest.fixture
def synthetic_match():
    rng = np.random.default_rng(120)
    half = expected_half(1200, 1200)
    frames = half.size
    players = rng.uniform(0.1, 0.9, size=(frames, 46, 3)).astype(np.float32)
    players[:, :, 2] = rng.uniform(0.0, 8.0, size=(frames, 46)).astype(np.float32)
    ball = rng.uniform(0.1, 0.9, size=(frames, 3)).astype(np.float32)
    ball[:, 2] = rng.uniform(0.0, 5.0, frames)
    ball_z = np.full(frames, 0.5, dtype=np.float32)
    labels = rng.integers(0, 3, size=frames, dtype=np.int8)
    gap = half == 0
    players[gap] = np.nan
    ball[gap] = np.nan
    ball_z[gap] = np.nan
    labels[gap] = 3
    slots = np.r_[np.zeros(24, dtype=np.int8), np.ones(22, dtype=np.int8)]
    return ExternalMatch("SYNTHETIC", players, ball, ball_z, labels, half, slots,
                         np.array([105.0, 68.0], dtype=np.float64))


def test_degrade_is_deterministic_and_does_not_mutate(synthetic_match, cfg, tmp_path,
                                                       monkeypatch):
    monkeypatch.setattr(paths, "processed_dir", lambda: tmp_path)
    before = {field: getattr(synthetic_match, field).copy()
              for field in ("players", "ball", "ball_z", "labels", "half",
                            "slot_team_ids", "pitch")}
    first = degrade(synthetic_match, cfg, 42)
    repeated = degrade(synthetic_match, cfg, 42)
    other = degrade(synthetic_match, cfg, 43)
    for field in before:
        actual = getattr(first, field)
        np.testing.assert_equal(actual, getattr(repeated, field))
        np.testing.assert_equal(getattr(synthetic_match, field), before[field])
    np.testing.assert_array_equal(first.labels, synthetic_match.labels)
    assert not np.array_equal(first.ball, other.ball, equal_nan=True)
    assert not np.array_equal(first.players, other.players, equal_nan=True)


def test_degrade_hits_visibility_tracking_and_injection_targets(synthetic_match, cfg,
                                                                 tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "processed_dir", lambda: tmp_path)
    cfg["gap_profile"]["lognormal_median_frames"] = 12.0
    cfg["gap_profile"]["max_frames"] = 40
    result = degrade(synthetic_match, cfg, 8)
    block_frames = result.half != 0
    ball_visible = np.isfinite(result.ball[:, 0]) & np.isfinite(result.ball[:, 1])
    visibility = ball_visible[block_frames].mean()
    assert abs(visibility - cfg["target_ball_visible"]) <= cfg["target_ball_visible_tolerance"]
    assert np.isnan(result.ball[block_frames, 2]).all()

    tracked = np.isfinite(result.players[:, :, 0]).sum(axis=1)
    mean_tracked = tracked[block_frames].mean()
    assert abs(mean_tracked - cfg["target_tracked_mean"]) <= cfg["target_tracked_mean_tolerance"]
    duplicate_fraction = (tracked[block_frames] > cfg["overtracked_threshold"]).mean()
    assert duplicate_fraction >= cfg["dup_frac"] * 0.8
    same_position = np.zeros(result.n_frames, dtype=bool)
    same_position[1:] = (np.isfinite(result.ball[1:, :2]).all(axis=1) &
                         np.isfinite(result.ball[:-1, :2]).all(axis=1) &
                         np.all(result.ball[1:, :2] == result.ball[:-1, :2], axis=1))
    ghost_like = same_position & block_frames
    assert ghost_like.sum() >= cfg["phantom_frac"] * block_frames.sum() * 0.5
    assert np.isnan(result.ball[~block_frames]).all()
    assert np.isnan(result.players[~block_frames]).all()


def test_degrade_config_rejects_unknown_missing_and_invalid_values(cfg):
    invalid = dict(cfg, unexpected_parameter=1)
    with pytest.raises(ValueError, match="unknown keys"):
        validate_degrade_config(invalid)

    invalid = dict(cfg)
    del invalid["p_air"]
    with pytest.raises(ValueError, match="missing keys"):
        validate_degrade_config(invalid)

    invalid = dict(cfg, p_air=1.5)
    with pytest.raises(ValueError, match="between 0 and 1"):
        validate_degrade_config(invalid)
