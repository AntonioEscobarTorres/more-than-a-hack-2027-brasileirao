"""Synthetic-only tests for IDSSE degradation."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
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
    # The 2 x 1200-frame fixture is too short for gaps of up to 1500 frames to hit the
    # visibility tolerance; real matches (~70k frames per half) use the default profile.
    config["gap_profile"]["lognormal_median_frames"] = 12.0
    config["gap_profile"]["max_frames"] = 40
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


@pytest.fixture
def realistic_match():
    """11 v 11 random walks (the other slots empty), as in a real tracking feed."""
    rng = np.random.default_rng(5)
    half = expected_half(2500, 2500)
    frames = half.size
    players = np.full((frames, 46, 3), np.nan, dtype=np.float32)
    on = np.r_[np.arange(11), 24 + np.arange(11)]
    start = rng.uniform(0.1, 0.9, size=(22, 2))
    walk = start + np.cumsum(rng.normal(0.0, 0.001, size=(frames, 22, 2)), axis=0)
    players[:, on, :2] = np.clip(walk, 0.02, 0.98)
    players[:, on, 2] = 3.0
    ball = np.full((frames, 3), np.nan, dtype=np.float32)
    ball[:, :2] = np.clip(0.5 + np.cumsum(rng.normal(0.0, 0.003, size=(frames, 2)), axis=0), 0, 1)
    ball[:, 2] = 5.0
    ball_z = np.where(rng.random(frames) < 0.05, 3.0, 0.2).astype(np.float32)
    labels = np.zeros(frames, dtype=np.int8)
    gap = half == 0
    players[gap] = np.nan
    ball[gap] = np.nan
    ball_z[gap] = np.nan
    labels[gap] = 3
    slots = np.r_[np.zeros(24, dtype=np.int8), np.ones(22, dtype=np.int8)]
    return ExternalMatch("REALISTIC", players, ball, ball_z, labels, half, slots,
                         np.array([105.0, 68.0], dtype=np.float64))


def test_default_config_on_realistic_match_hits_targets_with_single_duplicates(
        realistic_match, tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "processed_dir", lambda: tmp_path)
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "degrade_default.yaml")
    cfg["gap_profile"]["max_frames"] = 200  # 2 x 2500 frames: keep single gaps small
    result = degrade(realistic_match, cfg, 11)
    block = result.half != 0
    visible = (np.isfinite(result.ball[:, 0]) & np.isfinite(result.ball[:, 1]))[block].mean()
    assert abs(visible - cfg["target_ball_visible"]) <= cfg["target_ball_visible_tolerance"]
    tracked = np.isfinite(result.players[:, :, 0]).sum(axis=1)[block]
    assert abs(tracked.mean() - cfg["target_tracked_mean"]) <= cfg["target_tracked_mean_tolerance"]
    assert (tracked > cfg["overtracked_threshold"]).mean() >= cfg["dup_frac"]
    # One duplicate per frame at most: the same seed without duplicates differs by <= 1 object.
    no_dup = degrade(realistic_match, dict(cfg, dup_frac=0.0), 11)
    extra = (np.isfinite(result.players[:, :, 0]).sum(axis=1) -
             np.isfinite(no_dup.players[:, :, 0]).sum(axis=1))
    assert extra.min() == 0 and extra.max() == 1
    assert (extra[block] == 1).mean() >= cfg["dup_frac"]


def test_degrade_raises_when_tolerance_is_missed(synthetic_match, cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "processed_dir", lambda: tmp_path)
    strict = copy.deepcopy(cfg)
    strict["target_ball_visible_tolerance"] = 0.0  # 0.593 * 2400 is not an integer
    with pytest.raises(ValueError, match="ball visibility"):
        degrade(synthetic_match, strict, 42)
    strict = copy.deepcopy(cfg)
    strict["target_tracked_mean"] = 20.4999  # not k / 2400 for any integer k
    strict["target_tracked_mean_tolerance"] = 0.0
    with pytest.raises(ValueError, match="tracked mean"):
        degrade(synthetic_match, strict, 42)


def test_ball_gap_profile_counts_raw_gaps_not_interpolated_ball():
    from tools.ball_gap_profile import gap_lengths_by_half

    half = np.r_[np.ones(10), np.zeros(3), np.full(10, 2)].astype(np.int8)
    observed = np.ones(half.size, dtype=np.int8)
    jump_removed = np.zeros(half.size, dtype=np.int8)
    observed[[2, 3]] = 0            # short gap (interpolated into ball_x by preprocessing)
    observed[[6]] = 0
    jump_removed[[6]] = 1           # jump-filtered frame: the ball was there, not a gap
    observed[[8, 9]] = 0            # gap that runs into halftime ...
    observed[[13, 14, 15]] = 0      # ... and a separate one at the start of half 2
    master = pd.DataFrame({"half": half, "ball_observed": observed,
                           "ball_jump_removed": jump_removed,
                           "ball_x": np.where(half > 0, 0.5, np.nan)})  # fully filled
    assert sorted(gap_lengths_by_half(master)) == [2, 2, 3]
