"""Unit tests for dead-ball unfreezing in modeling/external/degrade.py ([F4-04] step 1).

All data is synthetic (no $MTAH_DATA_DIR, no ./data). Randomness is seeded through
degrade(..., seed) -> np.random.default_rng(seed).

Frame layout follows contract C9: first half, GAP_FRAMES empty frames (half == 0, label X,
NaN positions), second half. Dead-ball frames use the C0 code LABEL_CODES["D"].
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from modeling.contracts import LABEL_CODES
from modeling.external import degrade as degrade_module
from modeling.external.degrade import degrade, validate_degrade_config
from modeling.external.files import (
    GAP_FRAMES, N_HOME_SLOTS, X_CODE, ExternalMatch, expected_half)
from preprocessing.schema import N_SLOTS


def _find_configs_dir() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "configs"
        if (candidate / "degrade_default.yaml").is_file():
            return candidate
    raise FileNotFoundError("configs/degrade_default.yaml not found above the tests directory")


CONFIGS_DIR = _find_configs_dir()
CONFIG_NAMES = ("default", "harder", "easier", "hidelo", "hidehi")
NEW_KEYS = ("p_unfreeze_hide", "restart_visible_s", "unfreeze_drift_mps", "unfreeze_drift_tau_s")

DEAD_LABEL = LABEL_CODES["D"]
LIVE_LABEL = next(code for name, code in LABEL_CODES.items() if name not in ("D", "X"))

N_FIRST = 200
N_SECOND = 200
SECOND = N_FIRST + GAP_FRAMES  # first frame of the second half
N_FRAMES = N_FIRST + GAP_FRAMES + N_SECOND
PITCH_M = np.array([105.0, 68.0], dtype=np.float64)
FROZEN = {1: (0.25, 0.30), 2: (0.75, 0.70)}  # distinct frozen ball position per half
ATOL_M = 1e-3
ATOL_NORMALIZED = 1e-6
STEP_SIGMA_LIMIT = 8.0  # max allowed per-axis step / displacement, in sigmas
MIN_FRAMES_FOR_SPEED_STATS = 50
RMS_SPEED_BAND = (0.5, 1.5)  # accepted ratio of measured RMS speed to unfreeze_drift_mps

# Inclusive absolute frame ranges of dead-ball segments.
LAYOUTS = [
    pytest.param([(50, 119)], id="single_segment_inside_half"),
    pytest.param([(100, N_FIRST - 1), (SECOND, SECOND + 99)], id="segments_touching_halftime_gap"),
    pytest.param([(150, N_FIRST - 1)], id="segment_ending_first_half"),
    pytest.param([(SECOND, SECOND + 49)], id="segment_starting_second_half"),
    pytest.param([(20, 59), (150, N_FIRST - 1), (SECOND + 20, SECOND + 79)], id="multiple_segments"),
    pytest.param([(60, 69)], id="segment_shorter_than_restart_window"),
]


# --------------------------------------------------------------------------- helpers
@pytest.fixture(autouse=True)
def _no_external_gap_profile(monkeypatch):
    """Never read ball_gap_profile.json from the processed data directory."""
    monkeypatch.setattr(degrade_module, "_load_optional_profile", lambda: None)


def _load_config(name: str) -> dict:
    with (CONFIGS_DIR / "degrade_{}.yaml".format(name)).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _make_match(dead_ranges):
    """Build a C9-valid two-half synthetic match; ``dead_ranges`` are inclusive frame ranges."""
    half = expected_half(N_FIRST, N_SECOND)
    playable = half != 0
    dead = np.zeros(N_FRAMES, dtype=bool)
    for first, last in dead_ranges:
        dead[first:last + 1] = True
    assert not (dead & ~playable).any(), "dead-ball frames must not fall on the halftime gap"

    labels = np.full(N_FRAMES, LIVE_LABEL, dtype=np.int8)
    labels[~playable] = X_CODE
    labels[dead] = DEAD_LABEL

    frames = np.arange(N_FRAMES)
    ball = np.full((N_FRAMES, 3), np.nan, dtype=np.float32)
    ball[playable, 0] = 0.10 + 0.0005 * frames[playable]  # slowly moving ball in live play
    ball[playable, 1] = 0.50
    ball[playable, 2] = 0.0
    for half_id, (x, y) in FROZEN.items():
        frozen = dead & (half == half_id)
        ball[frozen, 0] = x
        ball[frozen, 1] = y
    ball_z = np.where(playable, 0.0, np.nan).astype(np.float32)

    slot_team_ids = np.r_[np.zeros(N_HOME_SLOTS, np.int8),
                          np.ones(N_SLOTS - N_HOME_SLOTS, np.int8)]
    match = ExternalMatch(
        match_id="synthetic",
        players=np.full((N_FRAMES, N_SLOTS, 3), np.nan, dtype=np.float32),
        ball=ball,
        ball_z=ball_z,
        labels=labels,
        half=half,
        slot_team_ids=slot_team_ids,
        pitch=PITCH_M.copy(),
    )
    return match, dead


def _playable(match):
    return match.half != 0


def _expected_hidden(dead, restart_frames):
    """Frames a hidden segment must mask: each run minus its last ``restart_frames`` frames."""
    hidden = np.zeros(N_FRAMES, dtype=bool)
    for lo, hi in _dead_segments(dead):
        hidden[lo:max(lo, hi - restart_frames)] = True
    return hidden


def _dead_segments(dead):
    """Contiguous dead-ball runs as [lo, hi); the halftime gap is never dead, so runs are half-local."""
    padded = np.diff(np.r_[False, dead, False].astype(np.int8))
    return list(zip(np.flatnonzero(padded == 1).tolist(), np.flatnonzero(padded == -1).tolist()))


def _visible_target(match, hidden):
    """target_ball_visible that leaves the background hazard at ~0 (mirrors degrade's maths)."""
    playable = _playable(match)
    return float((playable & ~hidden).sum() / playable.sum())


def _cfg(p_unfreeze_hide, *, ball_noise_m=0.0, unfreeze_drift_mps=2.0, restart_visible_s=1.0):
    """Config that leaves ball visibility untouched except for the unfreeze step.

    No air, crowd, phantom, duplicate or measurement noise, and a ~zero background hazard, so
    the output mask equals the unfreeze mask. Set ``target_ball_visible`` afterwards.
    """
    cfg = copy.deepcopy(_load_config("default"))
    cfg.update(
        p_unfreeze_hide=p_unfreeze_hide,
        restart_visible_s=restart_visible_s,
        unfreeze_drift_mps=unfreeze_drift_mps,
        target_ball_visible=1.0,
        ball_noise_m=ball_noise_m,
        player_noise_m=0.0,
        phantom_frac=0.0,
        dup_frac=0.0,
        crowd_hazard=0.0,
        bg_hazard_init=1e-12,
        bisection_iterations=5,
        player_bisection_iterations=3,
    )
    # The synthetic match has no players (all NaN), so the tracked mean is 0. Players are not
    # under test here: widen the tolerance so the player calibration accepts that.
    cfg["target_tracked_mean_tolerance"] = cfg["target_tracked_mean"]
    return cfg


def _meters(ball):
    return ball[:, :2].astype(np.float64) * PITCH_M


def _restart_frames(cfg):
    return int(round(cfg["restart_visible_s"] * cfg["fps"]))


def test_dead_label_comes_from_the_c0_contract():
    assert degrade_module._DEAD_LABEL == LABEL_CODES["D"]


# ------------------------------------------------- 1. hidden dead-ball segments
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("dead_ranges", LAYOUTS)
def test_hidden_dead_segment_masks_everything_except_restart_window(dead_ranges, seed):
    match, dead = _make_match(dead_ranges)
    cfg = _cfg(1.0)
    hidden = _expected_hidden(dead, _restart_frames(cfg))
    cfg["target_ball_visible"] = _visible_target(match, hidden)
    playable = _playable(match)

    out = degrade(match, cfg, seed)

    assert np.isnan(out.ball[hidden | ~playable, :2]).all()
    assert np.isfinite(out.ball[playable & ~hidden, :2]).all()
    # Restart window keeps the frozen restart position; live play is untouched.
    np.testing.assert_allclose(out.ball[playable & ~hidden, :2],
                               match.ball[playable & ~hidden, :2], atol=ATOL_NORMALIZED)


def test_hidden_segment_visible_window_has_exactly_restart_visible_s_seconds():
    match, dead = _make_match([(50, 149)])
    cfg = _cfg(1.0, restart_visible_s=2.0)
    restart_frames = _restart_frames(cfg)
    hidden = _expected_hidden(dead, restart_frames)
    cfg["target_ball_visible"] = _visible_target(match, hidden)

    out = degrade(match, cfg, 0)

    visible_dead = np.isfinite(out.ball[:, 0]) & dead
    assert int(visible_dead.sum()) == restart_frames
    assert visible_dead[150 - restart_frames:150].all()


# ------------------------------------------------- 2. drifted dead-ball segments
def _drift_bound_m(cfg, n_frames):
    """Generous per-axis bound on the drift around the straight line between the segment ends.

    The velocity is an AR(1) process with per-axis s.d. ``unfreeze_drift_mps / sqrt(2)`` and
    correlation time ``unfreeze_drift_tau_s``: over ``T`` seconds the position spreads by at
    most ``sigma * T`` (ballistic) and about ``sigma * sqrt(2 * tau * T)`` (diffusive).
    """
    sigma_v = cfg["unfreeze_drift_mps"] / math.sqrt(2.0)
    duration = n_frames / cfg["fps"]
    spread = sigma_v * min(duration, math.sqrt(2.0 * cfg["unfreeze_drift_tau_s"] * duration))
    return STEP_SIGMA_LIMIT * spread


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("dead_ranges", LAYOUTS)
def test_drifted_dead_segment_stays_visible_moves_smoothly_and_ends_at_restart(
        dead_ranges, seed):
    match, dead = _make_match(dead_ranges)
    cfg = _cfg(0.0)
    sigma_step = cfg["unfreeze_drift_mps"] / cfg["fps"] / math.sqrt(2.0)

    out = degrade(match, cfg, seed)
    meters = _meters(out.ball)

    assert np.isfinite(out.ball[dead, :2]).all()
    np.testing.assert_allclose(out.ball[~dead, :2], match.ball[~dead, :2],
                               atol=ATOL_NORMALIZED)
    for lo, hi in _dead_segments(dead):
        frozen = np.array(FROZEN[int(match.half[lo])]) * PITCH_M
        positions = meters[lo:hi]
        # Pinned at both ends: starts where the ball froze, ends at the restart position.
        np.testing.assert_allclose(positions[0], frozen, atol=ATOL_M)
        np.testing.assert_allclose(positions[-1], frozen, atol=ATOL_M)
        # Leaves the frozen state: it is a trajectory, not a constant.
        assert np.ptp(positions, axis=0).max() > ATOL_M
        # Smooth: bounded per-frame steps and bounded total displacement.
        steps = np.diff(positions, axis=0)
        assert np.abs(steps).max() < STEP_SIGMA_LIMIT * sigma_step
        assert np.abs(positions - frozen).max() < _drift_bound_m(cfg, hi - lo)


DRIFT_STATS_FRAMES = 25 * 60 * 5  # five minutes of dead ball: enough for stable statistics
ONE_SECOND_FRAMES = 25


def _long_drift(cfg, seed, begin=(30.0, 20.0), restart=(30.0, 20.0)):
    """Run the unfreeze step on one long drifted D segment; return positions in metres."""
    n = DRIFT_STATS_FRAMES
    ball = np.zeros((n, 3), dtype=np.float32)
    ball[:, 0], ball[:, 1] = restart[0] / PITCH_M[0], restart[1] / PITCH_M[1]
    ball[0, :2] = (begin[0] / PITCH_M[0], begin[1] / PITCH_M[1])
    visible = np.ones(n, dtype=bool)
    labels = np.full(n, DEAD_LABEL, dtype=np.int8)
    rng = np.random.default_rng(seed)
    degrade_module._unfreeze_dead_ball(ball, visible, labels, [(1, 0, n - 1)], cfg, rng,
                                       float(PITCH_M[0]), float(PITCH_M[1]))
    assert visible.all()
    return ball[:, :2].astype(np.float64) * PITCH_M


@pytest.mark.parametrize("seed", [0, 1])
def test_drift_speed_matches_config_at_the_one_second_scale(seed):
    """RMS speed ~ unfreeze_drift_mps, and the ball really travels ~that far in one second.

    Guards against a walk with independent per-frame steps, whose one-second displacement is
    only ~unfreeze_drift_mps / sqrt(fps): it keeps the ball looking stationary to statistic (c)
    of tools/compare_external_realism.py, which is what the drift is meant to undo.
    """
    cfg = _cfg(0.0, unfreeze_drift_mps=0.5)
    positions = _long_drift(cfg, seed)

    speeds = np.hypot(*np.diff(positions, axis=0).T) * cfg["fps"]
    rms = math.sqrt(float(np.mean(speeds ** 2)))
    assert RMS_SPEED_BAND[0] * cfg["unfreeze_drift_mps"] < rms < \
        RMS_SPEED_BAND[1] * cfg["unfreeze_drift_mps"]
    one_second = np.hypot(*(positions[ONE_SECOND_FRAMES:] - positions[:-ONE_SECOND_FRAMES]).T)
    # AR(1) velocity with tau = 1 s: median 1 s displacement ~0.71 x unfreeze_drift_mps.
    assert np.median(one_second) > 0.5 * cfg["unfreeze_drift_mps"] * 1.0


def test_drift_joins_different_freeze_and_restart_positions_continuously():
    cfg = _cfg(0.0, unfreeze_drift_mps=0.5)
    positions = _long_drift(cfg, 0, begin=(10.0, 5.0), restart=(40.0, 50.0))

    np.testing.assert_allclose(positions[0], (10.0, 5.0), atol=ATOL_M)
    np.testing.assert_allclose(positions[-1], (40.0, 50.0), atol=ATOL_M)
    # No teleport: the largest step stays a plausible slow-ball step.
    max_step_mps = np.hypot(*np.diff(positions, axis=0).T).max() * cfg["fps"]
    assert max_step_mps < STEP_SIGMA_LIMIT * cfg["unfreeze_drift_mps"]


def test_zero_drift_speed_keeps_the_ball_on_the_frozen_position():
    match, _ = _make_match([(50, 119)])
    cfg = _cfg(0.0, unfreeze_drift_mps=0.0)

    out = degrade(match, cfg, 0)

    np.testing.assert_allclose(out.ball[:, :2], match.ball[:, :2], atol=ATOL_NORMALIZED)


# ------------------------------------------------- 3. half boundary isolation
TOUCHING_GAP = [(100, N_FIRST - 1), (SECOND, SECOND + 99)]  # D right before and right after halftime


def test_hiding_never_leaks_masks_across_the_halftime_gap():
    match, dead = _make_match(TOUCHING_GAP)
    cfg = _cfg(1.0)
    restart_frames = _restart_frames(cfg)
    hidden = _expected_hidden(dead, restart_frames)
    cfg["target_ball_visible"] = _visible_target(match, hidden)

    out = degrade(match, cfg, 0)
    visible = np.isfinite(out.ball[:, 0])

    # Half 1 segment: hidden up to its own restart window, which ends at the gap.
    assert not visible[100:N_FIRST - restart_frames].any()
    assert visible[N_FIRST - restart_frames:N_FIRST].all()
    # The gap stays empty and half 2 hides its own segment from its first frame.
    assert not visible[N_FIRST:SECOND].any()
    assert not visible[SECOND:SECOND + 100 - restart_frames].any()
    assert visible[SECOND + 100 - restart_frames:SECOND + 100].all()
    # Live play outside the segments is untouched in both halves.
    live = _playable(match) & ~dead
    np.testing.assert_allclose(out.ball[live, :2], match.ball[live, :2], atol=ATOL_NORMALIZED)


def test_drift_never_leaks_positions_or_random_walks_across_the_halftime_gap():
    match, _ = _make_match(TOUCHING_GAP)
    cfg = _cfg(0.0)
    frozen1 = np.array(FROZEN[1]) * PITCH_M
    frozen2 = np.array(FROZEN[2]) * PITCH_M

    out = degrade(match, cfg, 0)
    meters = _meters(out.ball)

    # The gap stays NaN: no walk is extended into it.
    assert np.isnan(out.ball[N_FIRST:SECOND]).all()
    # Each half drifts around its own restart position only.
    bound = _drift_bound_m(cfg, N_FIRST - 100)
    assert np.abs(meters[100:N_FIRST] - frozen1).max() < bound
    assert np.abs(meters[SECOND:SECOND + 100] - frozen2).max() < bound
    # Each half ends exactly at its own restart position.
    np.testing.assert_allclose(meters[N_FIRST - 1], frozen1, atol=ATOL_M)
    np.testing.assert_allclose(meters[SECOND + 99], frozen2, atol=ATOL_M)
    # No continuity across halftime: the second-half walk starts near its own frozen spot.
    boundary_jump = np.linalg.norm(meters[SECOND] - meters[N_FIRST - 1])
    assert boundary_jump > 0.5 * np.linalg.norm(frozen2 - frozen1)


@pytest.mark.parametrize("p_unfreeze_hide", [0.0, 1.0])
def test_second_half_is_untouched_when_only_the_first_half_has_dead_frames(p_unfreeze_hide):
    match, dead = _make_match([(150, N_FIRST - 1)])  # dead segment ends at the halftime gap
    cfg = _cfg(p_unfreeze_hide)
    # Only the hide branch removes frames; with p_unfreeze_hide == 0 the ball stays fully visible.
    hidden = _expected_hidden(dead, _restart_frames(cfg)) if p_unfreeze_hide == 1.0 \
        else np.zeros(N_FRAMES, dtype=bool)
    cfg["target_ball_visible"] = _visible_target(match, hidden)

    out = degrade(match, cfg, 0)

    second_half = match.half == 2
    np.testing.assert_allclose(out.ball[second_half, :2], match.ball[second_half, :2],
                               atol=ATOL_NORMALIZED)
    assert np.isnan(out.ball[match.half == 0]).all()


# ------------------------------------------------- 4. determinism
def _determinism_cfg():
    cfg = copy.deepcopy(_load_config("default"))
    cfg.update(p_unfreeze_hide=0.5, target_ball_visible=0.3, dup_frac=0.0,
               bisection_iterations=10, player_bisection_iterations=3)
    cfg["target_tracked_mean_tolerance"] = cfg["target_tracked_mean"]  # no synthetic players
    # The 2 x 200-frame fixture is too short for gaps of up to 1500 frames to hit the
    # visibility tolerance (same fix as tests/test_modeling_degrade.py).
    cfg["gap_profile"]["lognormal_median_frames"] = 12.0
    cfg["gap_profile"]["max_frames"] = 40
    return cfg


DETERMINISM_LAYOUT = [(20, 59), (150, N_FIRST - 1), (SECOND + 20, SECOND + 79)]


def test_same_seed_gives_identical_output():
    match, _ = _make_match(DETERMINISM_LAYOUT)
    cfg = _determinism_cfg()

    first = degrade(match, cfg, 7)
    second = degrade(match, cfg, 7)

    for name in ("players", "ball", "ball_z", "labels", "half"):
        assert np.array_equal(getattr(first, name), getattr(second, name), equal_nan=True)


def test_different_seeds_give_different_output():
    match, _ = _make_match(DETERMINISM_LAYOUT)
    cfg = _determinism_cfg()

    first = degrade(match, cfg, 7)
    second = degrade(match, cfg, 8)

    assert not np.array_equal(first.ball, second.ball, equal_nan=True)


def test_degrade_does_not_mutate_its_inputs():
    match, _ = _make_match(DETERMINISM_LAYOUT)
    cfg = _determinism_cfg()
    ball_before = match.ball.copy()
    labels_before = match.labels.copy()
    cfg_before = copy.deepcopy(cfg)

    degrade(match, cfg, 3)

    assert np.array_equal(match.ball, ball_before, equal_nan=True)
    assert np.array_equal(match.labels, labels_before)
    assert cfg == cfg_before


def test_hide_probability_does_not_shift_the_random_stream_of_later_stages():
    """Hide and drift branches consume the same draws, so later noise stays comparable."""
    match, dead = _make_match([(100, N_FIRST - 1)])
    hide_cfg, drift_cfg = _cfg(1.0, ball_noise_m=0.25), _cfg(0.0, ball_noise_m=0.25)
    hide_cfg["target_ball_visible"] = _visible_target(
        match, _expected_hidden(dead, _restart_frames(hide_cfg)))

    hidden_out = degrade(match, hide_cfg, 5)
    drift_out = degrade(match, drift_cfg, 5)

    # Live-play frames get identical measurement noise in both branches.
    live = _playable(match) & ~dead
    np.testing.assert_allclose(hidden_out.ball[live, :2], drift_out.ball[live, :2],
                               atol=ATOL_NORMALIZED)


# ------------------------------------------------- 5. config validation
CONFIG_KEYS = sorted(_load_config("default"))


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_shipped_config_passes_validation(name):
    cfg = _load_config(name)

    assert validate_degrade_config(cfg) is cfg
    assert cfg["name"] == name
    for key in NEW_KEYS:
        assert key in cfg


def test_hide_probability_of_each_variant():
    default = _load_config("default")

    assert default["p_unfreeze_hide"] == 0.5
    assert _load_config("hidelo")["p_unfreeze_hide"] == 0.25
    assert _load_config("hidehi")["p_unfreeze_hide"] == 0.75
    assert _load_config("harder")["p_unfreeze_hide"] == default["p_unfreeze_hide"]
    assert _load_config("easier")["p_unfreeze_hide"] == default["p_unfreeze_hide"]


def test_default_has_near_zero_player_noise_and_unchanged_ball_noise():
    default = _load_config("default")

    assert default["ball_noise_m"] == 0.25
    assert 0.0 <= default["player_noise_m"] <= 0.1 * default["ball_noise_m"]


@pytest.mark.parametrize("name", ["hidelo", "hidehi"])
def test_hide_variants_differ_from_default_only_in_name_and_hide_probability(name):
    default, variant = _load_config("default"), _load_config(name)

    assert {k: v for k, v in variant.items() if k not in ("name", "p_unfreeze_hide")} == \
        {k: v for k, v in default.items() if k not in ("name", "p_unfreeze_hide")}


def test_harder_and_easier_scale_only_the_noise_relative_to_default():
    default = _load_config("default")
    harder, easier = _load_config("harder"), _load_config("easier")
    noise_keys = ("ball_noise_m", "player_noise_m")

    for key in noise_keys:
        assert harder[key] > default[key] > easier[key]
    # One common scale factor per variant for both noise sources.
    assert math.isclose(harder["ball_noise_m"] / default["ball_noise_m"],
                        harder["player_noise_m"] / default["player_noise_m"])
    assert math.isclose(easier["ball_noise_m"] / default["ball_noise_m"],
                        easier["player_noise_m"] / default["player_noise_m"])
    for variant in (harder, easier):
        assert {k: v for k, v in variant.items() if k not in ("name",) + noise_keys} == \
            {k: v for k, v in default.items() if k not in ("name",) + noise_keys}


@pytest.mark.parametrize("key", CONFIG_KEYS)
def test_missing_key_is_rejected(key):
    cfg = _load_config("default")
    del cfg[key]

    with pytest.raises(ValueError, match="missing keys"):
        validate_degrade_config(cfg)


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_unknown_key_is_rejected(name):
    cfg = _load_config(name)
    cfg["p_unfreeze_hid"] = 0.5  # typo of p_unfreeze_hide

    with pytest.raises(ValueError, match="unknown keys"):
        validate_degrade_config(cfg)


@pytest.mark.parametrize("key,value", [
    ("p_unfreeze_hide", -0.1),
    ("p_unfreeze_hide", 1.1),
    ("p_unfreeze_hide", float("nan")),
    ("p_unfreeze_hide", True),
    ("p_unfreeze_hide", "0.5"),
    ("restart_visible_s", -1.0),
    ("restart_visible_s", float("inf")),
    ("restart_visible_s", None),
    ("unfreeze_drift_mps", -0.1),
    ("unfreeze_drift_mps", float("nan")),
    ("unfreeze_drift_mps", False),
    ("unfreeze_drift_tau_s", 0.0),
    ("unfreeze_drift_tau_s", -1.0),
    ("unfreeze_drift_tau_s", float("inf")),
])
def test_invalid_unfreeze_values_are_rejected(key, value):
    cfg = _load_config("default")
    cfg[key] = value

    with pytest.raises(ValueError):
        validate_degrade_config(cfg)


@pytest.mark.parametrize("key,value", [
    ("p_unfreeze_hide", 0.0),
    ("p_unfreeze_hide", 1.0),
    ("restart_visible_s", 0.0),
    ("unfreeze_drift_mps", 0.0),
])
def test_boundary_unfreeze_values_are_accepted(key, value):
    cfg = _load_config("default")
    cfg[key] = value

    assert validate_degrade_config(cfg) is cfg

def test_dead_ball_mask_detects_c0_dead_label_on_int8_labels():
    labels = np.array([LIVE_LABEL, DEAD_LABEL, DEAD_LABEL, X_CODE], dtype=np.int8)

    np.testing.assert_array_equal(degrade_module._dead_ball_mask(labels),
                                  [False, True, True, False])
