"""Tests for the broadcast alignment (F4-05): ``read_atd`` n_slots, ``align_broadcast`` and the
``tools/broadcast_feasibility.py`` gate.

Everything here is synthetic: arrays are generated in memory from ``np.random.default_rng(seed)``,
and the only files written are ATD text files inside pytest's ``tmp_path``. No test reads the match
data directory (the CI has none), and no test depends on ``$MTAH_DATA_DIR`` or ``./data``.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

import games
from modeling.broadcast import align
from modeling.broadcast.align import Tracking, align_broadcast, tracking_from_normalized
from modeling.config import ConfigError, load_config
from preprocessing import prep_tracking
from preprocessing.schema import N_SLOTS

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "broadcast.yaml"
TOOL_PATH = REPO_ROOT / "tools" / "broadcast_feasibility.py"

SEED = 20261112
PITCH = tuple(games.GAMES[games.GAME]["pitch"])

# -- synthetic scenario layout ------------------------------------------------------------------
# Tactical timeline: two short halves with a gap between them.
HALF_BOUNDS = {1: (500, 3499), 2: (4000, 6999)}
N_TACTICAL_FRAMES = 7500
N_PLAYERS = 22
N_BROADCAST_SLOTS = 51
BLOCK_FRAMES = 50

# Broadcast timeline, as half-open frame ranges [start, stop). Offsets are
# ``tactical_frame - broadcast_frame``.
OFFSET_1 = 600
OFFSET_2 = 2500
SEGMENT_1 = (0, 1500)         # 30 blocks, shows tactical frames 600..2099 (half 1)
REPLAY = (1500, 1650)         # 3 blocks, repeats earlier tactical frames 1200..1349
REPLAY_SOURCE_START = 1200
SEGMENT_2 = (1650, 3650)      # 40 blocks, shows tactical frames 4150..6149 (half 2)
N_BROADCAST_FRAMES = 3650

VISIBLE_RADIUS_M = 38.0       # the broadcast only shows players this close to the ball
NOISE_SIGMA_M = 0.3
BALL_SEEN_PROB = 0.6
MAX_SHIFT_ERROR_FRAMES = 1
MIN_LIVE_MAPPED_RATIO = 0.8   # blocks with fewer than 6 visible players are legitimately skipped


@dataclass
class Scenario:
    tactical: Tracking
    broadcast: Tracking
    master: pd.DataFrame      # stand-in for the BAR-ATH master: ball_observed + half
    truth: np.ndarray         # int[T_broadcast]: true tactical frame, -1 for the replay


# -- synthetic data -----------------------------------------------------------------------------

def _tactical_world(rng):
    """Smooth 22-player motion on the whole tactical timeline plus a ball that hops between players."""
    length, width = PITCH
    t = np.arange(N_TACTICAL_FRAMES, dtype=np.float64)
    base = np.column_stack([rng.uniform(10.0, length - 10.0, N_PLAYERS),
                            rng.uniform(8.0, width - 8.0, N_PLAYERS)])
    phase = rng.uniform(0.0, 2.0 * np.pi, (N_PLAYERS, 2))
    period = rng.uniform(200.0, 500.0, (N_PLAYERS, 2))
    amplitude = rng.uniform(6.0, 12.0, (N_PLAYERS, 2))
    players = base[None] + amplitude[None] * np.sin(
        2.0 * np.pi * t[:, None, None] / period[None] + phase[None])
    players = np.clip(players, [0.0, 0.0], [length, width])
    ball = np.empty((N_TACTICAL_FRAMES, 2))
    for start in range(0, N_TACTICAL_FRAMES, 120):
        carrier = int(rng.integers(N_PLAYERS))
        ball[start:start + 120] = players[start:start + 120, carrier]
    ball += rng.normal(0.0, 0.1, ball.shape)
    return players, ball


def build_scenario(seed=SEED, rotated=False, ball_always_observed=False) -> Scenario:
    """A tactical stream and a 51-slot broadcast stream with two live segments and a replay.

    The broadcast shows only the players near the ball, with Gaussian noise, and sees the ball on
    part of the frames. With ``rotated`` the broadcast normalised coordinates are rotated by 180
    degrees (x -> 1 - x, y -> 1 - y), as if the camera used the opposite pitch orientation.
    """
    rng = np.random.default_rng(seed)
    length, width = PITCH
    scale = np.array([length, width])
    players, ball = _tactical_world(rng)
    team = np.repeat([0, 1], N_PLAYERS // 2)

    half = np.zeros(N_TACTICAL_FRAMES, dtype=np.int8)
    for half_id, (first, last) in HALF_BOUNDS.items():
        half[first:last + 1] = half_id

    # Tactical camera: players inside the halves, ball seen on part of the frames.
    t_players = players.copy()
    t_players[half == 0] = np.nan
    tactical_seen = (rng.random(N_TACTICAL_FRAMES) < BALL_SEEN_PROB) & (half != 0)
    t_ball = np.where(tactical_seen[:, None], ball, np.nan)
    tactical = tracking_from_normalized(t_players / scale, t_ball / scale, team, PITCH, half=half)

    # Which tactical frame each broadcast frame shows.
    shown = np.full(N_BROADCAST_FRAMES, -1, dtype=np.int64)
    shown[SEGMENT_1[0]:SEGMENT_1[1]] = np.arange(*SEGMENT_1) + OFFSET_1
    shown[SEGMENT_2[0]:SEGMENT_2[1]] = np.arange(*SEGMENT_2) + OFFSET_2
    shown[REPLAY[0]:REPLAY[1]] = REPLAY_SOURCE_START + np.arange(REPLAY[1] - REPLAY[0])
    truth = shown.copy()
    truth[REPLAY[0]:REPLAY[1]] = -1

    # Broadcast camera: nearby players only, noisy; the other slots stay empty (team id 2).
    b_players = np.full((N_BROADCAST_FRAMES, N_BROADCAST_SLOTS, 2), np.nan)
    near = np.linalg.norm(players[shown] - ball[shown][:, None, :], axis=2) < VISIBLE_RADIUS_M
    shown_players = players[shown] + rng.normal(0.0, NOISE_SIGMA_M, (N_BROADCAST_FRAMES, N_PLAYERS, 2))
    shown_players[~near] = np.nan
    b_players[:, :N_PLAYERS] = shown_players
    b_ball = ball[shown] + rng.normal(0.0, NOISE_SIGMA_M, (N_BROADCAST_FRAMES, 2))
    b_ball[rng.random(N_BROADCAST_FRAMES) >= BALL_SEEN_PROB] = np.nan
    b_team = np.full(N_BROADCAST_SLOTS, 2, dtype=np.int64)
    b_team[:N_PLAYERS] = team

    b_players_norm = b_players / scale
    b_ball_norm = b_ball / scale
    if rotated:
        b_players_norm = 1.0 - b_players_norm
        b_ball_norm = 1.0 - b_ball_norm
    broadcast = tracking_from_normalized(b_players_norm, b_ball_norm, b_team, PITCH,
                                         valid=np.ones(N_BROADCAST_FRAMES, dtype=bool))

    observed = np.ones(N_TACTICAL_FRAMES) if ball_always_observed else \
        np.isfinite(tactical.ball).all(axis=1).astype(np.float64)
    master = pd.DataFrame({"ball_observed": observed, "half": half})
    return Scenario(tactical, broadcast, master, truth)


# -- fixtures -----------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def identity_scenario():
    return build_scenario(rotated=False)


@pytest.fixture(scope="module")
def rot180_scenario():
    return build_scenario(rotated=True)


@pytest.fixture(scope="module")
def identity_mapping(identity_scenario, cfg):
    return align_broadcast(identity_scenario.tactical, identity_scenario.broadcast, cfg)


@pytest.fixture(scope="module")
def rot180_mapping(rot180_scenario, cfg):
    return align_broadcast(rot180_scenario.tactical, rot180_scenario.broadcast, cfg)


@pytest.fixture(scope="module", params=["identity", "rot180"])
def run(request, identity_scenario, identity_mapping, rot180_scenario, rot180_mapping):
    """(scenario, mapping) for both orientations, so shared properties are checked on each."""
    if request.param == "identity":
        return identity_scenario, identity_mapping
    return rot180_scenario, rot180_mapping


def _assert_recovers_known_shifts(scenario, mapping):
    """Live frames are mapped within one frame of the truth and each segment's offset is recovered."""
    mapped = mapping.tactical_frame
    assert mapped.dtype == np.int32
    assert mapped.shape == (N_BROADCAST_FRAMES,)
    live = scenario.truth >= 0
    live_mapped = live & (mapped >= 0)
    assert live_mapped.sum() >= MIN_LIVE_MAPPED_RATIO * live.sum()
    error = np.abs(mapped[live_mapped].astype(np.int64) - scenario.truth[live_mapped])
    assert error.max() <= MAX_SHIFT_ERROR_FRAMES
    frames = np.arange(N_BROADCAST_FRAMES)
    for (start, stop), offset in ((SEGMENT_1, OFFSET_1), (SEGMENT_2, OFFSET_2)):
        in_segment = (frames >= start) & (frames < stop) & (mapped >= 0)
        recovered = np.median(mapped[in_segment].astype(np.int64) - frames[in_segment])
        assert abs(recovered - offset) <= MAX_SHIFT_ERROR_FRAMES
    # The replay shows earlier tactical frames: none of its frames may be mapped.
    assert np.all(mapped[REPLAY[0]:REPLAY[1]] == -1)


# -- 1. read_atd regression ---------------------------------------------------------------------

def _write_atd(path, players, ball):
    """Write arrays as an ATD text file: ``frame_id:slot;slot;...:ball`` with ``x,y,s`` groups."""
    lines = []
    for frame in range(players.shape[0]):
        groups = ["N" if np.isnan(players[frame, slot, 0])
                  else "{:.6f},{:.6f},{:.3f}".format(*players[frame, slot])
                  for slot in range(players.shape[1])]
        ball_text = "N" if np.isnan(ball[frame, 0]) else "{:.6f},{:.6f},{:.3f}".format(*ball[frame])
        lines.append("{}:{}:{}".format(frame + 1, ";".join(groups), ball_text))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _random_atd_arrays(rng, n_frames, n_slots):
    players = rng.uniform(0.0, 1.0, (n_frames, n_slots, 3))
    players[rng.random((n_frames, n_slots)) < 0.2] = np.nan
    ball = rng.uniform(0.0, 1.0, (n_frames, 3))
    ball[rng.random(n_frames) < 0.3] = np.nan
    return players, ball


def _reference_read(path, n_frames, n_slots):
    """Independent parse of the ATD text, the expected output of ``read_atd``."""
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        frame_id, groups, ball_text = line.split(":")
        assert int(frame_id) == index + 1
        for slot, group in enumerate(groups.split(";")):
            if group != "N":
                players[index, slot] = np.array([float(v) for v in group.split(",")], dtype=np.float32)
        if ball_text != "N":
            ball[index] = np.array([float(v) for v in ball_text.split(",")], dtype=np.float32)
    return players, ball


def _patch_teams(monkeypatch, n_slots):
    teams = np.array([0] * (n_slots // 2) + [1] * (n_slots - n_slots // 2))
    monkeypatch.setattr(prep_tracking, "team_of_each_track", lambda meta_path: teams)
    return teams


N_ATD_TEST_FRAMES = 40


def test_read_atd_default_slots_is_bit_for_bit_unchanged(tmp_path, monkeypatch):
    rng = np.random.default_rng(SEED)
    players, ball = _random_atd_arrays(rng, N_ATD_TEST_FRAMES, N_SLOTS)
    raw = tmp_path / "tactical_like.txt"
    _write_atd(raw, players, ball)
    teams = _patch_teams(monkeypatch, N_SLOTS)

    got_players, got_ball, got_teams = prep_tracking.read_atd(
        raw, "unused_meta.xml", n_frames=N_ATD_TEST_FRAMES)
    want_players, want_ball = _reference_read(raw, N_ATD_TEST_FRAMES, N_SLOTS)

    assert got_players.dtype == np.float32 and got_ball.dtype == np.float32
    assert got_players.shape == (N_ATD_TEST_FRAMES, N_SLOTS, 3)
    assert got_players.tobytes() == want_players.tobytes()
    assert got_ball.tobytes() == want_ball.tobytes()
    assert got_teams is teams

    explicit_players, explicit_ball, _ = prep_tracking.read_atd(
        raw, "unused_meta.xml", n_frames=N_ATD_TEST_FRAMES, n_slots=N_SLOTS)
    assert explicit_players.tobytes() == got_players.tobytes()
    assert explicit_ball.tobytes() == got_ball.tobytes()


def test_read_atd_parses_51_slot_broadcast_file(tmp_path, monkeypatch):
    rng = np.random.default_rng(SEED + 1)
    players, ball = _random_atd_arrays(rng, N_ATD_TEST_FRAMES, N_BROADCAST_SLOTS)
    raw = tmp_path / "broadcast_like.txt"
    _write_atd(raw, players, ball)
    _patch_teams(monkeypatch, N_BROADCAST_SLOTS)

    got_players, got_ball, got_teams = prep_tracking.read_atd(
        raw, "unused_meta.xml", n_frames=N_ATD_TEST_FRAMES, n_slots=N_BROADCAST_SLOTS)
    want_players, want_ball = _reference_read(raw, N_ATD_TEST_FRAMES, N_BROADCAST_SLOTS)

    assert got_players.shape == (N_ATD_TEST_FRAMES, N_BROADCAST_SLOTS, 3)
    assert got_players.tobytes() == want_players.tobytes()
    assert got_ball.tobytes() == want_ball.tobytes()
    assert len(got_teams) == N_BROADCAST_SLOTS


def test_read_atd_slot_count_mismatches_are_errors(tmp_path, monkeypatch):
    rng = np.random.default_rng(SEED + 2)
    players, ball = _random_atd_arrays(rng, N_ATD_TEST_FRAMES, N_BROADCAST_SLOTS)
    raw = tmp_path / "broadcast_like.txt"
    _write_atd(raw, players, ball)

    # The default layout rejects a 51-slot file: first through the metadata ...
    _patch_teams(monkeypatch, N_BROADCAST_SLOTS)
    with pytest.raises(ValueError, match="Expected {} track slots".format(N_SLOTS)):
        prep_tracking.read_atd(raw, "unused_meta.xml", n_frames=N_ATD_TEST_FRAMES)
    # ... and through the file itself when the metadata does match the requested layout.
    _patch_teams(monkeypatch, N_SLOTS)
    with pytest.raises(ValueError, match="expected {} slots".format(N_SLOTS)):
        prep_tracking.read_atd(raw, "unused_meta.xml", n_frames=N_ATD_TEST_FRAMES)
    # Asking for 51 slots with 46-slot metadata is rejected as well.
    with pytest.raises(ValueError, match="Expected {} track slots".format(N_BROADCAST_SLOTS)):
        prep_tracking.read_atd(raw, "unused_meta.xml", n_frames=N_ATD_TEST_FRAMES,
                               n_slots=N_BROADCAST_SLOTS)


# -- config -------------------------------------------------------------------------------------

def _write_config(tmp_path, mutate):
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    mutate(raw)
    path = tmp_path / "broadcast_test.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_repo_config_loads_with_all_documented_keys(cfg):
    assert cfg["broadcast_n_slots"] == N_BROADCAST_SLOTS
    assert cfg["block_frames"] == BLOCK_FRAMES
    assert cfg["block_frames"] == cfg["block_seconds"] * cfg["source_fps"]


def test_config_rejects_unknown_and_missing_keys(tmp_path):
    with pytest.raises(ConfigError, match="unknown keys"):
        load_config(_write_config(tmp_path, lambda raw: raw.update(not_a_key=1)))
    with pytest.raises(ConfigError, match="missing keys"):
        load_config(_write_config(tmp_path, lambda raw: raw.pop("segment_min_blocks")))


def test_config_rejects_inconsistent_block_length(tmp_path):
    with pytest.raises(ConfigError, match="block_frames"):
        load_config(_write_config(tmp_path, lambda raw: raw.update(block_frames=40)))


# -- 2. live alignment and replay rejection, identity orientation -------------------------------

def test_identity_orientation_is_selected(identity_mapping):
    costs = identity_mapping.diagnostics.orientation_costs
    assert identity_mapping.orientation == "identity"
    assert costs["identity"] < costs["rot180"]


def test_identity_recovers_both_offsets_and_rejects_the_replay(identity_scenario, identity_mapping):
    _assert_recovers_known_shifts(identity_scenario, identity_mapping)


def test_coarse_shifts_are_close_to_the_true_offsets(identity_mapping, cfg):
    coarse = identity_mapping.diagnostics.coarse
    assert abs(coarse[1][0] - OFFSET_1) <= cfg["refine_search_radius_frames"]
    assert abs(coarse[2][0] - OFFSET_2) <= cfg["refine_search_radius_frames"]


def test_segments_respect_the_minimum_length(identity_mapping, cfg):
    segments = identity_mapping.diagnostics.segments
    assert segments
    assert all(last - first + 1 >= cfg["segment_min_blocks"] for first, last, _, _ in segments)


def test_alignment_is_deterministic(identity_scenario, identity_mapping, cfg):
    again = align_broadcast(identity_scenario.tactical, identity_scenario.broadcast, cfg)
    assert again.orientation == identity_mapping.orientation
    assert np.array_equal(again.tactical_frame, identity_mapping.tactical_frame)


def test_blocks_without_a_valid_homography_stay_unmapped(identity_scenario, cfg):
    scenario = build_scenario(rotated=False)
    invalid = np.zeros(N_BROADCAST_FRAMES, dtype=bool)
    invalid[200:400] = True                      # blocks 4..7 of live segment 1
    broadcast = Tracking(scenario.broadcast.players, scenario.broadcast.ball,
                         scenario.broadcast.team, ~invalid, scenario.broadcast.half)
    mapping = align_broadcast(scenario.tactical, broadcast, cfg)
    mapped = mapping.tactical_frame

    assert np.all(mapped[200:400] == -1)
    # Blocks 0..3 are a run shorter than ``segment_min_blocks``, so they are dropped as well.
    assert 4 < cfg["segment_min_blocks"]
    assert np.all(mapped[:200] == -1)
    after = np.arange(400, SEGMENT_1[1])
    kept = after[mapped[after] >= 0]
    assert kept.size > 0
    assert np.abs(mapped[kept].astype(np.int64) - (kept + OFFSET_1)).max() <= MAX_SHIFT_ERROR_FRAMES


# -- 3. 180-degree rotated orientation ----------------------------------------------------------

def test_rot180_orientation_is_selected(rot180_mapping):
    costs = rot180_mapping.diagnostics.orientation_costs
    assert rot180_mapping.orientation == "rot180"
    assert costs["rot180"] < costs["identity"]


def test_rot180_recovers_the_same_shifts(rot180_scenario, rot180_mapping):
    _assert_recovers_known_shifts(rot180_scenario, rot180_mapping)


def test_rotation_does_not_change_the_recovered_mapping(identity_mapping, rot180_mapping):
    both = (identity_mapping.tactical_frame >= 0) & (rot180_mapping.tactical_frame >= 0)
    assert both.any()
    difference = np.abs(identity_mapping.tactical_frame[both].astype(np.int64)
                        - rot180_mapping.tactical_frame[both])
    assert difference.max() <= MAX_SHIFT_ERROR_FRAMES


# -- 4. half-boundary isolation -----------------------------------------------------------------

def test_mapped_frames_never_leave_an_evaluated_half(run):
    scenario, mapping = run
    mapped = mapping.tactical_frame[mapping.tactical_frame >= 0]
    assert mapped.size > 0
    assert np.all(scenario.tactical.half[mapped] != 0)


def test_every_block_maps_into_exactly_one_half(run):
    scenario, mapping = run
    for start in range(0, N_BROADCAST_FRAMES - BLOCK_FRAMES + 1, BLOCK_FRAMES):
        block = mapping.tactical_frame[start:start + BLOCK_FRAMES]
        if np.all(block == -1):
            continue
        assert np.all(block >= 0)                # a block is mapped whole or not at all
        assert np.unique(scenario.tactical.half[block]).size == 1


def test_segments_never_bridge_the_two_halves(run):
    scenario, mapping = run
    diag = mapping.diagnostics
    segments = diag.segments
    assert {half for _, _, half, _ in segments} == {1, 2}
    for first, last, half, shift in segments:
        blocks = np.arange(first, last + 1)
        assert np.all(diag.block_half[blocks] == half)
        assert np.all(diag.block_in_segment[blocks])
        frames = np.concatenate([np.arange(b * BLOCK_FRAMES, (b + 1) * BLOCK_FRAMES) for b in blocks])
        segment_halves = np.unique(scenario.tactical.half[mapping.tactical_frame[frames]])
        assert segment_halves.tolist() == [half]
        expected_offset = OFFSET_1 if half == 1 else OFFSET_2
        assert abs(shift - expected_offset) <= MAX_SHIFT_ERROR_FRAMES


def test_live_segment_1_maps_to_half_1_and_live_segment_2_to_half_2(run):
    scenario, mapping = run
    mapped = mapping.tactical_frame
    first = mapped[SEGMENT_1[0]:SEGMENT_1[1]]
    second = mapped[SEGMENT_2[0]:SEGMENT_2[1]]
    assert np.all(scenario.tactical.half[first[first >= 0]] == 1)
    assert np.all(scenario.tactical.half[second[second >= 0]] == 2)


def test_search_window_never_lets_a_block_cross_the_half_edges():
    first, last = HALF_BOUNDS[1]
    start, radius = 1000, 25
    inside = align._window(1500, (first, last), start, BLOCK_FRAMES, radius)
    assert inside.size == 2 * radius + 1
    near_end = align._window(last - BLOCK_FRAMES + 1 - start + 10, (first, last), start,
                             BLOCK_FRAMES, radius)
    near_start = align._window(first - start - 10, (first, last), start, BLOCK_FRAMES, radius)
    for shifts in (near_end, near_start):
        assert shifts.size > 0
        assert shifts.min() + start >= first
        assert shifts.max() + start + BLOCK_FRAMES - 1 <= last
    assert align._window(10 ** 6, (first, last), start, BLOCK_FRAMES, radius).size == 0
    assert align._window(-10 ** 6, (first, last), start, BLOCK_FRAMES, radius).size == 0


# -- 5. feasibility CLI -------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("broadcast_feasibility_tool", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module            # dataclasses look the module up by name
    spec.loader.exec_module(module)
    return module


CRITERION_LINE = re.compile(r"(PASS|FAIL)  \((i|ii|iii|iv)\) [A-Za-z ]+")
CRITERIA_ORDER = ["i", "ii", "iii", "iv"]


def _run_tool(tool, monkeypatch, capsys, tactical, broadcast, master, verbose):
    monkeypatch.setattr(tool, "load_broadcast_inputs",
                        lambda config, data_dir=None: (tactical, broadcast))
    monkeypatch.setattr(tool, "io", SimpleNamespace(load_master=lambda path, n_frames=None: master))
    argv = ["--config", str(CONFIG_PATH)] + (["--verbose"] if verbose else [])
    code = tool.main(argv)
    lines = capsys.readouterr().out.splitlines()
    return code, lines


def _assert_private_output(lines):
    """Default output: four PASS/FAIL lines then GO / NO-GO, with no digit anywhere."""
    assert len(lines) == 5
    keys = []
    for line in lines[:4]:
        match = CRITERION_LINE.fullmatch(line)
        assert match, line
        keys.append(match.group(2))
    assert keys == CRITERIA_ORDER
    assert lines[4] in ("GO", "NO-GO")
    assert not re.search(r"\d", "\n".join(lines))


def _assert_verbose_output(lines):
    assert lines[-1] in ("GO", "NO-GO")
    assert re.search(r"\d", "\n".join(lines))            # numbers are shown on request
    assert len(lines) <= 12
    assert all(len(line) <= 200 and "[" not in line for line in lines)   # no array dumps


def test_cli_go_on_a_good_synthetic_broadcast(tool, monkeypatch, capsys, identity_scenario):
    s = identity_scenario
    code, lines = _run_tool(tool, monkeypatch, capsys, s.tactical, s.broadcast, s.master, False)
    assert code == 0
    _assert_private_output(lines)
    assert [line.split()[0] for line in lines[:4]] == ["PASS"] * 4
    assert lines[4] == "GO"


def test_cli_go_verbose_shows_numbers_and_exits_zero(tool, monkeypatch, capsys, identity_scenario):
    s = identity_scenario
    code, lines = _run_tool(tool, monkeypatch, capsys, s.tactical, s.broadcast, s.master, True)
    assert code == 0
    _assert_verbose_output(lines)
    assert lines[-1] == "GO"
    assert any(line.startswith("PASS  (i)") for line in lines)


def test_cli_go_with_the_rotated_broadcast(tool, monkeypatch, capsys, rot180_scenario):
    s = rot180_scenario
    code, lines = _run_tool(tool, monkeypatch, capsys, s.tactical, s.broadcast, s.master, False)
    assert code == 0
    _assert_private_output(lines)
    assert lines[4] == "GO"


def test_cli_no_go_when_the_tactical_ball_is_never_missing(tool, monkeypatch, capsys):
    s = build_scenario(ball_always_observed=True)       # no ball-less frames: (iii) cannot pass
    code, lines = _run_tool(tool, monkeypatch, capsys, s.tactical, s.broadcast, s.master, False)
    assert code == 0
    _assert_private_output(lines)
    assert lines[2].startswith("FAIL  (iii)")
    assert lines[4] == "NO-GO"


def test_cli_no_go_for_an_unrelated_broadcast(tool, monkeypatch, capsys, identity_scenario):
    unrelated = build_scenario(seed=SEED + 99)          # different players and ball: nothing aligns
    s = identity_scenario
    code, lines = _run_tool(tool, monkeypatch, capsys, s.tactical, unrelated.broadcast, s.master, False)
    assert code == 0
    _assert_private_output(lines)
    assert lines[0].startswith("FAIL  (i)")
    assert lines[4] == "NO-GO"


def test_cli_no_go_verbose_exits_zero(tool, monkeypatch, capsys):
    s = build_scenario(ball_always_observed=True)
    code, lines = _run_tool(tool, monkeypatch, capsys, s.tactical, s.broadcast, s.master, True)
    assert code == 0
    _assert_verbose_output(lines)
    assert lines[-1] == "NO-GO"


def test_cli_exits_one_when_the_inputs_cannot_be_loaded(tool, monkeypatch, capsys):
    def missing(config, data_dir=None):
        raise OSError("match data not found")

    monkeypatch.setattr(tool, "load_broadcast_inputs", missing)
    assert tool.main(["--config", str(CONFIG_PATH)]) == 1
    assert capsys.readouterr().out.startswith("FAIL  could not load inputs")


def test_mapped_pairs_drop_frames_outside_the_evaluated_halves(tool):
    evaluated = np.zeros(10, dtype=bool)
    evaluated[3:7] = True
    mapping = SimpleNamespace(tactical_frame=np.array([-1, 2, 3, 6, 7, -1], dtype=np.int32))
    b_idx, t_idx = tool.mapped_pairs(mapping, evaluated)
    assert b_idx.tolist() == [2, 3]
    assert t_idx.tolist() == [3, 6]
    with pytest.raises(ValueError):
        tool.mapped_pairs(SimpleNamespace(tactical_frame=np.array([99], dtype=np.int32)), evaluated)