"""Tests for the broadcast ball fusion (F4-06).

Everything here is synthetic and lives in memory or under ``tmp_path``: no test opens the match
data directory (the CI has none). Heavy modules are imported inside the tests that need them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from modeling.broadcast.align import IDENTITY, ROT180, Mapping
from modeling.broadcast.fuse import fused_ball, orient_broadcast_ball

UNIT_EXTENT = (1.0, 1.0)


def _tactical(n_frames, observed):
    """Tactical ball (T, 3), NaN everywhere except ``observed`` = {frame: (x, y, speed)}."""
    ball = np.full((n_frames, 3), np.nan, dtype=np.float64)
    for frame, values in observed.items():
        ball[frame] = values
    return ball


def _mapping(frames, orientation=IDENTITY):
    return Mapping(tactical_frame=np.asarray(frames, dtype=np.int32), orientation=orientation,
                   diagnostics=None)


# -- fusion rules -------------------------------------------------------------------------------

def test_fused_ball_never_overwrites_tactical():
    tactical = _tactical(6, {0: (0.10, 0.20, 3.0), 2: (0.30, 0.40, 0.0), 5: (0.50, 0.60, 7.5)})
    snapshot = tactical.copy()
    # Every broadcast frame is valid and maps onto a tactical frame, including observed ones.
    broadcast = np.array([[0.91, 0.92], [0.81, 0.82], [0.71, 0.72],
                          [0.61, 0.62], [0.51, 0.52], [0.41, 0.42]])
    out = fused_ball(tactical, broadcast, np.arange(6))

    observed = np.array([0, 2, 5])
    assert np.array_equal(out[observed], snapshot[observed])        # x, y and speed untouched
    assert np.array_equal(tactical, snapshot, equal_nan=True)       # the input is not mutated
    assert out.dtype == np.float64 and out.shape == (6, 3)


def test_fused_ball_fills_only_mapped_missing_frames():
    n_tactical = 10
    tactical = _tactical(n_tactical, {0: (0.1, 0.1, 1.0), 5: (0.5, 0.5, 2.0), 9: (0.9, 0.9, 3.0)})
    raw_broadcast = np.array([
        [0.11, 0.11],   # 0: valid ball, but the frame is unmapped
        [0.22, 0.22],   # 1: maps to tactical 1, valid -> fills
        [0.33, 0.33],   # 2: maps to tactical 2, invalid homography -> stays NaN
        [0.44, 0.44],   # 3: maps to tactical 3, lowest broadcast index of the pair -> wins
        [0.55, 0.55],   # 4: also maps to tactical 3, loses the tie
        [0.66, 0.66],   # 5: maps to tactical 5, which is observed -> untouched
        [0.77, 0.77],   # 6: maps outside the tactical timeline -> ignored
        [0.88, np.nan],  # 7: maps to tactical 7, y missing -> stays NaN
    ])
    valid = np.array([True, True, False, True, True, True, True, True])
    broadcast = orient_broadcast_ball(raw_broadcast, IDENTITY, UNIT_EXTENT, valid=valid)
    mapping = [-1, 1, 2, 3, 3, 5, 99, 7]

    out = fused_ball(tactical, broadcast, mapping)

    filled = np.array([1, 3])
    observed = np.array([0, 5, 9])
    assert np.array_equal(out[filled, :2], np.array([[0.22, 0.22], [0.44, 0.44]]))
    assert np.isnan(out[filled, 2]).all()                            # speed of filled frames is NaN
    assert np.array_equal(out[observed], tactical[observed])
    untouched = np.setdiff1d(np.arange(n_tactical), np.r_[filled, observed])
    assert np.isnan(out[untouched]).all()                            # nothing else gets a value


def test_fused_ball_rot180_mapping_is_oriented_before_filling():
    tactical = _tactical(3, {})
    broadcast = np.array([[0.2, 0.3]])
    out = fused_ball(tactical, broadcast, _mapping([1], ROT180), extent=UNIT_EXTENT)
    np.testing.assert_allclose(out[1, :2], [0.8, 0.7])
    assert np.isnan(out[1, 2])
    with pytest.raises(ValueError):
        fused_ball(tactical, broadcast, _mapping([1], ROT180))      # rot180 needs the extent


def test_fused_ball_rejects_bad_shapes():
    with pytest.raises(ValueError):
        fused_ball(np.zeros((4, 2)), np.zeros((4, 2)), np.arange(4))
    with pytest.raises(ValueError):
        fused_ball(np.zeros((4, 3)), np.zeros((4, 2)), np.arange(3))


# -- fusion happens before the jump filter, per half ---------------------------------------------

N_HALF_FRAMES = 120
GAP_FIRST, GAP_LAST = 40, 59
PLAUSIBLE_FRAME, SPIKE_FRAME = 45, 50


def _trajectory(frame):
    """Slow ball path in normalized units along x (about 0.26 m/s), identical in both halves."""
    return 0.20 + 1.0e-4 * (frame % N_HALF_FRAMES)


def _synthetic_match():
    """Two contiguous halves with players placed so that the second half is reflected."""
    from preprocessing.schema import N_SLOTS

    n_frames = 2 * N_HALF_FRAMES
    teams = np.arange(N_SLOTS) % 2
    rank = np.arange(N_SLOTS) // 2
    left = 0.10 + 0.30 * rank / (N_SLOTS // 2)          # x in the left half of the pitch
    y = 0.10 + 0.80 * np.arange(N_SLOTS) / N_SLOTS      # distinct y, so no duplicate removal
    players = np.zeros((n_frames, N_SLOTS, 3), dtype=np.float32)
    for first, home_left in ((0, True), (N_HALF_FRAMES, False)):
        home_x = np.where(teams == 0, left, 1.0 - left)
        x = home_x if home_left else 1.0 - home_x
        players[first:first + N_HALF_FRAMES, :, 0] = x
        players[first:first + N_HALF_FRAMES, :, 1] = y
    blocks = [(1, 0, N_HALF_FRAMES - 1), (2, N_HALF_FRAMES, n_frames - 1)]
    return players, teams, blocks


def _tactical_ball_with_gap():
    ball = np.zeros((2 * N_HALF_FRAMES, 3), dtype=np.float64)
    frames = np.arange(2 * N_HALF_FRAMES)
    ball[:, 0] = _trajectory(frames)
    ball[:, 1] = 0.50
    ball[GAP_FIRST:GAP_LAST + 1, :] = np.nan
    return ball


def test_fused_ball_enters_before_jump_filter_and_respects_halves():
    from preprocessing.prep_tracking import prepare_tracking

    players, teams, blocks = _synthetic_match()
    tactical = _tactical_ball_with_gap()

    # Broadcast: a plausible position at PLAUSIBLE_FRAME and an anomalous far jump at SPIKE_FRAME.
    n_frames = tactical.shape[0]
    broadcast = np.full((n_frames, 2), np.nan)
    broadcast[PLAUSIBLE_FRAME] = (_trajectory(PLAUSIBLE_FRAME), 0.50)
    broadcast[SPIKE_FRAME] = (0.95, 0.50)
    fused = fused_ball(tactical, broadcast, np.arange(n_frames))
    assert np.isfinite(fused[[PLAUSIBLE_FRAME, SPIKE_FRAME], :2]).all()

    frame = prepare_tracking(players, fused.astype(np.float32), teams, blocks=blocks)[0]
    removed = np.flatnonzero(frame["ball_jump_removed"].to_numpy() == 1)
    observed = frame["ball_observed"].to_numpy() == 1

    # A4 saw the fused spike and removed it; the plausible fused frame survives as an observation.
    assert removed.tolist() == [SPIKE_FRAME]
    assert not observed[SPIKE_FRAME]
    assert observed[PLAUSIBLE_FRAME]
    # No tactical observation was removed in either half. Half 2 is reflected relative to half 1,
    # so a filter that crossed the boundary would cut one of the two halves.
    assert not (removed >= N_HALF_FRAMES).any()

    # Control: the same match with the tactical ball only has no removals at all.
    control = prepare_tracking(players, tactical.astype(np.float32), teams, blocks=blocks)[0]
    assert int(control["ball_jump_removed"].sum()) == 0


# -- default path stays untouched ------------------------------------------------------------------

def _stub_run_all_steps(monkeypatch):
    """Replace every pipeline step of run_all by a recorder; returns the step-A argv list."""
    from preprocessing import (build_master_table, export_tagging, prep_homography,
                               prep_proximities, prep_tracking, run_all)

    tracking_calls = []
    monkeypatch.setattr(export_tagging, "export_tagging_data", lambda *a, **k: None)
    monkeypatch.setattr(prep_homography, "export_homography_data", lambda *a, **k: None)
    monkeypatch.setattr(prep_tracking, "main", lambda argv: tracking_calls.append(list(argv)) or 0)
    monkeypatch.setattr(prep_proximities, "main", lambda argv: 0)
    monkeypatch.setattr(build_master_table, "main", lambda argv: 0)
    return run_all, tracking_calls


def test_tactical_ball_source_bit_identical_regression(tmp_path, monkeypatch):
    from preprocessing import prep_tracking

    # 1) build(): the tactical source returns exactly what read_atd returned, byte for byte,
    #    and never touches the broadcast code.
    rng = np.random.default_rng(0)
    players = rng.random((7, 4, 3), dtype=np.float32)
    ball = rng.random((7, 3), dtype=np.float32)
    ball[2] = np.nan
    teams = np.array([0, 1, 0, 1])
    for name in ("meta.xml", "raw.txt"):
        (tmp_path / name).write_text("synthetic")
    monkeypatch.setattr(prep_tracking, "asset_paths", lambda **k: {
        "atd_meta": str(tmp_path / "meta.xml"), "atd_raw": str(tmp_path / "raw.txt")})
    monkeypatch.setattr(prep_tracking, "read_atd", lambda raw, meta: (players, ball, teams))

    def _must_not_run(*args, **kwargs):
        raise AssertionError("the tactical source must not run the broadcast fusion")

    monkeypatch.setattr(prep_tracking, "fuse_broadcast_ball", _must_not_run)
    for kwargs in ({}, {"ball_source": "tactical"}):
        got_players, got_ball, got_teams = prep_tracking.build(**kwargs)
        assert got_ball.dtype == ball.dtype and got_ball.tobytes() == ball.tobytes()
        assert got_players.tobytes() == players.tobytes()
        assert np.array_equal(got_teams, teams)
    with pytest.raises(ValueError):
        prep_tracking.build(ball_source="broadcast")

    # 2) run_all: the step-A arguments of the default and of the explicit tactical source are
    #    exactly the pre-F4-06 ones; only the fused source adds a flag.
    run_all, tracking_calls = _stub_run_all_steps(monkeypatch)
    root, processed = tmp_path / "bundle", tmp_path / "bundle" / "processed"
    legacy = ["--data-dir", str(root), "--output-dir", str(processed), "--stage", "V1"]
    assert run_all._run_steps(root, processed, True) == 0
    assert run_all._run_steps(root, processed, True, ball_source="tactical") == 0
    assert run_all._run_steps(root, processed, True, ball_source="fused") == 0
    assert tracking_calls[0] == legacy
    assert tracking_calls[1] == legacy
    assert tracking_calls[2] == legacy + ["--ball-source", "fused"]


def test_run_all_cli_passes_ball_source_and_processed_dir(tmp_path, monkeypatch):
    from preprocessing import run_all

    calls = []
    monkeypatch.setattr(run_all, "run_pipeline", lambda **kwargs: calls.append(kwargs) or 0)
    target = tmp_path / "processed_fused"
    assert run_all.main(["--ball-source", "fused", "--processed-dir", str(target)]) == 0
    assert run_all.main([]) == 0
    assert calls[0]["ball_source"] == "fused" and calls[0]["processed_dir"] == str(target)
    assert calls[1]["ball_source"] == "tactical"
    with pytest.raises(SystemExit):
        run_all.main(["--ball-source", "broadcast"])


# -- make_submission routing ---------------------------------------------------------------------

_CONFIG_TEMPLATE = """name: {name}
pipeline: baseline
ball_source: {ball_source}
baseline:
  d_gap_frames: 100
  slow_speed_mps: 1.5
  min_spell_frames: 25
"""


def _prepare_make_submission(tmp_path, monkeypatch, ball_source):
    """Write a config under tmp_path and stub every step that would touch data."""
    import pandas as pd

    import games
    from modeling import io, make_submission

    name = "route_{}".format(ball_source)
    config = tmp_path / "{}.yaml".format(name)
    config.write_text(_CONFIG_TEMPLATE.format(name=name, ball_source=ball_source))

    pipeline_calls, master_calls = [], []
    monkeypatch.setattr("preprocessing.run_all.run_pipeline",
                        lambda **kwargs: pipeline_calls.append(kwargs) or 0)
    monkeypatch.setattr(io, "load_master", lambda path=None, n_frames=None: (
        master_calls.append((path, n_frames)) or pd.DataFrame({"half": np.array([1, 1, 2, 2])})))
    monkeypatch.setattr(make_submission, "predict_labels",
                        lambda master, cfg: np.array(["H", "A", "H", "A"]))

    def _write(labels, out):
        out = Path(out)
        out.write_bytes(b"synthetic")
        return out

    monkeypatch.setattr(io, "write_submission", _write)
    monkeypatch.setattr(make_submission, "label_summary", lambda labels, half: {
        "n_graded": 4, "shares": {s: 0.0 for s in make_submission.STATES}, "possessions": {}})
    n_frames = int(games.GAMES[games.GAME]["n_video_frames"])
    out = tmp_path / "{}.csv".format(name)
    argv = ["--config", str(config), "--out", str(out), "--data-dir", str(tmp_path)]
    return make_submission, io, argv, pipeline_calls, master_calls, n_frames


def test_make_submission_fused_routes_to_processed_fused(tmp_path, monkeypatch):
    make_submission, io, argv, pipeline_calls, master_calls, n_frames = _prepare_make_submission(
        tmp_path, monkeypatch, "fused")

    assert make_submission.main(argv) == 0

    assert pipeline_calls == [{"data_dir": str(tmp_path),
                               "processed_dir": tmp_path / "processed_fused",
                               "ball_source": "fused"}]
    assert master_calls == [(tmp_path / "processed_fused" / io.MASTER_FILENAME, n_frames)]


def test_make_submission_tactical_keeps_processed(tmp_path, monkeypatch):
    make_submission, io, argv, pipeline_calls, master_calls, n_frames = _prepare_make_submission(
        tmp_path, monkeypatch, "tactical")

    assert make_submission.main(argv) == 0

    assert pipeline_calls == [{"data_dir": str(tmp_path)}]          # exactly the existing call
    assert master_calls == [(tmp_path / "processed" / io.MASTER_FILENAME, n_frames)]


def test_make_submission_fused_without_preprocessing_loads_processed_fused(tmp_path, monkeypatch):
    make_submission, io, argv, pipeline_calls, master_calls, _ = _prepare_make_submission(
        tmp_path, monkeypatch, "fused")
    env_root = tmp_path / "env_bundle"
    monkeypatch.setenv("MTAH_DATA_DIR", str(env_root))
    argv = [a for a in argv if a != str(tmp_path)]
    argv.remove("--data-dir")

    assert make_submission.main(argv + ["--skip-preprocessing"]) == 0

    assert pipeline_calls == []
    assert master_calls[0][0] == env_root / "processed_fused" / io.MASTER_FILENAME