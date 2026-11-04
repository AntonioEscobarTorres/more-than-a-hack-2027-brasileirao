"""Synthetic tests for the streaming ATD load stage; no match assets are read."""
import numpy as np
import pandas as pd
import pytest

from preprocessing import prep_tracking as pt
from preprocessing.schema import slot_columns


def _write_atd(path, frames):
    rows = []
    for frame_id, slot_groups, ball in frames:
        rows.append(f"{frame_id}:{';'.join(slot_groups)}:{ball}\n")
    path.write_text("".join(rows), encoding="utf-8")


def test_read_atd_maps_frames_slots_ball_and_missing_to_nan(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, "team_of_each_track", lambda _: np.arange(pt.N_SLOTS) % 2)
    empty = ["NaN"] * pt.N_SLOTS
    first = empty.copy()
    first[0] = "0.1,0.2,3.5"
    _write_atd(tmp_path / "raw.txt", [
        (1, first, "0.3,0.4,0"),
        (2, empty, "NaN"),
    ])

    players, ball, teams = pt.read_atd(tmp_path / "raw.txt", "unused.xml", n_frames=2)

    assert players.shape == (2, pt.N_SLOTS, 3)
    np.testing.assert_allclose(players[0, 0], [0.1, 0.2, 3.5])
    assert np.isnan(players[0, 1]).all() and np.isnan(players[1]).all()
    np.testing.assert_allclose(ball[0], [0.3, 0.4, 0.0])
    assert np.isnan(ball[1]).all()
    np.testing.assert_array_equal(teams, np.arange(pt.N_SLOTS) % 2)


@pytest.mark.parametrize("frames", [
    [(2, ["NaN"] * pt.N_SLOTS, "NaN"), (1, ["NaN"] * pt.N_SLOTS, "NaN")],
    [(1, ["NaN"] * pt.N_SLOTS, "NaN"), (1, ["NaN"] * pt.N_SLOTS, "NaN")],
])
def test_read_atd_rejects_out_of_order_and_duplicate_frames(tmp_path, monkeypatch, frames):
    monkeypatch.setattr(pt, "team_of_each_track", lambda _: np.zeros(pt.N_SLOTS))
    _write_atd(tmp_path / "raw.txt", frames)

    with pytest.raises(ValueError, match="out of order|duplicate"):
        pt.read_atd(tmp_path / "raw.txt", "unused.xml", n_frames=2)


def test_build_uses_asset_paths_and_returns_raw_arrays(tmp_path, monkeypatch):
    raw, meta = tmp_path / "atd.txt", tmp_path / "atd.xml"
    raw.touch()
    meta.touch()
    monkeypatch.setattr(pt, "asset_paths", lambda **kwargs: {"atd_raw": str(raw), "atd_meta": str(meta)})
    expected = (np.zeros((2, pt.N_SLOTS, 3)), np.zeros((2, 3)), np.zeros(pt.N_SLOTS))
    monkeypatch.setattr(pt, "read_atd", lambda raw_path, meta_path: expected)

    players, ball, teams = pt.build(data_dir=str(tmp_path))

    assert players.shape == (2, pt.N_SLOTS, 3)
    assert ball.shape == (2, 3)
    assert teams.shape == (pt.N_SLOTS,)


def test_prepare_tracking_keeps_halves_independent_and_outside_empty(monkeypatch):
    # Adjacent blocks make any accidental full-array interpolation/gradient visible.
    blocks = [(1, 1, 3), (2, 4, 6)]
    players = np.full((8, pt.N_SLOTS, 3), np.nan, dtype=np.float32)
    teams = np.arange(pt.N_SLOTS) % 2
    players[1:4, 0, :2] = (0.2, 0.5)
    players[4:7, 0, :2] = (0.7, 0.5)
    ball = np.full((8, 3), np.nan, dtype=np.float32)
    ball[1, :2] = (0.2, 0.5)
    ball[3, :2] = (0.2, 0.5)
    ball[4, :2] = (0.8, 0.5)
    ball[6, :2] = (0.8, 0.5)
    monkeypatch.setattr(pt, "MAX_GAP_FRAMES", 3)

    frame, _ = pt.prepare_tracking(players, ball, teams, blocks=blocks, n_frames=8)

    assert frame["half"].tolist() == [0, 1, 1, 1, 2, 2, 2, 0]
    assert frame.loc[[0, 7], "ball_observed"].tolist() == [0, 0]
    assert frame.loc[[0, 7], "ball_filled"].tolist() == [0, 0]
    derived = ["ball_x", "ball_vx", "ball_acc", "gap_length",
               "centroid_home_x", "mean_speed_home"]
    assert frame.loc[[0, 7], derived].isna().all().all()
    assert frame.loc[2, "ball_filled"] == 1
    assert frame.loc[5, "ball_filled"] == 1
    # Constant trajectories in each half have zero speed; the boundary is never used.
    np.testing.assert_allclose(frame.loc[1:3, "s01_vx"], 0)
    np.testing.assert_allclose(frame.loc[4:6, "s01_vx"], 0)


def test_flip_orients_coordinates_and_velocity_and_halves_oppose():
    blocks = [(1, 0, 2), (2, 3, 5)]
    teams = np.arange(pt.N_SLOTS) % 2
    players = np.full((6, pt.N_SLOTS, 3), np.nan, np.float32)
    # Home moves toward low x in half 1, high x in half 2: only half 2 is reflected.
    players[0:3, 0, :2] = np.array([[0.30, .40], [.301, .40], [.302, .40]])
    players[3:6, 0, :2] = np.array([[.70, .40], [.699, .40], [.698, .40]])
    ball = np.full((6, 3), np.nan, np.float32)
    ball[0:3, :2] = np.array([[.2, .5], [.21, .5], [.22, .5]])
    ball[3:6, :2] = np.array([[.8, .5], [.79, .5], [.78, .5]])
    frame, _ = pt.prepare_tracking(players, ball, teams, blocks=blocks,
                                   kickoffs={1: 0, 2: 3}, n_frames=6)
    assert frame.groupby("half")["flip"].first().tolist() == [0, 1]
    assert frame["flip"].dtype == np.int8
    assert frame.loc[0, "s01_vx"] > 0
    assert frame.loc[3, "s01_vx"] > 0  # original negative velocity was reflected
    assert frame.loc[0, "ball_vx"] > 0
    assert frame.loc[3, "ball_vx"] > 0
    # Raw audit fields preserve the input orientation even when clean positions flip.
    np.testing.assert_allclose(frame.loc[3, "ball_x_raw"], 0.8 * pt.PITCH_L, rtol=1e-6)
    assert frame.loc[0:2, "centroid_home_x"].is_monotonic_increasing
    assert frame.loc[3:5, "centroid_home_x"].is_monotonic_increasing


def test_v0_matches_frozen_column_order_dtypes_and_raw_observations():
    n = 8
    blocks = [(1, 1, 3), (2, 4, 6)]
    teams = np.arange(pt.N_SLOTS) % 2
    players = np.full((n, pt.N_SLOTS, 3), np.nan, dtype=np.float32)
    # Home starts in its own half in both blocks; half 2 is reflected by A3.
    players[1:4, 0, :2] = np.column_stack((.2 + .01 * np.arange(3), [.4] * 3))
    players[4:7, 0, :2] = np.column_stack((.8 - .01 * np.arange(3), [.4] * 3))
    players[2, 1, :2] = (.3, .3)  # isolated observation remains in V0, with no derivative
    ball = np.full((n, 3), np.nan, dtype=np.float32)
    ball[1:4, :2] = np.column_stack((.1 + .01 * np.arange(3), [.5] * 3))
    ball[4:7, :2] = np.column_stack((.9 - .01 * np.arange(3), [.5] * 3))

    frame, _ = pt.prepare_tracking(
        players, ball, teams, blocks=blocks, n_frames=n,
        kickoffs={1: 1, 2: 4}, stage="V0",
    )

    expected = ["video_frame", "half", "flip", "ball_x", "ball_y", "ball_vx", "ball_vy"]
    expected += slot_columns()
    assert frame.columns.tolist() == expected
    assert frame["video_frame"].dtype == np.dtype("int32")
    assert frame["half"].dtype == np.dtype("int8")
    assert frame["flip"].dtype == np.dtype("int8")
    assert all(frame[f"s{i:02d}_break"].dtype == np.dtype("int8")
               for i in range(1, pt.N_SLOTS + 1))
    assert all(frame[c].dtype == np.dtype("float32") for c in
               ["ball_x", "ball_y", "ball_vx", "ball_vy"] +
               [c for c in expected if c.endswith(("_x", "_y", "_vx", "_vy"))])
    assert all(frame[f"s{i:02d}_break"].eq(0).all() for i in range(1, pt.N_SLOTS + 1))
    # Coordinates preserve observations; the single-frame track has no derivative.
    np.testing.assert_allclose(frame.loc[1:3, "s01_x"], [21.06, 22.113, 23.166], atol=1e-4)
    np.testing.assert_allclose(frame.loc[2, "s02_x"], .3 * pt.PITCH_L, atol=1e-5)
    assert frame.loc[2, "s02_vx"] != frame.loc[2, "s02_vx"]
    assert frame.loc[[0, 7], ["ball_x", "ball_y", "ball_vx", "ball_vy", "s01_x", "s01_vx"]].isna().all().all()
    # Derivatives use the reflected, metre-scaled positions and do not bridge halftime.
    np.testing.assert_allclose(frame.loc[1:3, "s01_vx"], pt.PITCH_L * .01 * pt.FPS, atol=1e-4)
    np.testing.assert_allclose(frame.loc[4:6, "s01_vx"], pt.PITCH_L * .01 * pt.FPS, atol=1e-4)


def test_parquet_writer_persists_schema_and_stage_metadata(tmp_path):
    import pyarrow.parquet as pq

    path = tmp_path / "tracking.parquet"
    pt.write_parquet_with_meta(pd.DataFrame({"video_frame": np.array([0], dtype=np.int32)}),
                               path, tracking_stage="V0")

    metadata = pq.read_schema(path).metadata
    assert metadata[b"schema_version"].decode() == pt.SCHEMA_VERSION
    assert metadata[b"tracking_stage"].decode() == "V0"


def test_v1_requires_exact_kickoffs_inside_c4_tag_windows(tmp_path):
    import json

    proc_dir = tmp_path / "processed"
    proc_dir.mkdir()
    payload = {
        "1": pt.KICKOFF_FALLBACK[1],
        "2": pt.KICKOFF_FALLBACK[2],
        "tag_windows": {
            "1": [pt.KICKOFF_FALLBACK[1] - 3, pt.KICKOFF_FALLBACK[1] + 3],
            "2": [pt.KICKOFF_FALLBACK[2] - 3, pt.KICKOFF_FALLBACK[2] + 3],
        },
    }
    path = proc_dir / "kickoffs.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert pt._kickoff_frames(str(tmp_path), required=True) == pt.KICKOFF_FALLBACK

    payload["1"] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="expected"):
        pt._kickoff_frames(str(tmp_path), required=True)


def test_kickoffs_are_read_from_a_custom_processed_dir(tmp_path):
    """run_all --processed-dir: export_tagging writes kickoffs.json there, so A must read it there."""
    import json

    custom = tmp_path / "custom_out"
    custom.mkdir()
    payload = {
        "1": pt.KICKOFF_FALLBACK[1],
        "2": pt.KICKOFF_FALLBACK[2],
        "tag_windows": {str(h): [f - 3, f + 3] for h, f in pt.KICKOFF_FALLBACK.items()},
    }
    (custom / "kickoffs.json").write_text(json.dumps(payload), encoding="utf-8")

    data_dir = tmp_path / "data"                     # <data-dir>/processed has no kickoffs.json
    with pytest.raises(FileNotFoundError):
        pt._kickoff_frames(str(data_dir), required=True)
    assert pt._kickoff_frames(str(data_dir), required=True,
                              processed_dir=str(custom)) == pt.KICKOFF_FALLBACK


def test_v1_rejects_kickoff_outside_c4_tag_window(tmp_path):
    import json

    proc_dir = tmp_path / "processed"
    proc_dir.mkdir()
    payload = {
        "1": pt.KICKOFF_FALLBACK[1],
        "2": pt.KICKOFF_FALLBACK[2],
        "tag_windows": {
            "1": [pt.KICKOFF_FALLBACK[1] + 1, pt.KICKOFF_FALLBACK[1] + 3],
            "2": [pt.KICKOFF_FALLBACK[2] - 3, pt.KICKOFF_FALLBACK[2] + 3],
        },
    }
    (proc_dir / "kickoffs.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="outside tag window"):
        pt._kickoff_frames(str(tmp_path), required=True)


def test_team_aggregates_are_invariant_to_slots_swapping_player_identity():
    # Each frame contains the same players, positions, and velocities. Their slot
    # assignments swap in frame 2, as can happen when track slots change identity.
    px = np.array([[10.0, 20.0, 70.0, 80.0],
                   [20.0, 10.0, 80.0, 70.0]], dtype=np.float32)
    py = np.array([[10.0, 30.0, 40.0, 50.0],
                   [30.0, 10.0, 50.0, 40.0]], dtype=np.float32)
    vx = np.array([[0.0, 1.0, 2.0, 3.0],
                   [1.0, 0.0, 3.0, 2.0]], dtype=np.float32)
    vy = np.zeros_like(vx)
    teams = np.array([0, 0, 1, 1])

    aggregates = pt._team_aggregates(px, py, vx, vy, teams, np.ones(4, dtype=bool))

    for values in aggregates.values():
        np.testing.assert_allclose(values[0], values[1])


def test_team_aggregates_use_eligibility_mask_and_zero_counts_when_none_tracked():
    px = np.array([[10.0, 100.0, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)
    py = np.array([[20.0, 100.0, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)
    vx = np.zeros_like(px)
    vy = np.zeros_like(py)
    teams = np.array([0, 0, 1])
    field_players = np.array([True, False, True])

    aggregates = pt._team_aggregates(px, py, vx, vy, teams, field_players)

    assert aggregates["centroid_home_x"][0] == 10.0
    assert aggregates["n_tracked_home"][0] == 1.0
    assert aggregates["n_still_home"][0] == 1.0
    assert aggregates["n_tracked_home"][1] == 0
    assert np.isnan(aggregates["centroid_away_x"]).all()
    assert np.all(aggregates["n_tracked_away"] == 0)
    assert np.all(aggregates["n_still_away"] == 0)
    assert np.isnan(aggregates["mean_speed_away"]).all()


def test_all_eleven_slots_including_goalkeeper_contribute_to_aggregates():
    teams = np.r_[np.zeros(11, dtype=np.int8), np.ones(11, dtype=np.int8)]
    px = np.full((1, 22), np.nan, dtype=np.float32)
    py = np.full_like(px, np.nan)
    vx = np.zeros_like(px)
    vy = np.zeros_like(px)
    # Home's first slot represents the goalkeeper; all 11 players count.
    px[0, :11] = np.arange(1, 12, dtype=np.float32)
    py[0, :11] = 10
    px[0, 11:] = np.arange(20, 31, dtype=np.float32)
    py[0, 11:] = 20
    aggregates = pt._team_aggregates(px, py, vx, vy, teams, np.ones(22, dtype=bool))
    assert aggregates["n_tracked_home"][0] == 11
    assert aggregates["centroid_home_x"][0] == 6
    assert aggregates["spread_home_x"][0] == pytest.approx(np.sqrt(10))
    assert aggregates["centroid_home_y"][0] == 10
    assert aggregates["spread_home_y"][0] == 0


def test_prepare_tracking_defaults_all_slots_to_eligible_including_goalkeeper():
    teams = np.r_[np.zeros(11, dtype=np.int8), np.ones(pt.N_SLOTS - 11, dtype=np.int8)]
    players = np.full((2, pt.N_SLOTS, 3), np.nan, dtype=np.float32)
    players[:, 0, :2] = (0.02, 0.5)  # goalkeeper near the goal
    players[:, 1, :2] = (0.2, 0.5)
    ball = np.full((2, 3), np.nan, dtype=np.float32)
    frame, slots = pt.prepare_tracking(players, ball, teams, blocks=[(1, 0, 1)],
                                       kickoffs={1: 0}, n_frames=2)
    assert slots["is_field_player"].all()
    assert (frame["n_tracked_home"] == 2).all()
    np.testing.assert_allclose(frame["centroid_home_x"], 0.11 * pt.PITCH_L)


def test_no_tracked_players_yields_nan_metrics_and_zero_counts():
    teams = np.array([0, 1])
    missing = np.full((2, 2), np.nan, dtype=np.float32)
    zeros = np.zeros_like(missing)
    aggregates = pt._team_aggregates(missing, missing, zeros, zeros, teams,
                                     np.ones(2, dtype=bool))
    for team in ("home", "away"):
        for name in (f"centroid_{team}_x", f"centroid_{team}_y",
                     f"spread_{team}_x", f"spread_{team}_y", f"mean_speed_{team}"):
            assert np.isnan(aggregates[name]).all()
        assert np.all(aggregates[f"n_tracked_{team}"] == 0)
        assert np.all(aggregates[f"n_still_{team}"] == 0)


def test_team_aggregate_team_membership_does_not_mix_sides():
    px = np.array([[10.0, 90.0]], dtype=np.float32)
    py = np.array([[20.0, 40.0]], dtype=np.float32)
    zeros = np.zeros_like(px)
    mask = np.ones(2, dtype=bool)
    original = pt._team_aggregates(px, py, zeros, zeros, np.array([0, 1]), mask)
    reassigned = pt._team_aggregates(px, py, zeros, zeros, np.array([1, 1]), mask)
    assert original["centroid_home_x"][0] == 10
    assert original["centroid_away_x"][0] == 90
    assert np.isnan(reassigned["centroid_home_x"][0])
    assert reassigned["centroid_away_x"][0] == 50


def test_aggregate_duplicate_slots_are_removed_before_count_and_centroid():
    blocks = [(1, 0, 2)]
    teams = np.zeros(pt.N_SLOTS, dtype=np.int8)
    teams[2:] = 1
    players = np.full((3, pt.N_SLOTS, 3), np.nan, dtype=np.float32)
    players[:, 0, :2] = (0.2, 0.5)
    players[:, 1, :2] = (0.2 + pt.DEDUP_DIST / pt.PITCH_L / 2, 0.5)
    players[:, 2, :2] = (0.8, 0.5)
    ball = np.full((3, 3), np.nan, dtype=np.float32)
    frame, _ = pt.prepare_tracking(players, ball, teams, blocks=blocks,
                                   kickoffs={1: 0}, n_frames=3)
    assert (frame["n_tracked_home"] == 1).all()
    np.testing.assert_allclose(frame["centroid_home_x"], 0.2 * pt.PITCH_L)
    assert frame["s02_x"].isna().all()


def test_aggregate_columns_are_nan_outside_tracking_periods_and_repeat_deterministically():
    blocks = [(1, 1, 3), (2, 4, 6)]
    teams = np.arange(pt.N_SLOTS) % 2
    players = np.full((8, pt.N_SLOTS, 3), np.nan, dtype=np.float32)
    players[1:4, 0, :2] = (0.2, 0.4)
    players[4:7, 0, :2] = (0.8, 0.4)
    ball = np.full((8, 3), np.nan, dtype=np.float32)
    ball[1:4, :2] = (0.2, 0.5)
    ball[4:7, :2] = (0.8, 0.5)
    args = (players, ball, teams)
    kwargs = dict(blocks=blocks, kickoffs={1: 1, 2: 4}, n_frames=8)
    first, _ = pt.prepare_tracking(*args, **kwargs)
    second, _ = pt.prepare_tracking(*args, **kwargs)
    aggregate_columns = [c for c in first if c.startswith((
        "centroid_", "spread_", "mean_speed_", "n_still_", "n_tracked_"))]
    assert first.loc[[0, 7], aggregate_columns].isna().all().all()
    np.testing.assert_allclose(first[aggregate_columns], second[aggregate_columns], equal_nan=True)


def test_aggregates_keep_home_left_of_away_at_both_oriented_kickoffs():
    blocks = [(1, 0, 2), (2, 3, 5)]
    teams = np.arange(pt.N_SLOTS) % 2
    players = np.full((6, pt.N_SLOTS, 3), np.nan, dtype=np.float32)
    # Two players per side. Home starts left in half 1 and right in half 2;
    # orientation reflects half 2 so Home is left at both kickoffs.
    players[:, 0, :2] = (np.array([.20, .21, .22, .80, .79, .78])[:, None]
                          * np.array([pt.PITCH_L, pt.PITCH_W]))
    players[:, 2, :2] = (np.array([.25, .26, .27, .75, .74, .73])[:, None]
                          * np.array([pt.PITCH_L, pt.PITCH_W]))
    players[:, 1, :2] = (np.array([.75, .74, .73, .25, .26, .27])[:, None]
                          * np.array([pt.PITCH_L, pt.PITCH_W]))
    players[:, 3, :2] = (np.array([.80, .79, .78, .20, .21, .22])[:, None]
                          * np.array([pt.PITCH_L, pt.PITCH_W]))
    # prepare_tracking expects normalized positions.
    players[:, :, 0] /= pt.PITCH_L
    players[:, :, 1] /= pt.PITCH_W
    ball = np.full((6, 3), np.nan, dtype=np.float32)
    ball[:, :2] = (np.array([.3, .31, .32, .7, .69, .68])[:, None]
                   * np.array([pt.PITCH_L, pt.PITCH_W]))
    ball[:, 0] /= pt.PITCH_L
    ball[:, 1] /= pt.PITCH_W
    frame, _ = pt.prepare_tracking(players, ball, teams, blocks=blocks,
                                   kickoffs={1: 0, 2: 3}, n_frames=6)
    assert frame.loc[0, "centroid_home_x"] < frame.loc[0, "centroid_away_x"]
    assert frame.loc[3, "centroid_home_x"] < frame.loc[3, "centroid_away_x"]
    assert frame.groupby("half")["flip"].first().tolist() == [0, 1]


@pytest.mark.parametrize("field_player", [np.ones(pt.N_SLOTS - 1, dtype=bool),
                                           np.ones(pt.N_SLOTS, dtype=np.int8)])
def test_prepare_tracking_rejects_invalid_field_player_mask(field_player):
    players = np.full((1, pt.N_SLOTS, 3), np.nan, dtype=np.float32)
    ball = np.full((1, 3), np.nan, dtype=np.float32)
    teams = np.arange(pt.N_SLOTS) % 2
    with pytest.raises(ValueError, match="slot_is_field_player"):
        pt.prepare_tracking(players, ball, teams, blocks=[], n_frames=1,
                            slot_is_field_player=field_player)
"""Complementary tests for prep_tracking aligned with the actual implementation."""
import numpy as np
import pytest
from preprocessing import prep_tracking as pt


def test_kickoff_fallback_when_file_missing(tmp_path):
    """Ensures fallback to KICKOFF_FALLBACK when kickoffs.json does not exist in data_dir."""
    # Pass an empty directory
    kickoffs = pt._kickoff_frames(data_dir=str(tmp_path))
    assert kickoffs[1] == pt.KICKOFF_FALLBACK[1]
    assert kickoffs[2] == pt.KICKOFF_FALLBACK[2]


def test_kickoff_loads_from_valid_json(tmp_path):
    """Ensures correct reading when kickoffs.json exists."""
    import json
    proc_dir = tmp_path / "processed"
    proc_dir.mkdir(parents=True)
    payload = {"1": 12000, "2": 115000}
    (proc_dir / "kickoffs.json").write_text(json.dumps(payload), encoding="utf-8")

    kickoffs = pt._kickoff_frames(data_dir=str(tmp_path))
    assert kickoffs[1] == 12000
    assert kickoffs[2] == 115000


def test_coordinates_and_velocities_scaled_to_meters():
    """Ensures prepare_tracking outputs are in meters (PITCH_L x PITCH_W)."""
    blocks = [(1, 0, 1)]
    teams = np.arange(pt.N_SLOTS) % 2
    players = np.full((2, pt.N_SLOTS, 3), np.nan, dtype=np.float32)
    # Player 0 at normalized center (0.5, 0.5)
    players[:, 0, :2] = [0.5, 0.5]
    ball = np.full((2, 3), np.nan, dtype=np.float32)
    ball[:, :2] = [0.5, 0.5]

    frame, _ = pt.prepare_tracking(
        players, ball, teams, blocks=blocks, pitch=(105.3, 68.0), kickoffs={1: 0}, n_frames=2
    )

    # 0.5 * 105.3 = 52.65m; 0.5 * 68.0 = 34.0m
    np.testing.assert_allclose(frame.loc[0, "s01_x"], 52.65, rtol=1e-3)
    np.testing.assert_allclose(frame.loc[0, "s01_y"], 34.00, rtol=1e-3)
    np.testing.assert_allclose(frame.loc[0, "ball_x"], 52.65, rtol=1e-3)
    np.testing.assert_allclose(frame.loc[0, "ball_y"], 34.00, rtol=1e-3)


def test_orient_half_reflects_x_y_and_vectors():
    """Validates direct reflection by _orient_half on coordinates and velocities."""
    length, width = 105.3, 68.0
    players = np.full((5, pt.N_SLOTS, 3), np.nan, dtype=np.float32)
    ball = np.full((5, 3), np.nan, dtype=np.float32)
    teams = [0] + [1] * (pt.N_SLOTS - 1)  # slot 0 is home team (0)

    # Home team at high x (> length/2) -> Must force flip=True
    players[:, 0, :2] = [80.0, 20.0]
    players[:, 0, 2] = 5.0  # raw scalar speed
    ball[:, :2] = [70.0, 15.0]

    flipped = pt._orient_half(players, ball, teams, start=0, end=5, kick_frame=0, pitch=(length, width))

    assert flipped is True or flipped == 1
    # Reflected positions: L - x and W - y
    # Inputs and reflected arrays use float32, so allow their rounding error.
    np.testing.assert_allclose(players[0, 0, 0], length - 80.0, atol=1e-5)
    np.testing.assert_allclose(players[0, 0, 1], width - 20.0, atol=1e-5)
    np.testing.assert_allclose(ball[0, 0], length - 70.0, atol=1e-5)
    np.testing.assert_allclose(ball[0, 1], width - 15.0, atol=1e-5)
    # The third ATD player field is scalar speed and is unchanged by reflection.
    np.testing.assert_allclose(players[0, 0, 2], 5.0)


def test_remove_ball_jump_outliers_bridges_long_phantom_block_exceeding_k():
    """Ensures a block of >K (40 > 30) phantom frames is removed without breaking the real chain."""
    x = np.full(110, np.nan, dtype=np.float32)
    y = np.full(110, 34.0, dtype=np.float32)
    # Real ball before (35 frames) and after (35 frames), moving at 2.5 m/s (0.1 m/frame)
    x[0:35] = 10.0 + 0.1 * np.arange(35, dtype=np.float32)
    # Phantom block of 40 frames (> BALL_K = 30) ~35 m away
    x[35:75] = 48.0
    y[35:75] = 62.0
    # Real ball reappears in the natural continuation of the original trajectory
    x[75:110] = 10.0 + 0.1 * np.arange(75, 110, dtype=np.float32)

    xo, yo = pt.remove_ball_jump_outliers(x, y, k=30)

    assert np.isfinite(xo[0:35]).all() and np.isfinite(yo[0:35]).all()
    assert np.isnan(xo[35:75]).all() and np.isnan(yo[35:75]).all()
    assert np.isfinite(xo[75:110]).all() and np.isfinite(yo[75:110]).all()


def test_remove_ball_jump_outliers_keeps_disjoint_long_chains_after_multiball_jump():
    """Ensures two disjoint long chains (e.g., multi-ball restart) are preserved."""
    x = np.full(80, np.nan, dtype=np.float32)
    y = np.full(80, 34.0, dtype=np.float32)
    x[0:35] = 10.0 + 0.1 * np.arange(35, dtype=np.float32)
    x[35:75] = 80.0 - 0.1 * np.arange(40, dtype=np.float32)  # 2nd ball ~67 m away in 1 frame
    x[75:77] = 5.0  # short phantom spike of 2 frames at the end

    xo, _ = pt.remove_ball_jump_outliers(x, y, k=30, min_chain=30)

    assert np.isfinite(xo[0:35]).all()
    assert np.isfinite(xo[35:75]).all()
    assert np.isnan(xo[75:77]).all()


def test_ball_derivatives_masked_when_filled_and_smoothed_against_jitter():
    """Ensures velocities/acceleration are NaN where ball_filled == 1 and that Savitzky-Golay attenuates jitter."""
    n = 20
    blocks = [(1, 0, n - 1)]
    teams = np.arange(pt.N_SLOTS) % 2
    players = np.full((n, pt.N_SLOTS, 3), np.nan, dtype=np.float32)
    players[:, 0, :2] = [0.2, 0.5]
    ball = np.full((n, 3), np.nan, dtype=np.float32)

    # Linear trajectory with high-frequency jitter in the first 10 frames
    t_obs = np.arange(10, dtype=np.float32)
    jitter = (0.002 * np.sin(1.8 * t_obs)).astype(np.float32)
    ball[0:10, 0] = 0.2 + 0.002 * t_obs + jitter
    ball[0:10, 1] = 0.5
    # Short gap of 2 frames (frames 10 and 11 -> ball_filled == 1), followed by observations in 12..19
    ball[12:20, 0] = 0.2 + 0.002 * np.arange(12, 20, dtype=np.float32)
    ball[12:20, 1] = 0.5

    frame, _ = pt.prepare_tracking(players, ball, teams, blocks=blocks, kickoffs={1: 0}, n_frames=n)

    # Raw observed position (in meters) is not altered by the derivative filter
    np.testing.assert_allclose(frame.loc[0:9, "ball_x"], frame.loc[0:9, "ball_x_raw"], rtol=1e-6)
    # In interpolated frames (10 and 11), position exists but velocities and acceleration are NaN
    assert (frame.loc[10:11, "ball_filled"] == 1).all()
    assert frame.loc[10:11, ["ball_vx", "ball_vy", "ball_speed", "ball_acc"]].isna().all().all()
    # In observed frames (12..19), velocities and acceleration exist
    assert frame.loc[12:19, ["ball_vx", "ball_speed", "ball_acc"]].notna().all().all()
    # Smoothed acceleration in jitter segment is much lower than raw 2nd derivative
    raw_vx = np.gradient(frame.loc[0:9, "ball_x_raw"].to_numpy()) * pt.FPS
    raw_acc = np.abs(np.gradient(raw_vx) * pt.FPS)
    assert frame.loc[2:7, "ball_acc"].mean() < 0.5 * raw_acc[2:8].mean()


def test_ball_dir_change_detects_turn_and_masks_slow_ball():
    """Ensures ball_dir_change is ~0 in a straight line, >0 in a sharp turn, and NaN for a slow ball."""
    n = 14
    blocks = [(1, 0, n - 1)]
    teams = np.arange(pt.N_SLOTS) % 2
    players = np.full((n, pt.N_SLOTS, 3), np.nan, dtype=np.float32)
    players[:, 0, :2] = [0.2, 0.5]
    ball = np.full((n, 3), np.nan, dtype=np.float32)

    # Frames 0..6: fast ball (10.5 m/s) in a straight line along +x -> dir_change == 0 in 1..4
    for f in range(7):
        ball[f, :2] = [0.2 + 0.004 * f, 0.5]
    # Frames 7..10: 90-degree turn moving along +y (6.8 m/s)
    for f in range(7, 11):
        ball[f, :2] = [0.2 + 0.004 * 6, 0.5 + 0.004 * (f - 6)]
    # Frames 11..13: virtually stationary ball (< 1.0 m/s) -> dir_change must be NaN
    for f in range(11, 14):
        ball[f, :2] = [0.2 + 0.004 * 6, 0.5 + 0.004 * 4]

    frame, _ = pt.prepare_tracking(players, ball, teams, blocks=blocks, kickoffs={1: 0}, n_frames=n)

    np.testing.assert_allclose(frame.loc[1:2, "ball_dir_change"], 0.0, atol=1e-5)
    assert frame.loc[7, "ball_dir_change"] > 0.4  # sharp direction change at the corner
    assert frame.loc[12:13, "ball_dir_change"].isna().all()


def test_ghost_near_gap_end_is_removed_then_short_gap_is_interpolated():
    n = 24
    blocks = [(1, 0, n - 1)]
    players = np.full((n, pt.N_SLOTS, 3), np.nan, np.float32)
    players[:, 0, :2] = (0.2, 0.5)
    teams = np.arange(pt.N_SLOTS) % 2
    ball = np.full((n, 3), np.nan, np.float32)
    # Smooth 2.5 m/s path with a short missing interval and one distant ghost.
    true_x = 0.20 + 0.001 * np.arange(n)
    for f in list(range(0, 8)) + list(range(12, n)):
        ball[f] = (true_x[f], 0.5, 0.0)
    ball[11] = (0.75, 0.9, 0.0)
    frame, _ = pt.prepare_tracking(players, ball, teams, blocks=blocks,
                                   kickoffs={1: 0}, n_frames=n)
    assert frame.loc[11, "ball_jump_removed"] == 1
    assert frame.loc[8:11, "ball_filled"].eq(1).all()
    np.testing.assert_allclose(frame.loc[8:11, "ball_x"], true_x[8:12] * pt.PITCH_L, atol=1e-4)


def test_v1_smoothing_reduces_synthetic_quantization_acceleration():
    n = 80
    blocks = [(1, 0, n - 1)]
    teams = np.arange(pt.N_SLOTS) % 2
    players = np.full((n, pt.N_SLOTS, 3), np.nan, np.float32)
    players[:, 0, :2] = (0.2, 0.5)
    ball = np.full((n, 3), np.nan, np.float32)
    frames = np.arange(n)
    # Deterministic high-frequency tracking noise atop a realistic linear path.
    ball[:, 0] = 0.20 + 0.0004 * frames + 0.005 * np.sin(1.8 * frames)
    ball[:, 1] = 0.5
    v0, _ = pt.prepare_tracking(players, ball, teams, blocks=blocks,
                                kickoffs={1: 0}, n_frames=n, stage="V0")
    v1, _ = pt.prepare_tracking(players, ball, teams, blocks=blocks,
                                kickoffs={1: 0}, n_frames=n, stage="V1")
    # V0 intentionally has no acceleration column; derive its unsmoothed
    # acceleration from the V0 velocity to compare with the V1 output.
    a0 = np.abs(np.gradient(v0.loc[10:69, "ball_vx"].to_numpy()) * pt.FPS)
    a1 = v1.loc[10:69, "ball_acc"].to_numpy()
    assert np.nanmax(a1) < np.nanmax(a0) / 2.0
    assert np.nanmax(a1) < 600.0


def test_gap_features_and_short_long_gap_invariants():
    n = 50
    blocks = [(1, 0, n - 1)]
    teams = np.arange(pt.N_SLOTS) % 2
    players = np.full((n, pt.N_SLOTS, 3), np.nan, np.float32)
    players[:, 0, :2] = (0.2, 0.5)
    ball = np.full((n, 3), np.nan, np.float32)
    # Leading/trailing gaps verify the boundary semantics. Internal gaps have
    # lengths 2 and MAX_GAP_FRAMES + 1 and remain within one continuous path.
    observed_frames = np.r_[np.arange(2, 10), np.arange(12, 20), np.arange(20 + pt.MAX_GAP_FRAMES + 1, n - 2)]
    true_x = 0.2 + np.arange(n, dtype=np.float32) * 0.0005
    ball[observed_frames, 0] = true_x[observed_frames]
    ball[observed_frames, 1] = 0.5

    frame, _ = pt.prepare_tracking(players, ball, teams, blocks=blocks,
                                   kickoffs={1: 0}, n_frames=n)
    assert frame.loc[10:11, "ball_filled"].eq(1).all()
    assert frame.loc[10:11, "ball_x"].notna().all()
    long_start, long_end = 20, 20 + pt.MAX_GAP_FRAMES
    assert frame.loc[long_start:long_end, "ball_filled"].eq(0).all()
    assert frame.loc[long_start:long_end, "ball_x"].isna().all()
    assert frame.loc[:1, "since_last_obs"].isna().all()
    assert frame.loc[n - 2:, "until_next_obs"].isna().all()
    assert (frame["ball_x"].isna().to_numpy() ==
            ((frame["ball_observed"].to_numpy() == 0) & (frame["ball_filled"].to_numpy() == 0))).all()
