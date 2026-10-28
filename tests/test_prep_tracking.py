"""Synthetic tests for the streaming ATD load stage; no match assets are read."""
import numpy as np
import pytest

from preprocessing import prep_tracking as pt


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
    players[0:3, 0, :2] = np.array([[0.30, .40], [.31, .40], [.32, .40]])
    players[3:6, 0, :2] = np.array([[.70, .40], [.69, .40], [.68, .40]])
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
