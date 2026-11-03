"""Synthetic tests for B1–B4. No match parquet is read."""
import numpy as np
import pandas as pd
import pytest

from preprocessing.params import N_FRAMES, PRESSURE_TAU
from preprocessing.prep_proximities import (
    ProximityInputs, assemble_inputs, contiguous_half_slices, load_proximity_inputs,
    masked_nearest, player_ball_distances, radius_pressure, relative_speeds,
    team_masks, team_proximity)
from preprocessing.prep_tracking import write_parquet_with_meta
from preprocessing.schema import N_SLOTS

T = 12
BLOCKS = [(1, 2, 5), (2, 8, 10)]


def _half_column(n=T, blocks=BLOCKS):
    half = np.zeros(n, dtype=np.int8)
    for half_id, first, last in blocks:
        half[first:last + 1] = half_id
    return half


def _frame(n=T, blocks=BLOCKS, outside_value=None, n_slots=N_SLOTS):
    half = _half_column(n, blocks)
    rng = np.random.default_rng(0)
    data = {
        "video_frame": np.arange(n, dtype=np.int32),
        "half": half,
    }
    for field in ("x", "y", "vx", "vy"):
        values = rng.random((n, n_slots)).astype(np.float32)
        values[half == 0] = np.nan
        if outside_value is not None:
            values[half == 0] = outside_value
        for slot in range(n_slots):
            data[f"s{slot + 1:02d}_{field}"] = values[:, slot]
    for name in ("ball_x", "ball_y", "ball_vx", "ball_vy"):
        ball = rng.random(n).astype(np.float32)
        ball[half == 0] = np.nan if outside_value is None else outside_value
        data[name] = ball
    return pd.DataFrame(data), half


def _slot_teams(n_slots=N_SLOTS, drop_field=()):
    is_home = np.arange(n_slots) < (n_slots // 2)
    is_field = np.ones(n_slots, dtype=bool)
    is_field[list(drop_field)] = False
    return pd.DataFrame({
        "slot": np.arange(1, n_slots + 1, dtype=np.int8),
        "is_home": is_home,
        "is_field_player": is_field,
    })


def _write(folder, frame, slot_teams, stage="V1"):
    tracking = folder / "tracking_clean.parquet"
    slots = folder / "slot_teams.parquet"
    write_parquet_with_meta(frame, tracking, tracking_stage=stage)
    write_parquet_with_meta(slot_teams, slots)
    return tracking, slots


def test_assemble_stacks_slots_in_order_and_keeps_nan():
    frame, half = _frame()
    frame.loc[3, "s01_x"] = np.nan
    frame.loc[3, "s02_x"] = 4.5
    teams = _slot_teams()

    inputs = assemble_inputs(frame, teams, "V1", n_frames=T)

    assert inputs.px.shape == (T, N_SLOTS)
    assert inputs.px.dtype == np.float32
    assert inputs.tracking_stage == "V1"
    assert np.isnan(inputs.px[3, 0])
    np.testing.assert_allclose(inputs.px[3, 1], 4.5)
    np.testing.assert_array_equal(inputs.half, half)
    assert inputs.half_slices() == ((1, slice(2, 6)), (2, slice(8, 11)))
    playing = half != 0
    np.testing.assert_array_equal(np.isfinite(inputs.ball_x), playing)


def test_frames_outside_play_are_forced_to_nan():
    frame, half = _frame(outside_value=99.0)
    inputs = assemble_inputs(frame, _slot_teams(), "V0", n_frames=T)

    outside = half == 0
    assert np.isnan(inputs.px[outside]).all()
    assert np.isnan(inputs.pvy[outside]).all()
    assert np.isnan(inputs.ball_x[outside]).all()
    assert np.isnan(inputs.ball_vy[outside]).all()
    assert np.isfinite(inputs.px[~outside, 0]).all()


def test_masks_follow_slot_number_not_row_order_and_drop_non_players():
    teams = _slot_teams(drop_field=(0, N_SLOTS - 1)).sample(frac=1, random_state=1)
    mask_home, mask_away = team_masks(teams)

    assert mask_home.shape == (N_SLOTS,)
    assert not mask_home[0] and not mask_away[0]
    assert not mask_home[-1] and not mask_away[-1]
    assert mask_home[1] and not mask_away[1]
    assert mask_away[N_SLOTS // 2] and not mask_home[N_SLOTS // 2]
    assert not np.any(mask_home & mask_away)


def test_load_reads_stage_metadata_and_roundtrips_values(tmp_path):
    frame, _ = _frame()
    frame.loc[4, "ball_x"] = 12.25
    tracking, slots = _write(tmp_path, frame, _slot_teams(), stage="V1")

    inputs = load_proximity_inputs(tracking, slots, n_frames=T)

    assert inputs.tracking_stage == "V1"
    np.testing.assert_allclose(inputs.ball_x[4], np.float32(12.25))
    assert int(inputs.mask_home.sum()) == N_SLOTS // 2


def test_default_length_rejects_a_short_file(tmp_path):
    tracking, slots = _write(tmp_path, *_frame()[:1], _slot_teams())

    with pytest.raises(ValueError, match=str(N_FRAMES)):
        load_proximity_inputs(tracking, slots)


def test_rejects_bad_stage_metadata(tmp_path):
    tracking, slots = _write(tmp_path, _frame()[0], _slot_teams(), stage="V9")

    with pytest.raises(ValueError, match="tracking_stage"):
        load_proximity_inputs(tracking, slots, n_frames=T)


def test_rejects_missing_stage_metadata(tmp_path):
    frame, _ = _frame()
    tracking = tmp_path / "tracking_clean.parquet"
    frame.to_parquet(tracking, index=False)
    slots = tmp_path / "slot_teams.parquet"
    _slot_teams().to_parquet(slots, index=False)

    with pytest.raises(ValueError, match="missing tracking_stage"):
        load_proximity_inputs(tracking, slots, n_frames=T)


def test_rejects_unordered_video_frame_and_interleaved_halves():
    frame, _ = _frame()
    teams = _slot_teams()
    swapped = frame.copy()
    swapped.loc[0, "video_frame"] = 1
    with pytest.raises(ValueError, match="video_frame"):
        assemble_inputs(swapped, teams, "V1", n_frames=T)

    interleaved = frame.copy()
    interleaved.loc[3, "half"] = 2
    with pytest.raises(ValueError, match="contiguous"):
        assemble_inputs(interleaved, teams, "V1", n_frames=T)


def test_half_slices_reject_a_second_half_that_starts_first():
    half = np.zeros(T, dtype=np.int8)
    half[1:4] = 2
    half[6:9] = 1
    with pytest.raises(ValueError, match="before half 1 ends"):
        contiguous_half_slices(half)


def _tiny_inputs():
    """Four frames, two home and two away slots. Hand-checked distances."""
    # slots: 0 home, 1 home, 2 away, 3 away
    px = np.array([
        [0.0, 10.0, 3.0, 8.0],   # home nearest 0 at 0; away nearest 2 at 3
        [5.0, 4.0, 1.0, 9.0],    # home nearest 1 at 4; away nearest 2 at 1
        [2.0, np.nan, np.nan, 6.0],  # home only slot 0; away only slot 3
        [1.0, 1.0, 1.0, 1.0],    # no ball
    ], dtype=np.float32)
    py = np.zeros_like(px)
    ball_x = np.array([0.0, 0.0, 0.0, np.nan], dtype=np.float32)
    ball_y = np.zeros(4, dtype=np.float32)
    half = np.array([1, 1, 1, 0], dtype=np.int8)
    mask_home = np.array([True, True, False, False])
    mask_away = np.array([False, False, True, True])
    zeros = np.zeros(4, dtype=np.float32)
    pvx = np.zeros_like(px)
    return ProximityInputs(
        video_frame=np.arange(4, dtype=np.int32),
        half=half,
        px=px, py=py, pvx=pvx, pvy=pvx.copy(),
        ball_x=ball_x, ball_y=ball_y, ball_vx=zeros, ball_vy=zeros.copy(),
        mask_home=mask_home, mask_away=mask_away,
        tracking_stage="V1",
    )


def test_masked_nearest_ignores_nan_slots_and_empty_mask():
    d = np.array([[1.0, np.nan, 4.0], [np.nan, np.nan, np.nan]], dtype=np.float32)
    mask = np.array([True, True, False])
    idx, dist = masked_nearest(d, mask)
    assert idx[0] == 0
    np.testing.assert_allclose(dist[0], 1.0)
    assert np.isnan(dist[1])


def test_team_proximity_picks_nearest_per_side_and_signs_diff():
    proximity = team_proximity(_tiny_inputs())

    np.testing.assert_allclose(proximity.d_near_home[:3], [0.0, 4.0, 2.0])
    np.testing.assert_allclose(proximity.d_near_away[:3], [3.0, 1.0, 6.0])
    np.testing.assert_allclose(proximity.d_diff[:3], [-3.0, 3.0, -4.0])
    assert proximity.d_near_home.dtype == np.float32
    assert proximity.d_diff.dtype == np.float32
    np.testing.assert_array_equal(proximity.nearest_home[:3], [0, 1, 0])
    np.testing.assert_array_equal(proximity.nearest_away[:3], [2, 2, 3])
    assert np.isnan(proximity.d_near_home[3])
    assert np.isnan(proximity.d_near_away[3])
    assert np.isnan(proximity.d_diff[3])


def test_team_proximity_nan_when_side_has_no_player():
    inputs = _tiny_inputs()
    inputs.px[2, 0] = np.nan  # only home slot left was 0; now home empty
    proximity = team_proximity(inputs)
    assert np.isnan(proximity.d_near_home[2])
    np.testing.assert_allclose(proximity.d_near_away[2], 6.0)
    assert np.isnan(proximity.d_diff[2])


def test_team_proximity_respects_field_player_mask():
    inputs = _tiny_inputs()
    # slot 0 is home and closer (0 m); drop it from the field-player mask
    mask_home = inputs.mask_home.copy()
    mask_home[0] = False
    inputs = inputs._replace(mask_home=mask_home)
    proximity = team_proximity(inputs)
    np.testing.assert_allclose(proximity.d_near_home[0], 10.0)
    assert proximity.nearest_home[0] == 1


def test_player_ball_distances_are_metres_and_nan_safe():
    bx = np.array([0.0, np.nan], dtype=np.float32)
    by = np.array([0.0, 0.0], dtype=np.float32)
    px = np.array([[3.0, np.nan], [1.0, 2.0]], dtype=np.float32)
    py = np.array([[4.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    d = player_ball_distances(bx, by, px, py)
    np.testing.assert_allclose(d[0, 0], 5.0)
    assert np.isnan(d[0, 1])
    assert np.isnan(d[1]).all()
    assert d.dtype == np.float32


def _pressure_inputs():
    """Distances: home 1 m and 4 m; away 2.5 m and 10 m; last frame no ball."""
    px = np.array([
        [1.0, 4.0, 2.5, 10.0],
        [0.5, 2.5, 4.5, 6.0],
        [9.0, 9.0, 9.0, 9.0],
        [1.0, 1.0, 1.0, 1.0],
    ], dtype=np.float32)
    py = np.zeros_like(px)
    ball_x = np.array([0.0, 0.0, 0.0, np.nan], dtype=np.float32)
    ball_y = np.zeros(4, dtype=np.float32)
    zeros = np.zeros(4, dtype=np.float32)
    pvx = np.zeros_like(px)
    return ProximityInputs(
        video_frame=np.arange(4, dtype=np.int32),
        half=np.array([1, 1, 1, 0], dtype=np.int8),
        px=px, py=py, pvx=pvx, pvy=pvx.copy(),
        ball_x=ball_x, ball_y=ball_y, ball_vx=zeros, ball_vy=zeros.copy(),
        mask_home=np.array([True, True, False, False]),
        mask_away=np.array([False, False, True, True]),
        tracking_stage="V1",
    )


def test_radius_counts_are_monotonic_and_match_nearest():
    inputs = _pressure_inputs()
    proximity = team_proximity(inputs)
    pressure = radius_pressure(inputs, distances=proximity.distances)

    np.testing.assert_allclose(pressure.n_home_r2[0], 1.0)
    np.testing.assert_allclose(pressure.n_home_r3[0], 1.0)
    np.testing.assert_allclose(pressure.n_home_r5[0], 2.0)
    np.testing.assert_allclose(pressure.n_away_r2[0], 0.0)
    np.testing.assert_allclose(pressure.n_away_r3[0], 1.0)
    np.testing.assert_allclose(pressure.n_away_r5[0], 1.0)
    assert pressure.n_home_r2.dtype == np.float32

    for side in ("home", "away"):
        r2 = getattr(pressure, f"n_{side}_r2")
        r3 = getattr(pressure, f"n_{side}_r3")
        r5 = getattr(pressure, f"n_{side}_r5")
        finite = np.isfinite(r2)
        assert np.all(r2[finite] <= r3[finite])
        assert np.all(r3[finite] <= r5[finite])
        near = getattr(proximity, f"d_near_{side}")
        for radius, counts in ((2.0, r2), (3.0, r3), (5.0, r5)):
            inside = finite & np.isfinite(near) & (near < radius)
            assert np.all(counts[inside] >= 1.0)


def test_pressure_is_continuous_sum_of_exponentials():
    inputs = _pressure_inputs()
    pressure = radius_pressure(inputs)
    expected_home = np.exp(-1.0 / PRESSURE_TAU) + np.exp(-4.0 / PRESSURE_TAU)
    expected_away = np.exp(-2.5 / PRESSURE_TAU) + np.exp(-10.0 / PRESSURE_TAU)
    np.testing.assert_allclose(pressure.pressure_home[0], expected_home, rtol=1e-6)
    np.testing.assert_allclose(pressure.pressure_away[0], expected_away, rtol=1e-6)
    # Nobody inside any useful range still has tiny pressure from distant players.
    assert pressure.pressure_home[2] > 0
    assert pressure.n_home_r5[2] == 0.0


def test_radius_and_pressure_are_nan_without_ball():
    pressure = radius_pressure(_pressure_inputs())
    for column in pressure:
        assert np.isnan(column[3])
        assert column.dtype == np.float32


def _speed_inputs():
    """Four frames: carry, opposite 7 m/s, approach, leave; then missing ball velocity."""
    # slots: 0 home (nearest), 1 home far, 2 away (nearest), 3 away far
    px = np.array([
        [0.0, 20.0, 5.0, 25.0],
        [0.0, 20.0, 5.0, 25.0],
        [2.0, 20.0, 5.0, 25.0],
        [2.0, 20.0, 5.0, 25.0],
        [0.0, 20.0, 5.0, 25.0],
    ], dtype=np.float32)
    py = np.zeros_like(px)
    pvx = np.array([
        [3.0, 0.0, -1.0, 0.0],   # carry with ball +3
        [-7.0, 0.0, 0.0, 0.0],   # opposite to ball +7
        [0.0, 0.0, 0.0, 0.0],    # standing; ball approaches from left
        [0.0, 0.0, 0.0, 0.0],    # standing; ball leaves to the right
        [3.0, 0.0, 0.0, 0.0],    # player moving; ball velocity missing
    ], dtype=np.float32)
    pvy = np.zeros_like(pvx)
    ball_x = np.array([0.0, 0.0, 0.0, 4.0, 0.0], dtype=np.float32)
    ball_y = np.zeros(5, dtype=np.float32)
    ball_vx = np.array([3.0, 7.0, 2.0, 2.0, np.nan], dtype=np.float32)
    ball_vy = np.zeros(5, dtype=np.float32)
    return ProximityInputs(
        video_frame=np.arange(5, dtype=np.int32),
        half=np.array([1, 1, 1, 1, 1], dtype=np.int8),
        px=px, py=py, pvx=pvx, pvy=pvy,
        ball_x=ball_x, ball_y=ball_y, ball_vx=ball_vx, ball_vy=ball_vy,
        mask_home=np.array([True, True, False, False]),
        mask_away=np.array([False, False, True, True]),
        tracking_stage="V1",
    )


def test_relative_speed_carry_and_opposite_vectors():
    speeds = relative_speeds(_speed_inputs())
    np.testing.assert_allclose(speeds.speed_rel_home[0], 0.0, atol=1e-6)
    np.testing.assert_allclose(speeds.speed_rel_home[1], 14.0, atol=1e-5)
    assert speeds.speed_rel_home.dtype == np.float32


def test_radial_speed_signs_for_approach_and_separation():
    speeds = relative_speeds(_speed_inputs())
    # Frame 2: player at x=2, ball at x=0 moving +2 -> closing (radial < 0)
    assert speeds.speed_radial_home[2] < 0
    # Frame 3: player at x=2, ball at x=4 moving +2 -> opening (radial > 0)
    assert speeds.speed_radial_home[3] > 0


def test_relative_speed_nan_without_ball_velocity_or_player_velocity():
    inputs = _speed_inputs()
    speeds = relative_speeds(inputs)
    assert np.isnan(speeds.speed_rel_home[4])
    assert np.isnan(speeds.speed_radial_home[4])

    broken = inputs._replace(
        pvx=inputs.pvx.copy(),
        ball_vx=np.array([3.0, 7.0, 2.0, 2.0, 1.0], dtype=np.float32),
    )
    broken.pvx[0, 0] = np.nan
    speeds = relative_speeds(broken)
    assert np.isnan(speeds.speed_rel_home[0])
    assert np.isfinite(speeds.speed_rel_home[1])
