"""Synthetic tests for B1. No match parquet is read."""
import numpy as np
import pandas as pd
import pytest

from preprocessing.params import N_FRAMES
from preprocessing.prep_proximities import (
    assemble_inputs, contiguous_half_slices, load_proximity_inputs, team_masks)
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
