"""F3-01 dead-ball features. Synthetic masters only: nothing from the match."""
import numpy as np
import pandas as pd
import pytest

from modeling.deadball.features import (DEADBALL_COLUMNS, GAP_AIR, GAP_EXIT, GAP_LONG, GAP_NONE,
                                        GOAL_KICKOFF_TYPE, REQUIRED_COLUMNS,
                                        build_deadball_features)
from preprocessing.make_mock_master import generate_mock_master
from tests.modeling_fixtures import (LONG_GAP, PITCH_L, PITCH_W, dead_params, deadball_master,
                                     with_gap)

P = dead_params()
LONG = LONG_GAP                            # shortest long gap


def gap_kinds(master, params=P) -> list:
    return build_deadball_features(master, params)["db_gap_kind"].tolist()


def kind_of_single_gap(master, params=P) -> int:
    kinds = set(gap_kinds(master, params)) - {GAP_NONE}
    assert len(kinds) == 1
    return kinds.pop()


MIDFIELD = (50.0, 34.0, 0.0, 0.0)


# -- shape and dtypes ---------------------------------------------------------------------------
def test_columns_dtypes_and_index():
    master = with_gap(MIDFIELD, MIDFIELD)
    master.index = pd.RangeIndex(100, 100 + len(master), name="video_frame")
    db = build_deadball_features(master, P)
    assert tuple(db.columns) == DEADBALL_COLUMNS
    assert db.index.equals(master.index)
    for c in ("db_gap_kind", "db_ball_rest", "db_fast_in_pitch", "db_goal_window"):
        assert db[c].dtype == np.int8
    for c in ("db_still_frac", "db_mean_speed", "db_sp_prox", "db_sp_after"):
        assert db[c].dtype == np.float32


@pytest.mark.parametrize("column", REQUIRED_COLUMNS)
def test_missing_column_is_an_error(column):
    master = with_gap(MIDFIELD, MIDFIELD).drop(columns=[column])
    with pytest.raises(ValueError, match=column):
        build_deadball_features(master, P)


def test_master_is_not_modified():
    master = with_gap(MIDFIELD, MIDFIELD)
    before = master.copy(deep=True)
    build_deadball_features(master, P)
    pd.testing.assert_frame_equal(master, before)


def test_runs_on_the_mock_master_and_is_deterministic():
    mock = generate_mock_master(seed=42)
    first = build_deadball_features(mock, P)
    second = build_deadball_features(mock, P)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == len(mock)
    assert set(first["db_gap_kind"].unique()) <= {GAP_NONE, GAP_EXIT, GAP_AIR, GAP_LONG}
    outside = mock["half"].to_numpy() == 0
    assert first.loc[outside, ["db_gap_kind", "db_ball_rest", "db_fast_in_pitch",
                               "db_goal_window"]].to_numpy().sum() == 0
    assert first.loc[outside, ["db_still_frac", "db_mean_speed", "db_sp_prox",
                               "db_sp_after"]].isna().all().all()


# -- db_gap_kind: which frames ------------------------------------------------------------------
def test_gap_kind_marks_exactly_the_long_gap_frames():
    kinds = gap_kinds(with_gap(MIDFIELD, MIDFIELD, gap=LONG, lead=5, trail=5))
    assert kinds == [GAP_NONE] * 5 + [GAP_LONG] * LONG + [GAP_NONE] * 5


def test_gap_of_exactly_long_gap_frames_is_not_long():
    kinds = gap_kinds(with_gap(MIDFIELD, MIDFIELD, gap=P["long_gap_frames"]))
    assert set(kinds) == {GAP_NONE}


def test_long_gap_threshold_comes_from_params():
    master = with_gap(MIDFIELD, MIDFIELD, gap=LONG)
    assert set(gap_kinds(master, dead_params(long_gap_frames=LONG))) == {GAP_NONE}


# -- db_gap_kind: exit (a) ----------------------------------------------------------------------
def test_exit_over_the_touchline():
    # 1 m from the y = 0 touchline, moving towards it at 4 m/s; reappears in midfield.
    assert kind_of_single_gap(with_gap((50.0, 1.0, 0.0, -4.0), MIDFIELD)) == GAP_EXIT


def test_exit_over_the_far_touchline():
    assert kind_of_single_gap(with_gap((50.0, PITCH_W - 1.0, 0.0, 4.0), MIDFIELD)) == GAP_EXIT


@pytest.mark.parametrize("x, vx", [(1.0, -4.0), (PITCH_L - 1.0, 4.0)])
def test_exit_over_a_goal_line(x, vx):
    # y = 20 is outside the goal area band, so only rule (a) can fire.
    assert kind_of_single_gap(with_gap((x, 20.0, vx, 0.0), MIDFIELD)) == GAP_EXIT


@pytest.mark.parametrize("before", [
    (50.0, 1.0, 0.0, 4.0),                 # near the y = 0 touchline, moving away from it
    (50.0, PITCH_W - 1.0, 0.0, -4.0),      # near the far touchline, moving away from it
    (1.0, 20.0, 4.0, 0.0),                 # near the x = 0 goal line, moving away from it
    (PITCH_L - 1.0, 20.0, -4.0, 0.0),      # near the far goal line, moving away from it
    (50.0, 1.0, 4.0, 0.0),                 # near the touchline, moving along it
    (50.0, 1.0, 0.0, -0.4),                # towards the line, but slower than exit_speed_mps
    (50.0, 4.0, 0.0, -4.0),                # towards the line, but farther than exit_line_m
])
def test_no_exit_without_approaching_a_near_line(before):
    assert kind_of_single_gap(with_gap(before, MIDFIELD)) == GAP_LONG


def test_exit_when_the_last_observed_frame_is_out_of_pitch():
    # 2 m beyond the touchline (out_of_pitch == 1), not moving.
    master = with_gap((50.0, -2.0, 0.0, 0.0), MIDFIELD)
    assert master["out_of_pitch"].iloc[4] == 1
    assert kind_of_single_gap(master) == GAP_EXIT


def test_ball_just_beyond_the_line_still_reads_its_side():
    # Inside the 1 m margin (out_of_pitch == 0, negative distance), still moving outwards.
    for before in [(50.0, -0.5, 0.0, -4.0), (50.0, PITCH_W + 0.5, 0.0, 4.0),
                   (-0.5, 20.0, -4.0, 0.0), (PITCH_L + 0.5, 20.0, 4.0, 0.0)]:
        master = with_gap(before, MIDFIELD)
        assert master["out_of_pitch"].iloc[4] == 0
        assert kind_of_single_gap(master) == GAP_EXIT


# -- db_gap_kind: exit (b) ----------------------------------------------------------------------
@pytest.mark.parametrize("after", [
    (50.0, 0.5, 0.0, 0.0),                 # throw-in
    (50.0, PITCH_W - 0.5, 0.0, 0.0),       # throw-in, far side
    (0.5, 0.5, 0.0, 0.0),                  # corner
])
def test_exit_when_the_ball_reappears_on_the_touchline(after):
    assert kind_of_single_gap(with_gap(MIDFIELD, after)) == GAP_EXIT


def test_exit_over_the_goal_line_with_restart_in_the_goal_area():
    # Leaves over the goal line (fast shot) and reappears for the goal kick.
    master = with_gap((1.0, 40.0, -20.0, 0.0), (5.0, 34.0, 0.0, 0.0))
    assert kind_of_single_gap(master) == GAP_EXIT


@pytest.mark.parametrize("x", [5.0, PITCH_L - 5.0])
def test_restart_in_the_goal_area_alone_is_an_exit(x):
    assert kind_of_single_gap(with_gap(MIDFIELD, (x, 34.0 + 9.0, 0.0, 0.0))) == GAP_EXIT


@pytest.mark.parametrize("after", [
    (7.0, 34.0, 0.0, 0.0),                 # deeper than goal_area_depth_m
    (5.0, 34.0 + 9.5, 0.0, 0.0),           # wider than goal_area_half_width_m
    (50.0, 2.0, 0.0, 0.0),                 # farther from the touchline than restart_line_m
])
def test_no_exit_when_the_ball_reappears_outside_restart_positions(after):
    assert kind_of_single_gap(with_gap(MIDFIELD, after)) == GAP_LONG


def test_goal_area_uses_pitch_width_from_params():
    after = (5.0, 39.0, 0.0, 0.0)          # the band is centred on pitch_w_m / 2
    assert kind_of_single_gap(with_gap(MIDFIELD, after)) == GAP_EXIT   # |39 - 34| < 9.2
    narrow = dead_params(pitch_w_m=40.0)                               # |39 - 20| > 9.2
    assert kind_of_single_gap(with_gap(MIDFIELD, after), narrow) == GAP_LONG


# -- db_gap_kind: high ball versus exit ---------------------------------------------------------
def test_high_ball():
    # A long pass at 20 m/s from midfield that comes down in play.
    master = with_gap((40.0, 34.0, 20.0, 0.0), (70.0, 30.0, 0.0, 0.0), gap=40)
    assert kind_of_single_gap(master) == GAP_AIR


def test_high_ball_needs_speed_above_air_speed():
    master = with_gap((40.0, 34.0, P["air_speed_mps"], 0.0), (70.0, 30.0, 0.0, 0.0), gap=40)
    assert kind_of_single_gap(master) == GAP_LONG


def test_high_ball_gap_length_limit_is_inclusive():
    fast = (40.0, 34.0, 20.0, 0.0)
    at_limit = with_gap(fast, MIDFIELD, gap=P["air_max_frames"])
    over_limit = with_gap(fast, MIDFIELD, gap=P["air_max_frames"] + 1)
    assert kind_of_single_gap(at_limit) == GAP_AIR
    assert kind_of_single_gap(over_limit) == GAP_LONG


def test_exit_takes_precedence_over_high_ball():
    # Fast ball (would be a high ball) that goes out over the touchline: it is an exit.
    out_before = with_gap((50.0, 1.0, 0.0, -20.0), MIDFIELD, gap=40)
    out_after = with_gap((40.0, 34.0, 20.0, 0.0), (70.0, 0.5, 0.0, 0.0), gap=40)
    assert kind_of_single_gap(out_before) == GAP_EXIT
    assert kind_of_single_gap(out_after) == GAP_EXIT


def test_fast_ball_parallel_to_a_near_line_is_a_high_ball():
    master = with_gap((50.0, 1.0, 20.0, 0.0), MIDFIELD, gap=40)
    assert kind_of_single_gap(master) == GAP_AIR


# -- db_gap_kind: half limits and missing sides -------------------------------------------------
def test_gap_at_the_start_of_a_half_uses_only_the_frame_after():
    exit_after = with_gap(MIDFIELD, (50.0, 0.5, 0.0, 0.0), lead=0)
    plain_after = with_gap(MIDFIELD, MIDFIELD, lead=0)
    assert kind_of_single_gap(exit_after) == GAP_EXIT
    assert kind_of_single_gap(plain_after) == GAP_LONG


def test_gap_at_the_end_of_a_half_uses_only_the_frame_before():
    exit_before = with_gap((50.0, 1.0, 0.0, -4.0), MIDFIELD, trail=0)
    air_before = with_gap((40.0, 34.0, 20.0, 0.0), MIDFIELD, gap=40, trail=0)
    assert kind_of_single_gap(exit_before) == GAP_EXIT
    assert kind_of_single_gap(air_before) == GAP_AIR


def test_half_with_no_ball_at_all_is_long():
    master = deadball_master(np.full(LONG, np.nan), np.nan)
    assert gap_kinds(master) == [GAP_LONG] * LONG


def test_gap_never_looks_across_the_limit_of_a_half():
    # Half 1 ends with the ball going out; half 2 starts with a gap and then midfield.
    # The gap of half 2 must not see the last frame of half 1.
    x = np.r_[np.full(5, 50.0), np.full(LONG, np.nan), np.full(5, 50.0)]
    y = np.r_[np.full(5, 1.0), np.full(LONG, np.nan), np.full(5, 34.0)]
    vy = np.r_[np.full(5, -4.0), np.full(LONG, np.nan), np.full(5, 0.0)]
    half = np.r_[np.full(5, 1), np.full(LONG + 5, 2)]
    assert gap_kinds(deadball_master(x, y, 0.0, vy, half=half)) == (
        [GAP_NONE] * 5 + [GAP_LONG] * LONG + [GAP_NONE] * 5)

    # Same trajectory inside one half: the exit is seen.
    assert gap_kinds(deadball_master(x, y, 0.0, vy, half=1)) == (
        [GAP_NONE] * 5 + [GAP_EXIT] * LONG + [GAP_NONE] * 5)


def test_gap_split_by_the_interval_is_classified_per_half():
    # One hole that spans the end of half 1, the interval and the start of half 2.
    x = np.r_[np.full(5, 50.0), np.full(3 * LONG, np.nan), np.full(5, 50.0)]
    y = np.r_[np.full(5, 1.0), np.full(3 * LONG, np.nan), np.full(5, 34.0)]
    vy = np.r_[np.full(5, -4.0), np.full(3 * LONG, np.nan), np.full(5, 0.0)]
    half = np.r_[np.full(5 + LONG, 1), np.full(LONG, 0), np.full(LONG + 5, 2)]
    assert gap_kinds(deadball_master(x, y, 0.0, vy, half=half)) == (
        [GAP_NONE] * 5 + [GAP_EXIT] * LONG + [GAP_NONE] * LONG + [GAP_LONG] * LONG
        + [GAP_NONE] * 5)


def test_nan_velocity_and_distances_give_no_exit_and_no_high_ball():
    master = with_gap((50.0, 1.0, 0.0, -4.0), (50.0, 0.5, 0.0, 0.0))
    for c in ("ball_vx", "ball_vy", "ball_speed", "d_touchline", "d_goalline", "out_of_pitch"):
        master[c] = np.nan
    assert kind_of_single_gap(master) == GAP_LONG


def test_two_gaps_in_one_half_are_classified_independently():
    first = with_gap((50.0, 1.0, 0.0, -4.0), MIDFIELD, trail=5)
    second = with_gap((40.0, 34.0, 20.0, 0.0), MIDFIELD, gap=40, lead=5)
    master = pd.concat([first, second], ignore_index=True)
    assert gap_kinds(master) == ([GAP_NONE] * 5 + [GAP_EXIT] * LONG + [GAP_NONE] * 10
                                 + [GAP_AIR] * 40 + [GAP_NONE] * 5)


# -- db_still_frac and db_mean_speed ------------------------------------------------------------
def still_master(n=60, **cols) -> pd.DataFrame:
    return deadball_master(np.full(n, 50.0), 34.0, **cols)


def test_still_frac_and_mean_speed_of_constant_inputs():
    db = build_deadball_features(still_master(), P)
    np.testing.assert_allclose(db["db_still_frac"], 4.0 / 20.0, rtol=1e-6)
    np.testing.assert_allclose(db["db_mean_speed"], 2.5, rtol=1e-6)


def test_moving_mean_is_centred_and_cut_at_the_block_ends():
    n, w = 60, P["smooth_frames"]
    speed = np.arange(n, dtype="float64")
    db = build_deadball_features(still_master(n, mean_speed_home=speed, mean_speed_away=speed), P)
    expected = [speed[max(0, i - w // 2):i + (w - 1) // 2 + 1].mean() for i in range(n)]
    np.testing.assert_allclose(db["db_mean_speed"], expected, rtol=1e-6)


def test_moving_mean_ignores_nan_and_is_nan_when_the_window_is_all_nan():
    n, w = 80, P["smooth_frames"]
    speed = np.full(n, 4.0)
    speed[10:10 + 2 * w] = np.nan                       # hole wider than the window
    speed[70] = np.nan                                  # isolated NaN
    db = build_deadball_features(still_master(n, mean_speed_home=speed, mean_speed_away=speed), P)
    out = db["db_mean_speed"].to_numpy()
    all_nan = np.zeros(n, dtype=bool)
    all_nan[10 + w // 2:10 + 2 * w - w // 2] = True
    assert np.isnan(out[all_nan]).all()
    np.testing.assert_allclose(out[~all_nan], 4.0, rtol=1e-6)


def test_mean_speed_uses_the_team_that_has_a_value():
    db = build_deadball_features(still_master(mean_speed_home=np.nan, mean_speed_away=2.0), P)
    np.testing.assert_allclose(db["db_mean_speed"], 2.0, rtol=1e-6)


@pytest.mark.parametrize("cols", [
    {"n_tracked_home": 0.0, "n_tracked_away": 0.0},     # zero denominator
    {"n_tracked_home": np.nan},                         # NaN denominator
    {"n_still_away": np.nan},                           # NaN numerator
])
def test_still_frac_is_nan_without_evidence(cols):
    db = build_deadball_features(still_master(**cols), P)
    assert db["db_still_frac"].isna().all()


def test_moving_mean_does_not_cross_the_limit_of_a_half():
    n = 40
    half = np.r_[np.full(n, 1), np.full(n, 2)]
    speed = np.r_[np.full(n, 1.0), np.full(n, 9.0)]
    still = np.r_[np.full(n, 0.0), np.full(n, 5.0)]
    master = deadball_master(np.full(2 * n, 50.0), 34.0, half=half, mean_speed_home=speed,
                             mean_speed_away=speed, n_still_home=still, n_still_away=still)
    db = build_deadball_features(master, P)
    np.testing.assert_allclose(db["db_mean_speed"][:n], 1.0, rtol=1e-6)
    np.testing.assert_allclose(db["db_mean_speed"][n:], 9.0, rtol=1e-6)
    np.testing.assert_allclose(db["db_still_frac"][:n], 0.0, atol=1e-9)
    np.testing.assert_allclose(db["db_still_frac"][n:], 0.5, rtol=1e-6)


def test_smooth_frames_one_is_the_raw_value():
    speed = np.arange(30, dtype="float64")
    master = still_master(30, mean_speed_home=speed, mean_speed_away=speed)
    db = build_deadball_features(master, dead_params(smooth_frames=1))
    np.testing.assert_allclose(db["db_mean_speed"], speed, rtol=1e-6)


# -- db_ball_rest and db_fast_in_pitch ----------------------------------------------------------
def test_ball_rest_needs_a_run_of_rest_min_frames():
    m = P["rest_min_frames"]
    speed = np.r_[np.full(5, 3.0), np.full(m, 0.1), np.full(5, 3.0), np.full(m - 1, 0.1),
                  np.full(5, 3.0)]
    db = build_deadball_features(still_master(len(speed), ball_speed=speed), P)
    assert db["db_ball_rest"].tolist() == [0] * 5 + [1] * m + [0] * (5 + m - 1 + 5)


def test_ball_rest_speed_threshold_is_strict():
    m = P["rest_min_frames"]
    db = build_deadball_features(still_master(m, ball_speed=P["rest_speed_mps"]), P)
    assert not db["db_ball_rest"].any()


def test_ball_rest_run_is_broken_by_nan_speed_and_by_unobserved_frames():
    m = P["rest_min_frames"]
    nan_speed = np.full(2 * m - 1, 0.1)
    nan_speed[m - 1] = np.nan
    db = build_deadball_features(still_master(len(nan_speed), ball_speed=nan_speed), P)
    assert not db["db_ball_rest"].any()

    x = np.full(2 * m - 1, 50.0)
    x[m - 1] = np.nan                                   # one frame without the ball
    master = deadball_master(x, 34.0, ball_speed=0.1)
    assert not build_deadball_features(master, P)["db_ball_rest"].any()


def test_ball_rest_run_does_not_cross_the_limit_of_a_half():
    m = P["rest_min_frames"]
    half = np.r_[np.full(m - 1, 1), np.full(m - 1, 2)]
    master = deadball_master(np.full(2 * m - 2, 50.0), 34.0, half=half, ball_speed=0.1)
    assert not build_deadball_features(master, P)["db_ball_rest"].any()


def test_fast_in_pitch():
    fast = P["fast_speed_mps"]
    master = still_master(6, ball_speed=[fast + 1, fast, fast + 1, fast + 1, np.nan, fast + 1])
    master.loc[2, "out_of_pitch"] = 1.0
    master.loc[3, "out_of_pitch"] = np.nan
    master.loc[5, "ball_observed"] = 0.0
    db = build_deadball_features(master, P)
    assert db["db_fast_in_pitch"].tolist() == [1, 0, 0, 0, 0, 0]


# -- set-piece tags -----------------------------------------------------------------------------
def test_sp_prox_and_sp_after():
    to_next = [0.0, 250.0, 500.0, np.nan]
    since = [0.0, 75.0, np.nan, 150.0]
    db = build_deadball_features(
        still_master(4, frames_to_next_sp=to_next, frames_since_sp_end=since), P)
    np.testing.assert_allclose(db["db_sp_prox"][:3], [1.0, np.exp(-1.0), np.exp(-2.0)], rtol=1e-6)
    assert np.isnan(db["db_sp_prox"][3])
    np.testing.assert_allclose(db["db_sp_after"].to_numpy()[[0, 1, 3]],
                               [1.0, np.exp(-1.0), np.exp(-2.0)], rtol=1e-6)
    assert np.isnan(db["db_sp_after"][2])


def test_goal_window_matches_the_exact_lower_case_type():
    assert GOAL_KICKOFF_TYPE == "kick off (after goal)"
    types = ["kick off (after goal)", "Kick Off (After Goal)", "kick off (start)", "goal kick",
             "", "kick off (after goal) "]
    db = build_deadball_features(
        still_master(len(types), frames_to_next_sp=100.0, next_sp_type=types), P)
    assert db["db_goal_window"].tolist() == [1, 0, 0, 0, 0, 0]


def test_goal_window_distance_limit_is_strict_and_nan_is_zero():
    w = P["goal_window_frames"]
    to_next = [0.0, w - 1.0, float(w), w + 1.0, np.nan]
    db = build_deadball_features(
        still_master(5, frames_to_next_sp=to_next, next_sp_type=GOAL_KICKOFF_TYPE), P)
    assert db["db_goal_window"].tolist() == [1, 1, 0, 0, 0]


def test_goal_window_handles_missing_strings():
    master = still_master(3, frames_to_next_sp=10.0, next_sp_type=GOAL_KICKOFF_TYPE)
    master["next_sp_type"] = pd.array([GOAL_KICKOFF_TYPE, pd.NA, ""], dtype="string")
    assert build_deadball_features(master, P)["db_goal_window"].tolist() == [1, 0, 0]
    master["next_sp_type"] = master["next_sp_type"].astype(object)
    assert build_deadball_features(master, P)["db_goal_window"].tolist() == [1, 0, 0]


# -- outside the halves -------------------------------------------------------------------------
def test_outside_the_halves_floats_are_nan_and_integers_are_zero():
    n = 30
    half = np.r_[np.full(n, 1), np.full(n, 0), np.full(n, 2)]
    master = deadball_master(np.full(3 * n, 50.0), 34.0, half=half, vx=10.0,
                             frames_to_next_sp=5.0, frames_since_sp_end=5.0,
                             next_sp_type=GOAL_KICKOFF_TYPE)
    # Columns that would fire if the frames outside the halves were not masked.
    master.loc[half == 0, ["ball_observed", "ball_speed", "out_of_pitch"]] = [1.0, 10.0, 0.0]
    db = build_deadball_features(master, P)
    outside = half == 0
    for c in ("db_gap_kind", "db_ball_rest", "db_fast_in_pitch", "db_goal_window"):
        assert not db.loc[outside, c].any(), c
    for c in ("db_still_frac", "db_mean_speed", "db_sp_prox", "db_sp_after"):
        assert db.loc[outside, c].isna().all(), c
        assert db.loc[~outside, c].notna().all(), c
    assert db.loc[~outside, "db_fast_in_pitch"].all()
    assert db.loc[~outside, "db_goal_window"].all()


def test_empty_master():
    db = build_deadball_features(still_master(3).iloc[:0], P)
    assert len(db) == 0 and tuple(db.columns) == DEADBALL_COLUMNS
