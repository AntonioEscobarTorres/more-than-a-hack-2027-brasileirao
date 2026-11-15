"""tools/jump_filter_edge_cases.py (task F3-07).

Synthetic data only: hand-built tables, and synthetic arrays run through the real preprocessing.
Every file lives in tmp_path; nothing reads the match data.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from preprocessing.params import FPS, MAX_GAP_FRAMES, PITCH_L, PITCH_W, PITCH_MARGIN
from preprocessing.pipeline_api import build_master_from_arrays
from preprocessing.prep_tracking import prepare_tracking
from preprocessing.schema import N_SLOTS
from tools import jump_filter_edge_cases as jf

REPO = Path(__file__).resolve().parent.parent
STILL = jf.MIN_STILL_FRAMES                 # 75
WINDOW = jf.JUMP_WINDOW_FRAMES              # 25
LIMIT = jf.MAX_COUNT_PER_HALF               # 10
PHANTOM = (55.0, -3.0)                      # 3 m outside the touchline
MIDFIELD = (50.0, 34.0)


# -- hand-built tables ---------------------------------------------------------------------------
def tables(x, y, removed=None, half=1, flip=0, player_speed=4.0):
    """Master and tracking_clean columns the tool reads, from the detections of the ball.

    ``x``/``y`` are the detected positions in the orientation of the master (NaN = no detection)
    and ``removed`` marks the detections the jump filter threw away. As in the preprocessing,
    gaps of at most MAX_GAP_FRAMES between kept observations are interpolated, the speed exists
    only on runs of two or more kept observations, and nothing crosses the limit of a half.
    """
    x, y = np.asarray(x, dtype="float64"), np.asarray(y, dtype="float64")
    n = len(x)
    half = np.broadcast_to(np.asarray(half, dtype=np.int8), (n,)).copy()
    flip = np.broadcast_to(np.asarray(flip, dtype=np.int8), (n,)).copy()
    removed = np.zeros(n, dtype=bool) if removed is None else np.asarray(removed, dtype=bool)
    detected = np.isfinite(x) & np.isfinite(y) & (half != 0)
    removed = removed & detected
    kept = detected & ~removed

    ball_x, ball_y = np.where(kept, x, np.nan), np.where(kept, y, np.nan)
    filled = np.zeros(n, dtype=bool)
    speed = np.full(n, np.nan)
    cuts = np.flatnonzero(np.diff(half) != 0) + 1
    for b0, b1 in zip(np.r_[0, cuts], np.r_[cuts, n]):
        if half[b0] == 0:
            continue
        idx = np.flatnonzero(kept[b0:b1]) + b0
        for a, b in zip(idx[:-1], idx[1:]):                      # gap between kept a and b
            if 1 < b - a <= MAX_GAP_FRAMES + 1:
                inner = np.arange(a + 1, b)
                ball_x[inner] = np.interp(inner, [a, b], [x[a], x[b]])
                ball_y[inner] = np.interp(inner, [a, b], [y[a], y[b]])
                filled[inner] = True
        edges = np.flatnonzero(np.diff(np.r_[False, kept[b0:b1], False].astype(np.int8))) + b0
        for r0, r1 in zip(edges[::2], edges[1::2]):
            if r1 - r0 >= 2:
                speed[r0:r1] = np.hypot(np.gradient(x[r0:r1]), np.gradient(y[r0:r1])) * FPS

    out = ((ball_x < -PITCH_MARGIN) | (ball_x > PITCH_L + PITCH_MARGIN)
           | (ball_y < -PITCH_MARGIN) | (ball_y > PITCH_W + PITCH_MARGIN)).astype("float64")
    out[~(kept | filled)] = np.nan
    players = np.broadcast_to(np.asarray(player_speed, dtype="float64"), (n,)).copy()
    master = pd.DataFrame({
        "half": half, "ball_x": ball_x, "ball_y": ball_y, "ball_speed": speed,
        "ball_observed": kept.astype(np.int8), "ball_jump_removed": removed.astype(np.int8),
        "ball_filled": filled.astype(np.int8), "out_of_pitch": out,
        "mean_speed_home": players, "mean_speed_away": players,
    })
    flipped = flip == 1
    tracking = pd.DataFrame({
        "video_frame": np.arange(n, dtype=np.int32), "half": half, "flip": flip,
        "ball_x_raw": np.where(detected, np.where(flipped, PITCH_L - x, x), np.nan),
        "ball_y_raw": np.where(detected, np.where(flipped, PITCH_W - y, y), np.nan),
    })
    return master, tracking


def at(point, n):
    return np.tile(np.asarray(point, dtype="float64"), (n, 1))


MISS = at((np.nan, np.nan), 1)             # one frame without a detection


def phantom(n_still, lead=20, trail=20, point=PHANTOM, **kw):
    """Ball in midfield, then ``n_still`` frames at ``point``, then midfield again.

    One frame without a detection separates the three parts, so each is its own observed run
    and the speed inside the middle one is not smeared by the jumps around it.
    """
    xy = np.vstack([at(MIDFIELD, lead), MISS, at(point, n_still), MISS, at(MIDFIELD, trail)])
    return tables(xy[:, 0], xy[:, 1], **kw)


def phantom_stats(master, half=1):
    return jf.phantom_stretches(master)[half]


def restart(n_removed=10, wait=4, jump_to=(20.0, 0.3), landing_speed=0.0, lead=60, trail=60,
            **kw):
    """Old ball running to the touchline, a new ball far away ``wait`` frames after it vanishes.

    The last ``n_removed`` detections of the old ball are marked as removed by the filter.
    Returns ``(master, tracking, landing_frame)``.
    """
    f = np.arange(lead)
    old = np.c_[np.full(lead, 50.0), 20.0 - 0.6 * f]                 # 15 m/s towards y = 0
    gap = np.full((wait, 2), np.nan)
    g = np.arange(trail)
    new = np.c_[jump_to[0] + landing_speed / FPS * g, np.full(trail, jump_to[1])]
    xy = np.vstack([old, gap, new])
    removed = np.zeros(len(xy), dtype=bool)
    removed[lead - n_removed:lead] = True
    master, tracking = tables(xy[:, 0], xy[:, 1], removed=removed, **kw)
    return master, tracking, lead + wait


def jump_stats(master, tracking, half=1):
    return jf.removed_before_jumps(master, tracking)[half]


# -- (a) still phantom outside the pitch ---------------------------------------------------------
def test_phantom_of_three_seconds_is_counted_and_a_shorter_one_is_not():
    long_enough = phantom_stats(phantom(STILL)[0])
    assert (long_enough.n_counted, long_enough.n_stretches, long_enough.longest_frames) == (
        1, 1, STILL)
    too_short = phantom_stats(phantom(STILL - 1)[0])
    assert (too_short.n_counted, too_short.n_stretches, too_short.longest_frames) == (0, 0, 0)


def test_phantom_is_not_counted_while_the_players_are_slow():
    slow = phantom_stats(phantom(STILL, player_speed=jf.PLAYERS_FAST_SPEED_MPS)[0])   # not above
    assert (slow.n_counted, slow.n_stretches) == (0, 1)
    no_players = phantom_stats(phantom(STILL, player_speed=np.nan)[0])
    assert (no_players.n_counted, no_players.n_stretches) == (0, 1)


def test_players_must_be_fast_on_average_over_three_seconds_of_the_stretch():
    n, lead = 200, 20

    def stats(n_fast):
        speed = np.full(lead + 1 + n + 1 + 20, 1.0)
        speed[lead + 60:lead + 60 + n_fast] = 4.0
        return phantom_stats(phantom(n, lead=lead, player_speed=speed)[0])

    # Best 75-frame window: (50 * 4 + 25 * 1) / 75 = 3.0, not above the threshold.
    assert stats(50).n_counted == 0
    assert stats(51).n_counted == 1
    assert stats(200).n_counted == 1


@pytest.mark.parametrize("point", [
    MIDFIELD,                               # still, but inside the pitch
    (55.0, -0.5),                           # beyond the line, inside the out-of-pitch margin
])
def test_still_ball_inside_the_pitch_is_not_a_phantom(point):
    s = phantom_stats(phantom(STILL, point=point)[0])
    assert (s.n_counted, s.n_stretches) == (0, 0)


def test_moving_ball_outside_the_pitch_is_not_a_phantom():
    n = 200
    x = 20.0 + np.arange(n) * 1.0 / FPS                     # 1 m/s along the touchline
    master, _ = tables(x, np.full(n, -3.0))
    s = phantom_stats(master)
    assert (s.n_counted, s.n_stretches) == (0, 0)
    slow = tables(20.0 + np.arange(n) * 0.25 / FPS, np.full(n, -3.0))[0]   # 0.25 m/s: still
    assert phantom_stats(slow).n_counted == 1


def test_phantom_stretch_bridges_the_interpolated_frames():
    # The phantom beats a sparse real ball: every 9th detection is the real ball, removed by the
    # filter, so the kept phantom is a chain of 8-frame runs joined by interpolated frames.
    n = 200
    x, y = np.full(n, PHANTOM[0]), np.full(n, PHANTOM[1])
    real = np.arange(0, n, 9)
    x[real], y[real] = 30.0 + 0.2 * real, 34.0
    removed = np.zeros(n, dtype=bool)
    removed[real] = True
    master, _ = tables(x, y, removed=removed)
    assert master["ball_filled"].sum() == len(real) - 1     # frame 0 has no kept frame before it
    runs = np.diff(np.flatnonzero(np.r_[True, master["ball_observed"].to_numpy() == 0, True])) - 1
    assert runs.max() == 8                                  # no observed run comes near 75
    s = phantom_stats(master)
    assert (s.n_counted, s.n_stretches, s.longest_frames) == (1, 1, n - 1)


def test_phantom_made_of_isolated_observations_is_counted():
    # The phantom is kept on every other frame only: no observed run has a speed in the master.
    n = 200
    x, y = np.full(n, PHANTOM[0]), np.full(n, PHANTOM[1])
    real = np.arange(1, n, 2)
    x[real], y[real] = 30.0, 34.0
    removed = np.zeros(n, dtype=bool)
    removed[real] = True
    master, _ = tables(x, y, removed=removed)
    assert master["ball_speed"].isna().all()
    assert phantom_stats(master).n_counted == 1


def test_phantom_stretch_does_not_cross_the_limit_of_a_half():
    n = STILL - 10
    half = np.r_[np.full(n, 1), np.full(n, 2)]
    master, _ = tables(np.full(2 * n, PHANTOM[0]), np.full(2 * n, PHANTOM[1]), half=half)
    stats = jf.phantom_stretches(master)
    assert stats[1].n_stretches == 0 and stats[2].n_stretches == 0
    one_half, _ = tables(np.full(2 * n, PHANTOM[0]), np.full(2 * n, PHANTOM[1]))
    assert phantom_stats(one_half).n_counted == 1


def test_phantom_counts_are_per_half():
    one = np.vstack([at(MIDFIELD, 5), MISS, at(PHANTOM, STILL), MISS, at(MIDFIELD, 5)])
    xy = np.vstack([one, at((np.nan, np.nan), 30), one, one])
    half = np.r_[np.full(len(one), 1), np.full(30, 0), np.full(2 * len(one), 2)]
    master, _ = tables(xy[:, 0], xy[:, 1], half=half)
    stats = jf.phantom_stretches(master)
    assert stats[1].n_counted == 1 and stats[2].n_counted == 2


def test_kept_ball_speed_falls_back_to_the_step_between_positions():
    # Kept at frames 0-1 (speed from the master), a 3-frame interpolated gap, then one isolated
    # kept frame: 5 m in 4 frames.
    x = np.array([10.0, 10.0, np.nan, np.nan, np.nan, 15.0])
    master, _ = tables(x, np.full(6, 34.0))
    assert master["ball_speed"].notna().tolist() == [True, True, False, False, False, False]
    speed = jf.kept_ball_speed(master)
    np.testing.assert_allclose(speed, [0.0, 0.0, 31.25, 31.25, 31.25, 31.25])
    # No position at all on a frame: no speed.
    alone, _ = tables(np.r_[10.0, np.full(20, np.nan), 15.0], np.full(22, 34.0))
    assert np.isnan(jf.kept_ball_speed(alone)).all()


def test_kept_ball_speed_does_not_step_across_a_half():
    half = np.r_[np.full(3, 1), np.full(3, 2)]
    x = np.array([10.0, np.nan, 10.0, 90.0, np.nan, 90.0])
    master, _ = tables(x, np.full(6, 34.0), half=half)
    speed = jf.kept_ball_speed(master)
    np.testing.assert_allclose(speed, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


# -- (b) observations removed before a multi-ball jump -------------------------------------------
def test_removed_tail_of_the_old_ball_is_counted():
    master, tracking, landing = restart(n_removed=10)
    s = jump_stats(master, tracking)
    assert (s.n_removed, s.n_jumps, s.n_continuing) == (10, 1, 10)
    assert master["ball_observed"].iloc[landing] == 1


def test_nothing_is_counted_without_removed_observations_or_without_a_jump():
    master, tracking, _ = restart(n_removed=0)
    s = jump_stats(master, tracking)
    assert (s.n_removed, s.n_jumps) == (0, 1)
    # Removed observations, but the ball comes back where it was: no jump.
    master, tracking, _ = restart(n_removed=10, jump_to=(50.0, 5.0))
    s = jump_stats(master, tracking)
    assert (s.n_removed, s.n_jumps) == (0, 0)


def test_jump_distance_threshold_is_strict():
    # The last kept observation is at (50, y0); land exactly 20 m and just over 20 m from it.
    master, tracking, landing = restart(n_removed=5)
    last = int(np.flatnonzero(master["ball_observed"].to_numpy()[:landing])[-1])
    y0 = float(master["ball_y"].iloc[last])
    exact = restart(n_removed=5, jump_to=(50.0 + jf.JUMP_MIN_DISTANCE_M, y0))
    over = restart(n_removed=5, jump_to=(50.0 + jf.JUMP_MIN_DISTANCE_M + 0.01, y0))
    assert jump_stats(*exact[:2]).n_jumps == 0
    assert jump_stats(*over[:2]).n_jumps == 1


def test_jump_must_be_followed_by_a_still_ball():
    moving = restart(landing_speed=2.0)                    # the new ball keeps moving
    assert jump_stats(*moving[:2]).n_jumps == 0
    slow = restart(landing_speed=0.25)                     # below the still threshold
    assert jump_stats(*slow[:2]).n_removed == 10


def test_ball_may_settle_within_the_window_after_the_jump():
    def settle_after(n_moving):
        f = np.arange(60)
        old = np.c_[np.full(60, 50.0), 20.0 - 0.6 * f]
        moving = np.c_[20.0 + 2.0 / FPS * np.arange(n_moving), np.full(n_moving, 0.3)]
        rest = at(moving[-1], 60)
        xy = np.vstack([old, at((np.nan, np.nan), 4), moving, rest])
        removed = np.zeros(len(xy), dtype=bool)
        removed[50:60] = True
        return jump_stats(*tables(xy[:, 0], xy[:, 1], removed=removed))

    assert settle_after(WINDOW - 5).n_removed == 10        # still inside the window
    assert settle_after(WINDOW + 5).n_removed == 0         # still only later: not a restart


def test_only_the_window_before_the_landing_is_counted():
    # 10 removed detections; the landing comes ``wait`` frames after the last of them.
    counts = {wait: jump_stats(*restart(n_removed=10, wait=wait)[:2]).n_removed
              for wait in (0, WINDOW - 10, WINDOW - 5, WINDOW)}
    assert counts == {0: 10, WINDOW - 10: 10, WINDOW - 5: 5, WINDOW: 0}


def test_removed_observations_are_counted_once_when_windows_overlap():
    # Old ball -> new ball B (still) -> 10 frames later another ball C (still): the 8 removed
    # detections are in the window of both jumps.
    f = np.arange(60)
    old = np.c_[np.full(60, 50.0), 20.0 - 0.6 * f]
    xy = np.vstack([old, at((20.0, 0.3), 10), at((90.0, 60.0), 60)])
    removed = np.zeros(len(xy), dtype=bool)
    removed[52:60] = True
    s = jump_stats(*tables(xy[:, 0], xy[:, 1], removed=removed))
    assert (s.n_removed, s.n_jumps, s.n_continuing) == (8, 2, 8)


def test_a_jump_is_never_measured_across_the_limit_of_a_half():
    # Half 1 ends with removed detections; half 2 starts with a still ball far away.
    f = np.arange(60)
    old = np.c_[np.full(60, 50.0), 20.0 - 0.6 * f]
    xy = np.vstack([old, at((20.0, 0.3), 60)])
    removed = np.zeros(120, dtype=bool)
    removed[50:60] = True
    half = np.r_[np.full(60, 1), np.full(60, 2)]
    stats = jf.removed_before_jumps(*tables(xy[:, 0], xy[:, 1], removed=removed, half=half))
    assert (stats[1].n_removed, stats[1].n_jumps) == (0, 0)
    assert (stats[2].n_removed, stats[2].n_jumps) == (0, 0)
    one_half = jf.removed_before_jumps(*tables(xy[:, 0], xy[:, 1], removed=removed))
    assert one_half[1].n_removed == 10


def test_removed_phantoms_before_a_jump_are_counted_but_do_not_continue_the_trajectory():
    # The removed detections are 60 m from the old ball, one frame after it: unreachable.
    master, tracking, landing = restart(n_removed=0, wait=6)
    xy = np.c_[tracking["ball_x_raw"], tracking["ball_y_raw"]].copy()
    blips = np.arange(landing - 5, landing - 1)
    xy[blips] = (100.0, 60.0)
    removed = np.zeros(len(xy), dtype=bool)
    removed[blips] = True
    s = jump_stats(*tables(xy[:, 0], xy[:, 1], removed=removed))
    assert (s.n_removed, s.n_jumps, s.n_continuing) == (4, 1, 0)


def test_removed_observations_before_the_last_kept_one_do_not_continue_it():
    # A removed detection in the window, but before the last kept observation of the old ball.
    master, tracking, landing = restart(n_removed=0, wait=3)
    removed = np.zeros(len(master), dtype=bool)
    removed[landing - 10] = True
    xy = np.c_[tracking["ball_x_raw"], tracking["ball_y_raw"]]
    s = jump_stats(*tables(xy[:, 0], xy[:, 1], removed=removed))
    assert (s.n_removed, s.n_jumps, s.n_continuing) == (1, 1, 0)


def test_flipped_half_gives_the_same_counts():
    plain = jump_stats(*restart(n_removed=10)[:2])
    master, tracking, _ = restart(n_removed=10, flip=1)
    kept = master["ball_observed"].to_numpy() == 1
    assert not np.allclose(tracking["ball_x_raw"][kept], master["ball_x"][kept])   # really flipped
    assert jump_stats(master, tracking) == plain


# -- inputs --------------------------------------------------------------------------------------
def test_tables_from_different_runs_are_rejected():
    master, tracking, _ = restart()
    jf.check_same_run(master, tracking)
    with pytest.raises(ValueError, match="rows"):
        jf.check_same_run(master, tracking.iloc[:-1])
    other_half = tracking.assign(half=np.int8(2))
    with pytest.raises(ValueError, match="disagree on half"):
        jf.check_same_run(master, other_half)
    with pytest.raises(ValueError, match="regenerate"):
        jf.check_same_run(master, tracking.assign(flip=np.int8(1)))     # wrong orientation
    with pytest.raises(ValueError, match="regenerate"):
        jf.check_same_run(master, tracking.assign(ball_x_raw=tracking["ball_x_raw"] + 0.5))
    with pytest.raises(ValueError, match="video_frame"):
        jf.check_same_run(master, tracking.assign(video_frame=tracking["video_frame"] + 1))


@pytest.mark.parametrize("column", jf.MASTER_COLUMNS)
def test_missing_master_column_is_an_error(column):
    master, tracking, _ = restart()
    with pytest.raises(ValueError, match=column):
        jf.run_checks(master.drop(columns=[column]), tracking)


@pytest.mark.parametrize("column", jf.TRACKING_COLUMNS)
def test_missing_tracking_column_is_an_error(column):
    master, tracking, _ = restart()
    with pytest.raises(ValueError, match=column):
        jf.run_checks(master, tracking.drop(columns=[column]))


def test_master_without_any_half_is_an_error():
    master, tracking = tables(np.full(5, 50.0), np.full(5, 34.0), half=0)
    with pytest.raises(ValueError, match="no frames inside a half"):
        jf.run_checks(master, tracking)


# -- PASS/FAIL -----------------------------------------------------------------------------------
def many_phantoms(k):
    one = np.vstack([MISS, at(PHANTOM, STILL), MISS, at(MIDFIELD, 5)])
    xy = np.vstack([at(MIDFIELD, 5)] + [one] * k)
    return tables(xy[:, 0], xy[:, 1])


def test_pass_below_the_limit_and_fail_at_the_limit():
    results = {r.key: r for r in jf.run_checks(*many_phantoms(LIMIT - 1))}
    assert results["a"].passed and results["b"].passed
    results = {r.key: r for r in jf.run_checks(*many_phantoms(LIMIT))}
    assert not results["a"].passed and results["b"].passed

    results = {r.key: r for r in jf.run_checks(*restart(n_removed=LIMIT - 1)[:2])}
    assert results["a"].passed and results["b"].passed
    results = {r.key: r for r in jf.run_checks(*restart(n_removed=LIMIT)[:2])}
    assert results["a"].passed and not results["b"].passed


def test_one_bad_half_is_enough_to_fail():
    # Half 1 is quiet; half 2 has the restart with LIMIT removed detections.
    bad_master, bad_tracking, _ = restart(n_removed=LIMIT)
    raw = np.c_[bad_tracking["ball_x_raw"], bad_tracking["ball_y_raw"]]
    xy = np.vstack([at(MIDFIELD, 30), raw])
    removed = np.r_[np.zeros(30, dtype=bool), bad_master["ball_jump_removed"].to_numpy() == 1]
    half = np.r_[np.full(30, 1), np.full(len(raw), 2)]
    results = {r.key: r for r in jf.run_checks(*tables(xy[:, 0], xy[:, 1], removed=removed,
                                                       half=half))}
    assert not results["b"].passed
    assert "half 1: 0 removed" in results["b"].detail
    assert "half 2: {} removed".format(LIMIT) in results["b"].detail


def test_report_shows_numbers_only_when_verbose():
    results = jf.run_checks(*restart(n_removed=LIMIT)[:2])
    quiet = jf.report(results)
    assert quiet == ["PASS  (a) no still phantom kept outside the pitch while play goes on",
                     "FAIL  (b) no observations removed just before a multi-ball jump",
                     "FAIL (1 of 2 checks)"]
    loud = jf.report(results, verbose=True)
    assert len(loud) > len(quiet) and any("half 1: 10 removed observations" in s for s in loud)
    assert jf.report(jf.run_checks(*restart(n_removed=0)[:2]))[-1] == "PASS"


# -- the two scenarios through the real preprocessing --------------------------------------------
GAP_FRAMES = 50
TEAM_IDS = (np.arange(N_SLOTS) % 2).astype(int)


def _players(n, speed_mps, mirror):
    """22 players oscillating along x at a mean speed of ``speed_mps``; home in the low-x half."""
    out = np.full((n, N_SLOTS, 3), np.nan, dtype=np.float32)
    t = np.arange(n) / FPS
    amplitude = 6.0
    for slot in range(22):
        x0 = (25.0 if slot % 2 == 0 else 75.0) + slot // 2
        x = x0 + amplitude * np.sin(speed_mps * np.pi / (2 * amplitude) * t + slot)
        out[:, slot, 0] = (PITCH_L - x if mirror else x) / PITCH_L
        out[:, slot, 1] = (5.0 + 2.7 * slot) / PITCH_W
    return out


def preprocess(ball_xy, speed_mps=4.0):
    """Real preprocessing of a synthetic match: ``ball_xy`` (metres) in half 1, a quiet half 2."""
    n1 = len(ball_xy)
    n = 2 * n1 + GAP_FRAMES
    players = np.full((n, N_SLOTS, 3), np.nan, dtype=np.float32)
    players[:n1] = _players(n1, speed_mps, mirror=False)
    players[n1 + GAP_FRAMES:] = _players(n1, speed_mps, mirror=True)
    ball = np.full((n, 3), np.nan, dtype=np.float32)
    ball[:n1, 0], ball[:n1, 1] = ball_xy[:, 0] / PITCH_L, ball_xy[:, 1] / PITCH_W
    ball[n1 + GAP_FRAMES:, :2] = 0.5
    blocks = [(1, 0, n1 - 1), (2, n1 + GAP_FRAMES, n - 1)]
    tracking, _ = prepare_tracking(players, ball, TEAM_IDS, blocks=blocks,
                                   kickoffs={1: 0, 2: n1 + GAP_FRAMES}, n_frames=n)
    master = build_master_from_arrays(players, ball, TEAM_IDS, blocks)
    assert tracking["flip"].iloc[0] == 0 and tracking["flip"].iloc[-1] == 1
    return master, tracking


def multi_ball_restart(wait):
    """Old ball crosses the touchline at 15 m/s; a new, still ball appears ``wait`` frames later."""
    n, t_out = 400, 150
    xy = np.full((n, 2), np.nan)
    f = np.arange(t_out)
    xy[f, 0] = 50.0
    xy[f, 1] = 10.0 - 0.6 * np.clip(f - (t_out - 20), 0, None)
    xy[t_out + wait:] = (20.0, 0.3)
    return preprocess(xy)


def sparse_real_ball_and_phantom():
    """200 frames where a still phantom outside the pitch hides a real ball seen every 9th frame."""
    n = 500
    f = np.arange(n)
    real = np.c_[30.0 + 0.2 * f, 34.0 + 10.0 * np.sin(f / 20.0)]
    xy = real.copy()
    xy[150:350] = PHANTOM
    sparse = np.arange(150, 350, 9)
    xy[sparse] = real[sparse]
    return preprocess(xy)


@pytest.fixture(scope="module")
def restart_fast():
    return multi_ball_restart(wait=5)


@pytest.fixture(scope="module")
def phantom_case():
    return sparse_real_ball_and_phantom()


@pytest.fixture(scope="module")
def ordinary_play():
    f = np.arange(400)
    return preprocess(np.c_[30.0 + 0.1 * f, 34.0 + 10.0 * np.sin(f / 20.0)])


def test_real_filter_removes_the_old_ball_tail_and_the_tool_counts_it(restart_fast):
    master, tracking = restart_fast
    n_removed = int(master["ball_jump_removed"].sum())
    assert n_removed >= LIMIT                               # the edge case does happen
    stats = jf.removed_before_jumps(master, tracking)
    assert (stats[1].n_removed, stats[1].n_jumps, stats[1].n_continuing) == (
        n_removed, 1, n_removed)
    assert (stats[2].n_removed, stats[2].n_jumps) == (0, 0)
    results = {r.key: r for r in jf.run_checks(master, tracking)}
    assert results["a"].passed and not results["b"].passed


def test_real_filter_keeps_the_old_ball_when_the_new_one_comes_late():
    master, tracking = multi_ball_restart(wait=30)
    assert master["ball_jump_removed"].sum() == 0
    stats = jf.removed_before_jumps(master, tracking)
    assert (stats[1].n_removed, stats[1].n_jumps) == (0, 1)
    assert all(r.passed for r in jf.run_checks(master, tracking))


def test_real_filter_keeps_the_phantom_and_the_tool_counts_it(phantom_case):
    master, tracking = phantom_case
    window = master.iloc[150:350]
    assert window["ball_jump_removed"].sum() > 15           # the real ball lost to the phantom
    assert (window["out_of_pitch"] == 1).sum() > 150
    stats = jf.phantom_stretches(master)
    assert (stats[1].n_counted, stats[1].n_stretches) == (1, 1)
    assert stats[1].longest_frames >= 150
    assert (stats[2].n_counted, stats[2].n_stretches) == (0, 0)
    # One stretch is below the limit: the check passes, the count is in the verbose detail.
    results = {r.key: r for r in jf.run_checks(master, tracking)}
    assert results["a"].passed and "half 1: 1 counted" in results["a"].detail
    # The phantom is also a "new ball" that appears one frame after the real one: the last
    # frames of the real ball before it are removed, which is the footprint of case (b).
    jumps = jf.removed_before_jumps(master, tracking)
    assert jumps[1].n_jumps >= 1 and jumps[1].n_removed == jumps[1].n_continuing > 0


def test_phantom_with_walking_players_is_not_counted():
    n = 300
    xy = np.tile(np.asarray(PHANTOM), (n, 1))
    master, tracking = preprocess(xy, speed_mps=1.5)
    stats = jf.phantom_stretches(master)
    assert (stats[1].n_counted, stats[1].n_stretches) == (0, 1)


def test_ordinary_play_has_no_edge_case(ordinary_play):
    master, tracking = ordinary_play
    assert master["ball_jump_removed"].sum() == 0
    phantom_stats_, jumps = jf.phantom_stretches(master), jf.removed_before_jumps(master, tracking)
    for h in (1, 2):
        assert (phantom_stats_[h].n_counted, phantom_stats_[h].n_stretches) == (0, 0)
        assert (jumps[h].n_removed, jumps[h].n_jumps) == (0, 0)
    assert jf.report(jf.run_checks(master, tracking))[-1] == "PASS"


# -- command line --------------------------------------------------------------------------------
def write_inputs(tmp_path, master, tracking):
    master_path, tracking_path = tmp_path / "master.parquet", tmp_path / "tracking_clean.parquet"
    master.to_parquet(master_path)
    tracking.to_parquet(tracking_path)
    return ["--master", str(master_path), "--tracking", str(tracking_path)]


def test_main_prints_only_pass_fail_by_default(tmp_path, capsys, restart_fast):
    args = write_inputs(tmp_path, *restart_fast)
    assert jf.main(args) == 1
    out = capsys.readouterr().out.splitlines()
    assert out == ["PASS  (a) no still phantom kept outside the pitch while play goes on",
                   "FAIL  (b) no observations removed just before a multi-ball jump",
                   "FAIL (1 of 2 checks)"]

    assert jf.main(args + ["--verbose"]) == 1
    verbose = capsys.readouterr().out
    assert "half 1:" in verbose and "removed observations" in verbose


def test_main_passes_on_ordinary_play(tmp_path, capsys, ordinary_play):
    assert jf.main(write_inputs(tmp_path, *ordinary_play)) == 0
    assert capsys.readouterr().out.splitlines() == [
        "PASS  (a) no still phantom kept outside the pitch while play goes on",
        "PASS  (b) no observations removed just before a multi-ball jump",
        "PASS"]


def test_main_reports_unusable_inputs_with_exit_code_2(tmp_path, capsys, restart_fast):
    master, tracking = restart_fast
    args = write_inputs(tmp_path, master, tracking)
    assert jf.main(["--master", str(tmp_path / "nope.parquet"), "--tracking", args[3]]) == 2
    assert capsys.readouterr().out.startswith("ERROR  could not load inputs")
    assert jf.main(["--master", args[1], "--tracking", str(tmp_path / "nope.parquet")]) == 2
    assert capsys.readouterr().out.startswith("ERROR  could not load inputs")

    stale = tmp_path / "stale"
    stale.mkdir()
    stale_args = write_inputs(stale, master,
                              tracking.assign(ball_x_raw=tracking["ball_x_raw"] + 0.5))
    assert jf.main(stale_args) == 2
    assert "regenerate" in capsys.readouterr().out

    narrow = tmp_path / "narrow"
    narrow.mkdir()
    narrow_args = write_inputs(narrow, master, tracking.drop(columns=["ball_x_raw"]))
    assert jf.main(narrow_args) == 2
    assert capsys.readouterr().out.startswith("ERROR  could not load inputs")


def test_script_runs_from_the_command_line(tmp_path, restart_fast):
    args = write_inputs(tmp_path, *restart_fast)
    done = subprocess.run([sys.executable, str(REPO / "tools" / "jump_filter_edge_cases.py")]
                          + args, capture_output=True, text=True, cwd=str(tmp_path))
    assert done.returncode == 1, done.stderr
    assert done.stdout.splitlines()[-1] == "FAIL (1 of 2 checks)"
