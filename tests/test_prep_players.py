"""Synthetic player tracking regressions; no match assets are read."""
import numpy as np

from preprocessing import prep_tracking as pt


def _old_segment_gradient(values, valid, breaks=None, fps=pt.FPS, smooth=True):
    out = np.full(len(values), np.nan, dtype=np.float32)
    values = np.asarray(values)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    starts = valid & ~np.r_[False, valid[:-1]]
    if breaks is not None:
        breaks = np.asarray(breaks, dtype=bool)
        starts |= valid & breaks
    for start in np.flatnonzero(starts):
        end = start + 1
        while end < len(values) and valid[end] and not (breaks is not None and breaks[end]):
            end += 1
        if end - start >= 2:
            seg = pt._binomial3_smooth(values[start:end]) if smooth else values[start:end]
            out[start:end] = np.gradient(seg) * fps
    return out


def test_vectorized_gradient_matches_old_loop_on_seeded_synthetic_data():
    rng = np.random.default_rng(20261031)
    for n in range(1, 80):
        values = rng.normal(size=n)
        valid = rng.random(n) > 0.25
        values[~valid] = np.nan
        breaks = rng.random(n) > 0.85
        for smooth in (False, True):
            expected = _old_segment_gradient(values, valid, breaks, smooth=smooth)
            actual = pt._segment_gradient(values, valid, breaks, smooth=smooth)
            np.testing.assert_array_equal(actual, expected)


def _synthetic(n=6, slots=pt.N_SLOTS):
    players = np.full((n, slots, 3), np.nan, dtype=np.float32)
    ball = np.full((n, 3), np.nan, dtype=np.float32)
    teams = np.arange(slots) % 2
    return players, ball, teams


def test_gradient_cuts_at_break_without_contaminating_either_side():
    x = np.array([10, 10, 10, 40, 40, 40], dtype=float)
    y = np.full(6, 10.0)
    valid = np.ones(6, dtype=bool)
    breaks = np.array([False, False, False, True, False, False])
    vx = pt._segment_gradient(x, valid, breaks=breaks, smooth=False)
    vy = pt._segment_gradient(y, valid, breaks=breaks, smooth=False)
    np.testing.assert_allclose(vx[[2, 3]], [0, 0])
    np.testing.assert_allclose(vy[[2, 3]], [0, 0])
    assert np.nanmax(np.abs(vx)) == 0


def test_permanent_position_jump_is_cut_even_when_native_speed_is_zero():
    """ATD's scalar speed must not hide a permanent position discontinuity."""
    players, ball, teams = _synthetic(n=6)
    players[:, 0, :2] = [0.2, 0.5]
    players[:, 0, 2] = 0.0  # native ATD speed field
    players[2:, 0, 0] = 0.7  # new, permanently stable location from t=2

    frame, _ = pt.prepare_tracking(
        players, ball, teams, blocks=[(1, 0, 5)], kickoffs={1: 0},
        n_frames=6, stage="V1")

    assert frame.loc[2, "s01_break"]
    assert not frame.loc[3, "s01_break"]
    # The new segment starts at t=2 and has zero vector speed; no cross-break
    # derivative may leak the large coordinate jump into t=1 or t=2.
    np.testing.assert_allclose(frame.loc[[2, 3], ["s01_vx", "s01_vy"]], 0.0)
    np.testing.assert_allclose(
        frame.loc[1, ["s01_vx", "s01_vy"]].to_numpy(dtype=float), 0.0)


def test_singleton_segment_has_no_derivative():
    values = np.array([1.0, np.nan, 2.0, np.nan, 3.0, 4.0])
    valid = np.isfinite(values)
    breaks = np.array([False, False, True, False, False, False])
    result = pt._segment_gradient(values, valid, breaks=breaks, smooth=False)
    assert np.isnan(result[0])
    assert np.isnan(result[2])
    assert result[4] == 25.0
    assert result[5] == 25.0


def test_prepare_removes_isolated_and_same_team_duplicate_before_aggregates():
    players, ball, teams = _synthetic()
    # Stable kickoff direction and simple stationary tracks.
    players[:, 0, :2] = [0.2, 0.5]
    players[:, 1, :2] = [0.205, 0.5]  # same team, within one metre
    teams[1] = teams[0]
    players[:, 2, :2] = [0.8, 0.5]
    teams[2] = 1 - teams[0]
    # Isolated excursion in slot 3; neighbors agree.
    players[:, 3, :2] = [0.4, 0.5]
    players[2, 3, :2] = [0.8, 0.5]
    teams[3] = 1
    frame, _ = pt.prepare_tracking(players, ball, teams, blocks=[(1, 0, 5)],
                                   kickoffs={1: 0}, n_frames=6, stage="V1")
    assert frame.loc[0, ["s02_x", "s02_y", "s02_vx", "s02_vy"]].isna().all()
    assert frame.loc[2, ["s04_x", "s04_y", "s04_vx", "s04_vy"]].isna().all()
    assert frame.loc[0, "n_tracked_home"] == 1
    assert not frame.loc[2, "s04_break"]
    assert frame.loc[3, "s04_break"]  # reacquisition after the removed isolated point


def test_quality_before_dedup_is_measured_after_isolated_removal(caplog):
    """An isolated point that overlaps another slot is excluded from the before count."""
    players, ball, teams = _synthetic(n=5)
    # 21 base tracks plus a duplicate partner and one isolated sample.
    teams[:] = 0
    players[:, :21, :2] = np.column_stack((np.linspace(0.1, 0.3, 21), np.full(21, 0.5)))
    players[:, 21, :2] = [0.8, 0.5]
    # Slot 23's one-frame spike duplicates slot 22, then is removed as isolated.
    players[:, 22, :2] = [0.4, 0.5]
    players[2, 22, :2] = [0.8, 0.5]
    with caplog.at_level("INFO", logger=pt.__name__):
        frame, _ = pt.prepare_tracking(
            players, ball, teams, blocks=[(1, 0, 4)], kickoffs={1: 0},
            n_frames=5, stage="V1")
    # Isolated samples are removed before counting, so frame 2 has exactly 22
    # objects. The other four frames have 23; a premature snapshot would count
    # frame 2 too and report five frames above the threshold.
    assert frame.loc[2, "n_tracked_home"] == 22
    assert "frames >22 objects before=4 (80.00%), after=4 (80.00%)" in caplog.text


def test_different_team_nearby_tracks_are_retained():
    players, ball, teams = _synthetic(3)
    players[:, 0, :2] = [0.2, 0.5]
    players[:, 1, :2] = [0.205, 0.5]
    teams[0], teams[1] = 0, 1
    players[:, 2, :2] = [0.8, 0.5]
    teams[2] = 1
    frame, _ = pt.prepare_tracking(players, ball, teams, blocks=[(1, 0, 2)],
                                   kickoffs={1: 0}, n_frames=3, stage="V0")
    assert np.isfinite(frame.loc[0, "s01_x"])
    assert np.isfinite(frame.loc[0, "s02_x"])


def test_v1_smooths_synthetic_player_velocity_deterministically():
    t = np.arange(80, dtype=float)
    values = 0.08 * t + 0.03 * np.sin(1.7 * t)
    valid = np.ones(len(t), dtype=bool)
    raw = pt._segment_gradient(values, valid, smooth=False)
    smooth_a = pt._segment_gradient(values, valid, smooth=True)
    smooth_b = pt._segment_gradient(values, valid, smooth=True)
    assert np.var(smooth_a[5:-5]) < np.var(raw[5:-5])
    np.testing.assert_array_equal(smooth_a, smooth_b)
