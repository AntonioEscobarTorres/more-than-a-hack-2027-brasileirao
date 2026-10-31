"""Synthetic regression tests for the A2 ball pipeline."""
import numpy as np

from preprocessing import prep_tracking as pt


def _linear_ball(n, observed):
    ball = np.full((n, 3), np.nan, dtype=np.float32)
    frames = np.arange(n, dtype=np.float32)
    ball[observed, 0] = 0.2 + frames[observed] * 0.0004
    ball[observed, 1] = 0.5
    return ball


def test_jump_filter_removes_ghost_at_gap_end_and_reconnects_after_long_burst():
    # A single ghost near the end of a short missing interval cannot replace the
    # consistent real trajectory on either side.
    n = 40
    x = 0.1 + np.arange(n, dtype=np.float64) * 0.02
    y = np.full(n, 20.0)
    present = np.r_[0:10, 13:n]
    x_input, y_input = np.full(n, np.nan), np.full(n, np.nan)
    x_input[present], y_input[present] = x[present], y[present]
    x_input[12], y_input[12] = 45.0, 55.0
    kept_x, kept_y = pt.remove_ball_jump_outliers(x_input, y_input, vmax=10, eps=0.1, k=5)
    assert np.isnan(kept_x[12]) and np.isnan(kept_y[12])
    np.testing.assert_allclose(kept_x[present], x[present])

    # A longer stationary phantom burst is discarded while the real trajectory
    # resumes through the saved pre-break anchor.
    n = 50
    x = 3.0 + np.arange(n, dtype=np.float64) * 0.02
    y = np.full(n, 20.0)
    x_input, y_input = x.copy(), y.copy()
    x_input[8:25], y_input[8:25] = 90.0, 60.0
    kept_x, kept_y = pt.remove_ball_jump_outliers(x_input, y_input, vmax=10, eps=0.1, k=4)
    assert np.isnan(kept_x[8:25]).all() and np.isnan(kept_y[8:25]).all()
    np.testing.assert_allclose(kept_x[np.r_[0:8, 25:n]], x[np.r_[0:8, 25:n]])


def test_gap_metadata_and_filled_position_invariants():
    n = 42
    blocks = [(1, 0, n - 1)]
    teams = np.arange(pt.N_SLOTS) % 2
    players = np.full((n, pt.N_SLOTS, 3), np.nan, dtype=np.float32)
    players[:, 0, :2] = (0.2, 0.5)
    observed = np.r_[3:10, 12:20, (20 + pt.MAX_GAP_FRAMES + 1):n - 3]
    ball = _linear_ball(n, observed)

    frame, _ = pt.prepare_tracking(players, ball, teams, blocks=blocks,
                                   kickoffs={1: 0}, n_frames=n)
    filled = frame["ball_filled"].to_numpy(dtype=bool)
    is_observed = frame["ball_observed"].to_numpy(dtype=bool)
    gap = frame["gap_length"].to_numpy()
    bx = frame["ball_x"].to_numpy()

    assert np.all(~filled | (gap <= pt.MAX_GAP_FRAMES))
    assert np.all(~(filled & is_observed))
    np.testing.assert_array_equal(np.isnan(bx), ~(is_observed | filled))
    assert np.isnan(frame.loc[:2, "since_last_obs"]).all()
    assert np.isnan(frame.loc[n - 3:, "until_next_obs"]).all()
    # The schema specifies finite lengths for edge gaps, too.
    assert (frame.loc[:2, "gap_length"] == 3).all()
    assert (frame.loc[n - 3:, "gap_length"] == 3).all()
