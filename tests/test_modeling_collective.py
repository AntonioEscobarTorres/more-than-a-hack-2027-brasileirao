"""F1-04 collective evidence. Synthetic masters only: nothing from the match."""
import numpy as np
import pandas as pd
import pytest

import games
from modeling.evidence.collective import COLLECTIVE_COLUMNS, build_collective_features
from modeling.evidence.team import ball_team_logit, team_logit_rules_v1
from preprocessing.params import MAX_GAP_FRAMES
from tests.modeling_fixtures import fuzz_master, make_master, swap_teams, team_params

P = team_params()
W = P["collective"]["win_frames"]
FPS = games.GAMES[games.GAME]["fps"]


def moving_master(n, seed, half=1):
    rng = np.random.default_rng(seed)
    return make_master(
        n, half=half,
        centroid_home_x=np.cumsum(rng.normal(0, 0.1, n)) + 40,
        centroid_away_x=np.cumsum(rng.normal(0, 0.1, n)) + 60,
        spread_home_x=rng.uniform(5, 20, n), spread_home_y=rng.uniform(5, 20, n),
        spread_away_x=rng.uniform(5, 20, n), spread_away_y=rng.uniform(5, 20, n),
        mean_speed_home=rng.uniform(1, 6, n), mean_speed_away=rng.uniform(1, 6, n))


def test_columns_and_dtypes():
    f = build_collective_features(moving_master(100, 0), P)
    assert tuple(f.columns) == COLLECTIVE_COLUMNS
    assert all(f[c].dtype == np.float32 for c in COLLECTIVE_COLUMNS)


def test_derivative_on_a_known_ramp():
    n, v_home, v_away = 200, 2.0, -1.5                    # m/s
    frames = np.arange(n)
    master = make_master(n, centroid_home_x=10 + v_home * frames / FPS,
                         centroid_away_x=90 + v_away * frames / FPS)
    f = build_collective_features(master, P)
    inner = slice(W, n - W)
    assert np.allclose(f["ev_adv_home"].iloc[inner], v_home, atol=1e-5)
    assert np.allclose(f["ev_adv_away"].iloc[inner], -v_away, atol=1e-5)   # away attacks to -x
    assert np.allclose(f["ev_coll_adv"].iloc[inner], v_home + v_away, atol=1e-5)
    assert f["ev_adv_home"].iloc[:W].isna().all() and f["ev_adv_home"].iloc[n - W:].isna().all()


def test_spread_is_a_difference_of_areas_and_speed_a_difference():
    master = make_master(3, spread_home_x=10.0, spread_home_y=10.0, spread_away_x=5.0,
                         spread_away_y=[20.0, 10.0, 30.0], mean_speed_home=3.0, mean_speed_away=1.0)
    f = build_collective_features(master, P)
    assert list(f["ev_coll_spread"]) == [0.0, 50.0, -50.0]
    assert list(f["ev_coll_speed"]) == [2.0, 2.0, 2.0]


def test_antisymmetric_when_the_teams_are_swapped():
    master = moving_master(300, seed=1)
    a = build_collective_features(master, P)
    b = build_collective_features(swap_teams(master), P)
    for c in ("ev_coll_adv", "ev_coll_spread", "ev_coll_speed"):
        assert np.allclose(b[c], -a[c], atol=1e-4, equal_nan=True), c
        assert a[c].notna().sum() > 200, c


def test_nothing_crosses_the_half_boundary():
    first, gap, second = moving_master(100, 2, half=1), moving_master(10, 3, half=0), \
        moving_master(120, 4, half=2)
    master = pd.concat([first, gap, second], ignore_index=True)
    f = build_collective_features(master, P)
    adv = f["ev_coll_adv"].to_numpy()
    assert np.isnan(adv[100 - W:100 + 10 + W]).all()      # window would leave its half
    assert np.isfinite(adv[W:100 - W]).all() and np.isfinite(adv[110 + W:230 - W]).all()
    assert f.iloc[100:110].isna().all().all()             # half == 0: no evidence

    other = master.copy()                                 # changing half 2 leaves half 1 alone
    other.loc[110:, ["centroid_home_x", "centroid_away_x"]] += 37.0
    g = build_collective_features(other, P)
    pd.testing.assert_frame_equal(f.iloc[:110], g.iloc[:110])

    # the same half value in two separate runs is not one window either
    split = master.copy()
    split.loc[100:109, "half"] = 1
    split.loc[105, "half"] = 0
    h = build_collective_features(split, P)["ev_coll_adv"].to_numpy()
    assert np.isnan(h[105 - W:105]).all()


def test_nan_inputs_give_nan():
    master = moving_master(100, 5)
    master.loc[60, "centroid_away_x"] = np.nan
    master.loc[70, "spread_home_y"] = np.nan
    f = build_collective_features(master, P)
    assert np.isnan(f["ev_coll_adv"].iloc[60 - W]) and np.isnan(f["ev_coll_adv"].iloc[60 + W])
    assert np.isnan(f["ev_coll_spread"].iloc[70]) and np.isfinite(f["ev_coll_speed"].iloc[70])


def test_missing_columns_raise():
    with pytest.raises(ValueError, match="missing columns"):
        build_collective_features(make_master(3).drop(columns=["mean_speed_away"]), P)


# -- integration in team_logit rules_v1 ----------------------------------------------------------

def gap_master(seed=6):
    """A match with a long ball gap (frames 100..159) and a short filled one (200..209)."""
    master = moving_master(300, seed)
    rng = np.random.default_rng(seed)
    master["d_near_home"], master["d_near_away"] = rng.uniform(0, 15, 300), rng.uniform(0, 15, 300)
    master["d_diff"] = master["d_near_home"] - master["d_near_away"]
    lost = np.arange(100, 160)
    master.loc[lost, ["ball_observed", "ball_x", "ball_y"]] = [0.0, np.nan, np.nan]
    master.loc[lost, ["d_near_home", "d_near_away", "d_diff", "n_home_r2", "n_away_r2",
                      "speed_rel_home", "speed_rel_away"]] = np.nan
    master.loc[lost, "gap_length"] = 60.0
    short = np.arange(200, 210)
    master.loc[short, ["ball_observed", "ball_filled", "gap_length"]] = [0.0, 1.0, 10.0]
    assert 10 <= MAX_GAP_FRAMES < 60
    return master


def test_zero_weights_are_bit_identical_to_the_ball_term():
    for master in (gap_master(), fuzz_master(300, seed=8)):
        t = team_logit_rules_v1(master, P)
        assert t.tobytes() == ball_team_logit(master, P).astype(np.float32).tobytes()


def test_collective_term_only_acts_where_the_ball_is_missing_or_the_gap_is_long():
    master = gap_master()
    off = team_logit_rules_v1(master, P)
    on = team_logit_rules_v1(master, team_params(w_adv=0.5, w_spread=0.01, w_speed=0.3))
    changed = np.flatnonzero(on != off)
    assert set(changed) <= set(range(100, 160)) and len(changed) == 60
    assert np.abs(on).max() <= P["kappa"] and np.isfinite(on).all()


def test_a_nan_term_contributes_zero_but_the_others_still_count():
    master = gap_master()
    master.loc[100:159, "centroid_home_x"] = np.nan      # ev_coll_adv NaN in the whole gap
    t = team_logit_rules_v1(master, team_params(w_adv=1.0, w_speed=0.3))
    expected = np.clip(0.3 * (master["mean_speed_home"] - master["mean_speed_away"]), -3, 3)
    assert np.allclose(t[100:160], expected.iloc[100:160], atol=1e-5)


def test_collective_logit_flips_sign_with_the_teams():
    master = gap_master()
    params = team_params(w_adv=0.5, w_spread=0.01, w_speed=0.3)
    a, b = team_logit_rules_v1(master, params), team_logit_rules_v1(swap_teams(master), params)
    assert np.allclose(b, -a, atol=1e-5)


def test_collective_term_is_zero_on_half_zero():
    master = gap_master()
    master.loc[120:130, "half"] = 0
    master.loc[120:130, "gap_length"] = np.nan
    t = team_logit_rules_v1(master, team_params(w_speed=1.0))
    assert (t[120:131] == 0).all()
