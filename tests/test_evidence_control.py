"""Properties of the control features. Synthetic masters only: nothing from the match."""
import numpy as np
import pandas as pd
import pytest

from modeling.evidence.control import CONTROL_COLUMNS, build_control_features
from modeling.evidence.params import ControlParams

P = ControlParams(r_ctrl=2.0, l_prox=3.0, v_rel=6.0, ball_conf_filled=0.6, rel_w_missing=0.5)


def make_master(n=1, **cols):
    """Valid-ball defaults; any column can be overridden with a scalar or an array."""
    base = {"d_near_home": 1.0, "d_near_away": 5.0, "n_home_r2": 1.0, "n_away_r2": 0.0,
            "speed_rel_home": 1.0, "speed_rel_away": 1.0, "ball_observed": 1, "ball_filled": 0,
            "since_last_obs": 0.0, "until_next_obs": 0.0}
    base.update(cols)
    return pd.DataFrame({k: np.broadcast_to(np.asarray(v, dtype="float64"), (n,)).copy()
                         for k, v in base.items()})


def test_columns_and_index_are_kept():
    master = make_master(3)
    master.index = pd.Index([10, 11, 12], name="video_frame")
    ev = build_control_features(master, P)
    assert list(ev.columns) == list(CONTROL_COLUMNS)
    assert ev.index.equals(master.index)


def test_ball_confidence_levels():
    master = make_master(3, ball_observed=[1, 0, 0], ball_filled=[0, 1, 0])
    assert list(build_control_features(master, P)["ev_ball_conf"]) == [1.0, 0.6, 0.0]


def test_dmin_and_proximity_weight():
    master = make_master(4, d_near_home=[1.0, 2.0, 5.0, np.nan], d_near_away=[4.0, 9.0, 3.0, np.nan])
    ev = build_control_features(master, P)
    assert np.allclose(ev["ev_dmin"].iloc[:3], [1.0, 2.0, 3.0]) and np.isnan(ev["ev_dmin"].iloc[3])
    assert np.allclose(ev["ev_prox_w"].iloc[:3], [1.0, 1.0, np.exp(-1.0 / 3.0)])   # full weight within r_ctrl
    assert np.isnan(ev["ev_prox_w"].iloc[3])                                      # NaN stays NaN


def test_one_missing_team_distance_is_ignored_by_dmin():
    ev = build_control_features(make_master(2, d_near_home=[2.5, np.nan], d_near_away=[np.nan, 7.0]), P)
    assert list(ev["ev_dmin"]) == [2.5, 7.0]


def test_relative_speed_is_taken_from_the_closer_team_and_ties_go_to_home():
    master = make_master(4, d_near_home=[1.0, 4.0, 3.0, np.nan], d_near_away=[4.0, 1.0, 3.0, 2.0],
                         speed_rel_home=[2.0, 2.0, 2.0, 2.0], speed_rel_away=[8.0, 8.0, 8.0, 8.0])
    assert list(build_control_features(master, P)["ev_rel_near"]) == [2.0, 8.0, 2.0, 8.0]
    both_nan = build_control_features(make_master(1, d_near_home=np.nan, d_near_away=np.nan), P)
    assert np.isnan(both_nan["ev_rel_near"].iloc[0])


def test_relative_speed_weight_decreases_and_missing_uses_the_configured_value():
    master = make_master(4, speed_rel_home=[0.0, 6.0, 30.0, np.nan])
    w = build_control_features(master, P)["ev_rel_w"].to_numpy()
    assert w[0] == 1.0 and np.isclose(w[1], 0.5) and w[2] < w[1] and w[3] == P.rel_w_missing
    assert w[0] > w[1] > w[2]


def test_contest_needs_both_teams_within_two_metres_and_nan_counts_are_not_contest():
    master = make_master(4, n_home_r2=[1, 2, 0, np.nan], n_away_r2=[1, 0, 1, 1])
    assert list(build_control_features(master, P)["ev_contest"]) == [True, False, False, False]


def test_since_uses_the_nearest_observation_in_either_direction():
    master = make_master(4, since_last_obs=[5.0, np.nan, 3.0, np.nan],
                         until_next_obs=[9.0, 4.0, np.nan, np.nan])
    ev = build_control_features(master, P)["ev_since"]
    assert list(ev.iloc[:3]) == [5.0, 4.0, 3.0] and np.isnan(ev.iloc[3])


def test_no_fillna_the_master_nans_do_not_become_zero_evidence_columns():
    ev = build_control_features(make_master(1, d_near_home=np.nan, d_near_away=np.nan,
                                             ball_observed=0, ball_filled=0), P)
    assert np.isnan(ev["ev_dmin"].iloc[0]) and np.isnan(ev["ev_prox_w"].iloc[0])


def test_each_row_is_independent_no_window_no_delay():
    rng = np.random.default_rng(0)
    master = make_master(50, d_near_home=rng.uniform(0, 20, 50), d_near_away=rng.uniform(0, 20, 50),
                         speed_rel_home=rng.uniform(0, 20, 50), speed_rel_away=rng.uniform(0, 20, 50))
    whole = build_control_features(master, P)
    single = pd.concat([build_control_features(master.iloc[[i]], P) for i in range(50)])
    pd.testing.assert_frame_equal(whole, single)
    shuffled = master.sample(frac=1.0, random_state=1)
    pd.testing.assert_frame_equal(build_control_features(shuffled, P), whole.loc[shuffled.index])


def test_missing_columns_are_reported():
    with pytest.raises(ValueError, match="missing columns"):
        build_control_features(make_master(2).drop(columns=["n_home_r2"]), P)


def test_params_reject_unknown_missing_and_out_of_range_values():
    good = dict(r_ctrl=2.0, l_prox=3.0, v_rel=6.0, ball_conf_filled=0.6, rel_w_missing=0.5)
    assert ControlParams.from_dict(good) == P
    with pytest.raises(ValueError, match="unknown keys"):
        ControlParams.from_dict({**good, "r_control": 1.0})
    with pytest.raises(ValueError, match="missing keys"):
        ControlParams.from_dict({k: v for k, v in good.items() if k != "v_rel"})
    with pytest.raises(ValueError, match="rel_w_missing"):
        ControlParams.from_dict({**good, "rel_w_missing": 1.5})


def swap_teams(master):
    """Exchange the roles of home and away: swap every *_home/*_away column and flip d_diff."""
    def swap(name):
        if "home" in name:
            return name.replace("home", "away")
        if "away" in name:
            return name.replace("away", "home")
        return name
    out = master.rename(columns=swap)
    return out[master.columns]


def _fuzz_master(n, seed):
    rng = np.random.default_rng(seed)
    return make_master(
        n, d_near_home=rng.uniform(0, 25, n), d_near_away=rng.uniform(0, 25, n),
        n_home_r2=rng.integers(0, 3, n), n_away_r2=rng.integers(0, 3, n),
        speed_rel_home=rng.uniform(0, 25, n), speed_rel_away=rng.uniform(0, 25, n),
        ball_observed=rng.integers(0, 2, n), ball_filled=rng.integers(0, 2, n),
        since_last_obs=rng.uniform(0, 50, n), until_next_obs=rng.uniform(0, 50, n))


def test_every_feature_is_symmetric_when_the_teams_are_swapped():
    master = _fuzz_master(300, seed=5)                 # continuous distances: no ties
    pd.testing.assert_frame_equal(build_control_features(swap_teams(master), P),
                                  build_control_features(master, P))


def test_bounds_of_every_column():
    master = _fuzz_master(400, seed=6)
    ev = build_control_features(master, P)
    assert set(np.unique(ev["ev_ball_conf"])) <= {0.0, P.ball_conf_filled, 1.0}
    assert ((ev["ev_prox_w"] > 0) & (ev["ev_prox_w"] <= 1)).all()
    assert ((ev["ev_rel_w"] > 0) & (ev["ev_rel_w"] <= 1)).all()
    assert (ev["ev_dmin"] >= 0).all() and (ev["ev_since"] >= 0).all()
    assert ev["ev_contest"].dtype == bool


def test_weights_are_monotonic_in_their_inputs():
    d = np.linspace(0.0, 20.0, 60)
    prox = build_control_features(make_master(60, d_near_home=d, d_near_away=d + 3.0), P)["ev_prox_w"].to_numpy()
    assert np.all(np.diff(prox) <= 0) and prox[0] == 1.0 and prox[-1] < prox[len(prox) // 2]
    v = np.linspace(0.0, 30.0, 60)
    rel = build_control_features(make_master(60, speed_rel_home=v), P)["ev_rel_w"].to_numpy()
    assert np.all(np.diff(rel) < 0) and rel[0] == 1.0


def test_nan_pattern_follows_the_ball_distances():
    master = make_master(3, d_near_home=[np.nan, np.nan, 3.0], d_near_away=[np.nan, 4.0, np.nan])
    ev = build_control_features(master, P)
    assert list(np.isnan(ev["ev_dmin"])) == [True, False, False]
    assert list(np.isnan(ev["ev_prox_w"])) == [True, False, False]
    assert list(np.isnan(ev["ev_rel_near"])) == [True, False, False]     # speed of the team that has a distance
