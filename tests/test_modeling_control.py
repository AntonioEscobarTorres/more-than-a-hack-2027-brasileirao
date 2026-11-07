"""F1-01 control features. Synthetic masters only: nothing from the match."""
import inspect

import numpy as np
import pandas as pd
import pytest

from modeling.evidence import collective, control, team
from modeling.evidence.control import CONTROL_COLUMNS, build_control_features
from tests.modeling_fixtures import fuzz_master, make_master, swap_teams, team_params

P = team_params()


def test_columns_dtypes_and_index():
    master = make_master(3)
    master.index = pd.Index([10, 11, 12], name="video_frame")
    ev = build_control_features(master, P)
    assert tuple(ev.columns) == CONTROL_COLUMNS
    assert ev.index.equals(master.index)
    assert ev["ev_ball_valid"].dtype == bool
    assert all(ev[c].dtype == np.float32 for c in CONTROL_COLUMNS if c != "ev_ball_valid")


def test_ball_validity_follows_ball_x_and_ball_y():
    master = make_master(4, ball_x=[1.0, np.nan, 1.0, np.nan], ball_y=[1.0, 1.0, np.nan, np.nan])
    assert list(build_control_features(master, P)["ev_ball_valid"]) == [True, False, False, False]


def test_ball_confidence_levels():
    master = make_master(3, ball_observed=[1, 0, 0], ball_filled=[0, 1, 0])
    assert np.allclose(build_control_features(master, P)["ev_ball_conf"], [1.0, 0.6, 0.0])


def test_dmin_and_proximity_weight():
    master = make_master(4, d_near_home=[1.0, 2.0, 5.0, np.nan], d_near_away=[4.0, 9.0, 3.0, np.nan])
    ev = build_control_features(master, P)
    assert np.allclose(ev["ev_dmin"].iloc[:3], [1.0, 2.0, 3.0]) and np.isnan(ev["ev_dmin"].iloc[3])
    assert np.allclose(ev["ev_prox_w"].iloc[:3], [1.0, 1.0, np.exp(-1.0 / 3.0)])
    assert np.isnan(ev["ev_prox_w"].iloc[3])


def test_one_missing_team_distance_is_ignored_by_dmin():
    ev = build_control_features(make_master(2, d_near_home=[2.5, np.nan], d_near_away=[np.nan, 7.0]), P)
    assert list(ev["ev_dmin"]) == [2.5, 7.0]


def test_relative_speed_comes_from_the_closer_team_and_ties_go_to_home():
    master = make_master(5, d_near_home=[1.0, 4.0, 3.0, np.nan, np.nan],
                         d_near_away=[4.0, 1.0, 3.0, 2.0, np.nan],
                         speed_rel_home=2.0, speed_rel_away=8.0)
    rel = build_control_features(master, P)["ev_rel_near"].to_numpy()
    assert list(rel[:4]) == [2.0, 8.0, 2.0, 8.0] and np.isnan(rel[4])


def test_relative_speed_weight_values():
    master = make_master(4, speed_rel_home=[0.0, 6.0, 30.0, np.nan])
    w = build_control_features(master, P)["ev_rel_w"].to_numpy()
    assert w[0] == 1.0 and np.isclose(w[1], 0.5) and w[2] < w[1]
    assert w[3] == np.float32(P["rel_w_when_nan"])            # NaN speed with a valid ball


def test_contest_is_a_float_flag():
    master = make_master(4, n_home_r2=[1, 2, 0, 1], n_away_r2=[1, 0, 1, 3])
    assert list(build_control_features(master, P)["ev_contest"]) == [1.0, 0.0, 0.0, 1.0]


def test_everything_is_nan_without_a_ball():
    master = make_master(1, ball_observed=0, ball_filled=0, d_near_home=np.nan,
                         d_near_away=np.nan, n_home_r2=np.nan, n_away_r2=np.nan,
                         speed_rel_home=np.nan, speed_rel_away=np.nan)
    ev = build_control_features(master, P).iloc[0]
    assert not ev["ev_ball_valid"] and ev["ev_ball_conf"] == 0.0
    for c in ("ev_dmin", "ev_prox_w", "ev_rel_near", "ev_rel_w", "ev_contest"):
        assert np.isnan(ev[c]), c


def test_weights_are_in_unit_interval():
    ev = build_control_features(fuzz_master(500, seed=6), P)
    for c in ("ev_ball_conf", "ev_prox_w", "ev_rel_w", "ev_contest"):
        x = ev[c].dropna()
        assert ((x >= 0) & (x <= 1)).all(), c
    assert set(ev["ev_contest"].dropna().unique()) <= {0.0, 1.0}


def test_proximity_weight_decreases_with_dmin():
    d = np.linspace(0.0, 20.0, 60)
    ev = build_control_features(make_master(60, d_near_home=d, d_near_away=d + 3.0), P)
    assert np.allclose(ev["ev_dmin"], d)
    prox = ev["ev_prox_w"].to_numpy()
    assert np.all(np.diff(prox) <= 0) and prox[0] == 1.0 and np.all(np.diff(prox[d > 2.0]) < 0)


def test_relative_speed_weight_decreases_with_relative_speed():
    v = np.linspace(0.0, 30.0, 60)
    rel = build_control_features(make_master(60, speed_rel_home=v), P)["ev_rel_w"].to_numpy()
    assert np.all(np.diff(rel) < 0) and rel[0] == 1.0


def test_swapping_teams_keeps_dmin_proximity_and_contest():
    master = fuzz_master(400, seed=5)
    a = build_control_features(master, P)
    b = build_control_features(swap_teams(master), P)
    pd.testing.assert_frame_equal(a, b)                  # continuous distances: no ties


def test_each_row_is_independent():
    master = fuzz_master(50, seed=7)
    whole = build_control_features(master, P)
    single = pd.concat([build_control_features(master.iloc[[i]], P) for i in range(50)])
    pd.testing.assert_frame_equal(whole, single)


def test_missing_columns_raise():
    with pytest.raises(ValueError, match="missing columns"):
        build_control_features(make_master(2).drop(columns=["n_home_r2"]), P)


@pytest.mark.parametrize("module", [control, collective, team])
def test_no_fillna_in_evidence_code(module):
    assert "fillna" not in inspect.getsource(module)
