"""Tests for collective evidence."""
import numpy as np
import pandas as pd

from modeling.evidence.collective import build_collective_score
from modeling.evidence.team_logit import team_logit_v1
from modeling.evidence.team_logit_params import TeamLogitParams
from tests.test_team_logit_v1 import make_master, swap_teams, P

def _make_collective_master(n=100):
    df = make_master(n)
    df["half"] = 1
    df["centroid_home_x"] = np.linspace(10, 50, n)
    df["centroid_away_x"] = np.linspace(60, 40, n)
    df["spread_home_x"] = np.random.uniform(10, 20, n)
    df["spread_away_x"] = np.random.uniform(10, 20, n)
    df["spread_home_y"] = np.random.uniform(10, 20, n)
    df["spread_away_y"] = np.random.uniform(10, 20, n)
    df["mean_speed_home"] = np.random.uniform(2, 6, n)
    df["mean_speed_away"] = np.random.uniform(2, 6, n)
    df["gap_length"] = np.zeros(n)
    return df

def test_collective_symmetry():
    master = _make_collective_master()
    score = build_collective_score(master)
    swapped = swap_teams(master)
    swapped["centroid_home_x"] = 105.3 - swapped["centroid_home_x"]
    swapped["centroid_away_x"] = 105.3 - swapped["centroid_away_x"]
    
    score_swapped = build_collective_score(swapped)
    assert np.allclose(score, -score_swapped, atol=1e-5)

def test_weight_zero_changes_nothing():
    master = _make_collective_master()
    master.loc[50:60, "ball_observed"] = 0 # create a gap
    
    t_zero = team_logit_v1(master, P)
    
    P_active = TeamLogitParams(**{**P.__dict__, "w_coll": 0.5})
    t_active = team_logit_v1(master, P_active)
    
    # Outside the gap, they are identical
    assert np.array_equal(t_zero[:49], t_active[:49])
    
    # Inside the gap, active weight provides a non-zero signal
    assert np.all(t_zero[50:60] == 0.0)
    assert not np.all(t_active[50:60] == 0.0)

def test_no_evidence_outside_play():
    master = _make_collective_master()
    master["half"] = 0
    assert np.all(build_collective_score(master) == 0.0)