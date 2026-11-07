"""Properties of team_logit v1. Synthetic masters only: nothing from the match."""
import numpy as np
import pandas as pd

from modeling.evidence.team_logit_params import TeamLogitParams
from modeling.evidence.team_logit import team_logit_v1

P = TeamLogitParams(kappa=3.0, sigma_d=2.0, r_ctrl=2.0, l_prox=3.0, v_rel=6.0,
                    ball_conf_filled=0.6, contest_factor=0.5, rel_w_missing=0.5)

def make_master(n=1, **cols):
    """Valid-ball defaults (home 1 m from the ball, away 5 m); d_diff follows the distances."""
    base = {"d_near_home": 1.0, "d_near_away": 5.0, "n_home_r2": 1.0, "n_away_r2": 0.0,
            "speed_rel_home": 1.0, "speed_rel_away": 1.0, "ball_observed": 1, "ball_filled": 0,
            "since_last_obs": 0.0, "until_next_obs": 0.0}
    base.update(cols)
    df = pd.DataFrame({k: np.broadcast_to(np.asarray(v, dtype="float64"), (n,)).copy()
                       for k, v in base.items()})
    if "d_diff" not in cols:
        df["d_diff"] = df["d_near_home"] - df["d_near_away"]          
    else:
        df["d_diff"] = np.broadcast_to(np.asarray(cols["d_diff"], dtype="float64"), (n,)).copy()
    return df


def swap_teams(master):
    """Exchange the roles of home and away: swap every *_home/*_away column and flip d_diff."""
    def swap(name):
        if "home" in name:
            return name.replace("home", "away")
        if "away" in name:
            return name.replace("away", "home")
        return name
    out = master.rename(columns=swap)
    out["d_diff"] = -out["d_diff"]
    return out[master.columns]


def test_sign_convention_home_closer_is_positive():
    t = team_logit_v1(make_master(3, d_near_home=[1.0, 5.0, 3.0], d_near_away=[5.0, 1.0, 3.0]), P)
    assert t[0] > 0 and t[1] < 0 and t[2] == 0
    assert not np.signbit(t[2])          # a tie is +0.0, never -0.0 


def test_swapping_the_teams_flips_the_sign_exactly():
    rng = np.random.default_rng(1)
    n = 200
    master = make_master(n, d_near_home=rng.uniform(0, 15, n), d_near_away=rng.uniform(0, 15, n),
                         speed_rel_home=rng.uniform(0, 12, n), speed_rel_away=rng.uniform(0, 12, n),
                         n_home_r2=rng.integers(0, 3, n), n_away_r2=rng.integers(0, 3, n))
    t, t_swapped = team_logit_v1(master, P), team_logit_v1(swap_teams(master), P)
    assert np.array_equal(t_swapped, -t + 0.0)


def test_no_valid_ball_gives_zero_and_a_missing_d_diff_gives_zero():
    master = make_master(4, ball_observed=[0, 0, 1, 1], ball_filled=[0, 0, 0, 0],
                         d_near_home=[1.0, np.nan, np.nan, 1.0], d_near_away=[5.0, np.nan, 5.0, 5.0],
                         d_diff=[-4.0, np.nan, np.nan, -4.0])
    t = team_logit_v1(master, P)
    assert t[0] == 0 and t[1] == 0 and t[2] == 0 and t[3] > 0


def test_magnitude_grows_with_the_distance_difference_and_is_bounded():
    d = np.linspace(0.1, 8.0, 40)
    t = team_logit_v1(make_master(40, d_near_home=1.0, d_near_away=1.0 + d, n_home_r2=1, n_away_r2=0), P)
    assert np.all(np.diff(t) > 0) and t.max() <= P.kappa
    rng = np.random.default_rng(2)
    n = 500
    wild = make_master(n, d_near_home=rng.uniform(0, 60, n), d_near_away=rng.uniform(0, 60, n),
                       speed_rel_home=rng.uniform(0, 50, n), speed_rel_away=rng.uniform(0, 50, n))
    assert np.abs(team_logit_v1(wild, P)).max() <= P.kappa


def test_confidence_contest_and_relative_speed_scale_the_evidence():
    base = team_logit_v1(make_master(1), P)[0]
    filled = team_logit_v1(make_master(1, ball_observed=0, ball_filled=1), P)[0]
    contested = team_logit_v1(make_master(1, n_away_r2=1), P)[0]
    fast = team_logit_v1(make_master(1, speed_rel_home=20.0), P)[0]
    assert np.isclose(filled, 0.6 * base, rtol=1e-6) and np.isclose(contested, 0.5 * base, rtol=1e-6)
    assert 0 < fast < base


def test_output_contract_float32_finite_and_no_negative_zero():
    rng = np.random.default_rng(3)
    n = 300
    master = make_master(n, d_near_home=rng.uniform(0, 30, n), d_near_away=rng.uniform(0, 30, n),
                         ball_observed=rng.integers(0, 2, n))
    master.loc[::7, "d_diff"] = np.nan
    t = team_logit_v1(master, P)
    assert t.dtype == np.float32 and t.shape == (n,) and np.isfinite(t).all()
    assert not np.signbit(t[t == 0]).any()                       # identical bytes run after run


def test_no_dependence_on_neighbouring_frames_and_deterministic():
    rng = np.random.default_rng(4)
    n = 80
    master = make_master(n, d_near_home=rng.uniform(0, 20, n), d_near_away=rng.uniform(0, 20, n))
    whole = team_logit_v1(master, P)
    single = np.concatenate([team_logit_v1(master.iloc[[i]], P) for i in range(n)])
    assert np.array_equal(whole, single)                         # no window: no delay
    assert team_logit_v1(master, P).tobytes() == whole.tobytes()
