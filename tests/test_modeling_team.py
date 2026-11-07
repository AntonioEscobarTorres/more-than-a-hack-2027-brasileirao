"""F1-02 team_logit rules_v1. Synthetic masters only: nothing from the match."""
import numpy as np
import pytest

from modeling import contracts, variants
from modeling.evidence.team import team_logit_rules_v1
from preprocessing.make_mock_master import generate_mock_master
from tests.modeling_fixtures import fuzz_master, make_master, swap_teams, team_params

P = team_params()
KAPPA = P["kappa"]


def logit(master, params=P):
    return team_logit_rules_v1(master, params, holdout=None)


def test_registered_as_rules_v1():
    variants.load_kinds()
    assert variants.TEAM_KINDS["rules_v1"] is team_logit_rules_v1


def test_home_closer_is_positive_and_a_tie_is_plus_zero():
    t = logit(make_master(3, d_near_home=[1.0, 5.0, 3.0], d_near_away=[5.0, 1.0, 3.0]))
    assert t[0] > 0 and t[1] < 0 and t[2] == 0 and not np.signbit(t[2])


def test_swapping_teams_flips_the_sign_exactly():
    master = fuzz_master(300, seed=1)
    t = logit(master)
    assert np.array_equal(logit(swap_teams(master)), -t + np.float32(0.0))
    assert (t != 0).sum() > 100                            # the test is not vacuous


def test_nan_inputs_and_half_zero_give_zero():
    master = make_master(5, ball_observed=[0, 1, 1, 1, 1], d_diff=[-4.0, np.nan, -4.0, -4.0, -4.0],
                         half=[1, 1, 0, 1, 2])
    t = logit(master)
    assert list(t[:3]) == [0.0, 0.0, 0.0] and t[3] > 0 and t[4] > 0
    assert not np.signbit(t).any()


def test_magnitude_grows_with_the_distance_difference():
    d = np.linspace(0.1, 8.0, 40)
    for sign in (1.0, -1.0):
        home, away = (1.0, 1.0 + d) if sign > 0 else (1.0 + d, 1.0)
        # rest fixed: dmin = 1 m, same relative speed for both teams, no contest
        t = logit(make_master(40, d_near_home=home, d_near_away=away, n_home_r2=0, n_away_r2=0))
        assert np.all(np.diff(np.abs(t)) > 0) and np.all(np.sign(t) == sign)


def test_bounded_by_kappa():
    t = logit(fuzz_master(500, seed=2))
    assert np.abs(t).max() <= KAPPA


def test_confidence_contest_and_relative_speed_scale_the_evidence():
    base = logit(make_master(1))[0]
    filled = logit(make_master(1, ball_observed=0, ball_filled=1))[0]
    contested = logit(make_master(1, n_away_r2=1))[0]
    fast = logit(make_master(1, speed_rel_home=20.0))[0]
    assert np.isclose(filled, 0.6 * base, rtol=1e-6)
    assert np.isclose(contested, 0.5 * base, rtol=1e-6)
    assert 0 < fast < base


def test_a_change_at_one_frame_changes_only_that_frame():
    rng = np.random.default_rng(4)
    n = 80                                          # every frame has an observed ball
    master = make_master(n, d_near_home=rng.uniform(0, 15, n), d_near_away=rng.uniform(0, 15, n),
                         n_home_r2=rng.integers(0, 3, n), n_away_r2=rng.integers(0, 3, n),
                         speed_rel_home=rng.uniform(0, 12, n), speed_rel_away=rng.uniform(0, 12, n))
    t = logit(master)
    changed = master.copy()
    changed.loc[40, "d_diff"] = -changed.loc[40, "d_diff"]
    diff = np.flatnonzero(logit(changed) != t)
    assert list(diff) == [40]


def test_output_contract_and_determinism():
    master = fuzz_master(300, seed=3)
    t = logit(master)
    assert t.dtype == np.float32 and t.shape == (300,) and np.isfinite(t).all()
    assert logit(master).tobytes() == t.tobytes()


def test_missing_d_diff_raises():
    with pytest.raises(ValueError, match="missing columns"):
        logit(make_master(2).drop(columns=["d_diff"]))


def test_c2_contract_on_the_mock_master():
    master = generate_mock_master(seed=7)
    half = master["half"].to_numpy()
    t = logit(master)
    contracts.validate_logits(t, np.zeros(len(master), dtype=np.float32), len(master), half=half)
    on = logit(master, team_params(w_adv=0.3, w_spread=0.01, w_speed=0.5))
    contracts.validate_logits(on, np.zeros(len(master), dtype=np.float32), len(master), half=half)
