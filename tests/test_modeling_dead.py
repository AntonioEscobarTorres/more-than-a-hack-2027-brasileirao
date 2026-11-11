"""F3-02 dead_logit rules_v1. Synthetic masters only: nothing from the match."""
import numpy as np
import pandas as pd
import pytest

from modeling import config, contracts, variants
from modeling.deadball.dead import (KINEMATIC_TERMS, RULES_V1_TERMS, SPEED_HIGH_RAMP,
                                    SPEED_LOW_RAMP, STILL_RAMP, TAG_TERMS, dead_logit_rules_v1,
                                    dead_logit_terms, dead_tag_prior)
from modeling.deadball.features import GOAL_KICKOFF_TYPE
from preprocessing.make_mock_master import generate_mock_master
from tests.modeling_fixtures import dead_params, deadball_master, with_gap

P = dead_params()
W = P["weights"]
RAW = dead_params(smooth_frames=1)          # no smoothing: the features are the per-frame values
MIDFIELD = (50.0, 34.0, 0.0, 0.0)


def logit(master, params=P):
    return dead_logit_rules_v1(master, params, holdout=None)


def neutral(n=30, **cols) -> pd.DataFrame:
    """Ball observed in midfield at 2 m/s, players at 2 m/s, a fifth of them still, no tags.

    Every term of the logit is 0. ``cols`` overrides any column (and ``half``).
    """
    base = {"ball_speed": 2.0, "mean_speed_home": 2.0, "mean_speed_away": 2.0}
    base.update(cols)
    return deadball_master(np.full(n, 50.0), 34.0, **base)


@pytest.fixture(scope="module")
def mock_master() -> pd.DataFrame:
    return generate_mock_master(seed=42)


# -- registration and term names ----------------------------------------------------------------
def test_registered_as_rules_v1():
    variants.load_kinds()
    assert variants.DEAD_KINDS["rules_v1"] is dead_logit_rules_v1


def test_terms_are_exactly_the_weights_of_the_config_schema():
    assert RULES_V1_TERMS == KINEMATIC_TERMS + TAG_TERMS
    assert RULES_V1_TERMS == config.DEAD_WEIGHT_KEYS
    assert tuple(dead_logit_terms(neutral(), P)) == RULES_V1_TERMS


def test_ramp_shapes_are_the_ones_of_appendix_a3():
    assert STILL_RAMP == (0.3, 0.4)
    assert SPEED_HIGH_RAMP == (2.5, 2.0)
    assert SPEED_LOW_RAMP == (1.5, 1.0)


# -- no evidence --------------------------------------------------------------------------------
def test_neutral_master_is_exactly_zero():
    master = neutral()
    d = logit(master)
    assert (d == 0).all() and not np.signbit(d).any()
    for name, term in dead_logit_terms(master, P).items():
        assert (term == 0).all() and not np.signbit(term).any(), name


def test_every_feature_nan_gives_exactly_zero():
    # No ball at all (shorter than a long gap), no tracked players, no tags.
    master = deadball_master(np.full(10, np.nan), np.nan, n_still_home=np.nan,
                             n_still_away=np.nan, n_tracked_home=np.nan, n_tracked_away=np.nan,
                             mean_speed_home=np.nan, mean_speed_away=np.nan)
    assert master["out_of_pitch"].isna().all() and master["ball_speed"].isna().all()
    d = logit(master)
    assert np.isfinite(d).all() and (d == 0).all() and not np.signbit(d).any()


@pytest.mark.parametrize("term, on, nan", [
    ("out_of_pitch", {"out_of_pitch": 1.0}, {"out_of_pitch": np.nan}),
    ("still", {"n_still_home": 10.0, "n_still_away": 10.0}, {"n_still_home": np.nan}),
    ("still", {"n_still_home": 10.0, "n_still_away": 10.0}, {"n_tracked_away": np.nan}),
    ("still", {"n_still_home": 10.0, "n_still_away": 10.0},
     {"n_tracked_home": 0.0, "n_tracked_away": 0.0}),
    ("speed_high", {"mean_speed_home": 6.0, "mean_speed_away": 6.0},
     {"mean_speed_home": np.nan, "mean_speed_away": np.nan}),
    ("speed_low", {"mean_speed_home": 0.2, "mean_speed_away": 0.2},
     {"mean_speed_home": np.nan, "mean_speed_away": np.nan}),
    ("ball_rest", {"ball_speed": 0.1}, {"ball_speed": np.nan}),
    ("fast_in_pitch", {"ball_speed": 10.0}, {"ball_speed": np.nan}),
    ("fast_in_pitch", {"ball_speed": 10.0}, {"ball_speed": 10.0, "out_of_pitch": np.nan}),
    ("tag_sp_prox", {"frames_to_next_sp": 0.0}, {"frames_to_next_sp": np.nan}),
    ("tag_sp_after", {"frames_since_sp_end": 0.0}, {"frames_since_sp_end": np.nan}),
])
def test_a_term_with_a_nan_feature_is_zero(term, on, nan):
    assert (logit(neutral(**on)) == np.float32(W[term])).all()      # the term does fire...
    d = logit(neutral(**{**on, **nan}))
    assert (d == 0).all() and not np.signbit(d).any()               # ...and NaN turns it off


def test_set_piece_flag_accepts_missing_values():
    master = neutral(3)
    master["is_set_piece"] = pd.array([True, pd.NA, False], dtype="boolean")
    assert logit(master).tolist() == [W["tag_set_piece"], 0.0, 0.0]


# -- one term at a time -------------------------------------------------------------------------
@pytest.mark.parametrize("term, cols", [
    ("out_of_pitch", {"out_of_pitch": 1.0}),
    ("ball_rest", {"ball_speed": 0.1}),
    ("fast_in_pitch", {"ball_speed": 10.0}),
    ("tag_set_piece", {"is_set_piece": True}),
])
def test_flag_term_alone_equals_its_weight(term, cols):
    master = neutral(**cols)
    assert (logit(master) == np.float32(W[term])).all()
    terms = dead_logit_terms(master, P)
    assert (terms[term] == W[term]).all()
    assert all((terms[other] == 0).all() for other in RULES_V1_TERMS if other != term)


def test_out_of_pitch_alone_is_strongly_positive():
    d = logit(neutral(out_of_pitch=1.0))
    assert (d > 0).all()
    assert (d == np.float32(max(W.values()))).all()                 # the strongest single term
    # Still positive against the strongest live evidence that can coexist with it.
    running = logit(neutral(out_of_pitch=1.0, mean_speed_home=6.0, mean_speed_away=6.0))
    assert (running == np.float32(W["out_of_pitch"] + W["speed_high"])).all() and (running > 0).all()


def test_fast_ball_in_pitch_is_negative():
    d = logit(neutral(ball_speed=10.0))
    assert (d < 0).all() and (d == np.float32(W["fast_in_pitch"])).all()
    # A fast ball outside the pitch is not evidence of live play.
    assert (logit(neutral(ball_speed=10.0, out_of_pitch=1.0)) == np.float32(W["out_of_pitch"])).all()


@pytest.mark.parametrize("term, before, gap", [
    ("gap_exit", (50.0, 1.0, 0.0, -4.0), None),       # leaves over the touchline
    ("gap_air", (40.0, 34.0, 20.0, 0.0), 40),         # long pass at 20 m/s
    ("gap_long", MIDFIELD, None),                     # undetermined
])
def test_gap_term_alone_equals_its_weight_on_the_gap_frames(term, before, gap):
    master = with_gap(before, MIDFIELD, gap=gap, mean_speed_home=2.0, mean_speed_away=2.0)
    in_gap = master["ball_observed"].to_numpy() == 0
    d = logit(master)
    assert in_gap.sum() > 0 and (d[in_gap] == np.float32(W[term])).all()
    terms = dead_logit_terms(master, P)
    assert (terms[term][~in_gap] == 0).all()
    for other in ("gap_exit", "gap_air", "gap_long"):
        if other != term:
            assert (terms[other] == 0).all()


def test_ball_out_is_positive_and_high_ball_is_negative():
    # The distinction that matters most: both are long gaps, only one is a dead ball.
    out = with_gap((50.0, 1.0, 0.0, -20.0), MIDFIELD, gap=40, mean_speed_home=2.0,
                   mean_speed_away=2.0)
    high = with_gap((50.0, 34.0, 0.0, -20.0), MIDFIELD, gap=40, mean_speed_home=2.0,
                    mean_speed_away=2.0)
    in_gap = out["ball_observed"].to_numpy() == 0
    assert (logit(out)[in_gap] > 0).all()
    assert (logit(high)[in_gap] < 0).all()


# -- monotonicity -------------------------------------------------------------------------------
def test_still_fraction_ramp_is_monotone():
    k = np.arange(21, dtype="float64")                   # still fraction k / 20
    d = logit(neutral(21, n_still_home=k, n_still_away=0.0), RAW)
    expected = W["still"] * np.clip((k / 20.0 - 0.3) / 0.4, 0.0, 1.0)
    np.testing.assert_allclose(d, expected, atol=1e-5)
    assert (np.diff(d) >= 0).all()
    assert d[0] == 0 and d[-1] == np.float32(W["still"])
    assert d[10] == np.float32(0.5 * W["still"])         # half-way up the ramp at 0.5


def test_mean_speed_ramps_are_monotone():
    v = np.arange(0.0, 6.01, 0.25)
    d = logit(neutral(len(v), mean_speed_home=v, mean_speed_away=v), RAW)
    expected = (W["speed_low"] * np.clip((1.5 - v) / 1.0, 0.0, 1.0)
                + W["speed_high"] * np.clip((v - 2.5) / 2.0, 0.0, 1.0))
    np.testing.assert_allclose(d, expected, atol=1e-5)
    assert (np.diff(d) <= 0).all()                       # slower players -> more dead
    assert d[0] == np.float32(W["speed_low"]) and d[-1] == np.float32(W["speed_high"])
    assert (d[(v >= 1.5) & (v <= 2.5)] == 0).all()       # dead zone between the two ramps
    terms = dead_logit_terms(neutral(len(v), mean_speed_home=v, mean_speed_away=v), RAW)
    assert (terms["speed_low"] >= 0).all() and (terms["speed_high"] <= 0).all()
    assert ((terms["speed_low"] == 0) | (terms["speed_high"] == 0)).all()   # never both


def test_set_piece_proximity_terms_are_monotone():
    to_next = np.array([0.0, 50.0, 125.0, 250.0, 500.0, 2000.0])
    d = logit(neutral(len(to_next), frames_to_next_sp=to_next))
    np.testing.assert_allclose(d, W["tag_sp_prox"] * np.exp(-to_next / P["sp_prox_tau_frames"]),
                               rtol=1e-6)
    assert (np.diff(d) < 0).all() and (d > 0).all()      # closer to the next set piece -> more D

    since = np.array([0.0, 25.0, 75.0, 150.0, 1000.0])
    d = logit(neutral(len(since), frames_since_sp_end=since))
    np.testing.assert_allclose(d, W["tag_sp_after"] * np.exp(-since / P["sp_after_tau_frames"]),
                               rtol=1e-6)
    assert (np.diff(d) > 0).all() and (d < 0).all()      # just after a restart -> more alive


def test_goal_window_adds_exactly_its_weight():
    cols = {"frames_to_next_sp": 100.0}
    goal = neutral(next_sp_type=GOAL_KICKOFF_TYPE, **cols)
    other = neutral(next_sp_type="goal kick", **cols)
    t_goal, t_other = dead_logit_terms(goal, P), dead_logit_terms(other, P)
    assert (t_goal["tag_goal_window"] == W["tag_goal_window"]).all()
    assert (t_other["tag_goal_window"] == 0).all()
    assert all(np.array_equal(t_goal[n], t_other[n]) for n in RULES_V1_TERMS
               if n != "tag_goal_window")
    np.testing.assert_allclose(logit(goal) - logit(other), W["tag_goal_window"], rtol=1e-6)


def test_every_term_has_the_sign_of_its_weight(mock_master):
    terms = dead_logit_terms(mock_master, P)
    fired = 0
    for name in RULES_V1_TERMS:
        assert (np.sign(W[name]) * terms[name] >= 0).all(), name
        assert not np.signbit(terms[name][terms[name] == 0]).any(), name
        fired += int((terms[name] != 0).any())
    assert fired >= 8                                    # the test is not vacuous


def test_turning_a_term_on_never_moves_the_logit_against_its_weight(mock_master):
    # Clipping included: with one weight switched off, the logit moves only in the direction
    # of that weight, on every frame.
    full = logit(mock_master)
    for name in RULES_V1_TERMS:
        without = logit(mock_master, dead_params(weights={name: 0.0}))
        assert (np.sign(W[name]) * (full - without) >= 0).all(), name


# -- half == 0 and limits -----------------------------------------------------------------------
def test_zero_outside_the_halves():
    half = np.r_[np.full(10, 1), np.full(10, 0), np.full(10, 2)]
    master = neutral(30, half=half, out_of_pitch=1.0, is_set_piece=True, frames_to_next_sp=0.0,
                     frames_since_sp_end=0.0, next_sp_type=GOAL_KICKOFF_TYPE)
    d, prior = logit(master), dead_tag_prior(master, P)
    outside = half == 0
    assert (d[outside] == 0).all() and not np.signbit(d[outside]).any()
    assert (prior[outside] == 0).all()
    assert (d[~outside] > 0).all() and (prior[~outside] > 0).all()
    for name, term in dead_logit_terms(master, P).items():
        assert (term[outside] == 0).all(), name


def test_clipped_at_the_upper_limit():
    # Ball at rest outside the pitch, everyone standing, kick-off after a goal about to be taken.
    master = neutral(out_of_pitch=1.0, ball_speed=0.1, n_still_home=10.0, n_still_away=10.0,
                     mean_speed_home=0.2, mean_speed_away=0.2, is_set_piece=True,
                     frames_to_next_sp=0.0, next_sp_type=GOAL_KICKOFF_TYPE)
    total = sum(dead_logit_terms(master, P).values())
    expected = sum(W[k] for k in ("out_of_pitch", "still", "speed_low", "ball_rest",
                                  "tag_set_piece", "tag_sp_prox", "tag_goal_window"))
    np.testing.assert_allclose(total, expected)
    assert expected > contracts.DEAD_LOGIT_MAX
    assert (logit(master) == np.float32(contracts.DEAD_LOGIT_MAX)).all()


def test_clipped_at_both_limits_with_large_weights():
    up = logit(neutral(out_of_pitch=1.0), dead_params(weights={"out_of_pitch": 50.0}))
    down = logit(neutral(ball_speed=10.0), dead_params(weights={"fast_in_pitch": -50.0}))
    assert (up == np.float32(contracts.DEAD_LOGIT_MAX)).all()
    assert (down == np.float32(-contracts.DEAD_LOGIT_MAX)).all()


def test_c2_contract_on_the_mock_master(mock_master):
    half = mock_master["half"].to_numpy()
    zeros = np.zeros(len(mock_master), dtype=np.float32)
    d = logit(mock_master)
    contracts.validate_logits(zeros, d, len(mock_master), half=half)
    assert (d > 0).any() and (d < 0).any()
    big = dead_params(weights={name: 40.0 * np.sign(W[name]) for name in RULES_V1_TERMS})
    contracts.validate_logits(zeros, logit(mock_master, big), len(mock_master), half=half)


# -- weights come from the config ---------------------------------------------------------------
def test_weights_come_from_params(mock_master):
    assert (logit(neutral(out_of_pitch=1.0), dead_params(weights={"out_of_pitch": 7.5}))
            == np.float32(7.5)).all()
    off = dead_params(weights={name: 0.0 for name in RULES_V1_TERMS})
    d = logit(mock_master, off)
    assert (d == 0).all() and not np.signbit(d).any()


# -- dead_tag_prior -----------------------------------------------------------------------------
def test_tag_prior_is_float64_and_zero_without_tags(mock_master):
    master = neutral(out_of_pitch=1.0, ball_speed=0.1)   # kinematic terms on, no tags
    prior = dead_tag_prior(master, P)
    assert prior.dtype == np.float64 and prior.shape == (len(master),)
    assert (prior == 0).all() and not np.signbit(prior).any()

    # An external (IDSSE) master has an empty tagging table.
    no_tags = mock_master.copy()
    no_tags["is_set_piece"] = False
    no_tags["frames_to_next_sp"] = np.float32(np.nan)
    no_tags["frames_since_sp_end"] = np.float32(np.nan)
    no_tags["next_sp_type"] = pd.array([""] * len(no_tags), dtype="string")
    assert (dead_tag_prior(no_tags, P) == 0).all()


def test_tag_prior_value_and_logit_with_only_tags():
    master = neutral(is_set_piece=True, frames_to_next_sp=100.0, frames_since_sp_end=30.0,
                     next_sp_type=GOAL_KICKOFF_TYPE)
    expected = (W["tag_set_piece"] + W["tag_sp_prox"] * np.exp(-100.0 / P["sp_prox_tau_frames"])
                + W["tag_sp_after"] * np.exp(-30.0 / P["sp_after_tau_frames"])
                + W["tag_goal_window"])
    prior = dead_tag_prior(master, P)
    np.testing.assert_allclose(prior, expected, rtol=1e-6)
    assert np.array_equal(logit(master), prior.astype(np.float32))


def test_tag_prior_ignores_the_kinematic_columns():
    tags = {"is_set_piece": True, "frames_to_next_sp": 100.0, "frames_since_sp_end": 30.0}
    quiet = dead_tag_prior(neutral(**tags), P)
    busy = dead_tag_prior(neutral(out_of_pitch=1.0, ball_speed=0.1, mean_speed_home=0.2,
                                  mean_speed_away=0.2, n_still_home=10.0, **tags), P)
    assert np.array_equal(quiet, busy)


def test_tag_prior_is_not_clipped():
    big = dead_params(weights={"tag_set_piece": 30.0})
    master = neutral(is_set_piece=True)
    assert (dead_tag_prior(master, big) == 30.0).all()
    assert (logit(master, big) == np.float32(contracts.DEAD_LOGIT_MAX)).all()


def test_logit_is_the_clipped_sum_of_kinematic_terms_and_tag_prior(mock_master):
    terms = dead_logit_terms(mock_master, P)
    prior = dead_tag_prior(mock_master, P)
    tag_sum = np.zeros(len(mock_master))
    for name in TAG_TERMS:
        tag_sum = tag_sum + terms[name]
    assert np.array_equal(prior, tag_sum)
    kinematic = np.zeros(len(mock_master))
    for name in KINEMATIC_TERMS:
        kinematic = kinematic + terms[name]
    limit = contracts.DEAD_LOGIT_MAX
    expected = (np.clip(kinematic + prior, -limit, limit) + 0.0).astype(np.float32)
    assert logit(mock_master).tobytes() == expected.tobytes()


# -- output contract ----------------------------------------------------------------------------
def test_output_contract_determinism_and_holdout(mock_master):
    before = mock_master.copy(deep=True)
    d = logit(mock_master)
    assert d.dtype == np.float32 and d.shape == (len(mock_master),) and np.isfinite(d).all()
    assert not np.signbit(d[d == 0]).any()
    assert logit(mock_master).tobytes() == d.tobytes()
    assert dead_logit_rules_v1(mock_master, P, holdout="J03WMX").tobytes() == d.tobytes()
    pd.testing.assert_frame_equal(mock_master, before)


def test_a_change_in_a_frame_wise_column_changes_only_that_frame():
    master = neutral(60)
    base = logit(master)
    for column, value in (("out_of_pitch", 1.0), ("is_set_piece", True),
                          ("frames_to_next_sp", 0.0), ("frames_since_sp_end", 0.0)):
        changed = master.copy()
        changed.loc[30, column] = value
        assert list(np.flatnonzero(logit(changed) != base)) == [30], column


@pytest.mark.parametrize("column", ["half", "out_of_pitch", "is_set_piece", "n_still_home",
                                    "frames_to_next_sp"])
def test_missing_column_raises(column):
    with pytest.raises(ValueError, match=column):
        logit(neutral().drop(columns=[column]))


def test_empty_master():
    d = logit(neutral(3).iloc[:0])
    assert d.dtype == np.float32 and d.shape == (0,)
    assert dead_tag_prior(neutral(3).iloc[:0], P).shape == (0,)
