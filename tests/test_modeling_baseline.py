"""Baseline labeller (task F5-02): steps (a)-(e) of modeling.baseline.baseline_labels.

Synthetic data only: small hand-built masters and preprocessing.make_mock_master. No test reads
MTAH_DATA_DIR.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import games
from modeling import contracts
from modeling.baseline import MAX_SMOOTH_PASSES, baseline_labels, half_blocks
from modeling.config import load_config
from modeling.paths import repo_root
from preprocessing.make_mock_master import generate_mock_master

N_GAME = games.GAMES[games.GAME]["n_video_frames"]
PARAMS = {"d_gap_frames": 100, "slow_speed_mps": 1.5, "min_spell_frames": 25}
NO_SMOOTH = {**PARAMS, "min_spell_frames": 0}


# -- helpers ------------------------------------------------------------------------------------

@pytest.fixture(scope="session")
def mock_master() -> pd.DataFrame:
    return generate_mock_master(seed=42)


@pytest.fixture(scope="session")
def baseline_params() -> dict:
    return load_config(repo_root() / "configs" / "baseline.yaml")["baseline"]


def _master(d_diff, half=None, **cols) -> pd.DataFrame:
    """Minimal master with the baseline columns; every dead-ball rule off unless overridden."""
    d_diff = np.asarray(d_diff, dtype=np.float32)
    n = d_diff.size
    half = np.ones(n, np.int8) if half is None else np.asarray(half, np.int8)
    base = {
        "half": half,
        "d_diff": d_diff,
        "out_of_pitch": np.zeros(n, np.float32),
        "pre_kickoff": np.zeros(n, bool),
        "gap_length": np.zeros(n, np.float32),
        "mean_speed_home": np.full(n, 3.0, np.float32),
        "mean_speed_away": np.full(n, 3.0, np.float32),
    }
    for key, value in cols.items():
        base[key] = np.asarray(value, dtype=base[key].dtype)
    return pd.DataFrame(base)


def _runs(spec) -> np.ndarray:
    """``[("H", 30), ("A", 5)]`` -> labels array."""
    return np.concatenate([np.full(n, lab, dtype="<U1") for lab, n in spec])


def _master_for_labels(spec, **kw) -> pd.DataFrame:
    """A master whose steps (a)-(c) give exactly ``spec`` (D through out_of_pitch)."""
    labels = _runs(spec)
    d_diff = np.where(labels == "H", -1.0, 1.0)
    out_of_pitch = (labels == "D").astype(np.float32)
    return _master(d_diff, out_of_pitch=out_of_pitch, **kw)


def _segments(labels):
    starts = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
    lengths = np.diff(np.r_[starts, labels.size])
    return labels[starts].tolist(), lengths.tolist()


def _short_spells_with_team_neighbour(labels, half, min_spell):
    """Short H/A segments inside a half that still have an H/A neighbour (must be none)."""
    bad = []
    for _, start, stop in half_blocks(half):
        labs, lens = _segments(labels[start:stop])
        for i, (lab, length) in enumerate(zip(labs, lens)):
            if lab == "D" or length >= min_spell:
                continue
            prev = labs[i - 1] if i > 0 else None
            nxt = labs[i + 1] if i + 1 < len(labs) else None
            if prev in ("H", "A") or nxt in ("H", "A"):
                bad.append((start, i, lab, length))
    return bad


# -- mock master ---------------------------------------------------------------------------------

def test_mock_labels_shape_values_and_x(mock_master, baseline_params):
    labels = baseline_labels(mock_master, baseline_params)
    assert labels.shape == (N_GAME,)
    assert labels.dtype == np.dtype("<U1")
    assert set(np.unique(labels)) <= {"H", "A", "D", "X"}
    half = mock_master["half"].to_numpy()
    np.testing.assert_array_equal(labels == "X", half == 0)
    contracts.validate_labels(labels, half)


def test_mock_has_no_short_spell(mock_master, baseline_params):
    labels = baseline_labels(mock_master, baseline_params)
    half = mock_master["half"].to_numpy()
    min_spell = baseline_params["min_spell_frames"]
    assert min_spell > 0
    for _, start, stop in half_blocks(half):
        assert stop - start >= min_spell
        labs, lens = _segments(labels[start:stop])
        short = [(lab, n) for lab, n in zip(labs, lens) if lab != "D" and n < min_spell]
        assert short == []


def test_mock_pre_kickoff_is_dead(mock_master, baseline_params):
    labels = baseline_labels(mock_master, baseline_params)
    pre = mock_master["pre_kickoff"].to_numpy()
    assert pre.any()
    assert (labels[pre] == "D").all()


def test_mock_two_runs_identical(mock_master, baseline_params):
    first = baseline_labels(mock_master, baseline_params)
    second = baseline_labels(mock_master.copy(), dict(baseline_params))
    np.testing.assert_array_equal(first, second)


def test_mock_input_is_not_modified(mock_master, baseline_params):
    before = mock_master.copy()
    baseline_labels(mock_master, baseline_params)
    pd.testing.assert_frame_equal(mock_master, before)


def test_mock_label_mix_is_plausible(mock_master, baseline_params):
    labels = baseline_labels(mock_master, baseline_params)
    graded = labels[mock_master["half"].to_numpy() != 0]
    assert {"H", "A", "D"} <= set(np.unique(graded))


# -- (a) and (b): team from d_diff, carried forward ----------------------------------------------

def test_sign_of_d_diff_gives_team():
    labels = baseline_labels(_master([-2.0, -0.1, 0.1, 3.0]), NO_SMOOTH)
    assert labels.tolist() == ["H", "H", "A", "A"]


def test_missing_or_zero_d_diff_inherits_previous_team():
    labels = baseline_labels(_master([-1.0, np.nan, 0.0, 1.0, np.nan, np.inf, 0.0]), NO_SMOOTH)
    assert labels.tolist() == ["H", "H", "H", "A", "A", "A", "A"]


def test_leading_frames_inherit_first_team():
    labels = baseline_labels(_master([np.nan, 0.0, np.nan, 2.0, -1.0]), NO_SMOOTH)
    assert labels.tolist() == ["A", "A", "A", "A", "H"]


def test_half_without_any_team_is_home():
    labels = baseline_labels(_master([np.nan, 0.0, np.nan]), NO_SMOOTH)
    assert labels.tolist() == ["H", "H", "H"]


def test_team_never_crosses_halftime():
    d_diff = [1.0, 1.0, -1.0, -1.0, np.nan, np.nan, 1.0]
    half = [1, 1, 0, 0, 2, 2, 2]
    labels = baseline_labels(_master(d_diff, half=half), NO_SMOOTH)
    # half 2 starts with NaN: it takes the first team of half 2 (A), not the last of half 1
    assert labels.tolist() == ["A", "A", "X", "X", "A", "A", "A"]


def test_half_two_without_team_is_home_even_after_away_half():
    labels = baseline_labels(_master([1.0, 1.0, np.nan, np.nan], half=[1, 1, 2, 2]), NO_SMOOTH)
    assert labels.tolist() == ["A", "A", "H", "H"]


# -- (c): dead-ball rules ------------------------------------------------------------------------

def test_out_of_pitch_is_dead_and_nan_is_not():
    master = _master([-1.0] * 4, out_of_pitch=[1.0, np.nan, 0.0, 1.0])
    assert baseline_labels(master, NO_SMOOTH).tolist() == ["D", "H", "H", "D"]


def test_pre_kickoff_is_dead():
    master = _master([1.0] * 4, pre_kickoff=[True, True, False, False])
    assert baseline_labels(master, NO_SMOOTH).tolist() == ["D", "D", "A", "A"]


def test_long_gap_with_slow_players_is_dead():
    gap = [100, 99, 100, 100, 100, np.nan, 500]
    home = [1.0, 1.0, 1.5, 1.0, np.nan, 1.0, 1.0]
    away = [1.0, 1.0, 1.0, 1.49, 1.0, 1.0, 1.0]
    master = _master([-1.0] * 7, gap_length=gap, mean_speed_home=home, mean_speed_away=away)
    labels = baseline_labels(master, NO_SMOOTH)
    # >= d_gap_frames and both < slow_speed_mps; NaN anywhere -> not dead
    assert labels.tolist() == ["D", "H", "H", "D", "H", "H", "D"]


def test_speed_threshold_compares_in_float64():
    # float32(1.3) < 1.3 in float64, so the frame is slow under the config threshold 1.3
    params = {**NO_SMOOTH, "slow_speed_mps": 1.3}
    master = _master([-1.0], gap_length=[100], mean_speed_home=[1.3], mean_speed_away=[1.3])
    assert baseline_labels(master, params).tolist() == ["D"]


def test_first_half_starts_with_possession():
    # pre_kickoff only applies before the 2nd-half kickoff in the real master; the 1st half
    # starts at the kickoff, so its first frame is a team, not D
    master = _master([-1.0] * 4 + [1.0] * 4, half=[1] * 4 + [2] * 4,
                     pre_kickoff=[False] * 4 + [True, True, False, False])
    assert baseline_labels(master, NO_SMOOTH).tolist() == ["H"] * 4 + ["D", "D", "A", "A"]


# -- (d): short spells ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec, expected", [
    # short spell between two spells of the other team -> previous team
    ([("H", 30), ("A", 5), ("H", 30)], [("H", 65)]),
    # previous neighbour is D -> next team
    ([("D", 10), ("A", 5), ("H", 30)], [("D", 10), ("H", 35)]),
    # no previous neighbour (start of the half) -> next team
    ([("A", 5), ("H", 30)], [("H", 35)]),
    # end of the half -> previous team
    ([("H", 30), ("A", 5)], [("H", 35)]),
    # previous team wins over the next one, even if the next is D
    ([("A", 30), ("H", 5), ("D", 30)], [("A", 35), ("D", 30)]),
    # chain of short spells collapses left to right
    ([("H", 30), ("A", 5), ("H", 5), ("A", 30)], [("H", 40), ("A", 30)]),
    # D is never changed, even when short
    ([("H", 30), ("D", 3), ("A", 30)], [("H", 30), ("D", 3), ("A", 30)]),
    # a short spell between two D has no team neighbour and is kept
    ([("D", 30), ("H", 5), ("D", 30)], [("D", 30), ("H", 5), ("D", 30)]),
    # short spell -> next team, merged with it; the result is still between D and kept
    ([("D", 30), ("H", 3), ("A", 4), ("D", 30)], [("D", 30), ("A", 7), ("D", 30)]),
    # long spells are untouched
    ([("H", 25), ("A", 25), ("D", 25)], [("H", 25), ("A", 25), ("D", 25)]),
])
def test_short_spell_rules(spec, expected):
    labels = baseline_labels(_master_for_labels(spec), PARAMS)
    np.testing.assert_array_equal(labels, _runs(expected))


def test_min_spell_zero_turns_smoothing_off():
    spec = [("H", 30), ("A", 1), ("H", 1), ("D", 1), ("A", 30)]
    labels = baseline_labels(_master_for_labels(spec), NO_SMOOTH)
    np.testing.assert_array_equal(labels, _runs(spec))


def test_short_spell_never_borrows_from_other_half():
    spec = [("A", 30), ("H", 5), ("A", 30)]
    labels_in = _runs(spec)
    half = np.r_[np.full(30, 1), np.full(5 + 30, 2)]
    # half 2 starts with a short H spell: no previous neighbour inside the half -> next (A)
    master = _master(np.where(labels_in == "H", -1.0, 1.0), half=half)
    labels = baseline_labels(master, PARAMS)
    assert (labels[30:] == "A").all()


def test_spell_shorter_than_its_half_is_kept():
    labels = baseline_labels(_master([-1.0] * 10), PARAMS)
    assert labels.tolist() == ["H"] * 10


def test_smoothing_converges_on_alternating_noise():
    rng = np.random.default_rng(0)
    d_diff = rng.choice([-1.0, 1.0], size=5000)
    out = (rng.random(5000) < 0.02).astype(np.float32)
    master = _master(d_diff, out_of_pitch=out)
    labels = baseline_labels(master, PARAMS)
    dead = out == 1
    assert (labels[dead] == "D").all()                 # D never altered
    assert ((labels == "D") == dead).all()             # and never created by smoothing
    assert _short_spells_with_team_neighbour(labels, master["half"].to_numpy(), 25) == []


def test_max_passes_is_twenty():
    assert MAX_SMOOTH_PASSES == 20


# -- (e) and errors ------------------------------------------------------------------------------

def test_gap_frames_are_x_whatever_the_features():
    master = _master([-1.0, -1.0, -1.0], half=[1, 0, 2], out_of_pitch=[0.0, 1.0, 0.0],
                     pre_kickoff=[False, True, False])
    assert baseline_labels(master, NO_SMOOTH).tolist() == ["H", "X", "H"]


def test_missing_column_or_param_is_an_error():
    master = _master([-1.0, 1.0])
    with pytest.raises(ValueError, match="d_diff"):
        baseline_labels(master.drop(columns=["d_diff"]), PARAMS)
    with pytest.raises(ValueError, match="min_spell_frames"):
        baseline_labels(master, {"d_gap_frames": 100, "slow_speed_mps": 1.5})


def test_half_blocks():
    half = np.array([0, 1, 1, 0, 0, 2, 2, 2, 0], np.int8)
    assert half_blocks(half) == [(1, 1, 3), (2, 5, 8)]
    assert half_blocks(np.array([], np.int8)) == []
    assert half_blocks(np.array([1, 1, 2], np.int8)) == [(1, 0, 2), (2, 2, 3)]
