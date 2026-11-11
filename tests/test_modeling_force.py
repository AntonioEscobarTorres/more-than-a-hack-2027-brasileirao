"""F3-03 force_d mask (C4). Synthetic data only: the mock master and hand-built frames."""
import numpy as np
import pandas as pd
import pytest

import games
from modeling.contracts import validate_force_d
from modeling.deadball.force import build_force_d
from preprocessing.make_mock_master import generate_mock_master
from preprocessing.params import KICKOFF_FALLBACK


@pytest.fixture(scope="module")
def mock_master() -> pd.DataFrame:
    return generate_mock_master(seed=42)


def small_master(half, pre_kickoff) -> pd.DataFrame:
    return pd.DataFrame({"half": np.asarray(half, dtype=np.int8),
                         "pre_kickoff": np.asarray(pre_kickoff)})


def test_mock_mask_equals_pre_kickoff(mock_master):
    mask = build_force_d(mock_master)
    pre_kickoff = mock_master["pre_kickoff"].to_numpy()
    assert mask.dtype == np.bool_
    assert mask.shape == (len(mock_master),)
    assert int(mask.sum()) == int(pre_kickoff.sum())
    assert np.array_equal(mask, pre_kickoff)
    validate_force_d(mask, mock_master["half"].to_numpy())


def test_mock_mask_is_empty_outside_the_halves_and_in_the_first_half(mock_master):
    mask = build_force_d(mock_master)
    half = mock_master["half"].to_numpy()
    assert not mask[half == 0].any()
    assert not mask[half == 1].any()


def test_mock_mask_covers_the_second_half_up_to_its_kickoff(mock_master):
    mask = build_force_d(mock_master)
    _, first, _ = games.half_video_blocks(games.GAME)[1]
    kickoff = KICKOFF_FALLBACK[2]
    assert int(mask.sum()) == kickoff - first
    assert np.array_equal(np.flatnonzero(mask), np.arange(first, kickoff))


def test_all_false_when_there_is_no_pre_kickoff():
    # The external IDSSE masters have pre_kickoff False on every frame.
    master = small_master([1, 1, 0, 2, 2], [False] * 5)
    mask = build_force_d(master)
    assert mask.dtype == np.bool_
    assert not mask.any()


def test_pandas_boolean_dtype_is_accepted():
    master = small_master([1, 2, 2], [False, True, False])
    master["pre_kickoff"] = master["pre_kickoff"].astype("boolean")
    assert build_force_d(master).tolist() == [False, True, False]


def test_result_is_a_copy_and_the_master_is_untouched():
    master = small_master([2, 2, 2], [True, True, False])
    before = master.copy(deep=True)
    mask = build_force_d(master)
    mask[:] = False
    pd.testing.assert_frame_equal(master, before)
    assert build_force_d(master).tolist() == [True, True, False]


def test_rejects_true_outside_the_halves():
    with pytest.raises(ValueError, match="half == 0"):
        build_force_d(small_master([0, 2, 2], [True, True, False]))


@pytest.mark.parametrize("values", [[0, 1, 0], [0.0, 1.0, np.nan], ["", "x", ""]])
def test_rejects_non_boolean_pre_kickoff(values):
    with pytest.raises(ValueError, match="pre_kickoff must be boolean"):
        build_force_d(small_master([2, 2, 2], values))


def test_rejects_missing_boolean_values():
    master = small_master([2, 2, 2], [False, True, False])
    master["pre_kickoff"] = pd.array([False, pd.NA, True], dtype="boolean")
    with pytest.raises(ValueError, match="missing values"):
        build_force_d(master)


@pytest.mark.parametrize("column", ["half", "pre_kickoff"])
def test_rejects_missing_columns(column):
    master = small_master([1, 2], [False, True]).drop(columns=[column])
    with pytest.raises(ValueError, match=column):
        build_force_d(master)
