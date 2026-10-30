"""
Unit tests for the Smart Tagging reader. 
Uses ONLY SYNTHETIC data (no real match data).
"""
import numpy as np
import pandas as pd
import pytest

from preprocessing import prep_smart_tagging as st

BLOCKS = [(1, 11185, 90651), (2, 110808, 190197)]


def make_raw(rows):
    """Creates a DataFrame in kloppy format from rows: (start_s, end_s, code, team, type, half)."""
    return pd.DataFrame({
        "timestamp": pd.to_timedelta(np.array([r[0] for r in rows], dtype=float), unit="s"),
        "end_timestamp": pd.to_timedelta(np.array([r[1] for r in rows], dtype=float), unit="s"),
        "code": [r[2] for r in rows],
        "Team": [r[3] for r in rows],
        "Type": [r[4] for r in rows],
        "Half": [r[5] for r in rows],
    })


def test_conversion_to_frames_and_columns():
    ev = st.events_from_frame(make_raw([(10.0, 20.04, "SET PIECES", "N/A", "Throw-in", "1st Half")]))
    assert list(ev.columns) == st.EVENT_COLUMNS
    r = ev.iloc[0]
    assert (r.start_frame, r.end_frame) == (250, 501)


def test_text_normalization_and_unknown_team():
    raw = make_raw([
        (1, 2, " set pieces ", "N/A", "Kick Off (Start)", "1st Half"),
        (3, 4, "Progression", "ATLETIC CLUB", None, "2nd Half"),
        (5, 6, "BUILD UP", "Barcelona", np.nan, None),
        (7, 8, "GOALS", None, "", "1st Half"),
    ])
    ev = st.events_from_frame(raw).set_index("start_frame")
    
    assert ev.loc[25, "code"] == "SET PIECES" and ev.loc[25, "team"] == "U"
    assert ev.loc[25, "team_raw"] == "N/A" and ev.loc[25, "type"] == "kick off (start)"
    assert ev.loc[75, "code"] == "PROGRESSION" and ev.loc[75, "team"] == "A" and ev.loc[75, "half"] == 2
    assert ev.loc[75, "type"] == ""
    assert ev.loc[125, "team"] == "H" and ev.loc[125, "half"] == 0
    assert ev.loc[175, "team"] == "U" and ev.loc[175, "team_raw"] == ""


def test_deterministic_ordering():
    rows = [
        (5, 9, "BUILD UP", "Barcelona", "", "1st Half"), 
        (1, 2, "GOALS", "Atletic Club", "", "1st Half"),
        (5, 9, "BALL IN THE BOX", "Barcelona", "", "1st Half"), 
        (3, 4, "LONG BALL", "N/A", "", "1st Half")
    ]
    a = st.events_from_frame(make_raw(rows))
    b = st.events_from_frame(make_raw(rows[::-1]))
    
    pd.testing.assert_frame_equal(a, b)
    assert list(a["event_id"]) == [0, 1, 2, 3]


def test_missing_columns_and_end_before_start():
    with pytest.raises(ValueError, match="Missing columns"):
        st.events_from_frame(make_raw([(1, 2, "X", "N/A", "", "1st Half")]).drop(columns=["code"]))
        
    with pytest.raises(ValueError, match="events have end times before start times"):
        st.events_from_frame(make_raw([(5, 2, "X", "N/A", "", "1st Half")]))


def test_clip_within_valid_range():
    ev = st.events_from_frame(make_raw([(0, 99999, "X", "N/A", "", "1st Half")]))
    assert ev.loc[0, "end_frame"] == st.N_FRAMES  # Exclusive end can equal N_FRAMES


def _synthetic_published_frame():
    """Synthetic DataFrame that exactly reproduces the published counts in DATA.md."""
    rows = []
    team_pool = (["Barcelona"] * 314) + (["Atletic Club"] * 288) + (["N/A"] * 36)
    sp_types = [t for t, n in st.EXPECTED_SP_TYPES.items() for _ in range(n)]
    
    i = 0
    for code, n in st.EXPECTED_CODES.items():
        for _ in range(n):
            typ = sp_types.pop() if code == "SET PIECES" else ""
            rows.append((i * 10, i * 10 + 5, code, team_pool[i], typ, "1st Half"))
            i += 1
            
    return make_raw(rows)


def test_check_counts_passes_with_published_counts_and_detects_errors():
    ev = st.events_from_frame(_synthetic_published_frame())
    
    assert len(ev) == st.EXPECTED_TOTAL
    assert st.check_counts(ev) == []
    assert st.check_counts(ev.iloc[1:]) != []  # Missing one instance
    
    ev2 = ev.copy()
    ev2.loc[ev2["team_raw"] == "Atletic Club", "team_raw"] = "Athletic Club"  # different spelling
    msgs = " ".join(st.check_counts(ev2))

    assert "Athletic Club" in msgs and "Atletic Club" in msgs


def test_half_tag_consistency_with_blocks():
    raw = make_raw([
        (11185 / 25, 11300 / 25, "BUILD UP", "Barcelona", "", "1st Half"),
        (113742 / 25, 113800 / 25, "SET PIECES", "N/A", "Kick Off (Start)", "2nd Half")
    ])
    assert st.check_half_consistency(st.events_from_frame(raw), blocks=BLOCKS) == []
    
    raw.loc[1, "Half"] = "1st Half"  # Wrong tag for the 2nd half
    assert st.check_half_consistency(st.events_from_frame(raw), blocks=BLOCKS) != []


def test_starter_reader_and_raw_reader_produce_same_output():
    records = [
        {"code": "SET PIECES", "start_s": 10.0, "end_s": 20.04,
         "tags": {"Team": "N/A", "Type": "Throw-in", "Half": "1st Half"}},
        {"code": "build up", "start_s": 3.02, "end_s": 9.5,
         "tags": {"Team": "Atletic Club", "Half": "1st Half"}},
    ]
    a = st.events_from_records(records)
    
    raw = make_raw([
        (10.0, 20.04, "SET PIECES", "N/A", "Throw-in", "1st Half"),
        (3.02, 9.5, "build up", "Atletic Club", None, "1st Half")
    ])
    b = st.events_from_frame(raw)
    
    pd.testing.assert_frame_equal(a, b)
    assert st.compare_events(a, b)["start_frame"]["divergences"] == 0


def test_compare_events_detects_one_frame_divergence():
    a = st.events_from_frame(make_raw([(10.0, 20.0, "BUILD UP", "Barcelona", "", "1st Half")]))
    b = a.copy()
    b.loc[0, "end_frame"] += 1
    
    rep = st.compare_events(a, b)
    assert rep["end_frame"] == {"divergences": 1, "distinct_differences": [1]}
    assert rep["start_frame"]["divergences"] == 0


def test_exclusive_end_allows_slicing_without_plus_one():
    ev = st.events_from_frame(make_raw([(1.0, 1.2, "GOALS", "Barcelona", "", "1st Half")]))
    arr = np.zeros(100, int)
    
    arr[ev.loc[0, "start_frame"]:ev.loc[0, "end_frame"]] = 1
    assert arr.sum() == 5  # Frames 25 to 30 (exclusive end), sum is 5


def test_records_with_integer_float_before_non_integers():
    """Regression: pd.to_timedelta(list) parsed 1st value wrong (447.0 to 447 ns) in some pandas versions."""
    records = [
        {"code": "SET PIECES", "start_s": 447.0, "end_s": 452.0,
         "tags": {"Team": "Barcelona", "Type": "Kick Off (Start)", "Half": "1st Half"}},
        {"code": "PROGRESSION", "start_s": 500.02, "end_s": 510.5,
         "tags": {"Team": "Atletic Club", "Half": "1st Half"}},
    ]
    ev = st.events_from_records(records)
    
    assert list(ev["start_frame"]) == [11175, 12500]
    assert list(ev["end_frame"]) == [11300, 12762]  # Halves round to nearest even: 12762.5 -> 12762