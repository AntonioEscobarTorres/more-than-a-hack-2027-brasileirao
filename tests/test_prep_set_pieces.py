import numpy as np
import pandas as pd
from preprocessing.prep_set_pieces import process_set_pieces

BLOCKS = [(1, 10, 99), (2, 150, 249)]
N = 300

def _events(rows):
    """rows: (start_frame, end_frame, type) -> events DataFrame shaped like the C1 output."""
    return pd.DataFrame([
        {"start_frame": a, "end_frame": b, "code": "SET PIECES", "team": "H", "type": t, "half": 1}
        for a, b, t in rows
    ])

def test_nested_set_pieces_use_the_latest_end_not_the_last_start():
    out = process_set_pieces(_events([(20, 90, "free kick"), (30, 40, "throw-in")]), BLOCKS, n_frames=N)
    # the free kick ends after the nested throw-in
    assert out.loc[95, "frames_since_sp_end"] == 6
    assert out.loc[95, "prev_sp_type"] == "free kick"
    # while the free kick is still running, the most recent ended set piece is the throw-in
    assert out.loc[60, "frames_since_sp_end"] == 21
    assert out.loc[60, "prev_sp_type"] == "throw-in"

def test_input_order_does_not_change_the_result():
    rows = [(60, 70, "corner kick"), (20, 30, "throw-in"), (160, 170, "goal kick")]
    a = process_set_pieces(_events(rows), BLOCKS, n_frames=N)
    b = process_set_pieces(_events(rows[::-1]), BLOCKS, n_frames=N)
    pd.testing.assert_frame_equal(a, b)

def test_distances_never_cross_halftime():
    rows = [(20, 30, "throw-in"), (60, 70, "corner kick"), (160, 170, "goal kick")]
    out = process_set_pieces(_events(rows), BLOCKS, n_frames=N)
    assert out.loc[150, "frames_to_next_sp"] == 10            
    assert np.isnan(out.loc[150, "frames_since_sp_end"])      
    assert np.isnan(out.loc[120, "frames_to_next_sp"])       
    assert np.isnan(out.loc[99, "frames_to_next_sp"])         