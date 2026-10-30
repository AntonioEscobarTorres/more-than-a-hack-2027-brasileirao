"""
Uses ONLY SYNTHETIC data to prevent real data leakage.
"""
import numpy as np
import pandas as pd
import pytest

from preprocessing import prep_phases as pp

N_FRAMES = 100

def make_synthetic_events():
    return pd.DataFrame([
        # 1. Offensive H [10, 20) -> favors H
        {"start_frame": 10, "end_frame": 20, "code": "BUILD UP", "team": "H", "type": "", "half": 1},
        # 2. Defensive A [15, 25) -> favors H
        {"start_frame": 15, "end_frame": 25, "code": "DEFENSIVE TRANSITION", "team": "A", "type": "", "half": 1},
        # 3. Unknown team 'U' [30, 40) -> Ignored
        {"start_frame": 30, "end_frame": 40, "code": "PROGRESSION", "team": "U", "type": "", "half": 1},
        # 4. Set Pieces [50, 60) -> Ignored
        {"start_frame": 50, "end_frame": 60, "code": "SET PIECES", "team": "H", "type": "throw-in", "half": 1},
        # 5. Defensive H [70, 80) -> favors A
        {"start_frame": 70, "end_frame": 80, "code": "DEFENDING IN MIDDLE THIRD", "team": "H", "type": "", "half": 2},
    ])

def test_offensive_h_and_defensive_a_favors_h_additively():
    ev = make_synthetic_events()
    df = pp.build_phase_features(ev, n_frames=N_FRAMES)
    
    # Frames [10, 15): Only Event 1 is active (Offensive H = 1 favor H)
    assert df.loc[10, "n_tags_favor_home"] == 1
    assert df.loc[14, "n_tags_favor_home"] == 1
    assert df.loc[10, "n_tags_favor_away"] == 0
    
    # Frames [15, 20): Event 1 and Event 2 overlap (Offensive H + Defensive A = 2 favor H)
    assert df.loc[15, "n_tags_favor_home"] == 2
    assert df.loc[19, "n_tags_favor_home"] == 2
    assert df.loc[15, "n_tags_favor_away"] == 0
    
    # Frames [20, 25): Only Event 2 is active (Defensive A = 1 favor H)
    assert df.loc[20, "n_tags_favor_home"] == 1
    assert df.loc[24, "n_tags_favor_home"] == 1
    assert df.loc[20, "n_tags_favor_away"] == 0

def test_exclusive_end_frame_slicing():
    ev = make_synthetic_events()
    df = pp.build_phase_features(ev, n_frames=N_FRAMES)
    
    # Event 1 ends at 20. Frame 20 itself should not contain Event 1's tag.
    # At frame 20, only Event 2 (Defensive A) is active. So favor_home drops from 2 to 1.
    assert df.loc[19, "n_tags_favor_home"] == 2
    assert df.loc[20, "n_tags_favor_home"] == 1
    
    # Event 2 ends at 25. Frame 25 should drop to 0.
    assert df.loc[24, "n_tags_favor_home"] == 1
    assert df.loc[25, "n_tags_favor_home"] == 0

def test_unknown_teams_and_set_pieces_are_ignored():
    ev = make_synthetic_events()
    df = pp.build_phase_features(ev, n_frames=N_FRAMES)
    
    # Event 3 ('U' team) spans 30 to 40
    assert df.loc[35, "n_tags_favor_home"] == 0
    assert df.loc[35, "n_tags_favor_away"] == 0
    
    # Event 4 ('SET PIECES') spans 50 to 60
    assert df.loc[55, "n_tags_favor_home"] == 0
    assert df.loc[55, "n_tags_favor_away"] == 0

def test_defensive_home_favors_away():
    ev = make_synthetic_events()
    df = pp.build_phase_features(ev, n_frames=N_FRAMES)
    
    # Event 5 (Defensive H) spans 70 to 80
    assert df.loc[75, "n_tags_favor_home"] == 0
    assert df.loc[75, "n_tags_favor_away"] == 1


def test_build_phase_features_masks_outside_blocks_when_blocks_given():
    ev = make_synthetic_events()
    blocks = [(1, 15, 45), (2, 70, 75)]
    df = pp.build_phase_features(ev, n_frames=N_FRAMES, blocks=blocks)

    # Frames 10..14 and 76..79 are outside blocks -> forced to 0
    assert (df.loc[0:14, "n_tags_favor_home"] == 0).all()
    assert (df.loc[76:99, "n_tags_favor_away"] == 0).all()
    # Inside blocks -> preserved
    assert df.loc[15, "n_tags_favor_home"] == 2
    assert df.loc[75, "n_tags_favor_away"] == 1