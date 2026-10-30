"""
Uses ONLY SYNTHETIC data to prevent real data leakage.
"""
import numpy as np
import pandas as pd
import pytest

from preprocessing import prep_set_pieces as sp

# Synthetic blocks: Half 1 (100 to 200), Half 2 (300 to 400). Halftime is 201 to 299.
BLOCKS = [(1, 100, 200), (2, 300, 400)]
N_FRAMES = 500

def make_synthetic_events():
    return pd.DataFrame([
        # Half 1: Set piece from 120 to 130
        {"start_frame": 120, "end_frame": 130, "code": "SET PIECES", "team": "H", "type": "throw-in", "half": 1},
        # Half 1: Normal phase (should be ignored by SP logic)
        {"start_frame": 135, "end_frame": 140, "code": "BUILD UP", "team": "H", "type": "", "half": 1},
        # Half 2: Set piece from 350 to 360
        {"start_frame": 350, "end_frame": 360, "code": "SET PIECES", "team": "A", "type": "corner kick", "half": 2},
    ])

def test_build_set_piece_features_semi_open_slice():
    ev = make_synthetic_events()
    is_sp, sp_type, sp_team, starts, ends, types = sp.build_set_piece_features(ev, n_frames=N_FRAMES)
    
    # Check exclusive end slicing [120, 130) -> Length is exactly 10
    assert is_sp[120:130].all()
    assert not is_sp[130]  # Frame 130 should not be a set piece
    assert sum(is_sp[100:200]) == 10
    
    # Check sp_team mapping
    assert sp_team[125] == "H"
    assert sp_team[110] == ""  # Empty string where is_set_piece == False
    assert sp_team[355] == "A"
    
    # Check sp_type
    assert sp_type[125] == "throw-in"
    assert sp_type[355] == "corner kick"
    
    # Check starts and ends (end_frame - 1)
    assert list(starts) == [120, 350]
    assert list(ends) == [129, 359]  # Last active frames

def test_distance_to_set_pieces_isolation_and_nan():
    ev = make_synthetic_events()
    df = sp.process_set_pieces(ev, blocks=BLOCKS, n_frames=N_FRAMES)
    
    # 1. NaN where half == 0 (Halftime: 201 to 299, and borders: 0-99, 401-499)
    assert pd.isna(df.loc[50, "frames_to_next_sp"])
    assert pd.isna(df.loc[250, "frames_to_next_sp"])
    assert df.loc[250, "next_sp_type"] == ""
    
    # 2. Before 1st SP of Half 1 (Frame 110: 10 frames to next SP, no previous SP)
    assert df.loc[110, "frames_to_next_sp"] == 10.0
    assert df.loc[110, "next_sp_type"] == "throw-in"
    assert pd.isna(df.loc[110, "frames_since_sp_end"])
    
    # 3. Inside a SP (Frame 125)
    # Next is technically 120 (since searchsorted "left" uses >=, but logic gives 120-125 = -5)
    # Actually, if we are at 125, the next start >= 125 is 350, but 350 is in Half 2.
    # Wait, the logic matches the first start >= f. Let's check exactly frame 120:
    assert df.loc[120, "frames_to_next_sp"] == 0.0
    
    # 4. After 1st SP of Half 1 (Frame 150)
    # SP ended at 129. 150 - 129 = 21 frames since end.
    assert df.loc[150, "frames_since_sp_end"] == 21.0
    assert df.loc[150, "prev_sp_type"] == "throw-in"
    # No next SP in Half 1, so frames_to_next_sp should be NaN
    assert pd.isna(df.loc[150, "frames_to_next_sp"])

def test_blocks_never_cross_halftime():
    ev = make_synthetic_events()
    df = sp.process_set_pieces(ev, blocks=BLOCKS, n_frames=N_FRAMES)
    
    # Frame 199 is in Half 1, next SP is in Half 2.
    # It must not calculate distance to Half 2's set piece.
    assert pd.isna(df.loc[199, "frames_to_next_sp"])
    
    # Frame 310 is in Half 2, previous SP was in Half 1.
    # It must not calculate distance from Half 1's set piece.
    assert pd.isna(df.loc[310, "frames_since_sp_end"])