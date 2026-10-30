"""
Unit tests for Kickoff feature extraction and validation.
Uses synthetic data to prevent real data leakage.
"""
import json
import numpy as np
import pandas as pd
import pytest

from preprocessing import prep_kickoffs as pk

# Synthetic blocks: Half 1 (100 to 200), Half 2 (300 to 400).
BLOCKS = [(1, 100, 200), (2, 300, 400)]
N_FRAMES = 500

def make_synthetic_kickoff_events():
    """Creates a synthetic event DataFrame with kickoff tags."""
    return pd.DataFrame([
        {"start_frame": 120, "end_frame": 135, "code": "SET PIECES", "type": "kick off (start)", "half": 1},
        {"start_frame": 310, "end_frame": 325, "code": "SET PIECES", "type": "kick off (start)", "half": 2},
    ])

def test_kickoff_metadata_validation_passes():
    ev = make_synthetic_kickoff_events()
    exact_frames = {1: 125, 2: 320}  # Both fall safely inside their windows
    
    metadata = pk.get_kickoff_metadata(ev, exact_frames)
    
    assert metadata["1"] == 125
    assert metadata["2"] == 320
    assert metadata["tag_windows"]["1"] == [120, 135]
    assert metadata["tag_windows"]["2"] == [310, 325]

def test_kickoff_metadata_raises_on_out_of_bounds():
    ev = make_synthetic_kickoff_events()
    exact_frames = {1: 110, 2: 320}  # Frame 110 is before the window [120, 135]
    
    with pytest.raises(ValueError, match="outside tag window"):
        pk.get_kickoff_metadata(ev, exact_frames)


def test_kickoff_metadata_raises_when_half_window_missing():
    ev = make_synthetic_kickoff_events().iloc[:1]  # Only half 1 present
    with pytest.raises(ValueError, match="Missing 'kick off \\(start\\)' tag window"):
        pk.get_kickoff_metadata(ev, {1: 125, 2: 320})

def test_pre_kickoff_mask_generation():
    # Simulate the dictionary structure that gets saved to JSON
    metadata = {
        "1": 125, 
        "2": 320, 
        "tag_windows": {"1": [120, 135], "2": [310, 325]}
    }
    
    mask = pk.pre_kickoff_mask(metadata, blocks=BLOCKS, n_frames=N_FRAMES)
    
    # Half 1: Block starts at 100, kickoff is at 125.
    assert mask[100:125].all()
    assert not mask[125]
    assert sum(mask[100:200]) == 25
    
    # Half 2: Block starts at 300, kickoff is at 320.
    assert mask[300:320].all()
    assert not mask[320]
    assert sum(mask[300:400]) == 20
    
    assert sum(mask[0:100]) == 0
    assert sum(mask[201:299]) == 0