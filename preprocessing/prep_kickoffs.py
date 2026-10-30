"""
Kickoff Feature Extraction and Validation.

Extracts kickoff windows from smart tagging, validates them against
known kickoff frames, exports the configuration, and computes the 
pre-kickoff boolean mask per half.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from . import params

def get_kickoff_metadata(ev: pd.DataFrame, exact_frames: dict[int, int] | None = None) -> dict:
    """
    Extracts kickoff tag windows and validates that the exact kick frames 
    fall within these windows.
    
    Args:
        ev: Event DataFrame containing smart tagging data.
        exact_frames: Dictionary mapping half (int) to the exact kickoff frame.
                      Defaults to params.KICKOFF_FALLBACK if None.
                      
    Returns:
        Dictionary formatted for JSON export with exact frames and tag windows.
    """
    if exact_frames is None:
        exact_frames = params.KICKOFF_FALLBACK
        
    windows = {}
    ko_events = ev[(ev["code"] == "SET PIECES") & (ev["type"] == "kick off (start)")]
    
    for _, r in ko_events.iterrows():
        h = int(r["half"])
        if h in (1, 2):
            windows[str(h)] = [int(r["start_frame"]), int(r["end_frame"])]
            
    out = {"tag_windows": windows}
    
    for h_int, k_frame in exact_frames.items():
        h_str = str(h_int)
        out[h_str] = k_frame
        
        if h_str not in windows:
            raise ValueError(f"Missing 'kick off (start)' tag window for half {h_int}")
        a, b = windows[h_str]
        if not (a <= k_frame <= b):
            raise ValueError(
                f"Kickoff frame {k_frame} for half {h_int} is outside tag window [{a}, {b}]"
            )
                
    return out


def save_kickoffs_json(metadata: dict, output_path: str | Path) -> None:
    """
    Saves the validated kickoff metadata to a JSON file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


def pre_kickoff_mask(
    kick_frames: dict, 
    blocks: list[tuple[int, int, int]], 
    n_frames: int = params.N_FRAMES
) -> np.ndarray:
    """
    Creates a boolean mask for frames evaluated before the kickoff in each half.
    
    Args:
        kick_frames: Dictionary containing the exact kickoff frames (string or int keys).
        blocks: List of tuples (half, first_frame, last_frame).
        n_frames: Total number of frames.
        
    Returns:
        Boolean array where True indicates frames before the exact kickoff frame.
    """
    m = np.zeros(n_frames, dtype=bool)
    
    for half, f0, f1 in blocks:
        # Safely retrieve the kickoff frame whether the key is a string or integer
        k = kick_frames.get(str(half)) if str(half) in kick_frames else kick_frames.get(half)
        if k is not None and f0 <= k <= f1:
            m[f0:k] = True
            
    return m