"""
Set Pieces Feature Extraction (Task C2).

Calculates set piece presence, type, team, and temporal distances
(frames to next SP, frames since last SP end).

Strict constraints applied:
- Half-open intervals: [start_frame, end_frame) without '+ 1'.
- Distances are bounded by half_video_blocks (never crossing halftime).
- Last active frame is end_frame - 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import params

def build_set_piece_features(ev: pd.DataFrame, n_frames: int = params.N_FRAMES):
    """
    Extracts set piece arrays from the C1 events DataFrame.
    
    Args:
        ev: Event DataFrame from prep_smart_tagging.py
        n_frames: Total number of frames in the tracking/video data.
        
    Returns:
        is_sp: boolean array of set piece phases.
        sp_type: object array (lowercase string, empty if none).
        sp_team: object array ('H', 'A', 'U' if SP; empty if none).
        starts: array of start frames for each SP.
        ends: array of last active frames (end_frame - 1) for each SP.
        sp_types: array of SP types corresponding to starts/ends.
    """
    is_sp = np.zeros(n_frames, dtype=bool)
    sp_type = np.full(n_frames, "", dtype=object)
    sp_team = np.full(n_frames, "", dtype=object)
    
    sp_events = []
    
    for _, r in ev.iterrows():
        code = str(r["code"]).strip().upper()
        if code == "SET PIECES":
            a, b = int(r["start_frame"]), int(r["end_frame"])  # [a, b) semi-open
            team = str(r["team"]).strip().upper()
            typ = str(r["type"]).strip().lower()
            
            is_sp[a:b] = True
            sp_type[a:b] = typ
            sp_team[a:b] = team if team in ("H", "A", "U") else "U"
            
            if b > a:
                # b - 1 ensures we capture the LAST active frame of the set piece
                sp_events.append((a, b - 1, typ))
                
    sp_events.sort(key=lambda item: (item[0], item[1], item[2]))
    
    if sp_events:
        starts = np.asarray([s for s, _, _ in sp_events], dtype=int)
        ends = np.asarray([e for _, e, _ in sp_events], dtype=int)
        types = np.asarray([t for _, _, t in sp_events], dtype=object)
    else:
        starts = np.array([], dtype=int)
        ends = np.array([], dtype=int)
        types = np.array([], dtype=object)
        
    return is_sp, sp_type, sp_team, starts, ends, types


def distance_to_set_pieces(starts: np.ndarray, ends: np.ndarray, sp_types: np.ndarray, 
                           blocks: list[tuple[int, int, int]], n_frames: int = params.N_FRAMES):
    """
    Calculates distances and adjacent types strictly within each half block.
    Outside the blocks (e.g., halftime where half == 0), values remain NaN/empty.
    
    Args:
        starts: Array of SP start frames.
        ends: Array of SP last active frames.
        sp_types: Array of SP types.
        blocks: list of (half, first_frame, last_frame) from half_video_blocks.
        n_frames: Total number of frames.
    """
    starts = np.asarray(starts, dtype=int)
    ends = np.asarray(ends, dtype=int)
    sp_types = np.asarray(sp_types, dtype=object)
    
    to_next = np.full(n_frames, np.nan, dtype=np.float32)
    since_end = np.full(n_frames, np.nan, dtype=np.float32)
    next_type = np.full(n_frames, "", dtype=object)
    prev_type = np.full(n_frames, "", dtype=object)
    
    for _, f0, f1 in blocks:
        f = np.arange(f0, f1 + 1)
        
        m_start = (starts >= f0) & (starts <= f1)
        m_end = (ends >= f0) & (ends <= f1)
        
        s_half, ts_half = starts[m_start], sp_types[m_start]
        e_half, te_half = ends[m_end], sp_types[m_end]
        
        # 1. Frames to NEXT set piece
        i = np.searchsorted(s_half, f, side="left")
        ok_next = i < len(s_half)
        to_next[f[ok_next]] = (s_half[i[ok_next]] - f[ok_next]).astype(np.float32)
        next_type[f[ok_next]] = ts_half[i[ok_next]]
        
        # 2. Frames since PREVIOUS set piece ended
        j = np.searchsorted(e_half, f, side="right") - 1
        ok_prev = j >= 0
        since_end[f[ok_prev]] = (f[ok_prev] - e_half[j[ok_prev]]).astype(np.float32)
        prev_type[f[ok_prev]] = te_half[j[ok_prev]]
        
    return to_next, since_end, next_type, prev_type


def process_set_pieces(ev: pd.DataFrame, blocks: list[tuple[int, int, int]], n_frames: int = params.N_FRAMES):
    """
    Main pipeline function for Task C2. 
    Combines feature building and distance calculation.
    """
    is_sp, sp_type, sp_team, starts, ends, types = build_set_piece_features(ev, n_frames)
    
    to_next, since_end, next_type, prev_type = distance_to_set_pieces(
        starts, ends, types, blocks, n_frames
    )
    
    return pd.DataFrame({
        "is_set_piece": is_sp,
        "sp_type": sp_type,
        "sp_team": sp_team,
        "frames_to_next_sp": to_next,
        "frames_since_sp_end": since_end,
        "next_sp_type": next_type,
        "prev_sp_type": prev_type
    })