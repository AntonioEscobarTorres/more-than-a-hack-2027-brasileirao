"""
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

def build_set_piece_features(
    ev: pd.DataFrame,
    n_frames: int = params.N_FRAMES,
    blocks: list[tuple[int, int, int]] | None = None,
):
    """    
    Args:
        ev: Event DataFrame from prep_smart_tagging.py
        n_frames: Total number of frames in the tracking/video data.
        blocks: Optional list of (half, first_frame, last_frame) from half_video_blocks.
                When provided, set piece intervals are clipped to each half block so that
                outside frames (half == 0) remain False/empty.
        
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
            valid_team = team if team in ("H", "A", "U") else "U"

            if blocks is not None:
                slices = [
                    (max(a, f0), min(b, f1 + 1))
                    for _, f0, f1 in blocks
                    if min(b, f1 + 1) > max(a, f0)
                ]
            else:
                slices = [(a, b)] if b > a else []

            for ca, cb in slices:
                is_sp[ca:cb] = True
                sp_type[ca:cb] = typ
                sp_team[ca:cb] = valid_team
                # cb - 1 ensures we capture the last active frame of the set piece
                sp_events.append((ca, cb - 1, typ))
                
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
        ends: Array of SP last active frames (end_frame - 1).
        sp_types: Array of SP types.
        blocks: list of (half, first_frame, last_frame) from half_video_blocks.
        n_frames: Total number of frames.
    """
    starts = np.asarray(starts, dtype=int)
    ends = np.asarray(ends, dtype=int)
    sp_types = np.asarray(sp_types, dtype=object)

    # Do not rely on the caller's ordering: sort the set pieces by start frame here.
    order = np.argsort(starts, kind="stable")
    starts, ends, sp_types = starts[order], ends[order], sp_types[order]

    to_next = np.full(n_frames, np.nan, dtype=np.float32)
    since_end = np.full(n_frames, np.nan, dtype=np.float32)
    next_type = np.full(n_frames, "", dtype=object)
    prev_type = np.full(n_frames, "", dtype=object)

    for _, f0, f1 in blocks:
        f = np.arange(f0, f1 + 1)

        m_half = (starts <= f1) & (ends >= f0)
        s_half = np.maximum(starts[m_half], f0)
        e_half = np.minimum(ends[m_half], f1)
        t_half = sp_types[m_half]

        # Starts are sorted, but ends are NOT when one set piece is nested inside another
        # (np.searchsorted needs a sorted array), so sort the ends together with their types.
        end_order = np.argsort(e_half, kind="stable")
        e_sorted, t_by_end = e_half[end_order], t_half[end_order]

        # 1. Frames to next set piece (first start at or after this frame)
        i = np.searchsorted(s_half, f, side="left")
        ok_next = i < len(s_half)
        to_next[f[ok_next]] = (s_half[i[ok_next]] - f[ok_next]).astype(np.float32)
        next_type[f[ok_next]] = t_half[i[ok_next]]

        # 2. Frames since the most recent last-active frame of any set piece
        j = np.searchsorted(e_sorted, f, side="right") - 1
        ok_prev = j >= 0
        since_end[f[ok_prev]] = (f[ok_prev] - e_sorted[j[ok_prev]]).astype(np.float32)
        prev_type[f[ok_prev]] = t_by_end[j[ok_prev]]

    return to_next, since_end, next_type, prev_type


def process_set_pieces(ev: pd.DataFrame, blocks: list[tuple[int, int, int]], n_frames: int = params.N_FRAMES):
    """
    Combines feature building and distance calculation.
    """
    is_sp, sp_type, sp_team, starts, ends, types = build_set_piece_features(ev, n_frames, blocks=blocks)
    
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