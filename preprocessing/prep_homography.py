"""
Homography Feature Extraction.

Processes the raw homography data to extract the reprojection error.
Invalid frames are masked with NaN values. Half assignment is derived
from the standard video blocks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import params

def process_homography(
    raw_df: pd.DataFrame, 
    blocks: list[tuple[int, int, int]], 
    n_frames: int = params.N_FRAMES
) -> pd.DataFrame:
    """
    Extracts homography features from the raw dataframe.
    
    Args:
        raw_df: DataFrame loaded from the homography CSV containing 
                'frame', 'valid', and 'reproj_error'.
        blocks: List of tuples (half, first_frame, last_frame).
        n_frames: Total number of frames in the video.
        
    Returns:
        DataFrame containing 'video_frame', 'half', and 'reproj_error'.
    """
    video_frame = np.arange(n_frames, dtype=np.int32)
    half = np.zeros(n_frames, dtype=np.int8)
    reproj_error = np.full(n_frames, np.nan, dtype=np.float32)
    
    for h, f0, f1 in blocks:
        half[f0:f1 + 1] = h
        
    valid_mask = (raw_df["valid"] == 1) & (raw_df["frame"] >= 0) & (raw_df["frame"] < n_frames)
    
    valid_frames = raw_df.loc[valid_mask, "frame"].to_numpy(dtype=int)
    errors = raw_df.loc[valid_mask, "reproj_error"].to_numpy(dtype=np.float32)
    
    reproj_error[valid_frames] = errors
    
    return pd.DataFrame({
        "video_frame": video_frame,
        "half": half,
        "reproj_error": reproj_error
    })