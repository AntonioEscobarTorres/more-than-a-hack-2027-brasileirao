"""
Homography Feature Extraction.

Processes the raw homography data to extract the reprojection error.
Invalid frames are masked with NaN values. Half assignment is derived
from the standard video blocks.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

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
                'frame', 'valid', and 'reprojection_error' (or 'reproj_error').
        blocks: List of tuples (half, first_frame, last_frame).
        n_frames: Total number of frames in the video.
        
    Returns:
        DataFrame containing 'video_frame', 'half', and 'reproj_error'.
    """
    err_col = "reprojection_error" if "reprojection_error" in raw_df.columns else "reproj_error"
    if err_col not in raw_df.columns:
        raise ValueError(
            "Homography DataFrame must contain 'reprojection_error' or 'reproj_error'; "
            f"got columns {list(raw_df.columns)}"
        )

    video_frame = np.arange(n_frames, dtype=np.int32)
    half = np.zeros(n_frames, dtype=np.int8)
    reproj_error = np.full(n_frames, np.nan, dtype=np.float32)
    
    for h, f0, f1 in blocks:
        half[f0:f1 + 1] = h
        
    valid_mask = (raw_df["valid"] == 1) & (raw_df["frame"] >= 0) & (raw_df["frame"] < n_frames)
    
    valid_frames = raw_df.loc[valid_mask, "frame"].to_numpy(dtype=int)
    errors = raw_df.loc[valid_mask, err_col].to_numpy(dtype=np.float32)
    
    reproj_error[valid_frames] = errors
    reproj_error[half == 0] = np.nan
    
    return pd.DataFrame({
        "video_frame": video_frame,
        "half": half,
        "reproj_error": reproj_error
    })


def export_homography_data(
    output_dir: str | Path,
    blocks: list[tuple[int, int, int]],
    data_dir: str | Path | None = None,
    n_frames: int = params.N_FRAMES,
) -> None:
    """
    Loads the tactical homography CSV, extracts reprojection error features,
    and writes homography.parquet with schema_version metadata.
    """
    from games import GAME, require

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    (csv_path,) = require(["homography"], GAME, str(data_dir) if data_dir is not None else None)
    raw_df = pd.read_csv(csv_path)
    df = process_homography(raw_df, blocks, n_frames=n_frames)

    table = pa.Table.from_pandas(df)
    meta = table.schema.metadata or {}
    meta[b"schema_version"] = str(params.SCHEMA_VERSION).encode("utf-8")
    table = table.replace_schema_metadata(meta)

    pq.write_table(table, out_path / "homography.parquet")


if __name__ == "__main__":
    from games import GAME, half_video_blocks

    export_homography_data("data/processed", half_video_blocks(GAME))