"""
Tagging data exporter.

Combines set piece, phase, and kickoff features into a single DataFrame.
Enforces schema data types and writes to a Parquet file with deterministic
metadata for schema versioning.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from preprocessing import params
from preprocessing.prep_smart_tagging import load_tagging_events
from preprocessing.prep_set_pieces import process_set_pieces
from preprocessing.prep_phases import build_phase_features
from preprocessing.prep_kickoffs import get_kickoff_metadata, save_kickoffs_json, pre_kickoff_mask

def build_tagging_dataframe(
    ev: pd.DataFrame, 
    blocks: list[tuple[int, int, int]], 
    exact_kickoff_frames: dict[int, int],
    n_frames: int = params.N_FRAMES
) -> tuple[pd.DataFrame, dict]:
    """
    Assembles the complete tagging dataframe from individual feature modules.
    """
    df_sp = process_set_pieces(ev, blocks, n_frames)
    df_ph = build_phase_features(ev, n_frames)
    
    metadata = get_kickoff_metadata(ev, exact_kickoff_frames)
    mask_ko = pre_kickoff_mask(metadata, blocks, n_frames)
    
    video_frame = np.arange(n_frames, dtype=np.int32)
    half = np.zeros(n_frames, dtype=np.int8)
    
    for h, f0, f1 in blocks:
        half[f0:f1 + 1] = h
        
    df = pd.DataFrame({
        "video_frame": video_frame,
        "half": half,
        "is_set_piece": df_sp["is_set_piece"].astype(bool),
        "sp_type": df_sp["sp_type"].astype("string"),
        "sp_team": df_sp["sp_team"].astype("string"),
        "frames_to_next_sp": df_sp["frames_to_next_sp"].astype(np.float32),
        "frames_since_sp_end": df_sp["frames_since_sp_end"].astype(np.float32),
        "next_sp_type": df_sp["next_sp_type"].astype("string"),
        "prev_sp_type": df_sp["prev_sp_type"].astype("string"),
        "n_tags_favor_home": df_ph["n_tags_favor_home"].astype(np.int16),
        "n_tags_favor_away": df_ph["n_tags_favor_away"].astype(np.int16),
        "pre_kickoff": mask_ko.astype(bool),
    })
    
    return df, metadata

def export_tagging_data(
    output_dir: str | Path,
    blocks: list[tuple[int, int, int]],
    exact_kickoff_frames: dict[int, int] | None = None,
    data_dir: str | Path | None = None
) -> None:
    """
    Loads raw events, builds the structured dataframe, and exports to Parquet
    and JSON formats with schema metadata.
    """
    if exact_kickoff_frames is None:
        exact_kickoff_frames = params.KICKOFF_FALLBACK
        
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    ev = load_tagging_events(data_dir=data_dir, fps=params.FPS, n_frames=params.N_FRAMES)
    df, ko_metadata = build_tagging_dataframe(ev, blocks, exact_kickoff_frames, params.N_FRAMES)
    
    save_kickoffs_json(ko_metadata, out_path / "kickoffs.json")
    
    table = pa.Table.from_pandas(df)
    meta = table.schema.metadata or {}
    meta[b"schema_version"] = str(params.SCHEMA_VERSION).encode("utf-8")
    table = table.replace_schema_metadata(meta)
    
    pq.write_table(table, out_path / "tagging.parquet")

if __name__ == "__main__":
    from games import half_video_blocks, GAME
    blocks = half_video_blocks(GAME)
    export_tagging_data("data/processed", blocks)