"""
Unit tests for Homography feature extraction.
Uses synthetic data to prevent real data leakage.
"""
import numpy as np
import pandas as pd

from preprocessing import prep_homography as ph

BLOCKS = [(1, 10, 20), (2, 30, 40)]
N_FRAMES = 50

def make_synthetic_homography():
    """Creates a synthetic raw homography DataFrame."""
    return pd.DataFrame([
        {"frame": 12, "valid": 1, "reproj_error": 1.5},
        {"frame": 15, "valid": 0, "reproj_error": 8.0},
        {"frame": 35, "valid": 1, "reproj_error": 2.1},
        {"frame": 45, "valid": 1, "reproj_error": 1.1},
    ])

def test_process_homography_valid_masking():
    raw_df = make_synthetic_homography()
    df = ph.process_homography(raw_df, blocks=BLOCKS, n_frames=N_FRAMES)
    
    assert df.loc[12, "reproj_error"] == np.float32(1.5)
    
    assert pd.isna(df.loc[15, "reproj_error"])
    
    assert df.loc[35, "reproj_error"] == np.float32(2.1)
    # Frame 45 has valid == 1 but is outside BLOCKS (half == 0), so it must be NaN
    assert pd.isna(df.loc[45, "reproj_error"])


def test_process_homography_accepts_reprojection_error_csv_column():
    raw_df = pd.DataFrame([
        {"frame": 12, "valid": 1, "reprojection_error": 0.42},
        {"frame": 15, "valid": 0, "reprojection_error": 5.0},
        {"frame": 25, "valid": 1, "reprojection_error": 1.8},  # half == 0
    ])
    df = ph.process_homography(raw_df, blocks=BLOCKS, n_frames=N_FRAMES)

    assert df.loc[12, "reproj_error"] == np.float32(0.42)
    assert pd.isna(df.loc[15, "reproj_error"])
    assert pd.isna(df.loc[25, "reproj_error"])

def test_process_homography_half_assignment():
    raw_df = make_synthetic_homography()
    df = ph.process_homography(raw_df, blocks=BLOCKS, n_frames=N_FRAMES)
    
    assert df.loc[15, "half"] == 1
    assert df.loc[35, "half"] == 2
    
    assert df.loc[5, "half"] == 0
    assert df.loc[25, "half"] == 0

def test_process_homography_out_of_bounds_frames_are_ignored():
    raw_df = pd.DataFrame([
        {"frame": N_FRAMES + 10, "valid": 1, "reproj_error": 1.0},
        {"frame": -5, "valid": 1, "reproj_error": 1.0}
    ])
    
    df = ph.process_homography(raw_df, blocks=BLOCKS, n_frames=N_FRAMES)
    
    assert pd.isna(df["reproj_error"]).all()


def test_export_homography_data_writes_parquet_with_schema_metadata(tmp_path):
    import pyarrow.parquet as pq
    from preprocessing import params

    tactical_dir = tmp_path / "data" / "tactical"
    tactical_dir.mkdir(parents=True)
    csv_path = tactical_dir / "BAR-ATH_tactical_homography.csv"
    pd.DataFrame([
        {"frame": 12, "valid": 1, "reprojection_error": 0.35},
        {"frame": 25, "valid": 1, "reprojection_error": 1.40},
    ]).to_csv(csv_path, index=False)

    out_dir = tmp_path / "processed"
    ph.export_homography_data(out_dir, BLOCKS, data_dir=tmp_path / "data", n_frames=N_FRAMES)

    pq_path = out_dir / "homography.parquet"
    assert pq_path.exists()
    table = pq.read_table(pq_path)
    meta = table.schema.metadata
    assert meta[b"schema_version"] == str(params.SCHEMA_VERSION).encode("utf-8")

    df = table.to_pandas()
    assert len(df) == N_FRAMES
    assert df.loc[12, "reproj_error"] == np.float32(0.35)
    assert pd.isna(df.loc[25, "reproj_error"])