"""
Unit tests for tagging data export.
Uses synthetic data and temporary directories to validate parquet schema and metadata.
"""
import pandas as pd
import pyarrow.parquet as pq

from preprocessing import export_tagging as et
from preprocessing import params

def test_tagging_parquet_schema_and_metadata(tmp_path, monkeypatch):
    def mock_load_events(*args, **kwargs):
        return pd.DataFrame([
            {"start_frame": 10, "end_frame": 20, "code": "SET PIECES", "team": "H", "type": "kick off (start)", "half": 1}
        ])
        
    monkeypatch.setattr("preprocessing.export_tagging.load_tagging_events", mock_load_events)
    
    blocks = [(1, 5, 25)]
    kickoffs = {1: 15}
    
    et.export_tagging_data(tmp_path, blocks, kickoffs)
    
    pq_path = tmp_path / "tagging.parquet"
    json_path = tmp_path / "kickoffs.json"
    
    assert pq_path.exists()
    assert json_path.exists()
    
    table = pq.read_table(pq_path)
    meta = table.schema.metadata
    
    assert b"schema_version" in meta
    assert meta[b"schema_version"] == str(params.SCHEMA_VERSION).encode("utf-8")
    
    df = table.to_pandas()
    
    assert len(df) == params.N_FRAMES
    
    assert df["video_frame"].dtype == "int32"
    assert df["half"].dtype == "int8"
    assert df["n_tags_favor_home"].dtype == "int16"
    assert df["frames_to_next_sp"].dtype == "float32"
    assert df["sp_team"].dtype == "string"
    assert df["pre_kickoff"].dtype == "bool"
    
    assert df.loc[15, "half"] == 1
    assert df.loc[30, "half"] == 0