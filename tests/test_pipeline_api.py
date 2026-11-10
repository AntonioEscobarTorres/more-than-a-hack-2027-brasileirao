"""Synthetic regression tests for the in-memory preprocessing API."""
from pathlib import Path

import numpy as np

from preprocessing import build_master_table as bm
from preprocessing import schema
from preprocessing.export_tagging import empty_tagging_dataframe
from preprocessing.pipeline_api import build_master_from_arrays
from test_build_master_table import BLOCKS, N, make_intermediates


GOLDEN_MASTER_DATA_SHA256 = "b2ae7eca70223469e2bbc6dfaa362ffb1dda848f486aa0bf2147612d05ec7364"


def test_empty_tagging_dataframe_matches_tagging_dtypes():
    tagging = empty_tagging_dataframe([(1, 1, 3), (2, 5, 6)], n_frames=8)

    assert tagging.columns.tolist() == [
        "video_frame", "half", "is_set_piece", "sp_type", "sp_team",
        "frames_to_next_sp", "frames_since_sp_end", "next_sp_type", "prev_sp_type",
        "n_tags_favor_home", "n_tags_favor_away", "pre_kickoff",
    ]
    assert tagging.dtypes.astype(str).tolist() == [
        "int32", "int8", "bool", "string", "string", "float32", "float32",
        "string", "string", "int16", "int16", "bool",
    ]
    assert tagging["half"].tolist() == [0, 1, 1, 1, 0, 2, 2, 0]
    assert tagging["is_set_piece"].eq(False).all()
    assert tagging["pre_kickoff"].eq(False).all()
    assert tagging[["sp_type", "sp_team", "next_sp_type", "prev_sp_type"]].eq("").all().all()
    assert tagging[["frames_to_next_sp", "frames_since_sp_end"]].isna().all().all()
    assert tagging[["n_tags_favor_home", "n_tags_favor_away"]].eq(0).all().all()


def test_original_build_master_golden_sha256(tmp_path):
    """Freeze the original synthetic master parquet before the structural refactor."""
    make_intermediates(tmp_path, n=N, blocks=BLOCKS, homography=True)
    master = bm.build_master(tmp_path, BLOCKS, N)
    output = Path(tmp_path) / "master_features.parquet"
    bm.write_master(master, output)

    # Parquet bytes vary across pyarrow releases, even when the logical table is identical.
    # Hash the ordered values instead, and separately check the schema metadata on disk.
    import hashlib
    import pandas as pd
    import pyarrow.parquet as pq

    data_hash = hashlib.sha256(
        pd.util.hash_pandas_object(pd.read_parquet(output), index=False).to_numpy().tobytes()
    ).hexdigest()
    assert data_hash == GOLDEN_MASTER_DATA_SHA256
    assert pq.read_schema(output).metadata[b"schema_version"] == bm.params.SCHEMA_VERSION.encode()


def _synthetic_tracking(pitch):
    n_frames = 600
    players = np.full((n_frames, 6, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    blocks = [(1, 0, 249), (2, 350, 599)]
    for half, first, last in blocks:
        frames = np.arange(last - first + 1, dtype=np.float32)
        home_x = 0.20 + frames * 0.00001
        away_x = 0.75 - frames * 0.00001
        # Home attacks toward increasing x in half 1 and decreasing x in half 2.
        if half == 2:
            home_x = 0.80 - frames * 0.00001
            away_x = 0.25 + frames * 0.00001
        players[first:last + 1, :3, 0] = home_x[:, None] + np.array([0, .03, .06])
        players[first:last + 1, 3:, 0] = away_x[:, None] + np.array([0, .03, .06])
        players[first:last + 1, :3, 1] = np.array([.30, .40, .50])
        players[first:last + 1, 3:, 1] = np.array([.30, .40, .50])
        # Keep a small amount of ball motion while avoiding synthetic jumps.
        ball[first:last + 1, 0] = 0.45 + frames * 0.00001
        ball[first:last + 1, 1] = 0.50 + frames * 0.00001

    # Keep half 1 just beyond the goal line while the y coordinate moves smoothly.
    ball[:250, 0] = 106.2 / pitch[0]
    ball[:250, 1] = 0.48 + np.arange(250, dtype=np.float32) * 0.00001
    return players, ball, np.array([0, 0, 0, 1, 1, 1], dtype=np.int8), blocks


def test_build_master_from_arrays_uses_custom_pitch_and_frozen_schema():
    players, ball, teams, blocks = _synthetic_tracking((105.0, 68.0))
    players_inside, ball_inside, teams_inside, blocks_inside = _synthetic_tracking((105.3, 68.0))
    outside = build_master_from_arrays(
        players, ball, teams, blocks=blocks, pitch=(105.0, 68.0))
    inside = build_master_from_arrays(
        players_inside, ball_inside, teams_inside, blocks=blocks_inside, pitch=(105.3, 68.0))

    assert outside.shape == (600, 59)
    assert list(outside.columns) == schema.master_column_names(False)
    assert "reproj_error" not in outside.columns
    np.testing.assert_array_equal(outside["video_frame"], np.arange(600, dtype=np.int32))
    np.testing.assert_array_equal(outside["half"], np.r_[np.ones(250), np.zeros(100), np.full(250, 2)])
    assert outside.loc[20, "out_of_pitch"] == 1.0
    assert inside.loc[20, "out_of_pitch"] == 0.0
    assert outside.loc[250:349, "ball_x"].isna().all()
    assert outside.loc[250:349, "d_near_home"].isna().all()
    assert outside["sp_type"].dtype.name == "string"