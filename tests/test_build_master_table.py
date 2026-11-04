"""Unit tests for build_master_table. SYNTHETIC intermediates only: nothing from the match."""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from preprocessing import build_master_table as bm
from preprocessing import params, schema

N = 400
BLOCKS = [(1, 10, 99), (2, 150, 249)]


def _fake_columns(n, half):
    """One synthetic column per master column (except the index group), respecting the schema."""
    inside = half != 0
    idx = np.arange(n)
    cols = {}
    for group, (_, columns) in schema.MASTER_GROUPS.items():
        if group == "indice":
            continue
        for name, dtype in columns:
            if dtype == "float32":
                cols[name] = np.where(inside, np.linspace(0, 1, n), np.nan).astype("float32")
            elif dtype == "bool":
                cols[name] = inside & (idx % 3 == 0)
            elif dtype.startswith("int"):
                cols[name] = np.zeros(n, dtype=dtype)
            else:                                                     # string: '' means "none"
                cols[name] = np.where(inside & (idx % 7 == 0), "throw-in", "").astype(object)
    return cols


def _write(path, df, stage=None, version=b"1.0"):
    table = pa.Table.from_pandas(df, preserve_index=False)
    metadata = {**(table.schema.metadata or {})}
    if version is not None:
        metadata[b"schema_version"] = version
    if stage is not None:
        metadata[b"tracking_stage"] = stage
    pq.write_table(table.replace_schema_metadata(metadata), path)


def make_intermediates(folder, n=N, blocks=BLOCKS, homography=True):
    index = bm.index_columns(blocks, n)
    cols = _fake_columns(n, index["half"])
    keys = {"video_frame": index["video_frame"], "half": index["half"]}
    owner = {}
    for fname, groups in schema.INTERMEDIATE_FILES.items():
        for g in groups:
            owner[g] = fname
    files = {bm.TRACKING_FILE: dict(keys), "proximities.parquet": dict(keys),
             "tagging.parquet": dict(keys), bm.HOMOGRAPHY_FILE: dict(keys)}
    for group, (_, columns) in schema.MASTER_GROUPS.items():
        if group == "indice":
            continue
        for name, _ in columns:
            files[owner.get(group, bm.TRACKING_FILE)][name] = cols[name]
    # what tracking_clean V1 really carries besides the master columns
    track = files[bm.TRACKING_FILE]
    track.update({"is_graded": index["is_graded"], "frame_in_half": index["frame_in_half"],
                  "flip": np.zeros(n, dtype=np.int8), "ball_x_raw": np.zeros(n, dtype="float32"),
                  "s01_x": np.zeros(n, dtype="float32"), "s01_break": np.zeros(n, dtype=np.int8)})
    for fname, data in files.items():
        if fname == bm.HOMOGRAPHY_FILE and not homography:
            continue
        stage = b"V1" if fname == bm.TRACKING_FILE else None
        _write(folder / fname, pd.DataFrame(data), stage=stage)
    return files


def _rewrite(folder, fname, edit, **kw):
    df = pd.read_parquet(folder / fname)
    edit(df)
    _write(folder / fname, df, **kw)


def test_master_has_exactly_the_schema_columns_order_and_dtypes(tmp_path):
    make_intermediates(tmp_path)
    master = bm.build_master(tmp_path, BLOCKS, N)
    assert list(master.columns) == schema.master_column_names(True)
    assert len(master) == N and not master.columns.has_duplicates
    for name, dtype in schema.master_columns(True):
        assert str(master[name].dtype) == dtype, name
    for forbidden in ("flip", "ball_x_raw", "s01_x", "s01_break"):
        assert forbidden not in master.columns


def test_index_group_is_computed_from_the_blocks(tmp_path):
    make_intermediates(tmp_path)
    master = bm.build_master(tmp_path, BLOCKS, N)
    assert list(master["half"].iloc[[0, 10, 99, 100, 150, 249, 250]]) == [0, 1, 1, 0, 2, 2, 0]
    assert master["is_graded"].sum() == 90 + 100
    assert master["frame_in_half"].iloc[10] == 0 and master["frame_in_half"].iloc[99] == 89
    assert np.isnan(master["frame_in_half"].iloc[100])


def test_optional_homography_is_included_only_when_the_file_exists(tmp_path):
    make_intermediates(tmp_path, homography=False)
    master = bm.build_master(tmp_path, BLOCKS, N)
    assert list(master.columns) == schema.master_column_names(False) and "reproj_error" not in master
    make_intermediates(tmp_path, homography=True)
    assert bm.build_master(tmp_path, BLOCKS, N).columns[-1] == "reproj_error"


def test_half_mismatch_between_files_is_rejected(tmp_path):
    make_intermediates(tmp_path)
    def edit(df):
        df.loc[5, "half"] = 1
    _rewrite(tmp_path, "proximities.parquet", edit)
    with pytest.raises(ValueError, match="proximities.parquet: half differs"):
        bm.build_master(tmp_path, BLOCKS, N)


def test_video_frame_out_of_order_is_rejected(tmp_path):
    make_intermediates(tmp_path)
    _rewrite(tmp_path, "tagging.parquet", lambda df: df.__setitem__("video_frame", df["video_frame"][::-1].to_numpy()))
    with pytest.raises(ValueError, match="tagging.parquet: video_frame"):
        bm.build_master(tmp_path, BLOCKS, N)


def test_wrong_row_count_is_rejected(tmp_path):
    make_intermediates(tmp_path)
    _write(tmp_path / "proximities.parquet", pd.read_parquet(tmp_path / "proximities.parquet").iloc[:-1])
    with pytest.raises(ValueError, match="proximities.parquet: 399 rows, expected 400"):
        bm.build_master(tmp_path, BLOCKS, N)


def test_metadata_checks_schema_version_and_tracking_stage(tmp_path):
    make_intermediates(tmp_path)
    _rewrite(tmp_path, "tagging.parquet", lambda df: None, version=b"0.9")
    with pytest.raises(ValueError, match="schema_version"):
        bm.build_master(tmp_path, BLOCKS, N)
    make_intermediates(tmp_path)
    _rewrite(tmp_path, bm.TRACKING_FILE, lambda df: None, stage=b"V0")
    with pytest.raises(ValueError, match="prep_tracking --stage V1"):
        bm.build_master(tmp_path, BLOCKS, N)


def test_missing_file_and_missing_column_are_reported_by_name(tmp_path):
    make_intermediates(tmp_path)
    _rewrite(tmp_path, "proximities.parquet", lambda df: df.pop("d_diff"))
    with pytest.raises(ValueError, match=r"missing columns \['d_diff'\]"):
        bm.build_master(tmp_path, BLOCKS, N)
    (tmp_path / "tagging.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="tagging.parquet"):
        bm.build_master(tmp_path, BLOCKS, N)


def test_nan_in_an_integer_column_and_na_in_a_string_column_are_rejected(tmp_path):
    make_intermediates(tmp_path)
    def edit_int(df):
        df["ball_observed"] = df["ball_observed"].astype("float32")
        df.loc[20, "ball_observed"] = np.nan
    _rewrite(tmp_path, bm.TRACKING_FILE, edit_int, stage=b"V1")
    with pytest.raises(ValueError, match="ball_observed: has NaN but the schema says int8"):
        bm.build_master(tmp_path, BLOCKS, N)
    make_intermediates(tmp_path)
    def edit_str(df):
        df["sp_type"] = df["sp_type"].astype(object)
        df.loc[20, "sp_type"] = None
    _rewrite(tmp_path, "tagging.parquet", edit_str)
    with pytest.raises(ValueError, match="sp_type: string column has missing values"):
        bm.build_master(tmp_path, BLOCKS, N)


def test_values_outside_the_half_blocks_are_rejected(tmp_path):
    make_intermediates(tmp_path)
    def edit(df):
        df.loc[120, "d_near_home"] = 3.0                  # frame 120 is halftime (half == 0)
    _rewrite(tmp_path, "proximities.parquet", edit)
    with pytest.raises(ValueError, match=r"outside the half blocks.*d_near_home"):
        bm.build_master(tmp_path, BLOCKS, N)


def test_index_columns_written_by_tracking_are_cross_checked(tmp_path):
    make_intermediates(tmp_path)
    def edit(df):
        df.loc[50, "frame_in_half"] = 999.0
    _rewrite(tmp_path, bm.TRACKING_FILE, edit, stage=b"V1")
    with pytest.raises(ValueError, match="frame_in_half differs"):
        bm.build_master(tmp_path, BLOCKS, N)


def test_written_file_has_metadata_round_trips_and_is_deterministic(tmp_path):
    make_intermediates(tmp_path)
    master = bm.build_master(tmp_path, BLOCKS, N)
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    bm.write_master(master, a)
    bm.write_master(bm.build_master(tmp_path, BLOCKS, N), b)
    assert pq.read_schema(a).metadata[b"schema_version"] == b"1.0"
    assert bm.file_sha256(a) == bm.file_sha256(b)
    back = pd.read_parquet(a)
    assert list(back.columns) == list(master.columns)
    pd.testing.assert_series_equal(back["is_set_piece"], master["is_set_piece"])
    assert (back["sp_type"] == master["sp_type"]).all()               # '' stays '' (not NA)
    assert np.array_equal(back["ball_x"].to_numpy(), master["ball_x"].to_numpy(), equal_nan=True)