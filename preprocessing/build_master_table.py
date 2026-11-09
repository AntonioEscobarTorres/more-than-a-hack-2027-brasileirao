"""[D1] Build master_features.parquet from the intermediate preprocessing files.

Inputs, all in data/processed/ (git-ignored; match data never goes into the repository):
    tracking_clean.parquet   stage V1 (ball, confidence, teams, plus per-slot columns)
    proximities.parquet      proximity, relative_speed, pitch
    tagging.parquet          set_pieces, phases, kickoff
    homography.parquet       OPTIONAL (reproj_error)

Output: data/processed/master_features.parquet, exactly the columns of SCHEMA.md / schema.py, in
that order, one row per video frame (199,319), with `schema_version` in the Parquet metadata.

This table is the single input of every later modelling step, and a silent misalignment here would
corrupt everything downstream without any visible error. So every contract rule is checked and
fails loudly, naming the offending file or column. Nothing here is tuned: the script only joins,
selects, casts and verifies. It is deterministic (no randomness, fixed column order).

    python -m preprocessing.export_tagging
    python -m preprocessing.prep_tracking --stage V1
    python -m preprocessing.build_master_table [--check-determinism]
"""
from __future__ import annotations

import argparse
import hashlib
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import params, schema

TRACKING_FILE = "tracking_clean.parquet"
HOMOGRAPHY_FILE = "homography.parquet"
OUTPUT_FILE = "master_features.parquet"
REQUIRED_STAGE = b"V1"
KEY_COLUMNS = ("video_frame", "half")
INDEX_COLUMNS = tuple(name for name, _ in schema.MASTER_GROUPS["indice"][1])
V1_HINT = ("generate it with:\n"
           "  python -m preprocessing.export_tagging\n"
           "  python -m preprocessing.prep_tracking --stage V1")


def index_columns(blocks, n_frames=params.N_FRAMES):
    """The 'indice' group, computed from half_video_blocks (the master's own columns).

    half is 1 or 2 inside the blocks and 0 outside them (before the match, halftime, after the
    final whistle); frame_in_half counts frames from the first frame of the block, NaN outside.
    """
    video_frame = np.arange(n_frames, dtype=np.int32)
    half = np.zeros(n_frames, dtype=np.int8)
    is_graded = np.zeros(n_frames, dtype=bool)
    frame_in_half = np.full(n_frames, np.nan, dtype=np.float32)
    for h, first, last in blocks:
        half[first:last + 1] = h
        is_graded[first:last + 1] = True
        frame_in_half[first:last + 1] = np.arange(last - first + 1, dtype=np.float32)
    return {"video_frame": video_frame, "half": half, "is_graded": is_graded,
            "frame_in_half": frame_in_half}


def column_sources(include_optional):
    """Master column -> file that owns it (tracking_clean.parquet unless schema.py says otherwise)."""
    group_file = {group: name
                  for name, groups in schema.INTERMEDIATE_FILES.items() for group in groups}
    sources = {}
    for group, (_, columns) in schema.MASTER_GROUPS.items():
        if group == "indice":
            continue                                   
        for name, _dtype in columns:
            if not include_optional and name in schema.OPTIONAL_COLUMNS:
                continue
            sources[name] = group_file.get(group, TRACKING_FILE)
    return sources


def check_metadata(path, expect_stage=None):
    """schema_version must match params.SCHEMA_VERSION; tracking_clean must be stage V1."""
    metadata = pq.read_schema(path).metadata or {}
    expected = params.SCHEMA_VERSION.encode("utf-8")
    if metadata.get(b"schema_version") != expected:
        raise ValueError("{}: schema_version is {!r}, expected {!r}".format(
            Path(path).name, metadata.get(b"schema_version"), expected))
    if expect_stage is not None and metadata.get(b"tracking_stage") != expect_stage:
        raise ValueError("{}: tracking_stage is {!r}, expected {!r}; {}".format(
            Path(path).name, metadata.get(b"tracking_stage"), expect_stage, V1_HINT))


def _read(path, columns, n_frames):
    """Read only `columns` (tracking_clean is very wide), with a clear error if one is missing."""
    path = Path(path)
    available = set(pq.read_schema(path).names)
    missing = [c for c in columns if c not in available]
    if missing:
        raise ValueError("{}: missing columns {}".format(path.name, missing))
    df = pd.read_parquet(path, columns=list(columns))
    if len(df) != n_frames:
        raise ValueError("{}: {} rows, expected {}".format(path.name, len(df), n_frames))
    return df


def _check_keys(name, df, expected):
    """video_frame and half must be identical, value by value, in every file."""
    if not np.array_equal(df["video_frame"].to_numpy(), expected["video_frame"]):
        raise ValueError("{}: video_frame is not 0..n-1 in order".format(name))
    if not np.array_equal(df["half"].to_numpy(), expected["half"]):
        raise ValueError("{}: half differs from half_video_blocks".format(name))


def _cast(series, dtype, column):
    """Cast to the schema dtype, refusing silent damage (NaN into ints, NA into strings)."""
    if dtype == "string":
        if series.isna().any():
            raise ValueError("{}: string column has missing values; use '' for 'none'".format(column))
        return series.astype("string")
    if dtype == "bool":
        if series.isna().any():
            raise ValueError("{}: bool column has missing values".format(column))
        return series.astype(bool)
    if dtype.startswith("int"):
        if series.isna().any():
            raise ValueError("{}: has NaN but the schema says {}; NaN columns must be float32".format(
                column, dtype))
        return series.astype(dtype)
    return series.astype(dtype)                    


def assemble_master(tracking, proximities, tagging, homography, blocks, n_frames):
    """Join in-memory preprocessing frames into the schema-ordered master table."""
    include_optional = homography is not None
    sources = column_sources(include_optional)
    frames = {
        TRACKING_FILE: tracking,
        "proximities.parquet": proximities,
        "tagging.parquet": tagging,
    }
    if include_optional:
        frames[HOMOGRAPHY_FILE] = homography

    expected = index_columns(blocks, n_frames)
    columns = {}
    for name, df in frames.items():
        if len(df) != n_frames:
            raise ValueError("{}: {} rows, expected {}".format(name, len(df), n_frames))
        wanted = [column for column, source in sources.items() if source == name]
        missing = [column for column in list(KEY_COLUMNS) + wanted if column not in df.columns]
        if missing:
            raise ValueError("{}: missing columns {}".format(name, missing))
        _check_keys(name, df, expected)
        if name == TRACKING_FILE:
            extra = [column for column in INDEX_COLUMNS
                     if column in df.columns and column not in KEY_COLUMNS]
            for column in extra:
                if not np.array_equal(df[column].to_numpy(dtype="float64"),
                                      expected[column].astype("float64"), equal_nan=True):
                    raise ValueError("{}: {} differs from the one computed from half_video_blocks".format(
                        name, column))
        for column in wanted:
            columns[column] = df[column]

    dtypes = dict(schema.master_columns(include_optional))
    out = {}
    for column in dtypes:
        out[column] = pd.Series(expected[column]) if column in expected else columns[column]
    master = pd.DataFrame({
        column: _cast(out[column].reset_index(drop=True), dtypes[column], column)
        for column in dtypes
    })

    names = schema.master_column_names(include_optional)
    if list(master.columns) != names or master.columns.has_duplicates or len(master) != n_frames:
        raise ValueError("master does not match the schema (columns, order or row count)")

    outside = master["half"].to_numpy() == 0
    leaking = [column for column, dtype in dtypes.items()
               if dtype == "float32" and master[column][outside].notna().any()]
    if leaking:
        raise ValueError("columns with values outside the half blocks (half == 0): {}".format(leaking))
    return master


def build_master(processed_dir, blocks=None, n_frames=params.N_FRAMES):
    """Load intermediate files, validate metadata, and assemble the master table."""
    processed_dir = Path(processed_dir)
    if blocks is None:
        from games import GAME, half_video_blocks
        blocks = half_video_blocks(GAME)

    include_optional = (processed_dir / HOMOGRAPHY_FILE).exists()
    sources = column_sources(include_optional)
    files = sorted(set(sources.values()))
    for name in files:
        if not (processed_dir / name).exists():
            hint = "; " + V1_HINT if name == TRACKING_FILE else ""
            raise FileNotFoundError("missing {}{}".format(processed_dir / name, hint))
    for name in files:
        check_metadata(processed_dir / name, REQUIRED_STAGE if name == TRACKING_FILE else None)

    frames = {}
    for name in files:
        wanted = [column for column, source in sources.items() if source == name]
        extra = []
        if name == TRACKING_FILE:
            present = set(pq.read_schema(processed_dir / name).names)
            extra = [column for column in INDEX_COLUMNS
                     if column in present and column not in KEY_COLUMNS]
        frames[name] = _read(
            processed_dir / name, list(KEY_COLUMNS) + wanted + extra, n_frames)

    return assemble_master(
        frames[TRACKING_FILE], frames["proximities.parquet"], frames["tagging.parquet"],
        frames.get(HOMOGRAPHY_FILE), blocks, n_frames)


def write_master(master, path):
    """Write the Parquet file with schema_version in the metadata (pandas metadata kept)."""
    table = pa.Table.from_pandas(master, preserve_index=False)
    metadata = {**(table.schema.metadata or {}),
                b"schema_version": params.SCHEMA_VERSION.encode("utf-8")}
    table = table.replace_schema_metadata(metadata)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="snappy")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build master_features.parquet (task D1).")
    from games import DATA_DIR                     # honours MTAH_DATA_DIR, else ./data
    parser.add_argument("--processed-dir", default=str(Path(DATA_DIR) / "processed"),
                        help="default: <MTAH_DATA_DIR or ./data>/processed")
    parser.add_argument("--output", default=None, help="default: <processed-dir>/master_features.parquet")
    parser.add_argument("--check-determinism", action="store_true",
                        help="build a second time and compare the sha256 of both files")
    args = parser.parse_args(argv)

    processed = Path(args.processed_dir)
    output = Path(args.output) if args.output else processed / OUTPUT_FILE
    master = build_master(processed)
    write_master(master, output)
    print("wrote {} ({} rows, {} columns)".format(output, len(master), master.shape[1]))

    if args.check_determinism:
        with tempfile.TemporaryDirectory() as tmp:
            second = Path(tmp) / OUTPUT_FILE
            write_master(build_master(processed), second)
            same = file_sha256(output) == file_sha256(second)
        print("determinism: {}".format("OK (identical sha256)" if same else "FAILED (different sha256)"))
        return 0 if same else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
