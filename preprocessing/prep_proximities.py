"""Proximity features from cleaned tracking (script B).

B1 loads ``tracking_clean.parquet`` into ``ProximityInputs``. B2 turns those
arrays into nearest-player distances per team. Later tasks (B3–B6) reuse the
same inputs; features that look across time must iterate ``half_slices`` so
nothing crosses the interval.

Run from the repository root:

    python -m preprocessing.prep_proximities

The match parquet stays in ``data/processed`` (gitignored). The command logs
only shapes and counts.
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import NamedTuple

import numpy as np
import pandas as pd

from games import GAME, half_video_blocks
from .params import N_FRAMES, SCHEMA_VERSION
from .schema import N_SLOTS

LOGGER = logging.getLogger(__name__)

TRACKING_STAGES = ("V0", "V1")
BALL_COLUMNS = ("ball_x", "ball_y", "ball_vx", "ball_vy")
SLOT_FIELDS = ("x", "y", "vx", "vy")


class ProximityInputs(NamedTuple):
    """Frame-aligned arrays. Slot axis is ``s01`` .. ``s46`` in that order."""

    video_frame: np.ndarray
    half: np.ndarray
    px: np.ndarray
    py: np.ndarray
    pvx: np.ndarray
    pvy: np.ndarray
    ball_x: np.ndarray
    ball_y: np.ndarray
    ball_vx: np.ndarray
    ball_vy: np.ndarray
    mask_home: np.ndarray
    mask_away: np.ndarray
    tracking_stage: str

    def half_slices(self):
        """``(half_id, slice)`` for each contiguous half, in play order."""
        return contiguous_half_slices(self.half)


class TeamProximity(NamedTuple):
    """Nearest-player distances in metres (B2). Indices feed B4 later."""

    d_near_home: np.ndarray
    d_near_away: np.ndarray
    d_diff: np.ndarray
    nearest_home: np.ndarray
    nearest_away: np.ndarray
    distances: np.ndarray


def read_tracking_stage(path):
    """``tracking_stage`` stored on the Parquet schema by ``prep_tracking``."""
    import pyarrow.parquet as pq

    metadata = pq.read_schema(path).metadata or {}
    raw = metadata.get(b"tracking_stage")
    if raw is None:
        raise ValueError(f"{path} is missing tracking_stage metadata")
    stage = raw.decode("utf-8")
    if stage not in TRACKING_STAGES:
        raise ValueError(f"{path} has tracking_stage {stage!r}; expected one of {TRACKING_STAGES}")
    version = metadata.get(b"schema_version")
    if version is not None and version.decode("utf-8") != SCHEMA_VERSION:
        raise ValueError(
            f"{path} has schema_version {version.decode('utf-8')!r}; expected {SCHEMA_VERSION}")
    return stage


def contiguous_half_slices(half):
    """One slice per half. The two blocks must not overlap or interleave."""
    half = np.asarray(half)
    slices = []
    for half_id in (1, 2):
        idx = np.flatnonzero(half == half_id)
        if len(idx) == 0:
            raise ValueError(f"half {half_id} has no frames")
        start, stop = int(idx[0]), int(idx[-1]) + 1
        if stop - start != len(idx):
            raise ValueError(f"half {half_id} is not one contiguous block")
        slices.append((half_id, slice(start, stop)))
    if slices[0][1].stop > slices[1][1].start:
        raise ValueError("half 2 starts before half 1 ends")
    return tuple(slices)


def _slot_matrix(frame, field, n_slots):
    columns = [f"s{slot:02d}_{field}" for slot in range(1, n_slots + 1)]
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"tracking_clean is missing slot columns: {missing[:4]}")
    return frame.loc[:, columns].to_numpy(dtype=np.float32, copy=True)


def _series(frame, name, dtype):
    if name not in frame.columns:
        raise ValueError(f"tracking_clean is missing column {name}")
    return frame[name].to_numpy(dtype=dtype, copy=True)


def team_masks(slot_teams, n_slots=N_SLOTS):
    """``mask_home`` / ``mask_away`` aligned to slot columns ``s01`` .. ``sN``.

    A slot that is not a field player is in neither mask. Goalkeepers stay in,
    because ``is_field_player`` is true for every club slot in this match.
    """
    required = ("slot", "is_home", "is_field_player")
    missing = [name for name in required if name not in slot_teams.columns]
    if missing:
        raise ValueError(f"slot_teams is missing columns {missing}")
    if len(slot_teams) != n_slots:
        raise ValueError(f"slot_teams has {len(slot_teams)} rows; expected {n_slots}")
    if slot_teams["is_home"].isna().any() or slot_teams["is_field_player"].isna().any():
        raise ValueError("slot_teams has a missing is_home or is_field_player flag")

    ordered = slot_teams.sort_values("slot", kind="mergesort")
    slots = ordered["slot"].to_numpy()
    expected = np.arange(1, n_slots + 1)
    if not np.array_equal(slots, expected):
        raise ValueError(f"slot_teams slots must be 1..{n_slots} exactly once")

    is_home = ordered["is_home"].to_numpy(dtype=bool)
    is_field = ordered["is_field_player"].to_numpy(dtype=bool)
    mask_home = is_home & is_field
    mask_away = (~is_home) & is_field
    return mask_home, mask_away


def _blank_outside_play(half, *arrays):
    outside = np.asarray(half) == 0
    for array in arrays:
        array[outside] = np.nan
    return outside


def _assert_published_halves(half):
    expected = tuple((half_id, slice(first, last + 1))
                     for half_id, first, last in half_video_blocks(GAME))
    got = contiguous_half_slices(half)
    if got != expected:
        raise ValueError(
            f"half column does not match half_video_blocks({GAME!r}): "
            f"got {got}, expected {expected}")


def assemble_inputs(frame, slot_teams, tracking_stage, n_frames=N_FRAMES, n_slots=N_SLOTS):
    """Stack slot columns into ``(T, N)`` and force every non-playing frame to NaN."""
    if tracking_stage not in TRACKING_STAGES:
        raise ValueError(f"tracking_stage must be one of {TRACKING_STAGES}, got {tracking_stage!r}")
    if len(frame) != n_frames:
        raise ValueError(f"tracking_clean has {len(frame)} rows; expected {n_frames}")

    video_frame = _series(frame, "video_frame", np.int32)
    if not np.array_equal(video_frame, np.arange(n_frames, dtype=np.int32)):
        raise ValueError("video_frame must be 0..n-1 in order, one row per tactical frame")

    half = _series(frame, "half", np.int8)
    if not np.isin(half, (0, 1, 2)).all():
        raise ValueError("half must be 0, 1 or 2")
    contiguous_half_slices(half)
    if n_frames == N_FRAMES:
        _assert_published_halves(half)

    px = _slot_matrix(frame, "x", n_slots)
    py = _slot_matrix(frame, "y", n_slots)
    pvx = _slot_matrix(frame, "vx", n_slots)
    pvy = _slot_matrix(frame, "vy", n_slots)
    ball = [_series(frame, name, np.float32) for name in BALL_COLUMNS]
    _blank_outside_play(half, px, py, pvx, pvy, *ball)
    mask_home, mask_away = team_masks(slot_teams, n_slots)

    return ProximityInputs(
        video_frame=video_frame,
        half=half,
        px=px, py=py, pvx=pvx, pvy=pvy,
        ball_x=ball[0], ball_y=ball[1], ball_vx=ball[2], ball_vy=ball[3],
        mask_home=mask_home, mask_away=mask_away,
        tracking_stage=tracking_stage,
    )


def load_proximity_inputs(tracking_path, slot_teams_path, n_frames=N_FRAMES, n_slots=N_SLOTS):
    """Read the two parquet files and return ``ProximityInputs``.

    ``n_frames`` defaults to the match length. Tests pass a shorter length; the
    real command does not.
    """
    stage = read_tracking_stage(tracking_path)
    columns = ["video_frame", "half", *BALL_COLUMNS]
    columns.extend(f"s{slot:02d}_{field}" for slot in range(1, n_slots + 1) for field in SLOT_FIELDS)
    frame = pd.read_parquet(tracking_path, columns=columns)
    slot_teams = pd.read_parquet(slot_teams_path)
    inputs = assemble_inputs(frame, slot_teams, stage, n_frames=n_frames, n_slots=n_slots)
    LOGGER.info(
        "tracking_stage=%s rows=%d slots=%d home_slots=%d away_slots=%d",
        inputs.tracking_stage, len(inputs.video_frame), inputs.px.shape[1],
        int(inputs.mask_home.sum()), int(inputs.mask_away.sum()))
    return inputs


def player_ball_distances(bx, by, px, py):
    """Euclidean distances in metres: shape ``(T, N)``. Missing ball or slot -> NaN."""
    return np.hypot(px - bx[:, None], py - by[:, None]).astype(np.float32, copy=False)


def masked_nearest(d, mask):
    """Nearest finite distance under a slot mask.

    ``d`` is ``(T, N)``; ``mask`` is ``(N,)``. Returns slot indices and distances
    as float32. A frame with no eligible finite distance yields NaN.
    """
    d = np.asarray(d)
    mask = np.asarray(mask, dtype=bool)
    if d.ndim != 2:
        raise ValueError(f"distances must be 2-D (T, N); got shape {d.shape}")
    if mask.shape != (d.shape[1],):
        raise ValueError(f"mask length {mask.shape} does not match N={d.shape[1]}")

    dm = np.where(mask[None, :] & np.isfinite(d), d, np.inf)
    nearest = dm.argmin(axis=1).astype(np.int32)
    dist = dm[np.arange(len(d)), nearest]
    dist = np.where(np.isfinite(dist), dist, np.nan).astype(np.float32)
    return nearest, dist


def team_proximity(inputs):
    """B2: nearest home/away distance to the ball and their difference.

    ``d_diff = d_near_home - d_near_away`` (negative means home is closer).
    Frames without a ball, or without a tracked field player on that side, are
    NaN. Each frame is independent; ``half == 0`` is already blanked in B1.
    """
    distances = player_ball_distances(inputs.ball_x, inputs.ball_y, inputs.px, inputs.py)
    nearest_home, d_home = masked_nearest(distances, inputs.mask_home)
    nearest_away, d_away = masked_nearest(distances, inputs.mask_away)
    d_diff = (d_home - d_away).astype(np.float32)
    return TeamProximity(
        d_near_home=d_home,
        d_near_away=d_away,
        d_diff=d_diff,
        nearest_home=nearest_home,
        nearest_away=nearest_away,
        distances=distances,
    )


def _processed_dir(data_dir=None):
    root = data_dir or os.environ.get("MTAH_DATA_DIR") or os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data")
    return os.path.join(root, "processed")


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed-dir", help="directory with tracking_clean.parquet")
    args = parser.parse_args(argv)
    folder = args.processed_dir or _processed_dir()
    inputs = load_proximity_inputs(
        os.path.join(folder, "tracking_clean.parquet"),
        os.path.join(folder, "slot_teams.parquet"))
    proximity = team_proximity(inputs)
    for half_id, block in inputs.half_slices():
        ball_seen = int(np.isfinite(inputs.ball_x[block]).sum())
        home_near = int(np.isfinite(proximity.d_near_home[block]).sum())
        LOGGER.info(
            "half=%d frames=%d ball_finite=%d d_near_home_finite=%d",
            half_id, block.stop - block.start, ball_seen, home_near)
    outside = int(np.count_nonzero(inputs.half == 0))
    LOGGER.info(
        "outside_play=%d d_near_home_nan_outside=%d",
        outside, int(np.isnan(proximity.d_near_home[inputs.half == 0]).sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
