"""Rule-based baseline labeller (task F5-02): nearest team to the ball, plus simple dead-ball rules.

Steps, applied to each half block on its own (nothing crosses the halftime gap):

(a) ``H`` where ``d_diff < 0`` and ``A`` where ``d_diff > 0`` (finite ``d_diff`` only).
(b) Every other frame inherits the last H/A before it in the same half; the frames before the
    first H/A of a half inherit that first H/A; a half without any H/A is all ``H``.
(c) ``D`` where ``out_of_pitch == 1``, or ``pre_kickoff``, or (``gap_length >= d_gap_frames`` and
    both ``mean_speed_home`` and ``mean_speed_away`` ``< slow_speed_mps``). Any comparison with
    NaN is false, so a missing value never produces ``D``.
(d) H/A segments shorter than ``min_spell_frames`` take the label of the adjacent H/A segment
    before them or, if that neighbour is ``D`` or the start of the half, of the adjacent H/A
    segment after them. Scanned left to right until nothing changes (at most
    ``MAX_SMOOTH_PASSES`` passes). ``D`` is never changed. A short segment whose neighbours are
    both ``D`` (or a half boundary) has no H/A neighbour to take a label from and is kept.
(e) ``X`` where ``half == 0``.

The thresholds come from the ``baseline`` section of the config (``configs/baseline.yaml``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import LABEL_DTYPE

REQUIRED_COLUMNS = ("half", "d_diff", "out_of_pitch", "pre_kickoff", "gap_length",
                    "mean_speed_home", "mean_speed_away")
PARAM_KEYS = ("d_gap_frames", "slow_speed_mps", "min_spell_frames")

# Upper bound on the passes of step (d), fixed by the task specification (F5-02).
MAX_SMOOTH_PASSES = 20

_NONE, _H, _A = 0, 1, 2          # internal team codes for steps (a)-(b)


def half_blocks(half) -> list[tuple[int, int, int]]:
    """``[(half, start, stop)]``: maximal runs of one non-zero ``half``; ``stop`` is exclusive."""
    half = np.asarray(half)
    if half.ndim != 1:
        raise ValueError("half must be 1-D, got shape {}".format(half.shape))
    if half.size == 0:
        return []
    starts = np.flatnonzero(np.r_[True, half[1:] != half[:-1]])
    stops = np.r_[starts[1:], half.size]
    return [(int(half[a]), int(a), int(b)) for a, b in zip(starts, stops) if half[a] != 0]


def _team_fill(d_diff: np.ndarray) -> np.ndarray:
    """Steps (a)-(b) for one half: an array of ``"H"``/``"A"``."""
    team = np.full(d_diff.shape, _NONE, dtype=np.int8)
    finite = np.isfinite(d_diff)
    team[finite & (d_diff < 0)] = _H
    team[finite & (d_diff > 0)] = _A

    known = np.flatnonzero(team != _NONE)
    if known.size == 0:
        return np.full(d_diff.shape, "H", dtype=LABEL_DTYPE)
    idx = np.arange(team.size)
    carry = np.maximum.accumulate(np.where(team != _NONE, idx, -1))
    carry[carry < 0] = known[0]
    return np.where(team[carry] == _H, "H", "A").astype(LABEL_DTYPE)


def _dead_mask(block: pd.DataFrame, d_gap_frames: int, slow_speed_mps: float) -> np.ndarray:
    """Step (c) for one half. NaN compares false everywhere, so it never yields ``D``."""
    out_of_pitch = block["out_of_pitch"].to_numpy(dtype=np.float64)
    pre_kickoff = block["pre_kickoff"].to_numpy(dtype=bool)
    gap_length = block["gap_length"].to_numpy(dtype=np.float64)
    speed_home = block["mean_speed_home"].to_numpy(dtype=np.float64)
    speed_away = block["mean_speed_away"].to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore"):
        slow_gap = ((gap_length >= d_gap_frames) & (speed_home < slow_speed_mps)
                    & (speed_away < slow_speed_mps))
        return (out_of_pitch == 1) | pre_kickoff | slow_gap


def _segments(labels: np.ndarray) -> tuple[list[str], list[int]]:
    """Run-length encoding: (labels, lengths)."""
    starts = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
    lengths = np.diff(np.r_[starts, labels.size])
    return labels[starts].tolist(), lengths.tolist()


def _smooth_pass(labels: np.ndarray, min_spell_frames: int) -> np.ndarray:
    """One left-to-right pass of step (d) over one half. Returns a new array."""
    seg_labels, seg_lengths = _segments(labels)
    out_labels: list[str] = []
    out_lengths: list[int] = []
    n_seg = len(seg_labels)
    for i, (lab, length) in enumerate(zip(seg_labels, seg_lengths)):
        if lab != "D" and length < min_spell_frames:
            prev = out_labels[-1] if out_labels else None
            nxt = seg_labels[i + 1] if i + 1 < n_seg else None
            if prev is not None and prev != "D":
                lab = prev
            elif nxt is not None and nxt != "D":
                lab = nxt
        if out_labels and out_labels[-1] == lab:
            out_lengths[-1] += length
        else:
            out_labels.append(lab)
            out_lengths.append(length)
    return np.repeat(np.array(out_labels, dtype=LABEL_DTYPE), out_lengths)


def _smooth(labels: np.ndarray, min_spell_frames: int) -> np.ndarray:
    """Step (d) for one half: repeat passes until nothing changes (at most MAX_SMOOTH_PASSES)."""
    if min_spell_frames <= 0 or labels.size == 0:
        return labels
    for _ in range(MAX_SMOOTH_PASSES):
        new = _smooth_pass(labels, min_spell_frames)
        if np.array_equal(new, labels):
            break
        labels = new
    return labels


def baseline_labels(master: pd.DataFrame, params: dict) -> np.ndarray:
    """Baseline labels (``<U1``, one per master row) from the ``baseline`` config section."""
    missing = [c for c in REQUIRED_COLUMNS if c not in master.columns]
    if missing:
        raise ValueError("master is missing columns {}".format(missing))
    missing = [k for k in PARAM_KEYS if k not in params]
    if missing:
        raise ValueError("baseline params are missing keys {}".format(missing))
    d_gap_frames = params["d_gap_frames"]
    slow_speed_mps = float(params["slow_speed_mps"])
    min_spell_frames = params["min_spell_frames"]

    half = master["half"].to_numpy()
    d_diff = master["d_diff"].to_numpy(dtype=np.float64)
    labels = np.full(len(master), "X", dtype=LABEL_DTYPE)           # (e)
    for _, start, stop in half_blocks(half):
        block = labels[start:stop]
        block[:] = _team_fill(d_diff[start:stop])                   # (a), (b)
        dead = _dead_mask(master.iloc[start:stop], d_gap_frames, slow_speed_mps)
        block[dead] = "D"                                           # (c)
        block[:] = _smooth(block.copy(), min_spell_frames)          # (d)
    return labels


__all__ = ["MAX_SMOOTH_PASSES", "PARAM_KEYS", "REQUIRED_COLUMNS", "baseline_labels",
           "half_blocks"]
