"""Per-half possession post-processing (task F2-04).

``postprocess(labels, master, team_logit, dead_logit, force_d, cfg_post, *, holdout=None)``
applies optional, config-gated rules in a fixed order, then the final imposition
(``force_d`` → D, ``half == 0`` → X). Returns ``(labels, stats)``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import yaml

from modeling import paths
from modeling.contracts import LABEL_DTYPE
from modeling.pipeline import final_imposition

# Transition type keys used in ``configs/calib/boundary.yaml`` (F2-05).
BOUNDARY_TYPES = ("H>A", "A>H", "H>D", "A>D", "D>H", "D>A")
_MAX_MIN_SPELL_PASSES = 20


def _segments(labels: np.ndarray) -> list[tuple[str, int, int]]:
    """Contiguous ``(state, start, stop)`` runs (stop exclusive)."""
    n = int(labels.shape[0])
    if n == 0:
        return []
    out: list[tuple[str, int, int]] = []
    start = 0
    for i in range(1, n):
        if labels[i] != labels[start]:
            out.append((str(labels[start]), start, i))
            start = i
    out.append((str(labels[start]), start, n))
    return out


def _half_slices(half: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive-exclusive slices for contiguous runs with ``half in {1, 2}``."""
    slices: list[tuple[int, int]] = []
    n = half.shape[0]
    i = 0
    while i < n:
        h = int(half[i])
        j = i + 1
        while j < n and int(half[j]) == h:
            j += 1
        if h in (1, 2):
            slices.append((i, j))
        i = j
    return slices


def _count_changes(before: np.ndarray, after: np.ndarray) -> int:
    return int(np.count_nonzero(before != after))


def _zero_run_covering(logit: np.ndarray, boundary: int) -> tuple[int, int] | None:
    """Maximal ``team_logit == 0`` run that contains ``boundary`` or has it on an edge."""
    n = logit.shape[0]
    if boundary < 0 or boundary > n:
        return None
    # Inside: logit[boundary] == 0. On left edge: logit[boundary - 1] == 0 and
    # boundary is the first frame of the right segment (so the run ends at boundary - 1).
    if boundary < n and float(logit[boundary]) == 0.0:
        seed = boundary
    elif boundary > 0 and float(logit[boundary - 1]) == 0.0:
        seed = boundary - 1
    else:
        return None
    lo = seed
    hi = seed
    while lo > 0 and float(logit[lo - 1]) == 0.0:
        lo -= 1
    while hi + 1 < n and float(logit[hi + 1]) == 0.0:
        hi += 1
    return lo, hi


def _cuts(block: np.ndarray) -> tuple[list[str], list[int]]:
    """Segment labels and the internal cut positions (first frame of each later segment)."""
    segs = _segments(block)
    return [lab for lab, _a, _b in segs], [b0 for _lab, b0, _b1 in segs[1:]]


def _rebuild(block: np.ndarray, labs: list[str], cuts: list[int]) -> None:
    """Paint ``labs`` back into ``block`` between the (strictly increasing) ``cuts``."""
    edges = [0] + list(cuts) + [block.shape[0]]
    for k, lab in enumerate(labs):
        block[edges[k]:edges[k + 1]] = lab


def _clamp_cut(target: int, k: int, new_cuts: list[int], cuts: list[int], n: int) -> int:
    """Keep cut ``k`` strictly between its neighbours so every segment keeps >= 1 frame.

    Cuts are moved left to right: the left neighbour is already at its new place
    (``new_cuts[k - 1]``), the right neighbour still at its original one (``cuts[k + 1]``).
    Because each later cut is clamped against the earlier ones' new places, two neighbouring
    boundaries are never skipped or crossed, whatever the order of the moves.
    """
    lo = (new_cuts[k - 1] if k > 0 else 0) + 1
    hi = (cuts[k + 1] if k + 1 < len(cuts) else n) - 1
    return max(lo, min(hi, target))


def _apply_midpoint_neutral_switch(
    labels: np.ndarray,
    team_logit: np.ndarray,
    start: int,
    stop: int,
) -> int:
    """Move direct H↔A boundaries to the midpoint of covering neutral logit runs."""
    block = labels[start:stop].copy()
    logit = team_logit[start:stop]
    before = block.copy()
    labs, cuts = _cuts(block)
    new_cuts: list[int] = []
    for k, cut in enumerate(cuts):
        target = cut
        if {labs[k], labs[k + 1]} == {"H", "A"}:
            run = _zero_run_covering(logit, cut)
            if run is not None:
                lo, hi = run
                target = lo + (hi - lo + 1) // 2
        new_cuts.append(_clamp_cut(target, k, new_cuts, cuts, block.shape[0]))
    _rebuild(block, labs, new_cuts)
    labels[start:stop] = block
    return _count_changes(before, block)


def _load_boundary_shifts(path: str | Path, holdout: str | None) -> dict[str, int]:
    """Read median latencies and return per-type displacements ``-(median)``."""
    path = Path(path)
    if not path.is_absolute():
        path = paths.repo_root() / path
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("boundary_shift_file must be a mapping, got {}".format(type(raw).__name__))
    if holdout is not None:
        folds = raw.get("folds")
        if not isinstance(folds, dict) or holdout not in folds:
            raise ValueError("boundary_shift_file has no folds[{!r}]".format(holdout))
        table = folds[holdout]
    else:
        if "all" not in raw:
            raise ValueError("boundary_shift_file missing top-level key 'all'")
        table = raw["all"]
    if not isinstance(table, dict):
        raise ValueError("boundary latency table must be a mapping")
    shifts = {}
    for key in BOUNDARY_TYPES:
        median = int(table.get(key, 0))
        shifts[key] = -median
    return shifts


def _apply_boundary_shift(
    labels: np.ndarray,
    force_d: np.ndarray,
    start: int,
    stop: int,
    shifts: Mapping[str, int],
) -> int:
    """Shift each internal boundary by the typed displacement, clamped to neighbours.

    Every boundary of the initial segmentation is shifted exactly once (left to right), so
    consecutive boundaries (``H|A|D``) are all corrected. Boundaries adjacent to ``force_d``
    stay put, and a moved boundary never sweeps over a ``force_d`` frame.
    """
    block = labels[start:stop].copy()
    force = force_d[start:stop]
    n = block.shape[0]
    before = block.copy()
    labs, cuts = _cuts(block)
    new_cuts: list[int] = []
    for k, cut in enumerate(cuts):
        delta = int(shifts.get("{}>{}".format(labs[k], labs[k + 1]), 0))
        if delta == 0 or force[cut] or force[cut - 1]:
            new_cuts.append(_clamp_cut(cut, k, new_cuts, cuts, n))
            continue
        target = _clamp_cut(cut + delta, k, new_cuts, cuts, n)
        if target < cut:
            region = force[target:cut]
            if region.any():
                target = target + int(np.flatnonzero(region)[-1]) + 1
        elif target > cut:
            region = force[cut:target]
            if region.any():
                target = cut + int(np.flatnonzero(region)[0])
        new_cuts.append(_clamp_cut(target, k, new_cuts, cuts, n))
    _rebuild(block, labs, new_cuts)
    labels[start:stop] = block
    return _count_changes(before, block)


def _apply_d_bridge(
    labels: np.ndarray,
    force_d: np.ndarray,
    start: int,
    stop: int,
    max_len: int,
) -> int:
    """Short D between the same team becomes that team (never on force_d frames)."""
    if max_len <= 0:
        return 0
    block = labels[start:stop].copy()
    force = force_d[start:stop]
    before = block.copy()
    segs = _segments(block)
    for i in range(1, len(segs) - 1):
        lab, a, b = segs[i]
        if lab != "D" or (b - a) >= max_len:
            continue
        prev_lab = segs[i - 1][0]
        next_lab = segs[i + 1][0]
        if prev_lab != next_lab or prev_lab not in ("H", "A"):
            continue
        if force[a:b].any():
            continue
        block[a:b] = prev_lab
    labels[start:stop] = block
    return _count_changes(before, block)


def _apply_d_cross(
    labels: np.ndarray,
    force_d: np.ndarray,
    is_set_piece: np.ndarray,
    start: int,
    stop: int,
    max_len: int,
) -> int:
    """Short D between different teams is split at the midpoint (H|A or A|H)."""
    if max_len <= 0:
        return 0
    block = labels[start:stop].copy()
    force = force_d[start:stop]
    sp = is_set_piece[start:stop]
    before = block.copy()
    segs = _segments(block)
    for i in range(1, len(segs) - 1):
        lab, a, b = segs[i]
        length = b - a
        if lab != "D" or length >= max_len:
            continue
        prev_lab = segs[i - 1][0]
        next_lab = segs[i + 1][0]
        if {prev_lab, next_lab} != {"H", "A"} or prev_lab == next_lab:
            continue
        if force[a:b].any() or sp[a:b].any():
            continue
        # Odd length: middle frame goes to the previous team → ceil half for prev.
        mid = a + (length + 1) // 2
        block[a:mid] = prev_lab
        block[mid:b] = next_lab
    labels[start:stop] = block
    return _count_changes(before, block)


def _apply_min_spell(
    labels: np.ndarray,
    start: int,
    stop: int,
    min_len: int,
) -> int:
    """Absorb short H/A spells flanked by the same neighbour label.

    One pass scans the whole half left to right and absorbs every eligible spell; after an
    absorption the next segment (now merged into the neighbour) is skipped. Passes repeat until
    nothing changes, at most ``_MAX_MIN_SPELL_PASSES`` times.
    """
    if min_len <= 0:
        return 0
    block = labels[start:stop].copy()
    original = block.copy()
    for _ in range(_MAX_MIN_SPELL_PASSES):
        segs = _segments(block)
        changed = False
        i = 1
        while i < len(segs) - 1:
            lab, a, b = segs[i]
            prev_lab, next_lab = segs[i - 1][0], segs[i + 1][0]
            if lab in ("H", "A") and (b - a) < min_len and prev_lab == next_lab:
                block[a:b] = prev_lab
                changed = True
                i += 2  # segs[i + 1] is now part of the merged neighbour
            else:
                i += 1
        if not changed:
            break
    labels[start:stop] = block
    return _count_changes(original, block)


def postprocess(
    labels,
    master,
    team_logit,
    dead_logit,
    force_d,
    cfg_post: Mapping,
    *,
    holdout: str | None = None,
):
    """Apply config-gated post-processing rules per half, then the final imposition.

    Parameters
    ----------
    labels:
        Predicted labels ``<U1`` of length ``n``.
    master:
        Master table with at least ``half``; ``is_set_piece`` required when
        ``d_cross_frames > 0``.
    team_logit, dead_logit:
        float arrays of length ``n`` (``dead_logit`` reserved for future rules).
    force_d:
        bool array of length ``n``.
    cfg_post:
        ``cfg["postprocess"]`` mapping.
    holdout:
        External match id for fold-specific boundary shifts, or ``None`` for ``all``.

    Returns
    -------
    labels, stats
        New label array and a dict of change counts per rule (plus ``final_imposition``).
    """
    del dead_logit  # reserved; current rules do not use it
    labels = np.asarray(labels)
    half = np.asarray(master["half"])
    team_logit = np.asarray(team_logit)
    force = np.asarray(force_d, dtype=bool)
    n = labels.shape[0]
    if half.shape != (n,) or team_logit.shape != (n,) or force.shape != (n,):
        raise ValueError("labels, half, team_logit and force_d must share length {}".format(n))

    out = np.array(labels, dtype=LABEL_DTYPE, copy=True)
    stats = {
        "midpoint_neutral_switch": 0,
        "boundary_shift_file": 0,
        "d_bridge_frames": 0,
        "d_cross_frames": 0,
        "min_spell_frames": 0,
        "final_imposition": 0,
    }

    shifts = None
    if cfg_post.get("boundary_shift_file") is not None:
        shifts = _load_boundary_shifts(cfg_post["boundary_shift_file"], holdout)

    is_set_piece = None
    if int(cfg_post.get("d_cross_frames", 0)) > 0:
        if "is_set_piece" not in master.columns:
            raise ValueError("d_cross_frames > 0 requires master column 'is_set_piece'")
        is_set_piece = np.asarray(master["is_set_piece"]).astype(bool)

    for start, stop in _half_slices(half):
        if cfg_post.get("midpoint_neutral_switch"):
            stats["midpoint_neutral_switch"] += _apply_midpoint_neutral_switch(
                out, team_logit, start, stop)
        if shifts is not None:
            stats["boundary_shift_file"] += _apply_boundary_shift(
                out, force, start, stop, shifts)
        d_bridge = int(cfg_post.get("d_bridge_frames", 0))
        if d_bridge > 0:
            stats["d_bridge_frames"] += _apply_d_bridge(out, force, start, stop, d_bridge)
        d_cross = int(cfg_post.get("d_cross_frames", 0))
        if d_cross > 0:
            stats["d_cross_frames"] += _apply_d_cross(
                out, force, is_set_piece, start, stop, d_cross)
        min_spell = int(cfg_post.get("min_spell_frames", 0))
        if min_spell > 0:
            stats["min_spell_frames"] += _apply_min_spell(out, start, stop, min_spell)

    out, n_final = final_imposition(out, half, force)
    stats["final_imposition"] = n_final
    return out, stats


__all__ = ["BOUNDARY_TYPES", "postprocess"]
