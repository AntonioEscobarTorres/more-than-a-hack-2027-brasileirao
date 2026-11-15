<<<<<<< HEAD
"""Build a local histogram of ball gap lengths from the configured BAR-ATH master."""
=======
"""Build a local histogram of ball gap lengths from the configured BAR-ATH master.

A gap is a run of frames where the tracker reported no ball at all. ``ball_x`` cannot be used:
preprocessing interpolates short gaps (MAX_GAP_FRAMES) into it, which would erase every short gap
from the histogram. A frame removed by the jump filter did have a (bad) ball, so it is not a gap;
degrade() models those as phantoms. Prints only OK (local run, never shown to agents).
"""
>>>>>>> main
from __future__ import annotations

import json

import numpy as np

from modeling import io, paths


<<<<<<< HEAD
def main() -> int:
    master = io.load_master()
    half = master["half"].to_numpy()
    x = master["ball_x"].to_numpy(dtype=np.float64)
    y = master["ball_y"].to_numpy(dtype=np.float64)
    gap_lengths = []
=======
def gap_mask(master) -> np.ndarray:
    """True where the raw tracking had no ball (neither observed nor removed as a jump)."""
    observed = master["ball_observed"].to_numpy() == 1
    jump_removed = master["ball_jump_removed"].to_numpy() == 1
    return ~(observed | jump_removed)


def gap_lengths_by_half(master) -> list[int]:
    """Lengths of the raw ball gaps inside each half (a gap never crosses halftime)."""
    half = master["half"].to_numpy()
    raw_missing = gap_mask(master)
    lengths = []
>>>>>>> main
    for period in (1, 2):
        indices = np.flatnonzero(half == period)
        if indices.size == 0:
            continue
<<<<<<< HEAD
        missing = ~(np.isfinite(x[indices]) & np.isfinite(y[indices]))
        changes = np.diff(np.r_[False, missing, False].astype(np.int8))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        gap_lengths.extend((ends - starts).tolist())
=======
        missing = raw_missing[indices]
        changes = np.diff(np.r_[False, missing, False].astype(np.int8))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        lengths.extend((ends - starts).tolist())
    return lengths


def main() -> int:
    master = io.load_master()
    gap_lengths = gap_lengths_by_half(master)
>>>>>>> main
    lengths, counts = np.unique(np.asarray(gap_lengths, dtype=np.int64), return_counts=True)
    output = paths.processed_dir() / "ball_gap_profile.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"lengths": lengths.tolist(), "counts": counts.tolist()},
                                 separators=(",", ":")) + "\n", encoding="utf-8")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
