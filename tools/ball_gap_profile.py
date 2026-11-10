"""Build a local histogram of ball gap lengths from the configured BAR-ATH master."""
from __future__ import annotations

import json

import numpy as np

from modeling import io, paths


def main() -> int:
    master = io.load_master()
    half = master["half"].to_numpy()
    x = master["ball_x"].to_numpy(dtype=np.float64)
    y = master["ball_y"].to_numpy(dtype=np.float64)
    gap_lengths = []
    for period in (1, 2):
        indices = np.flatnonzero(half == period)
        if indices.size == 0:
            continue
        missing = ~(np.isfinite(x[indices]) & np.isfinite(y[indices]))
        changes = np.diff(np.r_[False, missing, False].astype(np.int8))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        gap_lengths.extend((ends - starts).tolist())
    lengths, counts = np.unique(np.asarray(gap_lengths, dtype=np.int64), return_counts=True)
    output = paths.processed_dir() / "ball_gap_profile.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"lengths": lengths.tolist(), "counts": counts.tolist()},
                                 separators=(",", ":")) + "\n", encoding="utf-8")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
