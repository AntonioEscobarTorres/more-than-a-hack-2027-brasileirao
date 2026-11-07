"""Collective evidence for ball gaps (task F1-04).

``params`` is ``cfg["team"]["params"]``; this module reads ``params["collective"]["win_frames"]``.

Computed per half: the centred derivative uses frames t - win_frames and t + win_frames, and is
NaN when either one falls outside the contiguous block of the same ``half`` (or on half == 0).
The master must be in frame order, one row per frame (SCHEMA.md).

Columns (same index as the master; float32; NaN = missing data or window outside the half):
    ev_adv_home     d(centroid_home_x)/dt in m/s
    ev_adv_away     -d(centroid_away_x)/dt in m/s (away attacks towards decreasing x)
    ev_coll_adv     ev_adv_home - ev_adv_away
    ev_coll_spread  spread_home_x * spread_home_y - spread_away_x * spread_away_y
    ev_coll_speed   mean_speed_home - mean_speed_away

The three ev_coll_* columns are antisymmetric: swapping the teams flips their sign.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import games

from .control import require_columns

REQUIRED_COLUMNS = (
    "half", "centroid_home_x", "centroid_away_x", "spread_home_x", "spread_home_y",
    "spread_away_x", "spread_away_y", "mean_speed_home", "mean_speed_away",
)
COLLECTIVE_COLUMNS = ("ev_adv_home", "ev_adv_away", "ev_coll_adv", "ev_coll_spread",
                      "ev_coll_speed")


def _half_runs(half: np.ndarray):
    """Yield (start, stop) of each maximal run of equal, non-zero ``half`` values."""
    if len(half) == 0:
        return
    edges = np.flatnonzero(half[1:] != half[:-1]) + 1
    starts = np.r_[0, edges]
    stops = np.r_[edges, len(half)]
    for start, stop in zip(starts, stops):
        if half[start] != 0:
            yield int(start), int(stop)


def centered_derivative(x: np.ndarray, half: np.ndarray, win_frames: int, fps: float) -> np.ndarray:
    """(x[t + w] - x[t - w]) / (2 w / fps) inside each half; NaN where the window leaves it."""
    out = np.full(len(x), np.nan)
    dt = 2.0 * win_frames / fps
    for start, stop in _half_runs(half):
        if stop - start > 2 * win_frames:
            seg = x[start:stop]
            out[start + win_frames:stop - win_frames] = (seg[2 * win_frames:] - seg[:-2 * win_frames]) / dt
    return out


def build_collective_features(master: pd.DataFrame, params: dict) -> pd.DataFrame:
    require_columns(master, REQUIRED_COLUMNS)
    win_frames = params["collective"]["win_frames"]
    fps = float(games.GAMES[games.GAME]["fps"])

    def col(name):
        return master[name].to_numpy(dtype="float64")

    half = master["half"].to_numpy()
    adv_home = centered_derivative(col("centroid_home_x"), half, win_frames, fps)
    adv_away = -centered_derivative(col("centroid_away_x"), half, win_frames, fps)
    spread = col("spread_home_x") * col("spread_home_y") - col("spread_away_x") * col("spread_away_y")
    speed = col("mean_speed_home") - col("mean_speed_away")
    outside = half == 0                       # no tracking-derived evidence outside the halves

    f32 = np.float32
    return pd.DataFrame({
        "ev_adv_home": adv_home.astype(f32),
        "ev_adv_away": adv_away.astype(f32),
        "ev_coll_adv": (adv_home - adv_away).astype(f32),
        "ev_coll_spread": np.where(outside, np.nan, spread).astype(f32),
        "ev_coll_speed": np.where(outside, np.nan, speed).astype(f32),
    }, index=master.index)


__all__ = ["COLLECTIVE_COLUMNS", "REQUIRED_COLUMNS", "build_collective_features",
           "centered_derivative"]
