"""Shared rolling-window features for evidence models (F1-07, also used by F3-08).

Computed per maximal run of equal, non-zero ``half`` values, so no window ever crosses a half
boundary or a ``half == 0`` stretch. For a window size ``W`` (frames):

- past ``[t - W, t]`` (``W + 1`` frames, ``t`` included);
- future ``[t, t + W]``;
- centred ``[t - W, t + W]`` (``2 W + 1`` frames).

At the start and end of a half the window keeps only the frames that exist. Statistics skip NaN
(``nanmean`` / ``nanmin`` / ``nanmax``) and are NaN where the window has no finite value. Every
column is NaN on ``half == 0``.

Columns, for each ``W`` in ``window_sizes`` (default :data:`WINDOW_SIZES`):

==================================  ==========  ===========================================
column                              team swap   meaning
==================================  ==========  ===========================================
``d_near_home_past_min_W``          pair        who was close to the ball recently
``d_near_away_past_min_W``          pair
``d_near_home_fut_min_W``           pair        who will be close to the ball soon
``d_near_away_fut_min_W``           pair
``d_diff_cent_mean_W``              odd         centred mean of ``d_diff``
``pressure_home_cent_mean_W``       pair
``pressure_away_cent_mean_W``       pair
``ball_obs_past_frac_W``            even        share of frames with an observed ball
``ball_obs_fut_frac_W``             even
==================================  ==========  ===========================================

"pair" columns exchange with their home/away counterpart under a team swap, "odd" columns flip
sign and "even" columns do not change (:func:`window_feature_roles`).
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .collective import _half_runs
from .control import require_columns

WINDOW_SIZES = (12, 25, 50)
STAT_NAMES = tuple("{}_{}".format(w, s) for w in ("past", "fut", "cent")
                   for s in ("mean", "min", "max"))
REQUIRED_COLUMNS = ("half", "d_near_home", "d_near_away", "d_diff", "pressure_home",
                    "pressure_away", "ball_observed")


def _check_window(w) -> int:
    w_int = int(w)
    if w_int != w or w_int < 1:
        raise ValueError("window sizes must be positive integers, got {!r}".format(w))
    return w_int


def rolling_window_stats(x: np.ndarray, half: np.ndarray, w: int,
                         stats: Sequence[str] = ("past_mean", "past_min", "fut_mean", "fut_min",
                                                 "cent_mean", "cent_min")) -> dict[str, np.ndarray]:
    """NaN-skipping mean/min/max over the past, future and centred windows, inside each half block.

    ``stats`` names are ``<past|fut|cent>_<mean|min|max>``.

    Returns float64 arrays keyed by the names in ``stats``; NaN on ``half == 0`` and where the
    window has no finite value.
    """
    w = _check_window(w)
    unknown = sorted(set(stats) - set(STAT_NAMES))
    if unknown:
        raise ValueError("unknown window statistics {}".format(unknown))
    x = np.asarray(x, dtype=np.float64)
    half = np.asarray(half)
    out = {name: np.full(len(x), np.nan, dtype=np.float64) for name in stats}
    for start, stop in _half_runs(half):
        seg = pd.Series(x[start:stop])
        rev = pd.Series(x[start:stop][::-1])
        past = seg.rolling(window=w + 1, min_periods=1)
        fut = rev.rolling(window=w + 1, min_periods=1)
        cent = seg.rolling(window=2 * w + 1, min_periods=1, center=True)
        compute = {
            "past_mean": lambda: past.mean().to_numpy(),
            "past_min": lambda: past.min().to_numpy(),
            "fut_mean": lambda: fut.mean().to_numpy()[::-1],
            "fut_min": lambda: fut.min().to_numpy()[::-1],
            "cent_mean": lambda: cent.mean().to_numpy(),
            "cent_min": lambda: cent.min().to_numpy(),
            "past_max": lambda: past.max().to_numpy(),
            "fut_max": lambda: fut.max().to_numpy()[::-1],
            "cent_max": lambda: cent.max().to_numpy(),
        }
        for name in stats:
            out[name][start:stop] = compute[name]()
    return out


def window_sizes_from(params: Mapping) -> tuple[int, ...]:
    """``params["window_sizes"]`` (validated) or :data:`WINDOW_SIZES`."""
    sizes = params.get("window_sizes", WINDOW_SIZES)
    sizes = tuple(_check_window(w) for w in sizes)
    if not sizes or len(set(sizes)) != len(sizes):
        raise ValueError("window_sizes must be a non-empty list of distinct sizes")
    return sizes


def window_feature_roles(window_sizes: Sequence[int] = WINDOW_SIZES) -> dict:
    """Team-swap roles of the window columns: ``{"pairs": [(home, away)], "odd": [], "even": []}``."""
    pairs, odd, even = [], [], []
    for w in window_sizes:
        pairs.append(("d_near_home_past_min_{}".format(w), "d_near_away_past_min_{}".format(w)))
        pairs.append(("d_near_home_fut_min_{}".format(w), "d_near_away_fut_min_{}".format(w)))
        odd.append("d_diff_cent_mean_{}".format(w))
        pairs.append(("pressure_home_cent_mean_{}".format(w),
                      "pressure_away_cent_mean_{}".format(w)))
        even.extend(["ball_obs_past_frac_{}".format(w), "ball_obs_fut_frac_{}".format(w)])
    return {"pairs": pairs, "odd": odd, "even": even}


def build_window_features(master: pd.DataFrame, params: Mapping) -> pd.DataFrame:
    """Window features (float64, same index as ``master``), see the module docstring."""
    require_columns(master, REQUIRED_COLUMNS)
    half = master["half"].to_numpy()
    sizes = window_sizes_from(params)

    def col(name):
        return master[name].to_numpy(dtype=np.float64)

    ball_obs = col("ball_observed")
    ball_obs = np.where(np.isfinite(ball_obs), (ball_obs > 0).astype(np.float64), np.nan)

    cols: dict[str, np.ndarray] = {}
    for w in sizes:
        for side in ("home", "away"):
            d = rolling_window_stats(col("d_near_" + side), half, w, ("past_min", "fut_min"))
            cols["d_near_{}_past_min_{}".format(side, w)] = d["past_min"]
            cols["d_near_{}_fut_min_{}".format(side, w)] = d["fut_min"]
        cols["d_diff_cent_mean_{}".format(w)] = rolling_window_stats(
            col("d_diff"), half, w, ("cent_mean",))["cent_mean"]
        for side in ("home", "away"):
            cols["pressure_{}_cent_mean_{}".format(side, w)] = rolling_window_stats(
                col("pressure_" + side), half, w, ("cent_mean",))["cent_mean"]
        b = rolling_window_stats(ball_obs, half, w, ("past_mean", "fut_mean"))
        cols["ball_obs_past_frac_{}".format(w)] = b["past_mean"]
        cols["ball_obs_fut_frac_{}".format(w)] = b["fut_mean"]

    out = pd.DataFrame(cols, index=master.index)
    out.loc[half == 0, :] = np.nan
    return out


__all__ = ["REQUIRED_COLUMNS", "STAT_NAMES", "WINDOW_SIZES", "build_window_features", "rolling_window_stats",
           "window_feature_roles", "window_sizes_from"]
