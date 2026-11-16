"""Shared rolling-window features for evidence modeling.

Computed per maximal run of equal, non-zero ``half`` values (never crossing half == 0).
Windows: past [t - W, t], future [t, t + W], and centered [t - W, t + W], where W in {12, 25, 50}.
Uses nanmean / nanmin so missing frames yield NaN explicitly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .collective import _half_runs
from .control import build_control_features, require_columns

WINDOW_SIZES = (12, 25, 50)

def _rolling_window_stats(x: np.ndarray, half: np.ndarray, w: int) -> dict[str, np.ndarray]:
    """Compute past, future, and centered nanmean and nanmin strictly within each half block."""
    n = len(x)
    past_mean = np.full(n, np.nan, dtype=np.float64)
    past_min = np.full(n, np.nan, dtype=np.float64)
    fut_mean = np.full(n, np.nan, dtype=np.float64)
    fut_min = np.full(n, np.nan, dtype=np.float64)
    cent_mean = np.full(n, np.nan, dtype=np.float64)
    cent_min = np.full(n, np.nan, dtype=np.float64)

    for start, stop in _half_runs(half):
        seg = x[start:stop]
        length = len(seg)
        if length == 0:
            continue

        # Use pandas rolling on the segment to respect boundaries cleanly
        s = pd.Series(seg)
        # past: [t - w, t] (min periods 1)
        pm = s.rolling(window=w + 1, min_periods=1, closed="both").mean().to_numpy()
        pmin = s.rolling(window=w + 1, min_periods=1, closed="both").min().to_numpy()

        # future: [t, t + w] -> computed by rolling on reversed series
        s_rev = pd.Series(seg[::-1])
        fm_rev = s_rev.rolling(window=w + 1, min_periods=1, closed="both").mean().to_numpy()
        fmin_rev = s_rev.rolling(window=w + 1, min_periods=1, closed="both").min().to_numpy()
        fm = fm_rev[::-1]
        fmin = fmin_rev[::-1]

        # centered: [t - w, t + w]
        cm = s.rolling(window=2 * w + 1, min_periods=1, center=True, closed="both").mean().to_numpy()
        cmin = s.rolling(window=2 * w + 1, min_periods=1, center=True, closed="both").min().to_numpy()

        past_mean[start:stop] = pm
        past_min[start:stop] = pmin
        fut_mean[start:stop] = fm
        fut_min[start:stop] = fmin
        cent_mean[start:stop] = cm
        cent_min[start:stop] = cmin

    return {
        "past_mean": past_mean,
        "past_min": past_min,
        "fut_mean": fut_mean,
        "fut_min": fut_min,
        "cent_mean": cent_mean,
        "cent_min": cent_min,
    }


def _ball_observed_fraction(observed: np.ndarray, half: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Fraction of frames with observed ball in past [t-W, t] and future [t, t+W]."""
    n = len(observed)
    past_frac = np.full(n, np.nan, dtype=np.float64)
    fut_frac = np.full(n, np.nan, dtype=np.float64)

    obs_f = np.where(np.isfinite(observed), observed, 0.0)
    for start, stop in _half_runs(half):
        seg = obs_f[start:stop]
        s = pd.Series(seg)
        pf = s.rolling(window=w + 1, min_periods=1, closed="both").mean().to_numpy()
        s_rev = pd.Series(seg[::-1])
        ff_rev = s_rev.rolling(window=w + 1, min_periods=1, closed="both").mean().to_numpy()
        ff = ff_rev[::-1]
        past_frac[start:stop] = pf
        fut_frac[start:stop] = ff

    return past_frac, fut_frac


def build_window_features(master: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Compute rolling window features for GBDT and evaluation models."""
    require_columns(master, ("d_diff", "half", "ball_observed"))
    half = master["half"].to_numpy()
    windows = params.get("window_sizes", WINDOW_SIZES)

    d_diff = master["d_diff"].to_numpy(dtype="float64")
    d_home = master["d_near_home"].to_numpy(dtype="float64") if "d_near_home" in master.columns else np.full(len(master), np.nan)
    d_away = master["d_near_away"].to_numpy(dtype="float64") if "d_near_away" in master.columns else np.full(len(master), np.nan)
    p_home = master["pressure_home"].to_numpy(dtype="float64") if "pressure_home" in master.columns else np.zeros(len(master))
    p_away = master["pressure_away"].to_numpy(dtype="float64") if "pressure_away" in master.columns else np.zeros(len(master))
    ball_obs = master["ball_observed"].to_numpy(dtype="float64")

    cols = {}
    for w in windows:
        dh_win = _rolling_window_stats(d_home, half, w)
        da_win = _rolling_window_stats(d_away, half, w)
        cols[f"d_near_home_past_min_{w}"] = dh_win["past_min"].astype(np.float32)
        cols[f"d_near_home_fut_min_{w}"] = dh_win["fut_min"].astype(np.float32)
        cols[f"d_near_away_past_min_{w}"] = da_win["past_min"].astype(np.float32)
        cols[f"d_near_away_fut_min_{w}"] = da_win["fut_min"].astype(np.float32)

        dd_win = _rolling_window_stats(d_diff, half, w)
        cols[f"d_diff_cent_mean_{w}"] = dd_win["cent_mean"].astype(np.float32)

        ph_win = _rolling_window_stats(p_home, half, w)
        pa_win = _rolling_window_stats(p_away, half, w)
        cols[f"pressure_home_cent_mean_{w}"] = ph_win["cent_mean"].astype(np.float32)
        cols[f"pressure_away_cent_mean_{w}"] = pa_win["cent_mean"].astype(np.float32)

        pf, ff = _ball_observed_fraction(ball_obs, half, w)
        cols[f"ball_obs_past_frac_{w}"] = pf.astype(np.float32)
        cols[f"ball_obs_fut_frac_{w}"] = ff.astype(np.float32)

    # Importação local para evitar importação circular
    from .team import build_pass_bridge_features
    bridge_df = build_pass_bridge_features(master, params)
    for c in bridge_df.columns:
        cols[c] = bridge_df[c]

    out = pd.DataFrame(cols, index=master.index)
    out.loc[half == 0, :] = np.nan
    return out

__all__ = ["WINDOW_SIZES", "build_window_features"]