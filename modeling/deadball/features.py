"""Dead-ball features (task F3-01), computed in memory from the master table.

``params`` is ``cfg["dead"]["params"]`` of a config read by :func:`modeling.config.load_config`.

Everything is computed per half block (a maximal run of frames with the same non-zero ``half``):
no window, run or neighbour search crosses the limit of a half. NaN means "no evidence" and is
never replaced by 0.

Columns (same index as the master):
    db_gap_kind       int8; kind of the long ball gap that contains the frame (see below)
    db_still_frac     float32; (n_still_home + n_still_away) / (n_tracked_home + n_tracked_away),
                      NaN if the denominator is 0 or NaN, then a centred moving nanmean of
                      ``smooth_frames`` frames (NaN if the whole window is NaN)
    db_mean_speed     float32; nanmean(mean_speed_home, mean_speed_away), same moving nanmean
    db_ball_rest      int8 0/1; the frame is in a run of at least ``rest_min_frames`` consecutive
                      frames with ball_observed == 1 and ball_speed < rest_speed_mps (a NaN speed
                      breaks the run)
    db_fast_in_pitch  int8 0/1; ball_observed == 1, ball_speed > fast_speed_mps, out_of_pitch == 0
    db_sp_prox        float32; exp(-frames_to_next_sp / sp_prox_tau_frames); NaN if that is NaN
    db_sp_after       float32; exp(-frames_since_sp_end / sp_after_tau_frames); NaN if that is NaN
    db_goal_window    int8 0/1; next_sp_type == "kick off (after goal)" (lower case, as in the
                      master) and frames_to_next_sp < goal_window_frames

Where ``half == 0`` the float columns are NaN and the integer columns are 0.

``db_gap_kind``. A long gap is a maximal run of frames, inside one half, with
``ball_observed == 0`` and ``gap_length > long_gap_frames``. Every frame of the run gets the same
kind; frames outside a long gap get ``GAP_NONE``:

* ``GAP_EXIT`` (2), the ball left the pitch, if (a) or (b) holds:
  (a) the last observed frame before the run has ``out_of_pitch == 1``, or is closer than
      ``exit_line_m`` to a line (``d_touchline`` or ``d_goalline``) while moving towards that
      line faster than ``exit_speed_mps`` (velocity component normal to the line);
  (b) the first observed frame after the run is in a restart position: ``d_touchline <
      restart_line_m`` (throw-in or corner), or inside the goal area (``d_goalline <
      goal_area_depth_m`` and ``|ball_y - pitch_w_m / 2| < goal_area_half_width_m``).
* ``GAP_AIR`` (3), a high ball, if it is not an exit, the ball speed at the last observed frame
  is above ``air_speed_mps`` and the run has at most ``air_max_frames`` frames.
* ``GAP_LONG`` (4), undetermined, otherwise.

The observed frames are the ones adjacent to the run, in the same half. If one side has none
(the run touches the limit of the half), only the other side is used; with neither, the kind is
``GAP_LONG``. Telling a high ball (possession goes on) from a ball out of play (D) is the point of
this column, so an exit always takes precedence over a high ball.

The master does not carry the pitch size, so the pitch width comes from ``pitch_w_m`` and the
side of the nearest line is read from the line distances themselves (see ``_towards_line``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GAP_NONE, GAP_EXIT, GAP_AIR, GAP_LONG = 0, 2, 3, 4
GOAL_KICKOFF_TYPE = "kick off (after goal)"

REQUIRED_COLUMNS = (
    "half", "ball_x", "ball_y", "ball_vx", "ball_vy", "ball_speed", "ball_observed",
    "gap_length", "out_of_pitch", "d_touchline", "d_goalline",
    "n_still_home", "n_still_away", "n_tracked_home", "n_tracked_away",
    "mean_speed_home", "mean_speed_away",
    "frames_to_next_sp", "frames_since_sp_end", "next_sp_type",
)
DEADBALL_COLUMNS = ("db_gap_kind", "db_still_frac", "db_mean_speed", "db_ball_rest",
                    "db_fast_in_pitch", "db_sp_prox", "db_sp_after", "db_goal_window")


def _f64(master: pd.DataFrame, name: str) -> np.ndarray:
    return master[name].to_numpy(dtype="float64")


def half_blocks(half: np.ndarray) -> list[tuple[int, int]]:
    """``[start, stop)`` of every maximal run of equal, non-zero ``half`` values."""
    n = len(half)
    if n == 0:
        return []
    cuts = np.flatnonzero(half[1:] != half[:-1]) + 1
    starts = np.r_[0, cuts]
    stops = np.r_[cuts, n]
    return [(int(a), int(b)) for a, b in zip(starts, stops) if half[a] != 0]


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``[start, stop)`` of every maximal run of True in a 1-D bool array."""
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def _centred_nanmean(values: np.ndarray, window: int) -> np.ndarray:
    """Centred moving nanmean of one half block; NaN where the whole window is NaN.

    The window of frame ``i`` is ``[i - window // 2, i + (window - 1) // 2]`` (symmetric for an
    odd ``window``), cut at the ends of the block.
    """
    n = len(values)
    finite = np.isfinite(values)
    total = np.r_[0.0, np.cumsum(np.where(finite, values, 0.0))]
    count = np.r_[0, np.cumsum(finite)]
    idx = np.arange(n)
    lo = np.maximum(idx - window // 2, 0)
    hi = np.minimum(idx + (window - 1) // 2 + 1, n)
    k = count[hi] - count[lo]
    out = np.full(n, np.nan)
    has = k > 0
    out[has] = (total[hi] - total[lo])[has] / k[has]
    return out


def _towards_line(coord: float, dist: float, velocity: float, pitch_w_m: float) -> float:
    """Velocity component towards the nearest line of one axis (positive = approaching it).

    ``dist`` is ``min(coord, size - coord)``. On the low side ``coord - dist`` is 0; on the high
    side it is ``size - 2 * dist``, which is larger than half the pitch width for either axis
    whenever the ball is near a line. This gives the side without knowing the pitch length.
    """
    low_side = 2.0 * (coord - dist) < pitch_w_m
    return -velocity if low_side else velocity


def _classify_gap(cols: dict, before: int | None, after: int | None, length: int,
                  params: dict) -> int:
    """Kind of one long gap from its adjacent observed frames (``None`` = no such frame)."""
    pitch_w = params["pitch_w_m"]
    if before is not None:
        if cols["out_of_pitch"][before] == 1:
            return GAP_EXIT
        for coord, dist, vel in (("ball_y", "d_touchline", "ball_vy"),
                                 ("ball_x", "d_goalline", "ball_vx")):
            towards = _towards_line(cols[coord][before], cols[dist][before], cols[vel][before],
                                    pitch_w)
            # A NaN distance or velocity compares False: no evidence, no exit.
            if cols[dist][before] < params["exit_line_m"] and towards > params["exit_speed_mps"]:
                return GAP_EXIT
    if after is not None:
        if cols["d_touchline"][after] < params["restart_line_m"]:
            return GAP_EXIT
        in_goal_area = (cols["d_goalline"][after] < params["goal_area_depth_m"]
                        and abs(cols["ball_y"][after] - pitch_w / 2.0)
                        < params["goal_area_half_width_m"])
        if in_goal_area:
            return GAP_EXIT
    if (before is not None and cols["ball_speed"][before] > params["air_speed_mps"]
            and length <= params["air_max_frames"]):
        return GAP_AIR
    return GAP_LONG


def _gap_kind(cols: dict, observed: np.ndarray, blocks, params: dict) -> np.ndarray:
    kind = np.zeros(len(observed), dtype=np.int8)
    long_gap = ~observed & (cols["gap_length"] > params["long_gap_frames"])   # NaN -> False
    for b0, b1 in blocks:
        for r0, r1 in _true_runs(long_gap[b0:b1]):
            start, stop = b0 + r0, b0 + r1
            before = start - 1 if start > b0 and observed[start - 1] else None
            after = stop if stop < b1 and observed[stop] else None
            kind[start:stop] = _classify_gap(cols, before, after, stop - start, params)
    return kind


def build_deadball_features(master: pd.DataFrame, params: dict) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in master.columns]
    if missing:
        raise ValueError("master is missing columns {}".format(missing))

    n = len(master)
    half = master["half"].to_numpy()
    in_half = half != 0
    blocks = half_blocks(half)
    cols = {c: _f64(master, c) for c in REQUIRED_COLUMNS if c not in ("half", "next_sp_type")}
    observed = cols["ball_observed"] == 1
    speed = cols["ball_speed"]

    gap_kind = _gap_kind(cols, observed, blocks, params)

    with np.errstate(divide="ignore", invalid="ignore"):
        n_tracked = cols["n_tracked_home"] + cols["n_tracked_away"]
        still_raw = (cols["n_still_home"] + cols["n_still_away"]) / n_tracked
    still_raw = np.where(n_tracked > 0, still_raw, np.nan)                    # 0 or NaN -> NaN

    sp_home, sp_away = cols["mean_speed_home"], cols["mean_speed_away"]
    ok_home, ok_away = np.isfinite(sp_home), np.isfinite(sp_away)
    n_ok = ok_home.astype("float64") + ok_away
    speed_sum = np.where(ok_home, sp_home, 0.0) + np.where(ok_away, sp_away, 0.0)
    speed_raw = np.full(n, np.nan)
    speed_raw[n_ok > 0] = speed_sum[n_ok > 0] / n_ok[n_ok > 0]

    still_frac = np.full(n, np.nan)
    mean_speed = np.full(n, np.nan)
    ball_rest = np.zeros(n, dtype=np.int8)
    slow = observed & (speed < params["rest_speed_mps"])                      # NaN -> False
    for b0, b1 in blocks:
        still_frac[b0:b1] = _centred_nanmean(still_raw[b0:b1], params["smooth_frames"])
        mean_speed[b0:b1] = _centred_nanmean(speed_raw[b0:b1], params["smooth_frames"])
        for r0, r1 in _true_runs(slow[b0:b1]):
            if r1 - r0 >= params["rest_min_frames"]:
                ball_rest[b0 + r0:b0 + r1] = 1

    fast_in_pitch = (observed & (speed > params["fast_speed_mps"])
                     & (cols["out_of_pitch"] == 0) & in_half)

    to_next = np.where(in_half, cols["frames_to_next_sp"], np.nan)
    since_end = np.where(in_half, cols["frames_since_sp_end"], np.nan)
    sp_prox = np.exp(-to_next / params["sp_prox_tau_frames"])                 # NaN stays NaN
    sp_after = np.exp(-since_end / params["sp_after_tau_frames"])

    next_is_goal = master["next_sp_type"].eq(GOAL_KICKOFF_TYPE).to_numpy(dtype=bool,
                                                                         na_value=False)
    goal_window = next_is_goal & (to_next < params["goal_window_frames"])     # NaN -> False

    f32 = np.float32
    return pd.DataFrame({
        "db_gap_kind": gap_kind,
        "db_still_frac": still_frac.astype(f32),
        "db_mean_speed": mean_speed.astype(f32),
        "db_ball_rest": ball_rest,
        "db_fast_in_pitch": fast_in_pitch.astype(np.int8),
        "db_sp_prox": sp_prox.astype(f32),
        "db_sp_after": sp_after.astype(f32),
        "db_goal_window": goal_window.astype(np.int8),
    }, index=master.index)


__all__ = ["DEADBALL_COLUMNS", "GAP_AIR", "GAP_EXIT", "GAP_LONG", "GAP_NONE",
           "GOAL_KICKOFF_TYPE", "REQUIRED_COLUMNS", "build_deadball_features", "half_blocks"]
