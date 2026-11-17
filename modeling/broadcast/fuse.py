"""Fuse the broadcast-camera ball into the tactical ball series (F4-06).

The F4-05 alignment gives, for every broadcast frame, the tactical frame it shows (``-1`` when
there is none) and the pitch orientation that makes the two cameras agree. This module uses that
mapping to fill tactical frames where the ball is not observed.

Rules:
- An observed tactical ball position (finite x and y) is never overwritten.
- Only tactical frames that a broadcast frame maps to can be filled, and only when the broadcast
  ball is observed there.
- Filled frames get speed = NaN; observed frames keep their original third channel. Speed is
  recomputed downstream, after the jump filter.
- Nothing crosses a half boundary: F4-05 guarantees every mapped block stays inside one tactical
  half, and this module performs only per-frame lookups, with no windows or interpolation.
- NaN keeps the meaning "no evidence"; nothing is imputed with zeros.
"""
from __future__ import annotations

import numpy as np

from modeling.broadcast.align import IDENTITY, ROT180, Mapping

BALL_CHANNELS = 3  # x, y, speed


def orient_broadcast_ball(broadcast_ball, orientation, extent, valid=None) -> np.ndarray:
    """Broadcast ball positions in the orientation chosen by F4-05.

    ``broadcast_ball`` is (T_b, >=2), unoriented. ``extent`` is (length, width) in the SAME units as
    the ball arrays (metres for ``Tracking.ball``). ``valid`` (bool[T_b], the broadcast homography
    mask) sets the ball to NaN on frames whose homography is invalid. Returns float64 (T_b, 2).
    """
    ball = np.array(np.asarray(broadcast_ball)[:, :2], dtype=np.float64, copy=True)
    if orientation == ROT180:
        ball = np.asarray(extent, dtype=np.float64) - ball
    elif orientation != IDENTITY:
        raise ValueError("unknown orientation {!r}".format(orientation))
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != (ball.shape[0],):
            raise ValueError("valid mask has shape {}, expected ({},)".format(
                valid.shape, ball.shape[0]))
        ball[~valid] = np.nan
    return ball


def fused_ball(tactical_ball, broadcast_ball, mapping, extent=None) -> np.ndarray:
    """Tactical ball (T, 3) with missing positions filled from the broadcast ball.

    Args:
        tactical_ball: (T, >=3) array with columns x, y, speed; NaN = not observed.
        broadcast_ball: (T_b, >=2) broadcast ball x, y, in the same units as ``tactical_ball``.
            If ``mapping`` is a ``Mapping``, the ball is expected UNORIENTED and is oriented here
            (rot180 needs ``extent``). If ``mapping`` is a raw int array, the ball is taken as
            already oriented. Frames with an invalid broadcast homography must already be NaN
            (see ``orient_broadcast_ball``).
        mapping: ``Mapping`` from F4-05, or an int array of length T_b with the tactical frame
            for each broadcast frame and a negative value where there is none.
        extent: (length, width) in the ball's units; only needed for a rot180 ``Mapping``.

    Returns:
        float64 (T, 3). Tactical observations are copied untouched; filled frames have
        x, y from the broadcast and speed = NaN.
    """
    tactical = np.asarray(tactical_ball)
    broadcast = np.asarray(broadcast_ball)
    if tactical.ndim != 2 or tactical.shape[1] < BALL_CHANNELS:
        raise ValueError("tactical_ball must have shape (T, >=3), got {}".format(tactical.shape))
    if broadcast.ndim != 2 or broadcast.shape[1] < 2:
        raise ValueError("broadcast_ball must have shape (T_b, >=2), got {}".format(broadcast.shape))

    if isinstance(mapping, Mapping):
        frame_map = np.asarray(mapping.tactical_frame)
        if mapping.orientation == ROT180 and extent is None:
            raise ValueError("a rot180 mapping needs the pitch extent to orient the ball")
        b_xy = orient_broadcast_ball(broadcast, mapping.orientation, extent)
    else:
        frame_map = np.asarray(mapping)
        b_xy = np.asarray(broadcast[:, :2], dtype=np.float64)
    if frame_map.ndim != 1 or frame_map.shape[0] != broadcast.shape[0]:
        raise ValueError("mapping has shape {}, expected ({},)".format(
            frame_map.shape, broadcast.shape[0]))

    out = np.array(tactical[:, :BALL_CHANNELS], dtype=np.float64, copy=True)
    n_tactical = out.shape[0]

    tactical_missing = ~np.isfinite(out[:, 0]) | ~np.isfinite(out[:, 1])
    mapped = (frame_map >= 0) & (frame_map < n_tactical)
    broadcast_seen = np.isfinite(b_xy[:, 0]) & np.isfinite(b_xy[:, 1])

    source = np.flatnonzero(mapped & broadcast_seen)          # broadcast frames, ascending
    target = frame_map[source].astype(np.int64)
    fillable = tactical_missing[target]                       # never overwrite an observation
    source, target = source[fillable], target[fillable]
    if source.size == 0:
        return out

    # Several broadcast frames may map to one tactical frame: the lowest broadcast index wins.
    order = np.argsort(target, kind="mergesort")
    source, target = source[order], target[order]
    first = np.ones(target.size, dtype=bool)
    first[1:] = target[1:] != target[:-1]
    source, target = source[first], target[first]

    out[target, 0] = b_xy[source, 0]
    out[target, 1] = b_xy[source, 1]
    out[target, 2] = np.nan
    return out


__all__ = ["fused_ball", "orient_broadcast_ball"]