"""Prepare Metrica ATD into frame-aligned, metre-based tracking tables.

The reader streams the raw ATD text file (one frame per line), assigns stable track slots from
the metadata XML, orients each half consistently, and writes ``tracking_clean.parquet`` plus
``slot_teams.parquet``. Coordinates and derived features are indexed by tactical ``video_frame``.

Run from the repository root with ``python -m preprocessing.prep_tracking``. Match assets are
expected under ``data/tactical`` (or ``MTAH_DATA_DIR``); outputs default to ``data/processed``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from games import GAME, GAMES, asset_paths, half_video_blocks
from starter.baseline import team_of_each_track
from .params import (BALL_EPS, BALL_K, BALL_VMAX, DEDUP_DIST, FPS, KICKOFF_FALLBACK, MAX_GAP_FRAMES,
                     MIN_BALL_SPEED_DIR, N_FRAMES, ORIENT_WIN, PITCH_L, PITCH_W, PLAYER_VMAX, STILL_SPEED)
from .schema import N_SLOTS
from .params import SCHEMA_VERSION

FIELD_SEP = ":"
GROUP_SEP = ";"
LOGGER = logging.getLogger(__name__)


def _kickoff_frames(data_dir=None):
    """Load C4 kickoff frames, or use published fallback values with a warning."""
    root = data_dir or os.environ.get("MTAH_DATA_DIR") or os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    path = os.path.join(root, "processed", "kickoffs.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as stream:
            payload = json.load(stream)
        result = {}
        for half in (1, 2):
            value = payload.get(str(half), payload.get(half,
                payload.get(f"kickoff_{half}", payload.get(f"half_{half}"))))
            if value is None and isinstance(payload.get("kickoffs"), dict):
                value = payload["kickoffs"].get(str(half), payload["kickoffs"].get(half))
            if isinstance(value, dict):
                value = next((value[k] for k in ("video_frame", "kickoff_frame", "frame") if k in value), None)
            if value is None:
                raise ValueError(f"{path} must contain a kickoff video frame for half {half}")
            result[half] = int(value)
        LOGGER.info("Using kickoff frames from %s: %s", path, result)
        return result
    LOGGER.warning("%s not found; using published KICKOFF_FALLBACK frames %s", path, KICKOFF_FALLBACK)
    return dict(KICKOFF_FALLBACK)


def read_atd(raw_path, meta_path, n_frames=N_FRAMES):
    """Read raw ATD to normalized arrays; ``frame_id`` 1 maps to ``video_frame`` 0."""
    teams = team_of_each_track(meta_path)
    if len(teams) != N_SLOTS:
        raise ValueError(f"Expected {N_SLOTS} track slots in metadata, got {len(teams)}")
    players = np.full((n_frames, N_SLOTS, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    seen = np.zeros(n_frames, dtype=bool)
    with open(raw_path, encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            parts = line.rstrip("\n").split(FIELD_SEP)
            if len(parts) != 3:
                raise ValueError(f"Malformed ATD line {line_no}: expected 3 colon-separated fields")
            frame = int(parts[0]) - 1
            if frame < 0 or frame >= n_frames or seen[frame]:
                raise ValueError(f"Unexpected or duplicate ATD frame_id {frame + 1} at line {line_no}")
            expected = line_no - 1
            if frame != expected:
                raise ValueError(f"ATD frames out of order at line {line_no}: expected video_frame {expected}, got {frame}")
            groups = parts[1].split(GROUP_SEP)
            if len(groups) != N_SLOTS:
                raise ValueError(f"Frame {frame + 1}: expected {N_SLOTS} slots, found {len(groups)}")
            for slot, group in enumerate(groups):
                values = _triple(group)
                if values is not None:
                    players[frame, slot] = values
            values = _triple(parts[2])
            if values is not None:
                ball[frame] = values
            seen[frame] = True
    if not seen.all():
        missing = np.flatnonzero(~seen)
        raise ValueError(f"ATD has {len(missing)} missing frames; first missing video_frame={missing[0]}")
    return players, ball, teams


def _triple(text):
    if not text or text[0] in "Nn":
        return None
    parts = text.split(",")
    if len(parts) < 3:
        raise ValueError(f"Malformed ATD object group: {text!r}")
    vals = np.asarray([float(v) for v in parts[:3]], dtype=np.float32)
    return vals if np.isfinite(vals[:2]).all() else None


def remove_ball_jump_outliers(x, y, vmax=BALL_VMAX, fps=FPS, eps=BALL_EPS, k=BALL_K, min_chain=None):
    """Keep mutually plausible chains of ball observations (O(n*k)).

    Tracks contiguous smooth runs so interior frames of a phantom burst cannot attach to
    an earlier trajectory that the burst start could not reach, always includes pre-break
    anchors so real play reconnects across phantom bursts longer than ``k`` observations,
    and retains disjoint chains of length >= ``min_chain`` (default ``k``) so a legitimate
    discontinuity (e.g. multi-ball restart) never discards a whole section of a half.
    """
    obs = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
    xo, yo = x.copy(), y.copy()
    m = len(obs)
    if m < 2:
        return xo, yo
    if min_chain is None:
        min_chain = k
    best = np.ones(m, dtype=np.int32)
    prev = np.full(m, -1, dtype=np.int32)
    run_start = np.zeros(m, dtype=np.int32)
    anchors = []  # end indices of previous smooth runs before a jump
    for i in range(1, m):
        dt_prev = (obs[i] - obs[i - 1]) / fps
        step_dist = np.hypot(x[obs[i]] - x[obs[i - 1]], y[obs[i]] - y[obs[i - 1]])
        if step_dist <= vmax * dt_prev + eps:
            run_start[i] = run_start[i - 1]
            # Inside a smooth run, only look back to the start of this run so a long
            # stationary phantom never hooks onto an earlier real segment partway through.
            js = np.arange(max(run_start[i], i - k), i, dtype=np.int32)
        else:
            run_start[i] = i
            anchors.append(i - 1)
            start_j = max(0, i - k)
            js = np.arange(start_j, i, dtype=np.int32)
            if anchors:
                # Include prior run-end anchors so real play bridges across phantom bursts > k.
                prior = np.asarray(anchors[-k:], dtype=np.int32)
                prior = prior[prior < start_j]
                if len(prior):
                    js = np.r_[prior, js]
        dt = (obs[i] - obs[js]) / fps
        dist = np.hypot(x[obs[i]] - x[obs[js]], y[obs[i]] - y[obs[js]])
        valid = js[dist <= vmax * dt + eps]
        if len(valid):
            j = valid[np.argmax(best[valid])]
            best[i], prev[i] = best[j] + 1, j

    keep = np.zeros(len(x), dtype=bool)
    blocked = np.zeros(m, dtype=bool)
    first_chain = True
    while not blocked.all():
        candidates = np.flatnonzero(~blocked)
        i = int(candidates[np.argmax(best[candidates])])
        chain = []
        while i >= 0 and not blocked[i]:
            chain.append(i)
            p = int(prev[i])
            # Never jump across an interval already claimed by a stronger valid chain.
            if p >= 0 and blocked[p:i + 1].any():
                break
            i = p
        if not first_chain and len(chain) < min_chain:
            blocked[chain] = True
            continue
        first_chain = False
        for idx in chain:
            keep[obs[idx]] = True
        blocked[min(chain):max(chain) + 1] = True

    xo[~keep], yo[~keep] = np.nan, np.nan
    return xo, yo


def _gap_features(obs):
    idx = np.arange(len(obs))
    last = np.maximum.accumulate(np.where(obs, idx, -1))
    nxt = np.minimum.accumulate(np.where(obs, idx, len(obs))[::-1])[::-1]
    since = np.where(last >= 0, idx - last, np.nan).astype(np.float32)
    until = np.where(nxt < len(obs), nxt - idx, np.nan).astype(np.float32)
    gap = np.zeros(len(obs), dtype=np.float32)
    changes = np.flatnonzero(np.diff(np.r_[True, obs, True].astype(np.int8)))
    for a, b in zip(changes[::2], changes[1::2]):
        gap[a:b] = b - a
    return gap, since, until


def _fill_short_gaps(values, max_gap):
    series = pd.Series(values, dtype="float64")
    # pandas may expose the interpolated values as a read-only view.
    filled = series.interpolate(method="linear", limit_area="inside").to_numpy(copy=True)
    idx = np.arange(len(values))
    valid = np.isfinite(values)
    if not valid.any():
        return values.copy()
    left = np.maximum.accumulate(np.where(valid, idx, -1))
    right = np.minimum.accumulate(np.where(valid, idx, len(values))[::-1])[::-1]
    gap = right - left - 1
    filled[(gap > max_gap) | (left < 0) | (right >= len(values))] = np.nan
    return filled.astype(np.float32)


def slot_breaks(px, py, fps=FPS, vmax=PLAYER_VMAX):
    """Return break markers and isolated one-frame position spikes."""
    valid = np.isfinite(px) & np.isfinite(py)
    jump = np.hypot(np.diff(px, axis=0), np.diff(py, axis=0)) * fps > vmax
    # A new observation after any missing run is a discontinuity too: never derive
    # velocity across an interval where the track identity/location was unknown.
    reacquired = valid[1:] & ~valid[:-1]
    jump |= reacquired
    broken = np.zeros(px.shape, bool)
    broken[1:] = jump
    skip = np.hypot(px[2:] - px[:-2], py[2:] - py[:-2]) * fps / 2
    isolated = np.zeros(px.shape, bool)
    isolated[1:-1] = jump[:-1] & jump[1:] & (skip <= vmax)
    return broken, isolated


def _binomial3_smooth(seg):
    """Zero-phase [0.25, 0.5, 0.25] smoother with linear boundary reflection (exact on lines)."""
    seg = np.asarray(seg, dtype=np.float64)
    if len(seg) < 3:
        return seg
    padded = np.empty(len(seg) + 2, dtype=np.float64)
    padded[0] = 2.0 * seg[0] - seg[1]
    padded[1:-1] = seg
    padded[-1] = 2.0 * seg[-1] - seg[-2]
    return 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]


def _segment_gradient(values, valid, fps=FPS, smooth=True):
    """Differentiate finite contiguous runs independently; singleton runs have no speed.

    When ``smooth=True``, applies a zero-phase 3-point binomial filter to each contiguous
    run before differentiating, attenuating 25 Hz position jitter without altering stored positions.
    """
    out = np.full(len(values), np.nan, dtype=np.float32)
    edges = np.flatnonzero(np.diff(np.r_[False, valid, False].astype(np.int8)))
    for start, end in zip(edges[::2], edges[1::2]):
        if end - start >= 2:
            seg = _binomial3_smooth(values[start:end]) if smooth else values[start:end]
            out[start:end] = np.gradient(seg) * fps
    return out


def _orient_half(players, ball, team_ids, start, end, kick_frame, pitch):
    """Reflect one half so home attacks toward increasing x; return whether reflected."""
    length, width = pitch
    a, b = max(start, kick_frame), min(end, kick_frame + ORIENT_WIN)
    window = players[a:b, :, 0]
    home = np.asarray(team_ids) == 0
    home_mean = np.nanmean(window[:, home]) if np.isfinite(window[:, home]).any() else np.nan
    if not np.isfinite(home_mean):
        raise ValueError(f"Cannot infer home direction near kickoff frame {kick_frame}")
    # At kickoff the home team is in its own half: low x means it attacks toward high x.
    flip = home_mean >= length / 2
    if flip:
        players[start:end, :, 0] = length - players[start:end, :, 0]
        players[start:end, :, 1] = width - players[start:end, :, 1]
        players[start:end, :, 2] = -players[start:end, :, 2]
        ball[start:end, 0] = length - ball[start:end, 0]
        ball[start:end, 1] = width - ball[start:end, 1]
    return flip


def prepare_tracking(players, ball, slot_team_ids, blocks=None, pitch=(PITCH_L, PITCH_W),
                     n_frames=N_FRAMES, kickoffs=None):
    """Build clean frame table and slot metadata from normalized 0..1 ATD arrays."""
    if blocks is None:
        blocks = half_video_blocks(GAME)
    players = players.astype(np.float32, copy=True)
    ball = ball.astype(np.float32, copy=True)
    length, width = pitch
    players[:, :, 0] *= length
    players[:, :, 1] *= width
    ball[:, 0] *= length
    ball[:, 1] *= width
    if kickoffs is None:
        kickoffs = _kickoff_frames()
    flips = []
    for half, first, last in blocks:
        kick = min(max(int(kickoffs.get(half, first)), first), last)
        flips.append(_orient_half(players, ball, slot_team_ids, first, last + 1, kick, pitch))
    if len(flips) >= 2 and flips[0] == flips[1]:
        raise ValueError(f"Half orientations should oppose each other, got {flips}")

    # Preserve the oriented ATD ball for audit, before filtering/interpolation.
    ball_raw = ball[:, :2].copy()

    t, slots = players.shape[:2]
    frame_data = {"video_frame": np.arange(t, dtype=np.int32)}
    half_col = np.zeros(t, dtype=np.int8)
    for half, first, last in blocks:
        half_col[first:last + 1] = half
    frame_data["half"] = half_col
    frame_data["is_graded"] = half_col > 0
    frame_data["frame_in_half"] = np.where(half_col > 0,
        np.arange(t) - np.where(half_col == 1, blocks[0][1], blocks[-1][1]), np.nan).astype(np.float32)

    # Ball filtering and confidence features are calculated independently within each half.
    bx = np.full(t, np.nan, np.float32); by = bx.copy()
    bvx = bx.copy(); bvy = bx.copy(); bspeed = bx.copy(); bacc = bx.copy(); bdir = bx.copy()
    observed = np.zeros(t, np.int8); jump_removed = np.zeros(t, np.int8); filled_flag = np.zeros(t, np.int8)
    gap = np.full(t, np.nan, np.float32); since = gap.copy(); until = gap.copy()
    for _, first, last in blocks:
        sl = slice(first, last + 1)
        ox, oy = ball[sl, 0].copy(), ball[sl, 1].copy()
        fx, fy = remove_ball_jump_outliers(ox, oy)
        obs = np.isfinite(fx) & np.isfinite(fy)
        observed[sl] = obs
        jump_removed[sl] = np.isfinite(ox) & np.isfinite(oy) & ~obs
        gl, st, un = _gap_features(obs)
        gap[sl], since[sl], until[sl] = gl, st, un
        ix, iy = _fill_short_gaps(fx, MAX_GAP_FRAMES), _fill_short_gaps(fy, MAX_GAP_FRAMES)
        filled_flag[sl] = ~obs & np.isfinite(ix) & np.isfinite(iy)
        # Keep measured/interpolated positions as-is; smooth only inside _segment_gradient,
        # and differentiate strictly over observed runs (masked to NaN where ball_filled == 1).
        bx[sl], by[sl] = ix, iy
        dvx, dvy = _segment_gradient(fx, obs), _segment_gradient(fy, obs)
        bvx[sl], bvy[sl] = dvx, dvy
        speed = np.hypot(dvx, dvy)
        bspeed[sl] = speed
        valid_vel = np.isfinite(dvx) & np.isfinite(dvy)
        ax, ay = _segment_gradient(dvx, valid_vel), _segment_gradient(dvy, valid_vel)
        bacc[sl] = np.hypot(ax, ay)
        dir_change = np.full(len(dvx), np.nan, dtype=np.float32)
        if len(dvx) >= 2:
            fast = (speed[:-1] >= MIN_BALL_SPEED_DIR) & (speed[1:] >= MIN_BALL_SPEED_DIR)
            cross = np.abs(dvx[:-1] * dvy[1:] - dvy[:-1] * dvx[1:])
            dot = dvx[:-1] * dvx[1:] + dvy[:-1] * dvy[1:]
            dir_change[1:] = np.where(fast, np.arctan2(cross, dot), np.nan).astype(np.float32)
        bdir[sl] = dir_change
    frame_data["ball_x"], frame_data["ball_y"] = bx, by
    frame_data["ball_vx"], frame_data["ball_vy"], frame_data["ball_speed"], frame_data["ball_acc"] = bvx, bvy, bspeed, bacc
    frame_data["ball_dir_change"] = bdir
    frame_data["ball_observed"], frame_data["ball_jump_removed"], frame_data["ball_filled"] = observed, jump_removed, filled_flag
    frame_data["gap_length"], frame_data["since_last_obs"], frame_data["until_next_obs"] = gap, since, until
    frame_data["ball_x_raw"], frame_data["ball_y_raw"] = ball_raw[:, 0], ball_raw[:, 1]
    frame_data["ball_x_raw"][half_col == 0] = np.nan
    frame_data["ball_y_raw"][half_col == 0] = np.nan

    px, py = players[:, :, 0], players[:, :, 1]
    valid_halves = np.zeros(t, dtype=bool)
    for _, first, last in blocks:
        valid_halves[first:last + 1] = True
    px[~valid_halves, :], py[~valid_halves, :] = np.nan, np.nan
    breaks = np.zeros(px.shape, dtype=bool)
    # Keep every player operation local to a half, just like the ball pipeline.
    # This makes the boundary rule explicit and robust if blocks ever become adjacent.
    members_by_team = [np.flatnonzero(np.asarray(slot_team_ids) == team) for team in (0, 1)]
    for _, first, last in blocks:
        sl = slice(first, last + 1)
        half_px, half_py = px[sl], py[sl]
        half_breaks, isolated = slot_breaks(half_px, half_py)
        half_px[isolated], half_py[isolated] = np.nan, np.nan
        breaks[sl] = half_breaks
        # Remove duplicates within this half, retaining the lower slot index.
        for members in members_by_team:
            for i_pos, i in enumerate(members):
                for j in members[i_pos + 1:]:
                    close = np.hypot(half_px[:, i] - half_px[:, j],
                                     half_py[:, i] - half_py[:, j]) < DEDUP_DIST
                    half_px[close, j], half_py[close, j] = np.nan, np.nan
    vx = np.full_like(px, np.nan); vy = np.full_like(py, np.nan)
    for _, first, last in blocks:
        sl = slice(first, last + 1)
        for s in range(slots):
            good = np.isfinite(px[sl, s]) & np.isfinite(py[sl, s])
            dx = _segment_gradient(px[sl, s], good)
            dy = _segment_gradient(py[sl, s], good)
            vx[sl, s], vy[sl, s] = dx, dy
            invalid = breaks[sl, s]
            vx[sl, s][invalid], vy[sl, s][invalid] = np.nan, np.nan
    for s in range(slots):
        for suffix, values in (("x", px[:, s]), ("y", py[:, s]), ("vx", vx[:, s]), ("vy", vy[:, s])):
            frame_data[f"s{s + 1:02d}_{suffix}"] = values.astype(np.float32)
        frame_data[f"s{s + 1:02d}_break"] = breaks[:, s]

    for team, label in ((0, "home"), (1, "away")):
        members = np.flatnonzero(np.asarray(slot_team_ids) == team)
        tx, ty, tv = px[:, members], py[:, members], np.hypot(vx[:, members], vy[:, members])
        valid = np.isfinite(tx) & np.isfinite(ty)
        count = valid.sum(axis=1)
        cx = np.divide(np.nansum(tx, axis=1), count, out=np.full(t, np.nan), where=count > 0)
        cy = np.divide(np.nansum(ty, axis=1), count, out=np.full(t, np.nan), where=count > 0)
        frame_data[f"centroid_{label}_x"], frame_data[f"centroid_{label}_y"] = cx, cy
        frame_data[f"spread_{label}_x"] = np.sqrt(np.divide(np.nansum((tx-cx[:, None])**2, axis=1), count,
            out=np.full(t, np.nan), where=count > 0))
        frame_data[f"spread_{label}_y"] = np.sqrt(np.divide(np.nansum((ty-cy[:, None])**2, axis=1), count,
            out=np.full(t, np.nan), where=count > 0))
        speed_count = (valid & np.isfinite(tv)).sum(axis=1)
        frame_data[f"mean_speed_{label}"] = np.divide(np.nansum(np.where(valid, tv, np.nan), axis=1), speed_count,
            out=np.full(t, np.nan), where=speed_count > 0)
        frame_data[f"n_still_{label}"] = ((tv < STILL_SPEED) & valid).sum(axis=1).astype(np.float32)
        frame_data[f"n_tracked_{label}"] = count.astype(np.float32)

    # Team aggregates are tracking-derived features. Frames outside the two
    # tracked periods must stay unknown (NaN), including the count columns.
    aggregate_columns = [
        f"{prefix}_{team}{suffix}"
        for prefix, suffixes in (("centroid", ("_x", "_y")), ("spread", ("_x", "_y")),
                                 ("mean_speed", ("",)), ("n_still", ("",)),
                                 ("n_tracked", ("",)))
        for team in ("home", "away")
        for suffix in suffixes
    ]
    outside_tracking = ~valid_halves
    for column in aggregate_columns:
        values = frame_data[column].astype(np.float32, copy=True)
        values[outside_tracking] = np.nan
        frame_data[column] = values

    slot_teams = pd.DataFrame({"slot": np.arange(1, slots + 1, dtype=np.int8),
        "is_home": np.asarray(slot_team_ids) == 0, "is_field_player": True})
    frame_data["half"] = np.asarray(frame_data["half"], dtype=np.int8)
    frame_data["is_graded"] = np.asarray(frame_data["is_graded"], dtype=bool)
    frame_data["flip"] = np.zeros(t, dtype=np.int8)
    for (half, first, last), flipped in zip(blocks, flips):
        frame_data["flip"][first:last + 1] = int(flipped)
    frame = pd.DataFrame(frame_data)
    return frame, slot_teams


def build(game=GAME, data_dir=None, output_dir=None):
    """Load raw frame-aligned tracking arrays for A2/A3.

    ``output_dir`` remains accepted for call compatibility; this load stage does not
    transform or persist the arrays.
    """
    if game != GAME:
        raise ValueError(f"This preprocessing module currently supports only {GAME}, got {game!r}")
    paths = asset_paths(game=game, data_dir=data_dir, source="tactical")
    required = ("atd_meta", "atd_raw")
    missing = [paths[name] for name in required if not os.path.isfile(paths[name])]
    if missing:
        raise FileNotFoundError("Required ATD assets not found:\n  " + "\n  ".join(missing))
    return read_atd(paths["atd_raw"], paths["atd_meta"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", help="data bundle root (defaults to MTAH_DATA_DIR or ./data)")
    parser.add_argument("--output-dir", help="processed output directory (defaults to <data-dir>/processed)")
    args = parser.parse_args(argv)
    players, ball, teams = build(data_dir=args.data_dir)
    frame, slot_teams = prepare_tracking(players, ball, teams, kickoffs=_kickoff_frames(args.data_dir))
    root = args.data_dir or os.environ.get("MTAH_DATA_DIR") or os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    output_dir = args.output_dir or os.path.join(root, "processed")
    os.makedirs(output_dir, exist_ok=True)
    frame.to_parquet(os.path.join(output_dir, "tracking_clean.parquet"), index=False)
    slot_teams.to_parquet(os.path.join(output_dir, "slot_teams.parquet"), index=False)
    print(f"players={players.shape}, ball={ball.shape}, slots={len(teams)}")
    print(f"tracking rows={len(frame):,}; flips={frame.groupby('half')['flip'].first().to_dict()}; output={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
