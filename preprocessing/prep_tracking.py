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
import tempfile

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from games import GAME, asset_paths, half_video_blocks
from starter.baseline import team_of_each_track
from .params import (BALL_EPS, BALL_K, BALL_VMAX, DEDUP_DIST, FPS, KICKOFF_FALLBACK, MAX_GAP_FRAMES,
                     MIN_BALL_SPEED_DIR, N_FRAMES, ORIENT_WIN, PITCH_L, PITCH_W, PLAYER_VMAX,
                     SCHEMA_VERSION, STILL_SPEED)
from .schema import N_SLOTS

FIELD_SEP = ":"
GROUP_SEP = ";"
LOGGER = logging.getLogger(__name__)


def _data_root(data_dir=None):
    return data_dir or os.environ.get("MTAH_DATA_DIR") or os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data")


def _kickoff_frames(data_dir=None, required=False, processed_dir=None):
    """Load C4 kickoff metadata and validate its exact frames against the tag windows.

    V0 can run without the C4 handoff and retains the published fallback. V1 requires
    the handoff so its orientation is tied to the validated kickoff file.
    ``processed_dir`` is where export_tagging wrote kickoffs.json (default <data-dir>/processed).
    """
    folder = processed_dir or os.path.join(_data_root(data_dir), "processed")
    path = os.path.join(folder, "kickoffs.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as stream:
            try:
                payload = json.load(stream)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ValueError(f"Invalid kickoff JSON in {path}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Kickoff JSON in {path} must be an object")
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
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Kickoff frame for half {half} in {path} must be an integer >= 0")
            result[half] = value
        expected = KICKOFF_FALLBACK
        if required:
            for half, frame in expected.items():
                if result[half] != frame:
                    raise ValueError(
                        f"Kickoff frame for half {half} in {path} is {result[half]}, expected {frame}"
                    )
        windows = payload.get("tag_windows")
        if required and not isinstance(windows, dict):
            raise ValueError(f"{path} must contain tag_windows for both halves")
        if windows is not None:
            for half, frame in result.items():
                window = windows.get(str(half), windows.get(half)) if isinstance(windows, dict) else None
                if (not isinstance(window, list) or len(window) != 2
                        or any(isinstance(v, bool) or not isinstance(v, int) for v in window)):
                    raise ValueError(f"{path} must contain a two-frame tag window for half {half}")
                if not window[0] <= frame <= window[1]:
                    raise ValueError(
                        f"Kickoff frame {frame} for half {half} is outside tag window {window} in {path}"
                    )
        LOGGER.info("Kickoffs loaded from %s", path)
        return result
    if required:
        raise FileNotFoundError(f"Required C4 kickoff metadata not found: {path}")
    LOGGER.warning("%s not found; using published kickoff fallback", path)
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
            if not line.strip():
                continue
            parts = line.rstrip("\r\n").split(FIELD_SEP)
            if len(parts) != 3:
                raise ValueError(f"Malformed ATD line {line_no}: expected 3 colon-separated fields")
            try:
                frame = int(parts[0]) - 1
            except ValueError as exc:
                raise ValueError(f"Invalid frame id at ATD line {line_no}") from exc
            if frame < 0 or frame >= n_frames or seen[frame]:
                raise ValueError(f"Unexpected or duplicate ATD frame id at line {line_no}")
            expected = line_no - 1
            if frame != expected:
                raise ValueError(f"ATD frames out of order at line {line_no}")
            groups = parts[1].split(GROUP_SEP)
            if len(groups) != N_SLOTS:
                raise ValueError(f"ATD line {line_no}: expected {N_SLOTS} slots, found {len(groups)}")
            for slot, group in enumerate(groups):
                values = _triple(group, line_no=line_no)
                if values is not None:
                    players[frame, slot] = values
            values = _triple(parts[2], line_no=line_no)
            if values is not None:
                ball[frame] = values
            seen[frame] = True
    if not seen.all():
        missing = np.flatnonzero(~seen)
        raise ValueError(f"ATD has {len(missing)} missing frames")
    return players, ball, teams


def _triple(text, line_no=None):
    if not text or text[0] in "Nn":
        return None
    parts = text.split(",")
    if len(parts) < 3:
        raise ValueError("Malformed ATD object group: expected three values")
    try:
        vals = np.asarray([float(v) for v in parts[:3]], dtype=np.float32)
    except ValueError as exc:
        where = f" at ATD line {line_no}" if line_no is not None else ""
        raise ValueError(f"Malformed numeric value{where}") from exc
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
    """Return gap lengths (including leading/trailing gaps) and bidirectional distances.

    The frozen schema defines gap_length for every in-half gap, including gaps at
    either edge; only since/until are NaN where the corresponding observation is absent.
    """
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


def _report_player_quality(before, after, blocks):
    """Log per-half aggregates of frames above 22 objects.

    ``before`` is after isolated-point removal but before deduplication;
    ``after`` is after deduplication. No frame-level values or coordinates are logged.
    """
    for half, first, last in blocks:
        pre = np.isfinite(before[first:last + 1]).all(axis=2).sum(axis=1) > 22
        post = np.isfinite(after[first:last + 1]).all(axis=2).sum(axis=1) > 22
        total = max(1, last - first + 1)
        LOGGER.info("Player quality half %s: frames >22 objects before=%d (%.2f%%), after=%d (%.2f%%)",
                    half, int(pre.sum()), 100 * pre.sum() / total,
                    int(post.sum()), 100 * post.sum() / total)


def _team_aggregates(px, py, vx, vy, slot_team_ids, is_field_player):
    """Aggregate every eligible player slot by team, independent of slot identity."""
    t = px.shape[0]
    features = {}
    slot_team_ids = np.asarray(slot_team_ids)
    is_field_player = np.asarray(is_field_player)
    if slot_team_ids.shape != (px.shape[1],) or is_field_player.shape != (px.shape[1],):
        raise ValueError("slot team and field-player metadata must have one value per slot")
    if is_field_player.dtype.kind != "b":
        raise ValueError("is_field_player must contain one boolean value per player slot")

    for team, label in ((0, "home"), (1, "away")):
        members = np.flatnonzero((slot_team_ids == team) & is_field_player)
        tx, ty = px[:, members], py[:, members]
        tv = np.hypot(vx[:, members], vy[:, members])
        valid = np.isfinite(tx) & np.isfinite(ty)
        count = valid.sum(axis=1)
        has_players = count > 0
        cx = np.divide(np.nansum(tx, axis=1), count, out=np.full(t, np.nan), where=has_players)
        cy = np.divide(np.nansum(ty, axis=1), count, out=np.full(t, np.nan), where=has_players)
        spread_x = np.sqrt(np.divide(np.nansum((tx-cx[:, None])**2, axis=1), count,
            out=np.full(t, np.nan), where=has_players))
        spread_y = np.sqrt(np.divide(np.nansum((ty-cy[:, None])**2, axis=1), count,
            out=np.full(t, np.nan), where=has_players))
        speed_valid = valid & np.isfinite(tv)
        speed_count = speed_valid.sum(axis=1)
        mean_speed = np.divide(np.nansum(np.where(speed_valid, tv, np.nan), axis=1), speed_count,
            out=np.full(t, np.nan), where=speed_count > 0)
        n_still = ((tv < STILL_SPEED) & speed_valid).sum(axis=1).astype(np.float32)

        n_tracked = count.astype(np.float32)
        # Spatial and speed metrics are undefined without a tracked slot. Counts
        # remain meaningful and are zero in that case.
        for values in (cx, cy, spread_x, spread_y, mean_speed):
            values[~has_players] = np.nan
        features.update({
            f"centroid_{label}_x": cx, f"centroid_{label}_y": cy,
            f"spread_{label}_x": spread_x, f"spread_{label}_y": spread_y,
            f"mean_speed_{label}": mean_speed, f"n_still_{label}": n_still,
            f"n_tracked_{label}": n_tracked,
        })
    return {name: np.asarray(values, dtype=np.float32) for name, values in features.items()}


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


def _segment_gradient(values, valid, breaks=None, fps=FPS, smooth=True):
    """Differentiate finite contiguous runs independently; singleton runs have no speed.

    A run ends at the first invalid frame or just before the next ``breaks`` frame, so
    ``np.gradient`` never crosses a slot break. When ``smooth=True``, a zero-phase
    3-point binomial filter is applied inside each run before differentiating.
    """
    values = np.asarray(values)
    out = np.full(len(values), np.nan, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    if breaks is None:
        cut = np.zeros(len(values), dtype=bool)
    else:
        cut = np.asarray(breaks, dtype=bool)
        if cut.shape != valid.shape:
            raise ValueError("breaks and valid must have the same shape")

    starts = valid & (~np.r_[False, valid[:-1]] | cut)
    start_idx = np.flatnonzero(starts)
    # Frames where a run cannot continue: invalid, or a new run begins (start).
    stop = ~valid | starts
    for start in start_idx:
        nxt = np.flatnonzero(stop[start + 1:])
        end = start + 1 + (nxt[0] if len(nxt) else len(values) - start - 1)
        if end - start >= 2:
            seg = _binomial3_smooth(values[start:end]) if smooth else values[start:end]
            out[start:end] = np.gradient(seg) * fps
    return out

def _smooth_ball_segment(seg):
    """Offline Savitzky-Golay smoothing for a single finite ball segment."""
    seg = np.asarray(seg, dtype=np.float64)
    if len(seg) >= 7:
        return savgol_filter(seg, window_length=7, polyorder=2, mode="interp")
    return seg.copy()


def _ball_gradient(values, valid, fps=FPS, smooth=True):
    """Differentiate contiguous ball observations, smoothing each V1 segment once."""
    out = np.full(len(values), np.nan, dtype=np.float32)
    edges = np.flatnonzero(np.diff(np.r_[False, valid, False].astype(np.int8)))
    for start, end in zip(edges[::2], edges[1::2]):
        if end - start >= 2:
            seg = values[start:end]
            if smooth:
                seg = _smooth_ball_segment(seg)
            out[start:end] = np.gradient(seg) * fps
    return out


def _report_ball_quality(frame):
    """Log aggregate ball quality metrics; never emit frame-level or coordinate data."""
    graded = frame["half"].to_numpy() > 0
    gap = frame.loc[graded, "gap_length"].to_numpy()
    observed = frame.loc[graded, "ball_observed"].to_numpy(dtype=bool)
    removed = frame.loc[graded, "ball_jump_removed"].to_numpy(dtype=bool)
    gap_frames = gap > 0
    long_gap_frames = gap > FPS  # strictly longer than one second
    starts = gap_frames & ~np.r_[False, gap_frames[:-1]]
    finite_gaps = gap[starts]
    gap_summary = ", ".join(
        f"{label}:{int(((finite_gaps >= lo) & (finite_gaps <= hi)).sum())}"
        for label, lo, hi in (("1f", 1, 1), ("2-6f", 2, 6), ("7-12f", 7, 12),
                              ("13-25f", 13, 25), (">25f", 26, np.inf)))
    removed_pct = 100.0 * removed.sum() / max(1, (removed | observed).sum())
    LOGGER.info("Ball quality: gap sizes (frames:count)=%s; frames in gaps >1s=%.2f%%; "
                "removed points=%.2f%%", gap_summary,
                100.0 * long_gap_frames.sum() / max(1, len(gap)), removed_pct)


def _report_filter_sensitivity(ball, blocks):
    """Summarize full-game speed and eps/K removal sensitivity for both halves.

    Interpret the report by choosing the smallest K and eps in a stable region:
    removed-point fractions should change little at neighboring settings, with
    enough distance from a small parameter change that causes an abrupt jump.
    These choices must be based on the complete match, never the public five minutes.
    """
    speeds = []
    for _, first, last in blocks:
        xy = ball[first:last + 1, :2]
        valid = np.isfinite(xy).all(axis=1)
        edges = np.flatnonzero(valid[1:] & valid[:-1]) + 1
        if len(edges):
            delta = xy[edges] - xy[edges - 1]
            speeds.extend((np.hypot(delta[:, 0] * PITCH_L, delta[:, 1] * PITCH_W) * FPS).tolist())
    speed = np.asarray(speeds, dtype=np.float64)
    if speed.size:
        counts, _ = np.histogram(speed, bins=(0, 5, 10, 20, 38, np.inf))
        LOGGER.info("Raw consecutive-observation speed histogram (m/s bins 0-5,5-10,10-20,20-38,>38): %s",
                    counts.tolist())
    total = 0
    for _, first, last in blocks:
        total += np.isfinite(ball[first:last + 1, :2]).all(axis=1).sum()
    for eps in (0.5, 1.0, 2.0):
        rates = []
        percentages = []
        for k in (20, 30, 40, 60):
            kept = 0
            for _, first, last in blocks:
                x, y = remove_ball_jump_outliers(ball[first:last + 1, 0] * PITCH_L,
                                                  ball[first:last + 1, 1] * PITCH_W,
                                                  eps=eps, k=k)
                kept += (np.isfinite(x) & np.isfinite(y)).sum()
            removed_pct = 100.0 * (total - kept) / max(1, total)
            percentages.append(removed_pct)
            rates.append(f"K={k}:{removed_pct:.2f}%")
        max_neighbor_delta = max((abs(a - b) for a, b in zip(percentages, percentages[1:])), default=0.0)
        LOGGER.info("Removed-point sensitivity at eps=%.1fm: %s; max neighboring K change=%.2f pp",
                    eps, ", ".join(rates), max_neighbor_delta)


def _orient_half(players, ball, team_ids, start, end, kick_frame, pitch):
    """Reflect one half so home attacks toward increasing x; return whether reflected."""
    length, width = pitch
    a, b = max(start, kick_frame), min(end, kick_frame + ORIENT_WIN)
    window = players[a:b, :, 0]
    home = np.asarray(team_ids) == 0
    home_mean = np.nanmean(window[:, home]) if np.isfinite(window[:, home]).any() else np.nan
    if not np.isfinite(home_mean):
        raise ValueError("Cannot infer home direction near kickoff")
    # At kickoff the home team is in its own half: low x means it attacks toward high x.
    flip = home_mean >= length / 2
    if flip:
        players[start:end, :, 0] = length - players[start:end, :, 0]
        players[start:end, :, 1] = width - players[start:end, :, 1]
        # ATD's third player field is scalar speed (s), so reflection does not change it.
        ball[start:end, 0] = length - ball[start:end, 0]
        ball[start:end, 1] = width - ball[start:end, 1]
    return flip


def _prepare_tracking_v0(players, ball, slot_team_ids, slot_is_field_player, flips, blocks):
    """Build the frozen V0 deliverable without A4/A5/A6 processing.

    Inputs are already scaled to metres and oriented in place. Differentiation is
    limited to contiguous observed runs inside each half, so neither missing
    observations nor the halftime interval can be bridged.
    """
    t, slots = players.shape[:2]
    half = np.zeros(t, dtype=np.int8)
    flip_col = np.zeros(t, dtype=np.int8)
    frame_data = {
        "video_frame": np.arange(t, dtype=np.int32),
        "half": half,
        "flip": flip_col,
    }
    ball_x = np.full(t, np.nan, dtype=np.float32)
    ball_y = np.full(t, np.nan, dtype=np.float32)
    ball_vx = np.full(t, np.nan, dtype=np.float32)
    ball_vy = np.full(t, np.nan, dtype=np.float32)
    slot_x = np.full((t, slots), np.nan, dtype=np.float32)
    slot_y = np.full((t, slots), np.nan, dtype=np.float32)
    slot_vx = np.full((t, slots), np.nan, dtype=np.float32)
    slot_vy = np.full((t, slots), np.nan, dtype=np.float32)

    for (half_id, first, last), flipped in zip(blocks, flips):
        sl = slice(first, last + 1)
        half[first:last + 1] = half_id
        flip_col[first:last + 1] = int(flipped)
        ball_x[sl], ball_y[sl] = ball[sl, 0], ball[sl, 1]
        ball_valid = np.isfinite(ball[sl, 0]) & np.isfinite(ball[sl, 1])
        ball_vx[sl] = _segment_gradient(ball[sl, 0], ball_valid, smooth=False)
        ball_vy[sl] = _segment_gradient(ball[sl, 1], ball_valid, smooth=False)
        slot_x[sl], slot_y[sl] = players[sl, :, 0], players[sl, :, 1]
        for slot in range(slots):
            valid = np.isfinite(players[sl, slot, 0]) & np.isfinite(players[sl, slot, 1])
            slot_vx[sl, slot] = _segment_gradient(players[sl, slot, 0], valid, smooth=False)
            slot_vy[sl, slot] = _segment_gradient(players[sl, slot, 1], valid, smooth=False)

    frame_data.update({"ball_x": ball_x, "ball_y": ball_y,
                       "ball_vx": ball_vx, "ball_vy": ball_vy})
    zero_break = np.zeros(t, dtype=np.int8)
    for slot in range(slots):
        frame_data[f"s{slot + 1:02d}_x"] = slot_x[:, slot]
        frame_data[f"s{slot + 1:02d}_y"] = slot_y[:, slot]
        frame_data[f"s{slot + 1:02d}_vx"] = slot_vx[:, slot]
        frame_data[f"s{slot + 1:02d}_vy"] = slot_vy[:, slot]
        frame_data[f"s{slot + 1:02d}_break"] = zero_break.copy()
    slot_teams = pd.DataFrame({
        "slot": np.arange(1, slots + 1, dtype=np.int8),
        "is_home": np.asarray(slot_team_ids) == 0,
        "is_field_player": np.asarray(slot_is_field_player, dtype=bool),
    })
    return pd.DataFrame(frame_data), slot_teams


def write_parquet_with_meta(df, path, **meta):
    """Write a pandas table as Parquet with the frozen schema metadata."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(df, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[b"schema_version"] = str(SCHEMA_VERSION).encode("utf-8")
    for key, value in meta.items():
        metadata[str(key).encode("utf-8")] = str(value).encode("utf-8")
    pq.write_table(table.replace_schema_metadata(metadata), path)


def _write_parquet_atomically(df, path, **meta):
    """Write metadata-bearing Parquet beside the target, then atomically replace it."""
    output_dir = os.path.dirname(path) or "."
    fd, temp_path = tempfile.mkstemp(prefix="." + os.path.basename(path) + ".",
                                     suffix=".parquet", dir=output_dir)
    os.close(fd)
    try:
        write_parquet_with_meta(df, temp_path, **meta)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def prepare_tracking(players, ball, slot_team_ids, blocks=None, pitch=(PITCH_L, PITCH_W),
                     n_frames=N_FRAMES, kickoffs=None, stage="V1", data_dir=None,
                     slot_is_field_player=None):
    """Build clean frame table and slot metadata from normalized 0..1 ATD arrays.

    ``slot_is_field_player`` controls eligibility for deduplication and team
    aggregates. The ATD's 46 slots are club players, including goalkeepers, so
    every slot is eligible by default.

    Ball kinematics are smoothed within observed runs (V1) rather than over the
    interpolated ``ix, iy`` series described in the issue. This keeps interpolation
    boundaries out of the derivative estimate and avoids edge artifacts.
    """
    stage = str(stage).upper()
    if stage not in {"V0", "V1"}:
        raise ValueError("stage must be 'V0' or 'V1'")
    players = np.asarray(players)
    slot_team_ids = np.asarray(slot_team_ids)
    if slot_team_ids.shape != (players.shape[1],) or not np.isin(slot_team_ids, (0, 1)).all():
        raise ValueError("slot_team_ids must have one 0/1 value per player slot")
    if slot_is_field_player is None:
        slot_is_field_player = np.ones(players.shape[1], dtype=bool)
    else:
        slot_is_field_player = np.asarray(slot_is_field_player)
    if slot_is_field_player.shape != (players.shape[1],):
        raise ValueError("slot_is_field_player must have one boolean value per player slot")
    if slot_is_field_player.dtype.kind != "b":
        raise ValueError("slot_is_field_player must have one boolean value per player slot")
    if blocks is None:
        blocks = half_video_blocks(GAME)
        if kickoffs is None:
            kickoffs = _kickoff_frames(data_dir)
    elif kickoffs is None:
        # Explicitly supplied blocks are commonly short synthetic intervals.
        # Their orientation starts at the first frame unless a test/caller gives
        # explicit kickoff frames; the published match frames cannot apply here.
        kickoffs = {half: first for half, first, _ in blocks}
    players = players.astype(np.float32, copy=True)
    ball = ball.astype(np.float32, copy=True)
    length, width = pitch
    players[:, :, 0] *= length
    players[:, :, 1] *= width
    ball[:, 0] *= length
    ball[:, 1] *= width
    # Audit coordinates are the original ATD positions in metres, before the
    # per-half orientation transform as well as before filtering/interpolation.
    ball_raw = ball[:, :2].copy()
    flips = []
    for half, first, last in blocks:
        kick = int(kickoffs.get(half, first))
        if not first <= kick <= last:
            raise ValueError(f"Kickoff frame {kick} for half {half} is outside tracking block [{first}, {last}]")
        flips.append(_orient_half(players, ball, slot_team_ids, first, last + 1, kick, pitch))
    if len(flips) >= 2 and flips[0] == flips[1]:
        raise ValueError("Half orientations should oppose each other")
    t, slots = players.shape[:2]
    frame_data = {"video_frame": np.arange(t, dtype=np.int32)}
    half_col = np.zeros(t, dtype=np.int8)
    for half, first, last in blocks:
        half_col[first:last + 1] = half
    frame_data["half"] = half_col
    frame_data["is_graded"] = half_col > 0
    frame_data["frame_in_half"] = np.where(half_col > 0,
        np.arange(t) - np.where(half_col == 1, blocks[0][1], blocks[-1][1]), np.nan).astype(np.float32)

    if stage == "V0":
        frame, slot_teams = _prepare_tracking_v0(
            players, ball, slot_team_ids, slot_is_field_player, flips, blocks)
        return frame, slot_teams

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
        # Keep stored positions untouched. V1 smooths each observed run with
        # Savitzky-Golay (window 7, order 2, mode='interp') inside _ball_gradient;
        # V0 preserves the legacy
        # unsmoothed baseline. Filled positions never contribute to kinematics.
        bx[sl], by[sl] = ix, iy
        smooth = stage == "V1"
        dvx, dvy = _ball_gradient(fx, obs, smooth=smooth), _ball_gradient(fy, obs, smooth=smooth)
        bvx[sl], bvy[sl] = dvx, dvy
        speed = np.hypot(dvx, dvy)
        bspeed[sl] = speed
        valid_vel = np.isfinite(dvx) & np.isfinite(dvy)
        # The observed position run was smoothed inside _ball_gradient; differentiate
        # velocity directly to avoid applying another filter to acceleration.
        ax, ay = (_segment_gradient(dvx, valid_vel, smooth=False),
                  _segment_gradient(dvy, valid_vel, smooth=False))
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
    frame_data["ball_x_raw"], frame_data["ball_y_raw"] = ball_raw[:, 0].copy(), ball_raw[:, 1].copy()
    frame_data["ball_x_raw"][half_col == 0] = np.nan
    frame_data["ball_y_raw"][half_col == 0] = np.nan

    px, py = players[:, :, 0], players[:, :, 1]
    positions_before_dedup = np.full((t, slots, 2), np.nan, dtype=np.float32)
    valid_halves = np.zeros(t, dtype=bool)
    for _, first, last in blocks:
        valid_halves[first:last + 1] = True
    px[~valid_halves, :], py[~valid_halves, :] = np.nan, np.nan
    breaks = np.zeros(px.shape, dtype=bool)
    # Keep every player operation local to a half, just like the ball pipeline.
    # This makes the boundary rule explicit and robust if blocks ever become adjacent.
    members_by_team = [np.flatnonzero((np.asarray(slot_team_ids) == team) & slot_is_field_player)
                       for team in (0, 1)]
    for _, first, last in blocks:
        sl = slice(first, last + 1)
        half_px, half_py = px[sl], py[sl]
        _, isolated = slot_breaks(half_px, half_py)
        half_px[isolated], half_py[isolated] = np.nan, np.nan
        # "Before" means after isolated-point removal and before duplicate removal.
        positions_before_dedup[sl] = np.stack((half_px, half_py), axis=2)
        # Remove duplicates within this half, retaining the lower slot index.
        for members in members_by_team:
            for i_pos, i in enumerate(members):
                for j in members[i_pos + 1:]:
                    close = np.hypot(half_px[:, i] - half_px[:, j],
                                     half_py[:, i] - half_py[:, j]) < DEDUP_DIST
                    half_px[close, j], half_py[close, j] = np.nan, np.nan
        # Recompute after every position removal so reacquisitions are explicit breaks.
        breaks[sl], _ = slot_breaks(half_px, half_py)
    _report_player_quality(positions_before_dedup, np.stack((px, py), axis=2), blocks)
    vx = np.full_like(px, np.nan); vy = np.full_like(py, np.nan)
    for _, first, last in blocks:
        sl = slice(first, last + 1)
        for s in range(slots):
            good = np.isfinite(px[sl, s]) & np.isfinite(py[sl, s])
            # A5 documents smoothed player velocities for V1; keep V0 unsmoothed.
            dx = _segment_gradient(px[sl, s], good, breaks=breaks[sl, s], smooth=(stage == "V1"))
            dy = _segment_gradient(py[sl, s], good, breaks=breaks[sl, s], smooth=(stage == "V1"))
            vx[sl, s], vy[sl, s] = dx, dy
    for s in range(slots):
        for suffix, values in (("x", px[:, s]), ("y", py[:, s]), ("vx", vx[:, s]), ("vy", vy[:, s])):
            frame_data[f"s{s + 1:02d}_{suffix}"] = values.astype(np.float32)
        frame_data[f"s{s + 1:02d}_break"] = breaks[:, s].astype(np.int8)

    slot_teams = pd.DataFrame({"slot": np.arange(1, slots + 1, dtype=np.int8),
        "is_home": np.asarray(slot_team_ids) == 0, "is_field_player": slot_is_field_player})
    # The same eligibility mask controls deduplication and team aggregates.
    frame_data.update(_team_aggregates(px, py, vx, vy, slot_team_ids,
                                       slot_teams["is_field_player"].to_numpy()))

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

    frame_data["half"] = np.asarray(frame_data["half"], dtype=np.int8)
    frame_data["is_graded"] = np.asarray(frame_data["is_graded"], dtype=bool)
    frame_data["flip"] = np.zeros(t, dtype=np.int8)
    for (half, first, last), flipped in zip(blocks, flips):
        frame_data["flip"][first:last + 1] = int(flipped)
    frame = pd.DataFrame(frame_data)
    # Keep the frozen 237-column V0 contract as an identical, ordered prefix.
    # V1-only confidence, audit, and team aggregate columns follow that prefix.
    v0_columns = ["video_frame", "half", "flip", "ball_x", "ball_y", "ball_vx", "ball_vy"]
    for slot in range(1, slots + 1):
        v0_columns.extend(f"s{slot:02d}_{suffix}" for suffix in ("x", "y", "vx", "vy", "break"))
    v1_columns = [column for column in frame.columns if column not in set(v0_columns)]
    frame = frame[v0_columns + v1_columns]
    return frame, slot_teams


def build(game=GAME, data_dir=None):
    """Load raw frame-aligned tracking arrays for A2/A3; writing is done by ``main``."""
    if game != GAME:
        raise ValueError(f"This preprocessing module currently supports only {GAME}, got {game!r}")
    paths = asset_paths(game=game, data_dir=data_dir, source="tactical")
    required = ("atd_meta", "atd_raw")
    missing = [paths[name] for name in required if not os.path.isfile(paths[name])]
    if missing:
        raise FileNotFoundError("Required ATD assets not found:\n  " + "\n  ".join(missing))
    return read_atd(paths["atd_raw"], paths["atd_meta"])


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", help="data bundle root (defaults to MTAH_DATA_DIR or ./data)")
    parser.add_argument("--output-dir", help="processed output directory (defaults to <data-dir>/processed)")
    parser.add_argument("--stage", choices=("V0", "V1"), default="V0",
                        help="tracking feature stage (default: V0)")
    parser.add_argument("--report-sensitivity", action="store_true",
                        help="run the optional eps/K ball-filter sensitivity report")
    args = parser.parse_args(argv)
    root = _data_root(args.data_dir)
    output_dir = args.output_dir or os.path.join(root, "processed")
    players, ball, teams = build(data_dir=args.data_dir)
    paths = asset_paths(game=GAME, data_dir=args.data_dir, source="tactical")
    # All 46 ATD slots belong to the two clubs. The flag means player vs
    # non-player and therefore includes goalkeepers for this match.
    is_field_player = np.ones(len(teams), dtype=bool)
    frame, slot_teams = prepare_tracking(players, ball, teams,
                                         kickoffs=_kickoff_frames(args.data_dir, required=args.stage == "V1",
                                                                  processed_dir=output_dir),
                                         stage=args.stage, slot_is_field_player=is_field_player)
    if args.stage == "V1":
        _report_ball_quality(frame)
    if args.report_sensitivity:
        if args.stage != "V1":
            parser.error("--report-sensitivity is available only with --stage V1")
        # Jump filtering is invariant to pitch reflection, so raw unoriented ball
        # coordinates produce the same keep/remove decisions as oriented coordinates.
        _report_filter_sensitivity(ball, half_video_blocks(GAME))
    os.makedirs(output_dir, exist_ok=True)
    _write_parquet_atomically(frame, os.path.join(output_dir, "tracking_clean.parquet"),
                              tracking_stage=args.stage)
    _write_parquet_atomically(slot_teams, os.path.join(output_dir, "slot_teams.parquet"))
    LOGGER.info("Tracking prepared: rows=%d, slots=%d", len(frame), len(teams))
    LOGGER.info("Tracking and slot metadata written to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
