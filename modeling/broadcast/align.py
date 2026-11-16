"""Align the broadcast (TV) tracking timeline to the tactical timeline (F4-05, part 1).

The broadcast feed has its own frame counter, cuts and replays, and there is no correspondence
table. This module recovers one: for every broadcast frame, the tactical video frame it shows
(``-1`` when there is none).

Pipeline, run once per candidate orientation (identity and 180-degree rotation):

1. Coarse alignment, per tactical half: a masked cross-correlation of the pitch-centred ball
   position series at ``coarse_fps``, using only samples where both cameras see the ball.
2. Block refinement: the broadcast timeline is cut into blocks of ``block_frames`` frames. Each
   eligible block searches integer shifts around the coarse shift (or around the previous accepted
   block) and keeps the shift that minimises the median distance from each visible broadcast
   player to the closest tactical player of the same team. A block is accepted only when the cost is
   small and the runner-up shift is clearly worse.
3. Live segments: consecutive accepted blocks with nearly equal shifts form a segment; short
   segments (replays, cuts) are dropped.

The orientation with the lower global cost wins.

Nothing here reads a file except ``load_broadcast_inputs``; every other function works on arrays,
so the tests use synthetic data only. No temporal window, correlation or block search crosses a
half boundary: a block maps into exactly one tactical half, and every mapped frame stays inside it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import games
from preprocessing.prep_tracking import read_atd

LOGGER = logging.getLogger(__name__)

IDENTITY = "identity"
ROT180 = "rot180"
ORIENTATIONS = (IDENTITY, ROT180)
TEAM_IDS = (0, 1)

_REQUIRED_CFG = (
    "tactical_n_slots", "broadcast_n_slots", "source_fps", "coarse_fps", "block_seconds",
    "block_frames", "min_visible_players", "refine_search_radius_frames", "align_max_cost_m",
    "second_best_min_offset_frames", "second_best_cost_ratio", "segment_max_offset_diff_frames",
    "segment_min_blocks")


# -- data containers ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Tracking:
    """One camera's tracking in metres on that camera's own frame counter.

    players: float64[T, S, 2]  x, y in metres, NaN = not visible.
    ball:    float64[T, 2]     x, y in metres, NaN = not observed.
    team:    int[S]            team id per slot; only ids in TEAM_IDS take part in pairing.
    valid:   bool[T]           homography valid (all True for the tactical camera).
    half:    int8[T]           half id per frame, 0 outside the halves. Required for the tactical
                               camera; the broadcast timeline has no half labels (all zeros).
    """

    players: np.ndarray
    ball: np.ndarray
    team: np.ndarray
    valid: np.ndarray
    half: np.ndarray


@dataclass(frozen=True)
class AlignDiagnostics:
    """Everything the feasibility tool needs to report, all derived from the alignment itself."""

    orientation_costs: dict          # orientation -> global cost (median best block cost), inf = none
    coarse: dict                     # half -> (shift_frames, normalised_corr, overlap_samples) or None
    block_frames: int
    block_eligible: np.ndarray       # bool[n_blocks]
    block_accepted: np.ndarray       # bool[n_blocks]
    block_half: np.ndarray           # int8[n_blocks]   half of the best candidate (0 = none)
    block_shift: np.ndarray          # int32[n_blocks]  tactical_frame - broadcast_frame
    block_cost_m: np.ndarray         # float64[n_blocks] best median pairing cost, inf = none
    block_in_segment: np.ndarray     # bool[n_blocks]   accepted and kept in a live segment
    segments: tuple = field(default_factory=tuple)  # (first_block, last_block, half, median_shift)


@dataclass(frozen=True)
class Mapping:
    """Result of ``align_broadcast``."""

    tactical_frame: np.ndarray       # int32[T_broadcast]; -1 where there is no tactical frame
    orientation: str                 # "identity" or "rot180"
    diagnostics: AlignDiagnostics

    def mapped_ratio(self) -> float:
        """Share of broadcast frames that carry a tactical frame."""
        if self.tactical_frame.size == 0:
            return 0.0
        return float(np.mean(self.tactical_frame >= 0))


# -- loading ------------------------------------------------------------------------------------

def tracking_from_normalized(players, ball, teams, pitch, valid=None, half=None) -> Tracking:
    """Convert ``read_atd`` output (0..1 coordinates, float32) to a metre-based ``Tracking``.

    No orientation is applied: coordinates stay as the camera delivered them.
    """
    length, width = pitch
    scale = np.array([length, width], dtype=np.float64)
    players_m = np.asarray(players)[:, :, :2].astype(np.float64) * scale
    ball_m = np.asarray(ball)[:, :2].astype(np.float64) * scale
    n_frames = ball_m.shape[0]
    return Tracking(
        players=players_m,
        ball=ball_m,
        team=np.asarray(teams),
        valid=np.ones(n_frames, dtype=bool) if valid is None else np.asarray(valid, dtype=bool),
        half=np.zeros(n_frames, dtype=np.int8) if half is None else np.asarray(half, dtype=np.int8),
    )


def _read_valid_mask(path, n_frames) -> np.ndarray:
    """The ``valid`` column of a homography CSV as bool[n_frames].

    Assumed layout (to be confirmed against DATA.md): one row per video frame, in order, with a
    ``valid`` column; an optional ``video_frame`` column is used as the row index when present.
    """
    table = pd.read_csv(path)
    columns = {str(c).strip().lower(): c for c in table.columns}
    if "valid" not in columns:
        raise ValueError("homography CSV {} has no 'valid' column; columns are {}".format(
            path, list(table.columns)))
    values = table[columns["valid"]].to_numpy().astype(bool)
    if "video_frame" in columns:
        index = table[columns["video_frame"]].to_numpy()
        if index.min() < 0 or index.max() >= n_frames:
            raise ValueError("homography CSV {} has video_frame values outside 0..{}".format(
                path, n_frames - 1))
        mask = np.zeros(n_frames, dtype=bool)
        mask[index.astype(np.int64)] = values
        return mask
    if len(values) != n_frames:
        raise ValueError("homography CSV {} has {} rows, expected {}".format(
            path, len(values), n_frames))
    return values


def load_broadcast_inputs(cfg, data_dir=None):
    """Read both cameras from the (gated) match bundle. Human-run only: this opens match data.

    Returns ``(tactical, broadcast)`` as ``Tracking`` objects in metres, both unoriented.
    """
    pitch = games.GAMES[games.GAME]["pitch"]

    b_paths = games.asset_paths(data_dir=data_dir, source="broadcast")
    games.require(("atd_raw", "atd_meta", "homography"), data_dir=data_dir, source="broadcast")
    n_broadcast = games.source_frames("broadcast")
    b_players, b_ball, b_teams = read_atd(
        b_paths["atd_raw"], b_paths["atd_meta"], n_frames=n_broadcast,
        n_slots=cfg["broadcast_n_slots"])
    valid = _read_valid_mask(b_paths["homography"], n_broadcast)
    broadcast = tracking_from_normalized(b_players, b_ball, b_teams, pitch, valid=valid)

    t_paths = games.asset_paths(data_dir=data_dir, source="tactical")
    games.require(("atd_raw", "atd_meta"), data_dir=data_dir, source="tactical")
    t_players, t_ball, t_teams = read_atd(t_paths["atd_raw"], t_paths["atd_meta"])
    if t_players.shape[1] != cfg["tactical_n_slots"]:
        raise ValueError(
            "tactical ATD has {} slots but config tactical_n_slots is {}; fix the config".format(
                t_players.shape[1], cfg["tactical_n_slots"]))
    half = np.zeros(t_players.shape[0], dtype=np.int8)
    for half_id, first, last in games.half_video_blocks(games.GAME):
        half[first:last + 1] = half_id
    tactical = tracking_from_normalized(t_players, t_ball, t_teams, pitch, half=half)
    return tactical, broadcast


# -- small helpers ------------------------------------------------------------------------------

def _check_cfg(cfg) -> None:
    missing = [key for key in _REQUIRED_CFG if key not in cfg]
    if missing:
        raise KeyError("alignment config is missing keys {}".format(missing))


def _half_blocks(half) -> list:
    """[(half_id, first_frame, last_frame)] from a per-frame half label; halves must be contiguous."""
    blocks = []
    for half_id in sorted(int(h) for h in np.unique(half) if h > 0):
        frames = np.flatnonzero(half == half_id)
        first, last = int(frames[0]), int(frames[-1])
        if last - first + 1 != frames.size:
            raise ValueError("half {} is not a contiguous block of frames".format(half_id))
        blocks.append((half_id, first, last))
    if not blocks:
        raise ValueError("tactical tracking has no half labelled > 0")
    return blocks


def _orient(players, ball, orientation, pitch):
    """Apply an orientation to metre coordinates: identity or a 180-degree pitch rotation."""
    if orientation == IDENTITY:
        return players, ball
    if orientation != ROT180:
        raise ValueError("unknown orientation {!r}".format(orientation))
    length, width = pitch
    flip = np.array([length, width], dtype=np.float64)
    return flip - players, flip - ball


def _split_by_team(players, team, valid) -> list:
    """Per-team position arrays [players of team 0, players of team 1], NaN on invalid frames."""
    team = np.asarray(team)
    out = []
    for team_id in TEAM_IDS:
        part = np.array(players[:, team == team_id, :], dtype=np.float64, copy=True)
        part[~valid] = np.nan
        out.append(part)
    return out


def _lower_median(values) -> int:
    """Median that is always an element of the input (lower middle), for integer shifts."""
    ordered = np.sort(np.asarray(values), kind="mergesort")
    return int(ordered[(ordered.size - 1) // 2])


# -- coarse alignment ---------------------------------------------------------------------------

def _xcorr(a, b):
    """Linear cross-correlation c[d] = sum_i a[i] * b[i + d] for d in [-(len(a)-1), len(b)-1]."""
    na, nb = a.size, b.size
    n = 1 << int(np.ceil(np.log2(na + nb - 1)))
    fa = np.fft.rfft(a, n)
    fb = np.fft.rfft(b, n)
    c = np.fft.irfft(np.conj(fa) * fb, n)
    lags = np.arange(-(na - 1), nb)
    return lags, c[lags % n]


def _coarse_shift(b_ball, t_ball, first_frame, stride, min_overlap, center):
    """Best shift (tactical_frame - broadcast_frame) for one half from the ball trajectories.

    Both series are already subsampled by ``stride``. Sample i of the broadcast series is broadcast
    frame i * stride; sample j of the tactical series is tactical frame first_frame + j * stride.

    The score of a lag d = j - i is the cross-correlation of the pitch-centred ball positions,
    sum over i of (pb - center) . (pt - center), restricted to samples where both cameras see the
    ball. Matching samples add a large positive amount, while samples from a replay, a cut or the
    other half behave like unrelated positions and average out, so they do not drag the peak away
    (a mean-distance score would be dominated by them). Lags with fewer than ``min_overlap`` common
    samples are ignored; ties go to the lowest lag.

    Returns (shift_frames, normalised_correlation, overlap_samples) or None when no lag qualifies.
    """
    mb = np.isfinite(b_ball).all(axis=1)
    mt = np.isfinite(t_ball).all(axis=1)
    bx = np.where(mb, b_ball[:, 0] - center[0], 0.0)
    by = np.where(mb, b_ball[:, 1] - center[1], 0.0)
    tx = np.where(mt, t_ball[:, 0] - center[0], 0.0)
    ty = np.where(mt, t_ball[:, 1] - center[1], 0.0)
    mbf, mtf = mb.astype(np.float64), mt.astype(np.float64)

    lags, count = _xcorr(mbf, mtf)
    _, cross_x = _xcorr(bx, tx)
    _, cross_y = _xcorr(by, ty)
    _, energy_b = _xcorr(bx * bx + by * by, mtf)
    _, energy_t = _xcorr(mbf, tx * tx + ty * ty)
    score = cross_x + cross_y
    count = np.rint(count)

    usable = count >= min_overlap
    if not usable.any():
        return None
    masked = np.where(usable, score, -np.inf)
    best = int(np.argmax(masked))
    norm = np.sqrt(max(energy_b[best], 0.0) * max(energy_t[best], 0.0))
    corr = float(score[best] / norm) if norm > 0.0 else 0.0
    return first_frame + int(lags[best]) * stride, corr, int(count[best])


# -- block refinement ---------------------------------------------------------------------------

def _block_costs(b_team, t_team, start, n_frames, shifts):
    """Median pairing cost of one broadcast block for every candidate shift (inf = no pairs).

    For each shift, every visible broadcast player is paired with the closest tactical player of the
    same team in the corresponding tactical frame; the cost is the median of those distances over the
    whole block.
    """
    costs = np.full(len(shifts), np.inf)
    for i, shift in enumerate(shifts):
        pooled = []
        for team_idx in range(len(TEAM_IDS)):
            bp = b_team[team_idx][start:start + n_frames]
            tp = t_team[team_idx][start + shift:start + shift + n_frames]
            if bp.shape[1] == 0 or tp.shape[1] == 0:
                continue
            dist = np.hypot(bp[:, :, None, 0] - tp[:, None, :, 0],
                            bp[:, :, None, 1] - tp[:, None, :, 1])
            dist = np.where(np.isnan(dist), np.inf, dist)
            closest = dist.min(axis=2)
            pooled.append(closest[np.isfinite(closest)])
        if pooled:
            joined = np.concatenate(pooled)
            if joined.size:
                costs[i] = float(np.median(joined))
    return costs


def _judge(costs, shifts, cfg):
    """(best_shift, best_cost, accepted) with the absolute-cost and runner-up-ratio tests."""
    best_idx = int(np.argmin(costs))          # first minimum = lowest shift
    best = float(costs[best_idx])
    if not np.isfinite(best):
        return None
    far = np.abs(shifts - shifts[best_idx]) > cfg["second_best_min_offset_frames"]
    second = float(costs[far].min()) if far.any() else np.inf
    accepted = best <= cfg["align_max_cost_m"] and second >= cfg["second_best_cost_ratio"] * best
    return int(shifts[best_idx]), best, accepted


def _window(center, half_bounds, start, n_frames, radius):
    """Integer shifts around ``center`` whose whole block lands inside the half, ascending."""
    first, last = half_bounds
    low = max(center - radius, first - start)
    high = min(center + radius, last - n_frames + 1 - start)
    if high < low:
        return np.empty(0, dtype=np.int64)
    return np.arange(low, high + 1, dtype=np.int64)


def _refine(b_team, t_team, eligible, coarse, bounds, cfg):
    """Per-block search. Returns half, shift, cost, accepted arrays over all blocks."""
    n_blocks = eligible.size
    n_frames = cfg["block_frames"]
    radius = cfg["refine_search_radius_frames"]
    block_half = np.zeros(n_blocks, dtype=np.int8)
    block_shift = np.zeros(n_blocks, dtype=np.int32)
    block_cost = np.full(n_blocks, np.inf)
    accepted = np.zeros(n_blocks, dtype=bool)
    previous = None                            # (half, shift) of the previous block if accepted

    def search(half_id, center, start):
        shifts = _window(center, bounds[half_id], start, n_frames, radius)
        if shifts.size == 0:
            return None
        costs = _block_costs(b_team, t_team, start, n_frames, shifts)
        verdict = _judge(costs, shifts, cfg)
        if verdict is None:
            return None
        shift, cost, ok = verdict
        return half_id, shift, cost, ok

    for block in range(n_blocks):
        if not eligible[block]:
            previous = None
            continue
        start = block * n_frames
        tried = []
        if previous is not None:
            tried.append(search(previous[0], previous[1], start))
        chosen = next((r for r in tried if r is not None and r[3]), None)
        if chosen is None:
            for half_id in sorted(coarse):
                if coarse[half_id] is not None:
                    tried.append(search(half_id, coarse[half_id][0], start))
            winners = [r for r in tried if r is not None and r[3]]
            if winners:
                chosen = min(winners, key=lambda r: (r[2], r[0]))
        seen = [r for r in tried if r is not None]
        if chosen is not None:
            block_half[block], block_shift[block], block_cost[block] = chosen[0], chosen[1], chosen[2]
            accepted[block] = True
            previous = (chosen[0], chosen[1])
        else:
            previous = None
            if seen:
                top = min(seen, key=lambda r: (r[2], r[0]))
                block_half[block], block_shift[block], block_cost[block] = top[0], top[1], top[2]
    return block_half, block_shift, block_cost, accepted


def _live_segments(accepted, block_half, block_shift, cfg):
    """Runs of consecutive accepted blocks with near-equal shifts, short runs dropped."""
    tolerance = cfg["segment_max_offset_diff_frames"]
    runs, start = [], None
    for block in range(accepted.size):
        continues = (start is not None and accepted[block] and accepted[block - 1]
                     and block_half[block] == block_half[block - 1]
                     and abs(int(block_shift[block]) - int(block_shift[block - 1])) <= tolerance)
        if continues:
            continue
        if start is not None:
            runs.append((start, block - 1))
            start = None
        if accepted[block]:
            start = block
    if start is not None:
        runs.append((start, accepted.size - 1))
    kept = []
    for first, last in runs:
        if last - first + 1 >= cfg["segment_min_blocks"]:
            kept.append((first, last, int(block_half[first]),
                         _lower_median(block_shift[first:last + 1])))
    return kept


# -- orientation passes -------------------------------------------------------------------------

def _align_one_orientation(orientation, tactical, broadcast, t_team, eligible, bounds, cfg, pitch):
    """Run coarse + refinement for one orientation; returns its blocks and global cost."""
    stride = cfg["source_fps"] // cfg["coarse_fps"]
    min_overlap = int(round(cfg["segment_min_blocks"] * cfg["block_seconds"] * cfg["coarse_fps"]))

    players, ball = _orient(broadcast.players, broadcast.ball, orientation, pitch)
    ball = np.where(broadcast.valid[:, None], ball, np.nan)
    b_team = _split_by_team(players, broadcast.team, broadcast.valid)
    b_ball_coarse = ball[::stride]

    coarse = {}
    for half_id, first, last in ((h, f, l) for h, (f, l) in sorted(bounds.items())):
        t_ball_coarse = tactical.ball[first:last + 1:stride]
        coarse[half_id] = _coarse_shift(b_ball_coarse, t_ball_coarse, first, stride, min_overlap,
                                        (pitch[0] / 2.0, pitch[1] / 2.0))

    block_half, block_shift, block_cost, accepted = _refine(
        b_team, t_team, eligible, coarse, bounds, cfg)
    scored = block_cost[eligible & np.isfinite(block_cost)]
    global_cost = float(np.median(scored)) if scored.size else np.inf
    return {"coarse": coarse, "half": block_half, "shift": block_shift, "cost": block_cost,
            "accepted": accepted, "global_cost": global_cost}


def align_broadcast(tactical: Tracking, broadcast: Tracking, cfg, pitch=None) -> Mapping:
    """Map every broadcast frame to a tactical frame (or -1) and pick the pitch orientation.

    A block is eligible when, over its frames, the median number of visible players (counting
    frames with an invalid homography as zero) is at least ``min_visible_players``. A candidate
    shift is only considered if the whole block stays inside one tactical half.
    """
    _check_cfg(cfg)
    pitch = tuple(games.GAMES[games.GAME]["pitch"]) if pitch is None else tuple(pitch)
    bounds = {h: (first, last) for h, first, last in _half_blocks(tactical.half)}
    n_frames = cfg["block_frames"]
    total = broadcast.ball.shape[0]
    n_blocks = total // n_frames

    t_team = _split_by_team(tactical.players, tactical.team, tactical.valid)
    b_visible = _split_by_team(broadcast.players, broadcast.team, broadcast.valid)
    per_frame = sum(np.isfinite(part[:, :, 0]).sum(axis=1) for part in b_visible)
    medians = np.median(per_frame[:n_blocks * n_frames].reshape(n_blocks, n_frames), axis=1) \
        if n_blocks else np.empty(0)
    eligible = medians >= cfg["min_visible_players"]
    del b_visible

    results = {}
    for orientation in ORIENTATIONS:
        results[orientation] = _align_one_orientation(
            orientation, tactical, broadcast, t_team, eligible, bounds, cfg, pitch)
        LOGGER.debug("orientation %s evaluated", orientation)

    costs = {o: results[o]["global_cost"] for o in ORIENTATIONS}
    chosen = ROT180 if costs[ROT180] < costs[IDENTITY] else IDENTITY   # ties keep identity
    best = results[chosen]

    segments = _live_segments(best["accepted"], best["half"], best["shift"], cfg)
    mapping = np.full(total, -1, dtype=np.int32)
    in_segment = np.zeros(n_blocks, dtype=bool)
    for first, last, _, _ in segments:
        for block in range(first, last + 1):
            start = block * n_frames
            mapping[start:start + n_frames] = (
                np.arange(start, start + n_frames, dtype=np.int64) + int(best["shift"][block]))
            in_segment[block] = True

    diagnostics = AlignDiagnostics(
        orientation_costs=costs,
        coarse=best["coarse"],
        block_frames=n_frames,
        block_eligible=eligible,
        block_accepted=best["accepted"],
        block_half=best["half"],
        block_shift=best["shift"],
        block_cost_m=best["cost"],
        block_in_segment=in_segment,
        segments=tuple(segments),
    )
    return Mapping(tactical_frame=mapping, orientation=chosen, diagnostics=diagnostics)


__all__ = ["AlignDiagnostics", "IDENTITY", "Mapping", "ORIENTATIONS", "ROT180", "Tracking",
           "align_broadcast", "load_broadcast_inputs", "tracking_from_normalized"]