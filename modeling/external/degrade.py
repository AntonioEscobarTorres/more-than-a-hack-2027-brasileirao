"""Apply deterministic, half-local tracking degradation to an IDSSE match."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .. import paths
from ..contracts import LABEL_CODES
from .files import ExternalMatch

# C0 code of the dead-ball label; the only label ``degrade`` reads (to unfreeze the DFL ball).
_DEAD_LABEL = LABEL_CODES["D"]


_CONFIG_KEYS = frozenset({
    "name", "target_ball_visible", "target_ball_visible_tolerance",
    "target_tracked_mean", "target_tracked_mean_tolerance", "z_air_m",
    "air_margin_frames", "p_air", "crowd_n", "crowd_r_m", "crowd_hazard",
    "bg_hazard_init", "hazard_max", "bisection_iterations",
    "player_bisection_iterations", "ball_noise_m", "phantom_len_frames",
    "phantom_frac", "phantom_margin_m", "player_visible_mean_s",
    "player_missing_mean_s_init", "player_missing_mean_s_max", "player_noise_m",
    "dup_offset_m", "dup_len_s", "dup_frac", "dup_max_attempts",
    "overtracked_threshold", "fps", "gap_profile",
    "p_unfreeze_hide", "restart_visible_s", "unfreeze_drift_mps", "unfreeze_drift_tau_s",
})


def validate_degrade_config(cfg: dict) -> dict:
    """Validate the complete degradation configuration and return it unchanged."""
    if not isinstance(cfg, dict):
        raise ValueError("degradation config must be a mapping")
    unknown = sorted(set(cfg) - _CONFIG_KEYS)
    missing = sorted(_CONFIG_KEYS - set(cfg))
    if unknown or missing:
        details = []
        if unknown:
            details.append("unknown keys {}".format(unknown))
        if missing:
            details.append("missing keys {}".format(missing))
        raise ValueError("degradation config: " + "; ".join(details))
    if not isinstance(cfg["name"], str) or not cfg["name"]:
        raise ValueError("name must be a non-empty string")

    numeric = _CONFIG_KEYS - {"name", "phantom_len_frames", "dup_len_s", "gap_profile"}
    for key in numeric:
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or \
                not math.isfinite(value):
            raise ValueError("{} must be a finite number".format(key))

    probabilities = ("p_air", "target_ball_visible", "phantom_frac", "dup_frac",
                     "crowd_hazard", "bg_hazard_init", "p_unfreeze_hide")
    for key in probabilities:
        if not 0.0 <= cfg[key] <= 1.0:
            raise ValueError("{} must be between 0 and 1".format(key))
    nonnegative = ("target_ball_visible_tolerance", "target_tracked_mean_tolerance",
                   "ball_noise_m", "player_noise_m", "dup_offset_m",
                   "restart_visible_s", "unfreeze_drift_mps")
    for key in nonnegative:
        if cfg[key] < 0.0:
            raise ValueError("{} must be non-negative".format(key))
    positive = ("z_air_m", "crowd_r_m", "hazard_max", "player_visible_mean_s",
                "player_missing_mean_s_init", "player_missing_mean_s_max", "fps",
                "phantom_margin_m", "unfreeze_drift_tau_s")
    for key in positive:
        if cfg[key] <= 0.0:
            raise ValueError("{} must be positive".format(key))
    if cfg["player_missing_mean_s_init"] > cfg["player_missing_mean_s_max"]:
        raise ValueError("player_missing_mean_s_init must not exceed player_missing_mean_s_max")
    if cfg["bg_hazard_init"] > cfg["hazard_max"]:
        raise ValueError("bg_hazard_init must not exceed hazard_max")
    if cfg["hazard_max"] > 1.0:
        raise ValueError("hazard_max must not exceed 1")
    if cfg["target_tracked_mean"] <= 0.0:
        raise ValueError("target_tracked_mean must be positive")

    integer_fields = ("air_margin_frames", "crowd_n", "bisection_iterations",
                      "player_bisection_iterations", "dup_max_attempts",
                      "overtracked_threshold", "fps")
    for key in integer_fields:
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("{} must be a non-negative integer".format(key))
    for key in ("bisection_iterations", "player_bisection_iterations", "dup_max_attempts",
                "crowd_n", "fps"):
        if cfg[key] == 0:
            raise ValueError("{} must be positive".format(key))

    for key in ("phantom_len_frames", "dup_len_s"):
        value = cfg[key]
        if not isinstance(value, (list, tuple)) or len(value) != 2 or any(
                isinstance(item, bool) or not isinstance(item, (int, float)) or
                not math.isfinite(item) for item in value):
            raise ValueError("{} must contain two finite numbers".format(key))
        if value[0] <= 0 or value[1] < value[0]:
            raise ValueError("{} must be positive and ordered".format(key))
    if any(not isinstance(value, int) for value in cfg["phantom_len_frames"]):
        raise ValueError("phantom_len_frames must contain integers")

    profile = cfg["gap_profile"]
    profile_keys = {"mix_geometric", "geometric_mean_frames", "lognormal_median_frames",
                    "lognormal_sigma", "max_frames"}
    if not isinstance(profile, dict) or set(profile) != profile_keys:
        raise ValueError("gap_profile must contain exactly {}".format(sorted(profile_keys)))
    for key, value in profile.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or \
                not math.isfinite(value):
            raise ValueError("gap_profile.{} must be a finite number".format(key))
    if not 0.0 <= profile["mix_geometric"] <= 1.0:
        raise ValueError("gap_profile.mix_geometric must be between 0 and 1")
    for key in ("geometric_mean_frames", "lognormal_median_frames", "lognormal_sigma",
                "max_frames"):
        if profile[key] <= 0:
            raise ValueError("gap_profile.{} must be positive".format(key))
    if int(profile["max_frames"]) != profile["max_frames"]:
        raise ValueError("gap_profile.max_frames must be an integer")
    return cfg


def _sample_gap_lengths(rng, count, cfg, profile):
    if profile is not None:
        lengths = np.asarray(profile["lengths"], dtype=np.int64)
        weights = np.asarray(profile["counts"], dtype=np.float64)
        return rng.choice(lengths, size=count, p=weights / weights.sum())
    params = cfg["gap_profile"]
    is_geometric = rng.random(count) < params["mix_geometric"]
    geometric = rng.geometric(1.0 / params["geometric_mean_frames"], size=count)
    lognormal = np.rint(rng.lognormal(np.log(params["lognormal_median_frames"]),
                                     params["lognormal_sigma"], size=count)).astype(np.int64)
    return np.clip(np.where(is_geometric, geometric, lognormal), 1,
                   params["max_frames"])


def _clear_bursts(mask, starts, lengths, blocks):
    out = mask.copy()
    for _, first, last in blocks:
        local = np.flatnonzero(starts[first:last + 1]) + first
        for start in local:
            out[start:min(last + 1, start + lengths[start])] = False
    return out


def _renewal_uniforms():
    """Create a lazy common-random-number cache for renewal event pairs."""
    return {}


def _player_visibility(uniforms, blocks, roster, visible_mean_frames, missing_mean_frames,
                       n_frames, rng):
    visible = np.zeros((n_frames, roster), dtype=bool)
    for player in range(roster):
        for block_index, (_, first, last) in enumerate(blocks):
            cursor = first
            event = 0
            is_visible = True
            while cursor <= last:
                key = (player, block_index)
                sequence = uniforms.setdefault(key, [])
                if event == len(sequence):
                    sequence.append(rng.random(2))
                pair = sequence[event]
                mean = visible_mean_frames if is_visible else missing_mean_frames
                duration = max(1, int(np.ceil(-np.log(max(pair[0], np.finfo(float).tiny)) * mean)))
                end = min(last + 1, cursor + duration)
                if is_visible:
                    visible[cursor:end, player] = True
                cursor = end
                is_visible = not is_visible
                event += 1
    return visible


def _assign_slots(players, visibility, slot_team_ids, blocks):
    result = np.full_like(players, np.nan)
    team_slots = [np.flatnonzero(slot_team_ids == team) for team in (0, 1)]
    team_sources = [np.flatnonzero(slot_team_ids == team) for team in (0, 1)]
    for team in (0, 1):
        slots = team_slots[team]
        sources = team_sources[team]
        for _, first, last in blocks:
            assigned = {}
            for frame in range(first, last + 1):
                active = [int(source) for source in sources if visibility[frame, source]]
                active_set = set(active)
                assigned = {source: slot for source, slot in assigned.items()
                            if source in active_set}
                occupied = set(assigned.values())
                free = [int(slot) for slot in slots if slot not in occupied]
                for source in active:
                    if source not in assigned:
                        if not free:
                            continue
                        assigned[source] = free.pop(0)
                for source, slot in assigned.items():
                    result[frame, slot] = players[frame, source]
    return result


def _count_players(players, blocks):
    counts = np.isfinite(players[:, :, 0]).sum(axis=1)
    return float(np.mean(np.concatenate([counts[first:last + 1] for _, first, last in blocks])))


def _load_optional_profile():
    path = paths.processed_dir() / "ball_gap_profile.json"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as stream:
        profile = json.load(stream)
    lengths = np.asarray(profile.get("lengths"), dtype=np.int64)
    counts = np.asarray(profile.get("counts"), dtype=np.float64)
    if lengths.ndim != 1 or counts.shape != lengths.shape or lengths.size == 0 or \
            (lengths < 1).any() or (counts < 0).any() or counts.sum() <= 0:
        raise ValueError("invalid ball gap profile")
    return {"lengths": lengths, "counts": counts}


def _dead_ball_mask(labels: np.ndarray) -> np.ndarray:
    """True on dead-ball (D) frames."""
    return np.asarray(labels) == _DEAD_LABEL


def _unfreeze_dead_ball(ball, ball_visible, labels, blocks, cfg, rng, length_m, width_m):
    """Undo the frozen ball in dead-ball (D) segments, in place and half by half.

    Each D segment gets one of two treatments, drawn with ``p_unfreeze_hide``:
      * hide: the ball is erased for the whole segment except the last ``restart_visible_s``
        seconds before the restart (the ball being placed);
      * drift: the ball stays visible but its frozen position is replaced by a slow, smooth
        drift pinned to the segment's first and last frozen positions (the last one is the
        restart spot), so the path joins live play continuously at both ends.
    The drift is a bridge built on a persistent-velocity walk: per axis, the velocity follows
    ``v[k] = rho * v[k-1] + sqrt(1 - rho**2) * sigma * eps[k]`` with
    ``sigma = unfreeze_drift_mps / sqrt(2)`` and ``rho = exp(-1 / (unfreeze_drift_tau_s * fps))``.
    The RMS speed is ``unfreeze_drift_mps`` and, because the velocity persists for about
    ``unfreeze_drift_tau_s``, the ball really travels about that far in one second; a walk with
    independent per-frame steps would cancel out (about ``unfreeze_drift_mps / sqrt(fps)`` per
    second) and leave the ball looking stationary. With ``unfreeze_drift_mps == 0`` the ball
    is left untouched.
    Both branches draw the same random numbers (one uniform plus ``n x 2`` normals), so
    ``p_unfreeze_hide`` never shifts the random stream of the later stages. Segments never
    cross a half boundary. Uses only the IDSSE labels to mimic the ATD behaviour.
    """
    fps = cfg["fps"]
    restart_frames = int(round(cfg["restart_visible_s"] * fps))
    sigma_mps = cfg["unfreeze_drift_mps"] / np.sqrt(2.0)
    rho = float(np.exp(-1.0 / (cfg["unfreeze_drift_tau_s"] * fps)))
    innovation = np.sqrt(1.0 - rho * rho) * sigma_mps
    scale = np.array([length_m, width_m], dtype=np.float64)
    for _, first, last in blocks:
        local = _dead_ball_mask(labels[first:last + 1])
        edges = np.diff(np.r_[False, local, False].astype(np.int8))
        for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            start, end = first + int(s), first + int(e)  # end is exclusive
            n = end - start
            hide = rng.random() < cfg["p_unfreeze_hide"]
            eps = rng.normal(0.0, 1.0, size=(n, 2))
            if hide:
                ball_visible[start:max(start, end - restart_frames)] = False
                continue
            if n < 2 or sigma_mps == 0.0:
                continue
            view = ball[start:end, :2]
            seg = view.astype(np.float64) * scale
            finite = np.isfinite(seg).all(axis=1)
            if not finite.any():
                continue
            begin_m = seg[int(np.argmax(finite))]
            anchor_m = seg[n - 1 - int(np.argmax(finite[::-1]))]
            velocity = np.empty((n, 2), dtype=np.float64)
            velocity[0] = sigma_mps * eps[0]
            for k in range(1, n):
                velocity[k] = rho * velocity[k - 1] + innovation * eps[k]
            walk = np.zeros((n, 2), dtype=np.float64)
            walk[1:] = np.cumsum(velocity[:-1], axis=0) / fps
            # Bridge: pin the walk to begin_m at k = 0 and to anchor_m at k = n - 1.
            t = (np.arange(n, dtype=np.float64) / (n - 1))[:, None]
            path = begin_m + (anchor_m - begin_m) * t + walk - walk[-1] * t
            view[finite] = (path / scale)[finite].astype(view.dtype)


def degrade(match: ExternalMatch, cfg: dict, seed: int) -> ExternalMatch:
    """Return a degraded copy of ``match`` using one seeded RNG and half-local processes."""
    return degrade_with_info(match, cfg, seed)[0]


def degrade_with_info(match: ExternalMatch, cfg: dict, seed: int) -> tuple[ExternalMatch, dict]:
    """Like ``degrade``, plus a dict of calibration results (``bg_hazard``: final hazard)."""
    validate_degrade_config(cfg)
    rng = np.random.default_rng(seed)
    blocks = match.blocks()
    length_m, width_m = match.pitch
    players = match.players.copy()
    ball = match.ball.copy()
    ball_z = match.ball_z.copy()
    labels = match.labels.copy()
    playable = np.zeros(match.n_frames, dtype=bool)
    for _, first, last in blocks:
        playable[first:last + 1] = True
    profile = _load_optional_profile()

    # High-ball gaps are drawn independently for contiguous above-threshold runs.
    ball_visible = np.isfinite(ball[:, 0]) & np.isfinite(ball[:, 1]) & playable
    above = np.isfinite(ball_z) & (ball_z > cfg["z_air_m"]) & playable
    for _, first, last in blocks:
        local = above[first:last + 1]
        edges = np.diff(np.r_[False, local, False].astype(np.int8))
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if rng.random() < cfg["p_air"]:
                lo = max(first, first + int(start) - cfg["air_margin_frames"])
                hi = min(last + 1, first + int(end) + cfg["air_margin_frames"])
                ball_visible[lo:hi] = False

    # Dead-ball freeze (DFL artifact): hide or jitter the ball in each D segment, before the
    # visibility calibration so that the bisection accounts for the hidden frames.
    _unfreeze_dead_ball(ball, ball_visible, labels, blocks, cfg, rng, length_m, width_m)

    # Crowd hazards and background hazards share one sampled gap-length sequence.
    x_m = ball[:, 0].astype(np.float64) * length_m
    y_m = ball[:, 1].astype(np.float64) * width_m
    px_m = players[:, :, 0].astype(np.float64) * length_m
    py_m = players[:, :, 1].astype(np.float64) * width_m
    proximity = np.zeros(match.n_frames, dtype=bool)
    for _, first, last in blocks:
        dx = px_m[first:last + 1] - x_m[first:last + 1, None]
        dy = py_m[first:last + 1] - y_m[first:last + 1, None]
        near = np.isfinite(dx) & np.isfinite(dy) & (dx * dx + dy * dy < cfg["crowd_r_m"] ** 2)
        proximity[first:last + 1] = near.sum(axis=1) >= cfg["crowd_n"]
    crowd_starts = playable & ball_visible & proximity & (rng.random(match.n_frames) < cfg["crowd_hazard"])
    lengths = _sample_gap_lengths(rng, match.n_frames, cfg, profile)
    visible_after_crowd = _clear_bursts(ball_visible, crowd_starts, lengths, blocks)
    bg_random = rng.random(match.n_frames)
    bg_lengths = _sample_gap_lengths(rng, match.n_frames, cfg, profile)

    # Phantom bursts (step 5) are drawn here so the background-gap calibration can target the
    # visibility the output really has: a phantom on a frame that is already visible adds
    # nothing, one on a gap makes it visible. They are applied after the noise (step 4).
    target_phantom_frames = int(np.ceil(cfg["phantom_frac"] * playable.sum()))
    phantom_mask = np.zeros(match.n_frames, dtype=bool)
    phantom_bursts = []
    phantom_min, phantom_max = cfg["phantom_len_frames"]
    while np.count_nonzero(phantom_mask) < target_phantom_frames:
        _, first, last = blocks[int(rng.integers(0, len(blocks)))]
        start = int(rng.integers(first, last + 1))
        duration = int(rng.integers(phantom_min, phantom_max + 1))
        end = min(last + 1, start + duration)
        phantom_mask[start:end] = True
        ghost_x = rng.uniform(-cfg["phantom_margin_m"], length_m + cfg["phantom_margin_m"])
        ghost_y = rng.uniform(-cfg["phantom_margin_m"], width_m + cfg["phantom_margin_m"])
        phantom_bursts.append((start, end, ghost_x / length_m, ghost_y / width_m))

    target_visible = cfg["target_ball_visible"]

    def bg_visible(hazard):
        starts = playable & visible_after_crowd & (bg_random < hazard)
        return _clear_bursts(visible_after_crowd, starts, bg_lengths, blocks)

    def final_fraction(hazard):
        return float((bg_visible(hazard) | phantom_mask)[playable].mean())

    if final_fraction(0.0) < target_visible - cfg["target_ball_visible_tolerance"]:
        raise ValueError("air and crowd gaps exceed the target ball visibility")
    low, high = 0.0, min(cfg["bg_hazard_init"], cfg["hazard_max"])
    while high < cfg["hazard_max"] and final_fraction(high) > target_visible:
        low, high = high, min(cfg["hazard_max"], high * 2.0)
    if final_fraction(high) > target_visible + cfg["target_ball_visible_tolerance"]:
        raise ValueError("hazard_max cannot reach target_ball_visible")
    for _ in range(cfg["bisection_iterations"]):
        mid = (low + high) / 2.0
        if final_fraction(mid) > target_visible:
            low = mid
        else:
            high = mid
    # The visible fraction is a step function of the hazard: keep the closer bracket end.
    hazard = min((low, high), key=lambda h: abs(final_fraction(h) - target_visible))
    ball_visible = bg_visible(hazard)
    ball[~ball_visible, :2] = np.nan
    ball[~ball_visible, 2] = np.nan

    noise = rng.normal(0.0, cfg["ball_noise_m"], size=(match.n_frames, 2))
    ball[:, 0] += (noise[:, 0] / length_m).astype(np.float32)
    ball[:, 1] += (noise[:, 1] / width_m).astype(np.float32)
    ball[~playable] = np.nan

    # Phantom bursts (drawn above) are clipped at each half boundary.
    for start, end, ghost_x, ghost_y in phantom_bursts:
        ball[start:end, :2] = (ghost_x, ghost_y)
    ball[playable, 2] = np.nan
    visible_fraction = float((np.isfinite(ball[playable, 0]) & np.isfinite(ball[playable, 1])).mean())
    if abs(visible_fraction - cfg["target_ball_visible"]) > cfg["target_ball_visible_tolerance"]:
        raise ValueError("ball visibility {:.4f} misses target_ball_visible {} +/- {}".format(
            visible_fraction, cfg["target_ball_visible"], cfg["target_ball_visible_tolerance"]))

    # Calibrate per-player alternating exponential renewals with common random numbers.
    roster = players.shape[1]
    uniforms = _renewal_uniforms()
    visible_mean = cfg["player_visible_mean_s"] * cfg["fps"]
    lo = np.finfo(float).eps
    hi = min(cfg["player_missing_mean_s_init"], cfg["player_missing_mean_s_max"])
    max_missing = cfg["player_missing_mean_s_max"]
    while hi < max_missing:
        initial_visibility = _player_visibility(
            uniforms, blocks, roster, visible_mean, hi * cfg["fps"], match.n_frames, rng)
        initial_players = _assign_slots(players, initial_visibility, match.slot_team_ids, blocks)
        if _count_players(initial_players, blocks) <= cfg["target_tracked_mean"]:
            break
        lo, hi = hi, min(max_missing, hi * 2.0)
    if hi == max_missing:
        max_visibility = _player_visibility(
            uniforms, blocks, roster, visible_mean, hi * cfg["fps"], match.n_frames, rng)
        max_players = _assign_slots(players, max_visibility, match.slot_team_ids, blocks)
        if _count_players(max_players, blocks) > cfg["target_tracked_mean"]:
            raise ValueError("player_missing_mean_s_max cannot reach target_tracked_mean")
    for _ in range(cfg["player_bisection_iterations"]):
        candidate = (lo + hi) / 2.0
        visibility = _player_visibility(uniforms, blocks, roster, visible_mean,
                                        candidate * cfg["fps"], match.n_frames, rng)
        degraded = _assign_slots(players, visibility, match.slot_team_ids, blocks)
        if _count_players(degraded, blocks) > cfg["target_tracked_mean"]:
            lo = candidate
        else:
            hi = candidate
    visibility = _player_visibility(uniforms, blocks, roster, visible_mean,
                                    ((lo + hi) / 2.0) * cfg["fps"], match.n_frames, rng)
    players = _assign_slots(players, visibility, match.slot_team_ids, blocks)
    tracked_mean = _count_players(players, blocks)
    if abs(tracked_mean - cfg["target_tracked_mean"]) > cfg["target_tracked_mean_tolerance"]:
        raise ValueError("tracked mean {:.3f} misses target_tracked_mean {} +/- {}".format(
            tracked_mean, cfg["target_tracked_mean"], cfg["target_tracked_mean_tolerance"]))

    player_noise = rng.normal(0.0, cfg["player_noise_m"], size=players[:, :, :2].shape)
    players[:, :, 0] += (player_noise[:, :, 0] / length_m).astype(np.float32)
    players[:, :, 1] += (player_noise[:, :, 1] / width_m).astype(np.float32)

    # Duplicate tracks: one copy of one player per burst, in the lowest free same-team slot, never
    # crossing a half boundary and never more than one duplicate per frame. Bursts are added until
    # dup_frac of the half frames carry more than overtracked_threshold objects.
    threshold = cfg["overtracked_threshold"]
    tracked = np.isfinite(players[:, :, 0]).sum(axis=1)
    target_dup_frames = int(np.ceil(cfg["dup_frac"] * playable.sum()))
    duplicate_covered = int(np.count_nonzero(playable & (tracked > threshold)))
    has_duplicate = np.zeros(match.n_frames, dtype=bool)
    duplicate_attempts = 0
    min_len = max(1, int(np.ceil(cfg["dup_len_s"][0] * cfg["fps"])))
    max_len = max(min_len, int(np.floor(cfg["dup_len_s"][1] * cfg["fps"])))
    while duplicate_covered < target_dup_frames and duplicate_attempts < cfg["dup_max_attempts"]:
        duplicate_attempts += 1
        _, first, last = blocks[int(rng.integers(0, len(blocks)))]
        start = int(rng.integers(first, last + 1))
        duration = int(rng.integers(min_len, max_len + 1))
        end = min(last + 1, start + duration)
        burst_frames = [frame for frame in range(start, end) if not has_duplicate[frame]]
        if not burst_frames:
            continue
        first_frame = burst_frames[0]
        active = np.flatnonzero(np.isfinite(players[first_frame, :, 0]) &
                                np.isfinite(players[first_frame, :, 1]))
        teams = [team for team in (0, 1) if np.any(match.slot_team_ids[active] == team)]
        if not teams:
            continue
        team = int(teams[int(rng.integers(0, len(teams)))])
        sources = active[match.slot_team_ids[active] == team]
        source = int(sources[int(rng.integers(0, sources.size))])
        offset = rng.normal(0.0, cfg["dup_offset_m"], size=2)
        normalized_offset = (offset / np.array([length_m, width_m], dtype=np.float64)).astype(
            np.float32)
        for frame in burst_frames:
            if not np.isfinite(players[frame, source, :2]).all():
                continue
            team_free = np.flatnonzero((match.slot_team_ids == team) &
                                       np.isnan(players[frame, :, 0]))
            if team_free.size == 0:
                continue
            target = int(team_free[0])
            players[frame, target, :2] = players[frame, source, :2] + normalized_offset
            players[frame, target, 2] = players[frame, source, 2]
            has_duplicate[frame] = True
            tracked[frame] += 1
            if tracked[frame] == threshold + 1:
                duplicate_covered += 1
    if duplicate_covered < target_dup_frames:
        raise ValueError("dup_max_attempts cannot reach dup_frac")
    players[~playable] = np.nan
    degraded = ExternalMatch(match_id=match.match_id, players=players, ball=ball,
                             ball_z=ball_z.copy(), labels=labels, half=match.half.copy(),
                             slot_team_ids=match.slot_team_ids.copy(), pitch=match.pitch.copy())
    return degraded, {"bg_hazard": float(hazard)}


__all__ = ["degrade", "degrade_with_info", "validate_degrade_config"]
