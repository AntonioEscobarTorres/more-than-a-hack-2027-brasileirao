"""How often the two known edge cases of the ball jump filter occur (F3-07).

Local execution only (a person runs it on the match data):

    python tools/jump_filter_edge_cases.py
    python tools/jump_filter_edge_cases.py --verbose

By default it prints only PASS/FAIL per check. ``--verbose`` adds the counts behind each check,
for the person at the terminal only: they never go to issues, PRs or agents.

The review of the preprocessing found, on synthetic data, two cases where
``preprocessing.prep_tracking.remove_ball_jump_outliers`` keeps the wrong observations:

  (a) a phantom that stands still for 3 s or more beats the real ball when the real ball is
      detected sparsely;
  (b) in a multi-ball restart, if the new ball appears within about 0.4 s, the last frames of the
      old ball (the crossing of the line) are removed.

This tool counts the footprint of each case, per half, in ``tracking_clean.parquet`` and in the
master. Nothing is computed across the limit of a half.

  (a) Stretches of at least ``MIN_STILL_FRAMES`` consecutive frames where the kept ball is still
      (speed < ``BALL_STILL_SPEED_MPS``) and outside the pitch (``out_of_pitch == 1``), and that
      contain ``MIN_STILL_FRAMES`` consecutive frames over which the mean speed of the players is
      above ``PLAYERS_FAST_SPEED_MPS``. The kept ball includes the short gaps the preprocessing
      interpolates: a phantom that beats a sparse real ball is a chain of short observed runs
      joined by interpolated frames (the removed real observations), never one long observed run.

  (b) Removed observations (``ball_jump_removed == 1``) in the ``JUMP_WINDOW_FRAMES`` frames
      before a jump of the kept ball of more than ``JUMP_MIN_DISTANCE_M`` that is followed by a
      still ball. A jump is measured between two consecutive kept observations; "followed by a
      still ball" means the kept ball is still at some frame of the ``JUMP_WINDOW_FRAMES`` from
      the landing observation. With ``--verbose``, the tool also says how many of those removed
      observations continue the kept trajectory within the physical limit of the filter
      (``BALL_VMAX``, ``BALL_EPS``), from their original positions in ``tracking_clean.parquet``:
      those are the ones a fix would readmit.

PASS if both counts are below ``MAX_COUNT_PER_HALF`` in each half.

Exit code 0 if every check passes, 1 if a check fails, 2 if the inputs could not be loaded.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import games  # noqa: E402
from modeling import io, paths  # noqa: E402
from modeling.deadball.features import half_blocks  # noqa: E402
from preprocessing.params import BALL_EPS, BALL_VMAX  # noqa: E402

TRACKING_FILENAME = "tracking_clean.parquet"

# Fixed by the task specification (F3-07), not tuned.
BALL_STILL_SPEED_MPS = 0.5        # the kept ball is still below this speed
MIN_STILL_FRAMES = 75             # (a) shortest stretch, 3 s
PLAYERS_FAST_SPEED_MPS = 3.0      # (a) play goes on: mean player speed above this
JUMP_WINDOW_FRAMES = 25           # (b) frames inspected before (and after) a jump, 1 s
JUMP_MIN_DISTANCE_M = 20.0        # (b) shortest jump of the kept ball
MAX_COUNT_PER_HALF = 10           # PASS if every count is below this

# Largest difference accepted between a kept position and the original position of the same
# observation; above it the two input files do not come from the same preprocessing run.
RAW_MATCH_TOL_M = 1e-3

MASTER_COLUMNS = ("half", "ball_x", "ball_y", "ball_speed", "ball_observed", "ball_jump_removed",
                  "ball_filled", "out_of_pitch", "mean_speed_home", "mean_speed_away")
TRACKING_COLUMNS = ("video_frame", "half", "flip", "ball_x_raw", "ball_y_raw")


@dataclass
class PhantomStats:
    """(a), one half. ``n_counted`` is the number compared with the limit."""
    n_counted: int = 0            # still-and-outside stretches while the players are fast
    n_stretches: int = 0          # still-and-outside stretches, whatever the players do
    longest_frames: int = 0       # longest still-and-outside stretch


@dataclass
class JumpStats:
    """(b), one half. ``n_removed`` is the number compared with the limit."""
    n_removed: int = 0            # removed observations in the window before a jump
    n_jumps: int = 0              # jumps of the kept ball followed by a still ball
    n_continuing: int = 0         # removed observations that continue the kept trajectory


@dataclass
class CheckResult:
    key: str
    title: str
    passed: bool
    detail: str = ""


def _fps() -> int:
    return int(games.GAMES[games.GAME]["fps"])


def _require(frame: pd.DataFrame, columns, name: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError("{} is missing columns {}".format(name, missing))


def _f64(frame: pd.DataFrame, name: str) -> np.ndarray:
    return frame[name].to_numpy(dtype="float64")


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``[start, stop)`` of every maximal run of True in a 1-D bool array."""
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def kept_ball_speed(master: pd.DataFrame, fps: int | None = None) -> np.ndarray:
    """Speed of the kept ball (m/s) on every frame where it has a position, NaN elsewhere.

    ``ball_speed`` where the master has it. It is NaN on interpolated frames and on isolated
    observations: there, the speed of the step from the previous frame (or to the next one) is
    used, which on an interpolated gap is the speed of the linear interpolation. Steps never
    cross the limit of a half.
    """
    _require(master, ("half", "ball_x", "ball_y", "ball_speed"), "master")
    fps = _fps() if fps is None else fps
    x, y, speed = _f64(master, "ball_x"), _f64(master, "ball_y"), _f64(master, "ball_speed")
    half = master["half"].to_numpy()
    n = len(master)
    step_in = np.full(n, np.nan)                     # speed of the step from t - 1 to t
    if n > 1:
        same_half = (half[1:] == half[:-1]) & (half[1:] != 0)
        step_in[1:] = np.where(same_half, np.hypot(np.diff(x), np.diff(y)) * fps, np.nan)
    step_out = np.r_[step_in[1:], np.nan]            # speed of the step from t to t + 1
    fallback = np.where(np.isfinite(step_in), step_in, step_out)
    return np.where(np.isfinite(speed), speed, fallback)


def _mean_player_speed(master: pd.DataFrame) -> np.ndarray:
    """Mean of the two team mean speeds, ignoring a team without tracked players."""
    home, away = _f64(master, "mean_speed_home"), _f64(master, "mean_speed_away")
    ok_home, ok_away = np.isfinite(home), np.isfinite(away)
    count = ok_home.astype("float64") + ok_away
    total = np.where(ok_home, home, 0.0) + np.where(ok_away, away, 0.0)
    out = np.full(len(master), np.nan)
    out[count > 0] = total[count > 0] / count[count > 0]
    return out


def _has_fast_window(player_speed: np.ndarray, window: int, threshold: float) -> bool:
    """True if some ``window`` consecutive frames have a mean (of the finite values) > threshold."""
    finite = np.isfinite(player_speed)
    total = np.r_[0.0, np.cumsum(np.where(finite, player_speed, 0.0))]
    count = np.r_[0, np.cumsum(finite)]
    sums = total[window:] - total[:-window]
    counts = count[window:] - count[:-window]
    has = counts > 0
    return bool((sums[has] / counts[has] > threshold).any())


def phantom_stretches(master: pd.DataFrame, fps: int | None = None) -> dict[int, PhantomStats]:
    """(a) per half: still ball outside the pitch for 3 s or more while the players run."""
    _require(master, MASTER_COLUMNS, "master")
    half = master["half"].to_numpy()
    has_position = (_f64(master, "ball_observed") == 1) | (_f64(master, "ball_filled") == 1)
    speed = kept_ball_speed(master, fps)
    still_outside = (has_position & (speed < BALL_STILL_SPEED_MPS)             # NaN -> False
                     & (_f64(master, "out_of_pitch") == 1))
    player_speed = _mean_player_speed(master)

    stats: dict[int, PhantomStats] = {}
    for b0, b1 in half_blocks(half):
        s = stats.setdefault(int(half[b0]), PhantomStats())
        for r0, r1 in _true_runs(still_outside[b0:b1]):
            length = r1 - r0
            if length < MIN_STILL_FRAMES:
                continue
            s.n_stretches += 1
            s.longest_frames = max(s.longest_frames, length)
            if _has_fast_window(player_speed[b0 + r0:b0 + r1], MIN_STILL_FRAMES,
                                PLAYERS_FAST_SPEED_MPS):
                s.n_counted += 1
    return stats


def oriented_raw_ball(tracking: pd.DataFrame, pitch=None) -> tuple[np.ndarray, np.ndarray]:
    """Original ball positions in the orientation of the master (``flip`` undone)."""
    _require(tracking, TRACKING_COLUMNS, "tracking_clean")
    length, width = games.GAMES[games.GAME]["pitch"] if pitch is None else pitch
    flip = _f64(tracking, "flip") == 1
    x, y = _f64(tracking, "ball_x_raw"), _f64(tracking, "ball_y_raw")
    return np.where(flip, length - x, x), np.where(flip, width - y, y)


def check_same_run(master: pd.DataFrame, tracking: pd.DataFrame, pitch=None) -> None:
    """Raise ``ValueError`` unless the two tables come from the same preprocessing run."""
    _require(master, MASTER_COLUMNS, "master")
    _require(tracking, TRACKING_COLUMNS, "tracking_clean")
    if len(master) != len(tracking):
        raise ValueError("master has {} rows and tracking_clean has {}".format(
            len(master), len(tracking)))
    if not np.array_equal(tracking["video_frame"].to_numpy(), np.arange(len(tracking))):
        raise ValueError("tracking_clean video_frame is not 0..n-1")
    if not np.array_equal(master["half"].to_numpy(), tracking["half"].to_numpy()):
        raise ValueError("master and tracking_clean disagree on half")
    observed = _f64(master, "ball_observed") == 1
    raw_x, raw_y = oriented_raw_ball(tracking, pitch)
    error = np.hypot(raw_x[observed] - _f64(master, "ball_x")[observed],
                     raw_y[observed] - _f64(master, "ball_y")[observed])
    if not (error <= RAW_MATCH_TOL_M).all():                                   # NaN -> mismatch
        raise ValueError("kept ball positions differ from tracking_clean: regenerate both files "
                         "with python -m preprocessing.run_all")


def removed_before_jumps(master: pd.DataFrame, tracking: pd.DataFrame, fps: int | None = None,
                         pitch=None) -> dict[int, JumpStats]:
    """(b) per half: observations removed just before a jump of the kept ball to a still ball."""
    check_same_run(master, tracking, pitch)
    fps = _fps() if fps is None else fps
    half = master["half"].to_numpy()
    x, y = _f64(master, "ball_x"), _f64(master, "ball_y")
    observed = _f64(master, "ball_observed") == 1
    removed = _f64(master, "ball_jump_removed") == 1
    still = kept_ball_speed(master, fps) < BALL_STILL_SPEED_MPS                # NaN -> False
    raw_x, raw_y = oriented_raw_ball(tracking, pitch)

    stats: dict[int, JumpStats] = {}
    for b0, b1 in half_blocks(half):
        s = stats.setdefault(int(half[b0]), JumpStats())
        kept = np.flatnonzero(observed[b0:b1]) + b0
        if kept.size < 2:
            continue
        distance = np.hypot(np.diff(x[kept]), np.diff(y[kept]))
        in_window = np.zeros(len(master), dtype=bool)      # frames before a jump, counted once
        continuing = np.zeros(len(master), dtype=bool)
        for q in np.flatnonzero(distance > JUMP_MIN_DISTANCE_M):
            last, landing = int(kept[q]), int(kept[q + 1])
            if not still[landing:min(landing + JUMP_WINDOW_FRAMES, b1)].any():
                continue
            s.n_jumps += 1
            in_window[max(b0, landing - JUMP_WINDOW_FRAMES):landing] = True
            # Removed observations that follow the last kept one within the reach of the filter.
            anchor_frame, anchor_x, anchor_y = last, x[last], y[last]
            for t in np.flatnonzero(removed[last + 1:landing]) + last + 1:
                reach = BALL_VMAX * (t - anchor_frame) / fps + BALL_EPS
                if np.hypot(raw_x[t] - anchor_x, raw_y[t] - anchor_y) <= reach:
                    continuing[t] = True
                    anchor_frame, anchor_x, anchor_y = int(t), raw_x[t], raw_y[t]
        s.n_removed += int((removed & in_window).sum())
        s.n_continuing += int((continuing & in_window).sum())
    return stats


def run_checks(master: pd.DataFrame, tracking: pd.DataFrame, fps: int | None = None,
               pitch=None) -> list[CheckResult]:
    """Both checks. Raises ``ValueError`` if the inputs are not usable."""
    phantom = phantom_stretches(master, fps)
    jumps = removed_before_jumps(master, tracking, fps, pitch)
    if not phantom:
        raise ValueError("master has no frames inside a half")

    detail_a = "; ".join(
        "half {}: {} counted (limit < {}); {} still-and-outside stretches of {}+ frames, "
        "longest {} frames".format(h, s.n_counted, MAX_COUNT_PER_HALF, s.n_stretches,
                                   MIN_STILL_FRAMES, s.longest_frames)
        for h, s in sorted(phantom.items()))
    detail_b = "; ".join(
        "half {}: {} removed observations (limit < {}) before {} jumps; {} of them continue the "
        "kept trajectory".format(h, s.n_removed, MAX_COUNT_PER_HALF, s.n_jumps, s.n_continuing)
        for h, s in sorted(jumps.items()))
    return [
        CheckResult("a", "no still phantom kept outside the pitch while play goes on",
                    all(s.n_counted < MAX_COUNT_PER_HALF for s in phantom.values()), detail_a),
        CheckResult("b", "no observations removed just before a multi-ball jump",
                    all(s.n_removed < MAX_COUNT_PER_HALF for s in jumps.values()), detail_b),
    ]


def report(results: list[CheckResult], verbose: bool = False) -> list[str]:
    lines = []
    for r in results:
        lines.append("{}  ({}) {}".format("PASS" if r.passed else "FAIL", r.key, r.title))
        if verbose and r.detail:
            lines.extend("        {}".format(part) for part in r.detail.split("; "))
    n_fail = sum(not r.passed for r in results)
    lines.append("PASS" if n_fail == 0 else "FAIL ({} of {} checks)".format(n_fail, len(results)))
    return lines


def load_tracking(path=None) -> pd.DataFrame:
    """The columns of ``tracking_clean.parquet`` this tool needs."""
    path = paths.processed_dir() / TRACKING_FILENAME if path is None else Path(path)
    if not path.is_file():
        raise FileNotFoundError("tracking_clean not found: {}".format(path))
    return pd.read_parquet(path, columns=list(TRACKING_COLUMNS))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PASS/FAIL count of the two known edge cases of the ball jump filter.")
    parser.add_argument("--verbose", action="store_true",
                        help="also print the counts behind each check (for the person only)")
    parser.add_argument("--master",
                        help="master parquet (default: the BAR-ATH master in MTAH_DATA_DIR)")
    parser.add_argument("--tracking",
                        help="tracking_clean parquet (default: the one in MTAH_DATA_DIR)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        master = io.load_master(args.master)
        tracking = load_tracking(args.tracking)
        results = run_checks(master, tracking)
    except (OSError, ValueError, KeyError) as exc:
        print("ERROR  could not load inputs: {}".format(exc))
        return 2
    for line in report(results, verbose=args.verbose):
        print(line)
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
