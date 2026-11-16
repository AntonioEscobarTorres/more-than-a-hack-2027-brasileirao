"""Feasibility gate for the broadcast camera (F4-05): GO / NO-GO.

Local execution only (a person runs it on the match data):

    python tools/broadcast_feasibility.py
    python tools/broadcast_feasibility.py --verbose

By default it prints only PASS/FAIL for each criterion and the final GO / NO-GO. ``--verbose`` adds
the numbers behind each criterion, for the person at the terminal only: they never go to issues,
PRs or agents. The tool prints no feature arrays, quantiles, trajectories or match rows, even with
``--verbose``: only aggregate ratios, medians and counts.

The broadcast timeline is mapped onto the tactical one by ``modeling.broadcast.align`` (the
parameters come from ``configs/broadcast.yaml``), and four criteria are evaluated, always inside
the evaluated halves only (tactical frames with ``half != 0``, from ``games.half_video_blocks``):

  (i)   coverage: the mapped broadcast frames cover at least ``min_coverage_ratio`` of the
        evaluated tactical frames;
  (ii)  player pairing: the median distance from each visible broadcast player to the closest
        tactical player of the same team, over mapped frames, is at most
        ``max_player_median_cost_m``;
  (iii) ball recovery potential: on evaluated tactical frames with ``ball_observed == 0``, the
        share that has a mapped broadcast frame which sees the ball is at least
        ``min_ball_recovery_ratio``;
  (iv)  ball agreement: where both cameras see the ball on mapped frames, the median distance
        between the two positions is at most ``max_ball_median_dist_m``.

GO if all four pass, NO-GO otherwise. A criterion that cannot be evaluated (no mapped frames, no
pairs) fails. Exit code 0 after a clean run (GO or NO-GO: the decision is the output, not an
error), 1 if the inputs could not be loaded.

The tactical ``ball_observed`` flag comes from the BAR-ATH master (``--master``, default: the one
in MTAH_DATA_DIR), as in ``tools/compare_external_realism.py``. Ball and player positions of both
cameras are the raw, unoriented ATD coordinates; the broadcast is rotated into the orientation
that ``align_broadcast`` chose.
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

import games  # noqa: E402
from modeling import io  # noqa: E402
from modeling.broadcast.align import (  # noqa: E402
    TEAM_IDS, Mapping, Tracking, align_broadcast, load_broadcast_inputs, orient_tracking)
from modeling.config import load_config  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "broadcast.yaml"

BALL_OBSERVED = 1
BALL_LOST = 0

# Memory only: how many mapped frames are paired at once. The result is a median over the pooled
# distances, so it does not depend on this value.
PAIRING_CHUNK_FRAMES = 2000


@dataclass
class CheckResult:
    key: str
    title: str
    passed: bool
    detail: str = ""


# -- helpers ------------------------------------------------------------------------------------

def _fmt(x: float | None) -> str:
    return "n/a" if x is None else "{:.4f}".format(x)


def _ball_observed(master, tactical: Tracking) -> np.ndarray:
    """The master's ``ball_observed`` column as float64[T], checked against the tactical halves."""
    observed = master["ball_observed"].to_numpy(dtype=np.float64)
    if observed.shape[0] != tactical.half.shape[0]:
        raise ValueError("master has {} frames but the tactical tracking has {}".format(
            observed.shape[0], tactical.half.shape[0]))
    if not np.array_equal(master["half"].to_numpy() != 0, tactical.half != 0):
        raise ValueError("master half blocks disagree with games.half_video_blocks")
    return observed


def evaluated_frames(tactical: Tracking) -> np.ndarray:
    """bool[T_tactical]: frames inside the evaluated halves."""
    return tactical.half != 0


def mapped_pairs(mapping: Mapping, evaluated: np.ndarray):
    """(broadcast_frames, tactical_frames) of every mapped frame that lies in an evaluated half."""
    b_idx = np.flatnonzero(mapping.tactical_frame >= 0)
    t_idx = mapping.tactical_frame[b_idx].astype(np.int64)
    if t_idx.size and t_idx.max() >= evaluated.size:
        raise ValueError("mapping points past the end of the tactical timeline")
    keep = evaluated[t_idx]
    return b_idx[keep], t_idx[keep]


def _ball_present(ball: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """bool[T]: a finite ball position on a frame with a valid homography."""
    return np.isfinite(ball).all(axis=1) & valid


def pairing_distances(tactical: Tracking, broadcast: Tracking, b_idx, t_idx) -> np.ndarray:
    """Distance from each visible broadcast player to the closest same-team tactical player.

    One value per (mapped frame, visible broadcast player). Broadcast frames with an invalid
    homography are skipped; a frame with no tactical player of the team contributes nothing.
    ``broadcast`` must already be in the orientation chosen by the alignment.
    """
    pooled = []
    for start in range(0, b_idx.size, PAIRING_CHUNK_FRAMES):
        b_chunk = b_idx[start:start + PAIRING_CHUNK_FRAMES]
        t_chunk = t_idx[start:start + PAIRING_CHUNK_FRAMES]
        usable = broadcast.valid[b_chunk]
        b_chunk, t_chunk = b_chunk[usable], t_chunk[usable]
        if b_chunk.size == 0:
            continue
        b_all = broadcast.players[b_chunk]
        t_all = tactical.players[t_chunk]
        for team_id in TEAM_IDS:
            bp = b_all[:, broadcast.team == team_id, :]
            tp = t_all[:, tactical.team == team_id, :]
            if bp.shape[1] == 0 or tp.shape[1] == 0:
                continue
            dist = np.hypot(bp[:, :, None, 0] - tp[:, None, :, 0],
                            bp[:, :, None, 1] - tp[:, None, :, 1])
            dist = np.where(np.isnan(dist), np.inf, dist)
            closest = dist.min(axis=2)
            pooled.append(closest[np.isfinite(closest)])
    return np.concatenate(pooled) if pooled else np.empty(0, dtype=np.float64)


# -- criteria -----------------------------------------------------------------------------------

def check_coverage(t_idx, evaluated, cfg) -> CheckResult:
    """(i) share of evaluated tactical frames that some mapped broadcast frame shows."""
    covered = np.zeros(evaluated.size, dtype=bool)
    covered[t_idx] = True
    n_eval = int(evaluated.sum())
    n_cov = int(covered[evaluated].sum())
    ratio = n_cov / n_eval if n_eval else None
    passed = ratio is not None and ratio >= cfg["min_coverage_ratio"]
    return CheckResult(
        "i", "mapping coverage of the evaluated frames", passed,
        "coverage {} (need >= {}); {} of {} evaluated frames covered".format(
            _fmt(ratio), _fmt(cfg["min_coverage_ratio"]), n_cov, n_eval))


def check_player_pairing(tactical, broadcast, b_idx, t_idx, cfg) -> CheckResult:
    """(ii) median player pairing distance over the mapped frames."""
    distances = pairing_distances(tactical, broadcast, b_idx, t_idx)
    median = float(np.median(distances)) if distances.size else None
    passed = median is not None and median <= cfg["max_player_median_cost_m"]
    return CheckResult(
        "ii", "median player pairing cost on mapped frames", passed,
        "median {} m (need <= {} m) over {} pairings".format(
            _fmt(median), _fmt(cfg["max_player_median_cost_m"]), distances.size))


def check_ball_recovery(broadcast, b_idx, t_idx, evaluated, tactical_observed, cfg) -> CheckResult:
    """(iii) share of ball-less tactical frames where the broadcast sees the ball."""
    seen = np.zeros(evaluated.size, dtype=bool)
    sees_ball = _ball_present(broadcast.ball, broadcast.valid)[b_idx]
    seen[t_idx[sees_ball]] = True
    lost = evaluated & (tactical_observed == BALL_LOST)
    n_lost = int(lost.sum())
    n_recovered = int(seen[lost].sum())
    ratio = n_recovered / n_lost if n_lost else None
    passed = ratio is not None and ratio >= cfg["min_ball_recovery_ratio"]
    return CheckResult(
        "iii", "broadcast ball on frames where the tactical ball is missing", passed,
        "recovery {} (need >= {}); {} of {} ball-less evaluated frames".format(
            _fmt(ratio), _fmt(cfg["min_ball_recovery_ratio"]), n_recovered, n_lost))


def check_ball_agreement(tactical, broadcast, b_idx, t_idx, tactical_observed, cfg) -> CheckResult:
    """(iv) median ball distance where both cameras see the ball on mapped frames."""
    b_sees = _ball_present(broadcast.ball, broadcast.valid)[b_idx]
    t_sees = (tactical_observed[t_idx] == BALL_OBSERVED) & np.isfinite(tactical.ball[t_idx]).all(axis=1)
    both = b_sees & t_sees
    if both.any():
        delta = broadcast.ball[b_idx[both]] - tactical.ball[t_idx[both]]
        median = float(np.median(np.hypot(delta[:, 0], delta[:, 1])))
    else:
        median = None
    passed = median is not None and median <= cfg["max_ball_median_dist_m"]
    return CheckResult(
        "iv", "ball agreement where both cameras see it", passed,
        "median {} m (need <= {} m) over {} frames".format(
            _fmt(median), _fmt(cfg["max_ball_median_dist_m"]), int(both.sum())))


def run_checks(tactical: Tracking, broadcast: Tracking, mapping: Mapping,
               tactical_observed: np.ndarray, cfg) -> list[CheckResult]:
    """The four criteria, evaluated inside the evaluated halves only."""
    evaluated = evaluated_frames(tactical)
    oriented = orient_tracking(broadcast, mapping.orientation)
    b_idx, t_idx = mapped_pairs(mapping, evaluated)
    return [
        check_coverage(t_idx, evaluated, cfg),
        check_player_pairing(tactical, oriented, b_idx, t_idx, cfg),
        check_ball_recovery(oriented, b_idx, t_idx, evaluated, tactical_observed, cfg),
        check_ball_agreement(tactical, oriented, b_idx, t_idx, tactical_observed, cfg),
    ]


def decision(results: list[CheckResult]) -> str:
    return "GO" if all(r.passed for r in results) else "NO-GO"


def report(results: list[CheckResult], mapping: Mapping | None = None,
           verbose: bool = False) -> list[str]:
    """Output lines. Without ``verbose`` only PASS/FAIL and GO/NO-GO: no numbers at all."""
    lines = []
    for r in results:
        lines.append("{}  ({}) {}".format("PASS" if r.passed else "FAIL", r.key, r.title))
        if verbose and r.detail:
            lines.append("        {}".format(r.detail))
    if verbose and mapping is not None:
        diag = mapping.diagnostics
        lines.append("        orientation {}; {} live segments; {:.4f} of broadcast frames mapped".format(
            mapping.orientation, len(diag.segments), mapping.mapped_ratio()))
    lines.append(decision(results))
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GO / NO-GO feasibility gate for the broadcast camera (PASS/FAIL only)."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="broadcast configuration (default: %(default)s)",
    )
    parser.add_argument(
        "--master",
        help="master parquet for the tactical ball_observed flag "
             "(default: the BAR-ATH master in MTAH_DATA_DIR)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="also print the numbers behind each criterion (for the person only)",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
        tactical, broadcast = load_broadcast_inputs(cfg)
        master = io.load_master(args.master, n_frames=games.source_frames("tactical"))
        tactical_observed = _ball_observed(master, tactical)
        mapping = align_broadcast(tactical, broadcast, cfg)
        results = run_checks(tactical, broadcast, mapping, tactical_observed, cfg)
    except (OSError, ValueError, KeyError) as exc:  # ConfigError is a ValueError
        print("FAIL  could not load inputs: {}".format(exc))
        return 1
    for line in report(results, mapping, verbose=args.verbose):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())