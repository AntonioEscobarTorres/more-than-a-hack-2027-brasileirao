"""Realism comparison of degraded external (IDSSE) masters against the BAR-ATH master (F4-04).

Local execution only (a person runs it on the match data):

    python tools/compare_external_realism.py
    python tools/compare_external_realism.py --degrade default hidelo hidehi --verbose

By default it prints only PASS/FAIL per statistic. ``--verbose`` adds the numbers behind each
statistic, for the person at the terminal only: they never go to issues, PRs or agents. No
ground-truth labels are used: every statistic comes from master features or from the baseline
predictions on the masters (``configs/baseline.yaml`` via ``predict_labels``).

The tested value is always the BAR-ATH master; the reference is the mean over the degraded
external masters (every selected match x degradation). All statistics use half frames
(``half != 0``) only.

Statistics:
  (a)  median of ``mean_speed_home`` and ``mean_speed_away`` combined, within +/-20%;
  (b)  median of ``ball_speed`` on frames with ``ball_observed == 1``, within +/-20%;
  (c)  fraction of frames with the ball stationary for 1 s (displacement between t-12 and t+12
       below 0.5 m, both frames observed, same half), within +/-20%;
  (d)  baseline possessions per minute, per half, within +/-20% (both halves must pass);
  (e)  fraction of frames inside ball gaps longer than 125 frames, within +/-20%;
  (f)  ball visibility P(ball_observed == 1) by player speed bin, absolute difference <= 0.10 in
       every bin holding at least 2% of the half frames (smaller bins are reported, never
       failed). Only valid when (a) passes; otherwise it fails.

Exit code 0 if every statistic passes, 1 otherwise.
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
import yaml  # noqa: E402

import games  # noqa: E402
from evaluation import possession_metrics as pm  # noqa: E402
from modeling import io  # noqa: E402
from modeling.baseline import baseline_labels  # noqa: E402
from modeling.external.files import IDSSE_MATCHES, load_external  # noqa: E402

DEFAULT_BASELINE_CONFIG = REPO_ROOT / "configs" / "baseline.yaml"
DEFAULT_DEGRADE = "default"

# Fixed by the task specification (F4-04), not tuned.
TOLERANCE_REL = 0.20
FRAMES_PER_SECOND = 25
SECONDS_PER_MINUTE = 60
HALVES = (1, 2)
BALL_OBSERVED = 1
STATIONARY_HALF_WINDOW_FRAMES = 12
STATIONARY_MAX_DISPLACEMENT_M = 0.5
MIN_LONG_GAP_FRAMES = 125
SPEED_BIN_EDGES_MPS = (1.0, 2.0)
MIN_BIN_SHARE = 0.02
MAX_BIN_VISIBILITY_DIFF = 0.10

SPEED_BIN_LABELS = (
    "v < {:.1f}".format(SPEED_BIN_EDGES_MPS[0]),
    "{:.1f} <= v < {:.1f}".format(*SPEED_BIN_EDGES_MPS),
    "v >= {:.1f}".format(SPEED_BIN_EDGES_MPS[-1]),
)


@dataclass
class CheckResult:
    key: str
    title: str
    passed: bool
    detail: str = ""


def predict_labels(master: pd.DataFrame, config: dict) -> np.ndarray:
    """Run the rule-based baseline labeller from a loaded baseline config dict."""
    params = config.get("baseline", config) if isinstance(config, dict) else config
    return baseline_labels(master, params)


# -- helpers ------------------------------------------------------------------------------------

def _n_frames() -> int:
    return int(games.GAMES[games.GAME]["n_video_frames"])


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else "{:.4f}".format(x)


def _mean_defined(values) -> float | None:
    """Mean of the values that are not None; None if there is none."""
    defined = [v for v in values if v is not None]
    return float(np.mean(defined)) if defined else None


def _within_tolerance(tested: float | None, reference: float | None) -> bool:
    """``|tested - reference| <= TOLERANCE_REL * |reference|``; False if either is missing."""
    if tested is None or reference is None:
        return False
    return abs(tested - reference) <= TOLERANCE_REL * abs(reference)


def _relative_diff(tested: float | None, reference: float | None) -> str:
    if tested is None or reference is None or reference == 0.0:
        return "n/a"
    return "{:+.1f}%".format(100.0 * (tested - reference) / abs(reference))


def _column(master: pd.DataFrame, name: str) -> np.ndarray:
    return master[name].to_numpy(dtype=np.float64)


def _observed(master: pd.DataFrame) -> np.ndarray:
    return _column(master, "ball_observed") == BALL_OBSERVED


def _gap_runs(missing: np.ndarray) -> np.ndarray:
    """Lengths of the runs of True in ``missing``."""
    changes = np.diff(np.r_[False, missing, False].astype(np.int8))
    return np.flatnonzero(changes == -1) - np.flatnonzero(changes == 1)


def _possession_runs(codes: np.ndarray) -> int:
    """Number of maximal runs of a constant home or away code (supports string or int codes)."""
    if codes.size == 0:
        return 0
    if codes.dtype.kind in ("U", "S", "O"):
        alive = (codes == "H") | (codes == "A")
    else:
        alive = (codes == pm.H) | (codes == pm.A)
    boundaries = np.r_[True, codes[1:] != codes[:-1]]
    return int(np.sum(alive & boundaries))


# -- per-master statistics ----------------------------------------------------------------------

def stat_speed_median(master: pd.DataFrame) -> float | None:
    graded = _column(master, "half") != 0
    speeds = np.concatenate([
        _column(master, "mean_speed_home")[graded],
        _column(master, "mean_speed_away")[graded],
    ])
    speeds = speeds[np.isfinite(speeds)]
    return float(np.median(speeds)) if speeds.size else None


def stat_ball_speed_median(master: pd.DataFrame) -> float | None:
    graded = _column(master, "half") != 0
    speeds = _column(master, "ball_speed")[graded & _observed(master)]
    speeds = speeds[np.isfinite(speeds)]
    return float(np.median(speeds)) if speeds.size else None


def stat_stationary_fraction(master: pd.DataFrame) -> float | None:
    """Share of valid frames whose ball moved less than the threshold between t-W and t+W."""
    half = _column(master, "half")
    x, y, observed = _column(master, "ball_x"), _column(master, "ball_y"), _observed(master)
    window = STATIONARY_HALF_WINDOW_FRAMES
    stationary = valid = 0
    for period in HALVES:
        idx = np.flatnonzero(half == period)
        if idx.size <= 2 * window:
            continue
        px, py, obs = x[idx], y[idx], observed[idx]
        before = slice(0, idx.size - 2 * window)
        after = slice(2 * window, idx.size)
        ok = (
            obs[before]
            & obs[after]
            & np.isfinite(px[before])
            & np.isfinite(py[before])
            & np.isfinite(px[after])
            & np.isfinite(py[after])
        )
        distance = np.hypot(px[after] - px[before], py[after] - py[before])
        stationary += int(np.sum(distance[ok] < STATIONARY_MAX_DISPLACEMENT_M))
        valid += int(ok.sum())
    return stationary / valid if valid else None


def stat_possessions_per_minute(master: pd.DataFrame, config: dict) -> dict[int, float | None]:
    half = _column(master, "half")
    codes = np.asarray(predict_labels(master, config))
    rates: dict[int, float | None] = {}
    for period in HALVES:
        idx = np.flatnonzero(half == period)
        minutes = idx.size / FRAMES_PER_SECOND / SECONDS_PER_MINUTE
        rates[period] = _possession_runs(codes[idx]) / minutes if idx.size else None
    return rates


def stat_long_gap_fraction(master: pd.DataFrame) -> float | None:
    half = _column(master, "half")
    observed = _observed(master)
    in_long_gaps = 0
    for period in HALVES:
        lengths = _gap_runs(~observed[half == period])
        in_long_gaps += int(lengths[lengths > MIN_LONG_GAP_FRAMES].sum())
    n_graded = int(np.sum(half != 0))
    return in_long_gaps / n_graded if n_graded else None


def stat_visibility_by_speed(master: pd.DataFrame) -> tuple[list[float | None], list[float]]:
    """(P(ball observed) per speed bin, share of half frames per bin)."""
    graded = _column(master, "half") != 0
    speed = (_column(master, "mean_speed_home") + _column(master, "mean_speed_away")) / 2.0
    observed = _observed(master)
    usable = graded & np.isfinite(speed)
    bins = np.digitize(speed[usable], SPEED_BIN_EDGES_MPS)
    n_graded = int(graded.sum())
    visibility: list[float | None] = []
    share: list[float] = []
    for index in range(len(SPEED_BIN_LABELS)):
        in_bin = bins == index
        visibility.append(float(observed[usable][in_bin].mean()) if in_bin.any() else None)
        share.append(float(in_bin.sum()) / n_graded if n_graded else 0.0)
    return visibility, share


def master_statistics(master: pd.DataFrame, config: dict) -> dict:
    visibility, share = stat_visibility_by_speed(master)
    return {
        "a": stat_speed_median(master),
        "b": stat_ball_speed_median(master),
        "c": stat_stationary_fraction(master),
        "d": stat_possessions_per_minute(master, config),
        "e": stat_long_gap_fraction(master),
        "f": visibility,
        "f_share": share,
    }


# -- comparison ---------------------------------------------------------------------------------

def _scalar_check(
    key: str, title: str, tested: float | None, references: list[float | None]
) -> CheckResult:
    reference = _mean_defined(references)
    return CheckResult(
        key,
        title,
        _within_tolerance(tested, reference),
        "BAR-ATH {}; degraded mean {}; diff {}".format(
            _fmt(tested), _fmt(reference), _relative_diff(tested, reference)
        ),
    )


def run_checks(reference: dict, degraded: list[dict]) -> list[CheckResult]:
    """All six statistics. ``reference`` is the BAR-ATH master, ``degraded`` the external ones."""
    results = [
        _scalar_check(
            "a",
            "median player speed (home and away)",
            reference["a"],
            [s["a"] for s in degraded],
        ),
        _scalar_check(
            "b",
            "median ball speed on observed frames",
            reference["b"],
            [s["b"] for s in degraded],
        ),
        _scalar_check(
            "c",
            "fraction of frames with the ball stationary for 1 s",
            reference["c"],
            [s["c"] for s in degraded],
        ),
    ]

    # (d) every half must be within tolerance.
    parts, ok_d = [], True
    for period in HALVES:
        tested = reference["d"][period]
        mean = _mean_defined([s["d"][period] for s in degraded])
        ok = _within_tolerance(tested, mean)
        ok_d &= ok
        parts.append(
            "half {}: BAR-ATH {} degraded mean {} diff {}".format(
                period, _fmt(tested), _fmt(mean), _relative_diff(tested, mean)
            )
        )
    results.append(
        CheckResult(
            "d", "baseline possessions per minute, per half", ok_d, "; ".join(parts)
        )
    )

    results.append(
        _scalar_check(
            "e",
            "fraction of frames in long ball gaps",
            reference["e"],
            [s["e"] for s in degraded],
        )
    )

    # (f) only valid when (a) passes.
    if not results[0].passed:
        results.append(
            CheckResult(
                "f", "ball visibility by player speed", False, "not valid: (a) failed"
            )
        )
        return results
    parts, ok_f, n_tested = [], True, 0
    for index, label in enumerate(SPEED_BIN_LABELS):
        tested = reference["f"][index]
        mean = _mean_defined([s["f"][index] for s in degraded])
        share = reference["f_share"][index]
        is_tested = share >= MIN_BIN_SHARE
        diff = abs(tested - mean) if tested is not None and mean is not None else None
        if is_tested:
            n_tested += 1
            ok_f &= diff is not None and diff <= MAX_BIN_VISIBILITY_DIFF
        parts.append(
            "{} ({:.1f}% of frames, {}): BAR-ATH {} degraded {} diff {}".format(
                label,
                100.0 * share,
                "tested" if is_tested else "reported only",
                _fmt(tested),
                _fmt(mean),
                _fmt(diff),
            )
        )
    results.append(
        CheckResult(
            "f",
            "ball visibility by player speed",
            ok_f and n_tested > 0,
            "; ".join(parts),
        )
    )
    return results


def report(results: list[CheckResult], verbose: bool = False) -> list[str]:
    lines = []
    for r in results:
        lines.append("{}  ({}) {}".format("PASS" if r.passed else "FAIL", r.key, r.title))
        if verbose and r.detail:
            lines.append("        {}".format(r.detail))
    n_fail = sum(not r.passed for r in results)
    lines.append(
        "PASS"
        if n_fail == 0
        else "FAIL ({} of {} statistics)".format(n_fail, len(results))
    )
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PASS/FAIL realism statistics of degraded external masters vs BAR-ATH."
    )
    parser.add_argument(
        "--degrade",
        nargs="+",
        default=[DEFAULT_DEGRADE],
        help="degradation names to average (default: %(default)s)",
    )
    parser.add_argument(
        "--match",
        nargs="+",
        default=list(IDSSE_MATCHES),
        help="external matches to average (default: all IDSSE matches)",
    )
    parser.add_argument(
        "--baseline-config",
        default=str(DEFAULT_BASELINE_CONFIG),
        help="baseline configuration (default: %(default)s)",
    )
    parser.add_argument(
        "--master",
        help="master parquet (default: the BAR-ATH master in MTAH_DATA_DIR)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="also print the numbers behind each statistic (for the person only)",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        with open(args.baseline_config, encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        reference = master_statistics(
            io.load_master(args.master, n_frames=_n_frames()), config
        )
        degraded = []
        for match in args.match:
            for degrade in args.degrade:
                master, _ = load_external(match, degrade)
                degraded.append(master_statistics(master, config))
    except (OSError, ValueError) as exc:
        print("FAIL  could not load inputs: {}".format(exc))
        return 1
    results = run_checks(reference, degraded)
    for line in report(results, verbose=args.verbose):
        print(line)
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())