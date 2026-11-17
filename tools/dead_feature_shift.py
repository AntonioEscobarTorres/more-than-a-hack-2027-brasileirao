"""Shift of the calibrated dead-ball inputs between the degraded IDSSE and the BAR-ATH (F3-04 review).

Local execution only (a person runs it on the match data):

    python tools/dead_feature_shift.py
    python tools/dead_feature_shift.py --degrade default hidelo hidehi --verbose

By default it prints only PASS/FAIL per check. ``--verbose`` adds the numbers behind each check,
for the person at the terminal only: they never go to issues, PRs or agents. No labels are read:
every statistic comes from master features or from the calibrated ``d_fit`` (model ``all``).

Why: the calibrated ``dead_logit`` (``configs/calib/dead*.yaml``) leans on the smoothed mean
player speed ``db_mean_speed``: in the model ``all`` of ``dead.yaml`` its slope at the training
means is about -2.8 per m/s, so a shift of 0.25 m/s moves ``d_fit`` by about 0.7. The F4-04
realism check (a) compares only the median of the raw team speeds, within +/-20%; this one looks
at the feature the model reads, across its distribution. All statistics use half frames
(``half != 0``) only. The tested value is the BAR-ATH master; the reference is the mean over the
degraded external masters (every selected match x degradation).

Eliminatory checks (decide the overall result and the exit code):

  (a)  ``db_mean_speed`` is finite on every half frame of the BAR-ATH;
  (b)  ``db_still_frac`` is finite on every half frame of the BAR-ATH;
  (c)  every quantile of ``QUANTILES`` of ``db_mean_speed`` is within ``SPEED_QUANTILE_TOL_MPS``
       of the reference.

Informative check (printed as PASS/FAIL, never changes the overall result: the true share of dead
ball may differ between matches):

  (d)  share of half frames with ``d_fit > 0`` (model ``all`` of ``--calibration``: a logistic
       F3-04 file or a GBDT F3-08 file such as ``configs/calib/dead_gbdt.yaml``) within
       ``D_EVIDENCE_SHARE_TOL_REL`` (relative) of the reference.

Exit code 0 on PASS, 1 on FAIL, 2 if the inputs could not be loaded.
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
from modeling import io  # noqa: E402
from modeling.calibration import dead as calibration_dead  # noqa: E402
from modeling.calibration.dead import calibrated_d_fit  # noqa: E402
from modeling.deadball import dead_gbdt  # noqa: E402
from modeling.config import load_config  # noqa: E402
from modeling.deadball.features import build_deadball_features  # noqa: E402
from modeling.external.files import IDSSE_MATCHES, load_external  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "v1.yaml"
DEFAULT_CALIBRATION = REPO_ROOT / "configs" / "calib" / "dead.yaml"
DEFAULT_DEGRADE = "default"

# Fixed by this check (see the module docstring), not tuned.
QUANTILES = (10.0, 50.0, 90.0)
SPEED_QUANTILE_TOL_MPS = 0.25
D_EVIDENCE_SHARE_TOL_REL = 0.20


@dataclass(frozen=True)
class CheckResult:
    key: str
    title: str
    passed: bool
    eliminatory: bool
    detail: str = ""


def _n_frames() -> int:
    return int(games.GAMES[games.GAME]["n_video_frames"])


def load_calibration(path) -> dict:
    """A calibrated dead file: logistic (F3-04) or GBDT (F3-08, ``variant: gbdt``)."""
    import yaml

    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if isinstance(raw, dict) and raw.get("variant") == dead_gbdt.VARIANT:
        return dead_gbdt.load_calibration(path)
    return calibration_dead.load_calibration(path)


def master_statistics(master: pd.DataFrame, params: dict, calibration: dict) -> dict:
    """Statistics of one master on its half frames (no labels)."""
    in_half = master["half"].to_numpy() != 0
    db = build_deadball_features(master, params)
    speed = db["db_mean_speed"].to_numpy(dtype="float64")[in_half]
    still = db["db_still_frac"].to_numpy(dtype="float64")[in_half]
    finite_speed = speed[np.isfinite(speed)]
    quantiles = ([float(q) for q in np.percentile(finite_speed, QUANTILES)]
                 if finite_speed.size else None)
    params_file = {**params, "calibrated_file": "unused"}
    if calibration.get("variant") == dead_gbdt.VARIANT:
        d = dead_gbdt.calibrated_gbdt_d_fit(master, params_file, calibration, None)[in_half]
    else:
        d = calibrated_d_fit(master, params_file, calibration, None)[in_half]
    return {
        "speed_finite": bool(np.isfinite(speed).all()),
        "still_finite": bool(np.isfinite(still).all()),
        "speed_quantiles": quantiles,
        "d_positive_share": float(np.mean(d > 0)) if d.size else None,
    }


def _mean_quantiles(stats: list[dict]) -> list[float] | None:
    rows = [s["speed_quantiles"] for s in stats if s["speed_quantiles"] is not None]
    return [float(v) for v in np.mean(rows, axis=0)] if rows else None


def _mean_share(stats: list[dict]) -> float | None:
    vals = [s["d_positive_share"] for s in stats if s["d_positive_share"] is not None]
    return float(np.mean(vals)) if vals else None


def run_checks(reference: dict, degraded: list[dict]) -> list[CheckResult]:
    """``reference`` is the BAR-ATH; ``degraded`` holds one statistics dict per external master."""
    results = [
        CheckResult("a", "db_mean_speed finite on every half frame", reference["speed_finite"],
                    True),
        CheckResult("b", "db_still_frac finite on every half frame", reference["still_finite"],
                    True),
    ]
    tested, ref = reference["speed_quantiles"], _mean_quantiles(degraded)
    if tested is None or ref is None:
        results.append(CheckResult("c", "db_mean_speed quantiles", False, True,
                                   "no finite db_mean_speed"))
    else:
        diffs = [t - r for t, r in zip(tested, ref)]
        ok = all(abs(d) <= SPEED_QUANTILE_TOL_MPS for d in diffs)
        detail = ", ".join("q{:g}: {:.3f} vs {:.3f} ({:+.3f})".format(q, t, r, d)
                           for q, t, r, d in zip(QUANTILES, tested, ref, diffs))
        results.append(CheckResult("c", "db_mean_speed quantiles within tolerance", ok, True,
                                   detail))
    tested, ref = reference["d_positive_share"], _mean_share(degraded)
    if tested is None or ref is None or ref <= 0:
        results.append(CheckResult("d", "share of positive d_fit (informative)", False, False,
                                   "undefined"))
    else:
        rel = tested / ref - 1.0
        results.append(CheckResult(
            "d", "share of positive d_fit within tolerance (informative)",
            abs(rel) <= D_EVIDENCE_SHARE_TOL_REL, False,
            "{:.4f} vs {:.4f} ({:+.1%})".format(tested, ref, rel)))
    return results


def report(results: list[CheckResult], verbose: bool = False) -> list[str]:
    lines = []
    for r in results:
        lines.append("{}  ({}) {}".format("PASS" if r.passed else "FAIL", r.key, r.title))
        if verbose and r.detail:
            lines.append("        {}".format(r.detail))
    n_fail = sum(not r.passed for r in results if r.eliminatory)
    n_elim = sum(r.eliminatory for r in results)
    lines.append("PASS" if n_fail == 0 else "FAIL ({} of {} eliminatory checks)".format(
        n_fail, n_elim))
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PASS/FAIL shift of the calibrated dead-ball inputs, IDSSE vs BAR-ATH.")
    parser.add_argument("--degrade", nargs="+", default=[DEFAULT_DEGRADE],
                        help="degradation names to average (default: %(default)s)")
    parser.add_argument("--match", nargs="+", default=list(IDSSE_MATCHES),
                        help="external matches to average (default: all IDSSE matches)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="config whose dead.params build the features (default: %(default)s)")
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION),
                        help="calibrated dead file (default: %(default)s)")
    parser.add_argument("--master", help="master parquet (default: the BAR-ATH master)")
    parser.add_argument("--verbose", action="store_true",
                        help="also print the numbers behind each check (for the person only)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        params = load_config(args.config)["dead"]["params"]
        calibration = load_calibration(args.calibration)
        reference = master_statistics(io.load_master(args.master, n_frames=_n_frames()), params,
                                      calibration)
        degraded = []
        for match in args.match:
            for degrade in args.degrade:
                master, _ = load_external(match, degrade)
                degraded.append(master_statistics(master, params, calibration))
    except (OSError, ValueError, KeyError, RuntimeError) as exc:  # ConfigError is a ValueError
        print("FAIL  could not load inputs: {}".format(type(exc).__name__))
        return 2
    results = run_checks(reference, degraded)
    for line in report(results, verbose=args.verbose):
        print(line)
    return 0 if all(r.passed for r in results if r.eliminatory) else 1


if __name__ == "__main__":
    raise SystemExit(main())
