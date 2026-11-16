"""Shift of the GBDT team_logit inputs and outputs between the degraded IDSSE and the BAR-ATH.

Local execution only (a person runs it on the match data; F1-07, decision D-023):

    python tools/gbdt_feature_shift.py
    python tools/gbdt_feature_shift.py --degrade default hidelo hidehi --verbose

By default it prints only PASS/FAIL per check, plus the *names* of the features that fail (a)
(names, never values). ``--verbose`` adds the numbers behind each check, for the person at the
terminal only: they never go to issues, PRs or agents. No labels are read.

Why: the GBDT (``configs/calib/team_gbdt.yaml``) is trained on the degraded IDSSE only. Trees do
not extrapolate: a BAR-ATH feature outside the IDSSE range gets the value of the last leaf, and a
feature whose meaning differs between the two sources (ball noise, camera coverage) silently
changes the evidence. All statistics use frames with a valid ball inside a half (the domain of
the GBDT ``ball`` model). On the IDSSE every match uses its leave-one-match-out fold models, so
the IDSSE side is out of sample like the BAR-ATH; the BAR-ATH uses the ``all`` models.

Checks:

  (a)  every GBDT feature: share of BAR-ATH frames outside the pooled IDSSE
       [``Q_LO``, ``Q_HI``] quantile range at most ``OUT_SHARE_MAX``, and NaN share within
       ``NAN_SHARE_TOL_ABS`` (absolute) of the IDSSE. Eliminatory.
  (b)  sign agreement between the GBDT and the F1-05 logistic ``team_logit`` (frames where the
       logistic is at least ``AGREE_MIN_LOGIT`` in absolute value) within ``AGREE_TOL_ABS``
       (absolute) of the IDSSE mean. A model that learned the possession signal agrees with the
       simpler logistic about as often on both sources. Eliminatory.
  (c)  share of frames with ``|team_logit| >= STRONG_LOGIT`` within ``STRONG_SHARE_TOL_REL``
       (relative) of the IDSSE mean. Informative (play styles differ).

If an eliminatory check FAILs, do not submit the GBDT config: keep ``configs/v2.yaml``.

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
from modeling.config import load_config  # noqa: E402
from modeling.evidence.team import team_logit_calibrated, team_logit_gbdt  # noqa: E402
from modeling.evidence.team_gbdt import (  # noqa: E402
    build_gbdt_features,
    load_gbdt_calibration,
    resolve_repo_path,
)
from modeling.external.files import IDSSE_MATCHES, load_external  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "v2_gbdt.yaml"
DEFAULT_DEGRADE = "default"

# Fixed by this check (see the module docstring), not tuned.
Q_LO, Q_HI = 0.005, 0.995
OUT_SHARE_MAX = 0.05
NAN_SHARE_TOL_ABS = 0.10
AGREE_MIN_LOGIT = 1.0
AGREE_TOL_ABS = 0.05
STRONG_LOGIT = 2.0
STRONG_SHARE_TOL_REL = 0.20
SUBSAMPLE = 5


@dataclass(frozen=True)
class CheckResult:
    key: str
    title: str
    passed: bool
    eliminatory: bool
    detail: str = ""
    names: tuple = ()


def _n_frames() -> int:
    return int(games.GAMES[games.GAME]["n_video_frames"])


def master_statistics(master: pd.DataFrame, cfg: dict, *, holdout: str | None) -> dict:
    """Features on valid-ball frames (1 every ``SUBSAMPLE``) and the logit statistics."""
    params = cfg["team"]["params"]
    gbdt_file = str(resolve_repo_path(cfg["team"]["calibrated_file"]))
    calib = load_gbdt_calibration(gbdt_file)
    feats, valid = build_gbdt_features(master, params, calib["feature_params"])
    t_g = team_logit_gbdt(master, {**params, "calibrated_file": gbdt_file},
                          holdout=holdout).astype(np.float64)
    t_l = team_logit_calibrated(
        master, {**params, "calibrated_file": str(resolve_repo_path(calib["logistic_file"]))},
        holdout=holdout).astype(np.float64)
    tg, tl = t_g[valid], t_l[valid]
    sure = np.abs(tl) >= AGREE_MIN_LOGIT
    return {
        "features": feats.to_numpy(dtype=np.float64)[valid][::SUBSAMPLE],
        "names": list(feats.columns),
        "agree": float(np.mean(np.sign(tg[sure]) == np.sign(tl[sure]))) if sure.any() else None,
        "strong": float(np.mean(np.abs(tg) >= STRONG_LOGIT)) if tg.size else None,
    }


def feature_check(reference: dict, degraded: list[dict]) -> CheckResult:
    pooled = np.vstack([d["features"] for d in degraded])
    X = reference["features"]
    failing, details = [], []
    for j, name in enumerate(reference["names"]):
        ref_col, x = pooled[:, j], X[:, j]
        ref_fin, x_fin = np.isfinite(ref_col), np.isfinite(x)
        nan_diff = float(np.mean(~x_fin) - np.mean(~ref_fin)) if len(x) else float("nan")
        if ref_fin.any() and x_fin.any():
            lo, hi = np.quantile(ref_col[ref_fin], [Q_LO, Q_HI])
            out = float(np.mean((x[x_fin] < lo) | (x[x_fin] > hi)))
        else:
            out = 0.0 if not x_fin.any() else 1.0
        ok = out <= OUT_SHARE_MAX and abs(nan_diff) <= NAN_SHARE_TOL_ABS
        if not ok:
            failing.append(name)
        details.append("{}: outside {:.3f}, NaN diff {:+.3f}".format(name, out, nan_diff))
    return CheckResult("a", "every GBDT feature inside the IDSSE range", not failing, True,
                       "\n        ".join(details), tuple(failing))


def _mean(stats: list[dict], key: str) -> float | None:
    vals = [s[key] for s in stats if s[key] is not None]
    return float(np.mean(vals)) if vals else None


def run_checks(reference: dict, degraded: list[dict]) -> list[CheckResult]:
    results = [feature_check(reference, degraded)]
    tested, ref = reference["agree"], _mean(degraded, "agree")
    if tested is None or ref is None:
        results.append(CheckResult("b", "GBDT/logistic sign agreement within tolerance", False,
                                   True, "undefined"))
    else:
        results.append(CheckResult("b", "GBDT/logistic sign agreement within tolerance",
                                   abs(tested - ref) <= AGREE_TOL_ABS, True,
                                   "{:.4f} vs {:.4f} ({:+.4f})".format(tested, ref, tested - ref)))
    tested, ref = reference["strong"], _mean(degraded, "strong")
    if tested is None or ref is None or ref <= 0:
        results.append(CheckResult("c", "strong GBDT logit share (informative)", False, False,
                                   "undefined"))
    else:
        rel = tested / ref - 1.0
        results.append(CheckResult("c", "strong GBDT logit share within tolerance (informative)",
                                   abs(rel) <= STRONG_SHARE_TOL_REL, False,
                                   "{:.4f} vs {:.4f} ({:+.1%})".format(tested, ref, rel)))
    return results


def report(results: list[CheckResult], verbose: bool = False) -> list[str]:
    lines = []
    for r in results:
        lines.append("{}  ({}) {}".format("PASS" if r.passed else "FAIL", r.key, r.title))
        if r.names:
            lines.append("        failing features: {}".format(", ".join(r.names)))
        if verbose and r.detail:
            lines.append("        {}".format(r.detail))
    n_fail = sum(not r.passed for r in results if r.eliminatory)
    n_elim = sum(r.eliminatory for r in results)
    lines.append("PASS" if n_fail == 0 else "FAIL ({} of {} eliminatory checks)".format(
        n_fail, n_elim))
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PASS/FAIL shift of the GBDT team_logit inputs/outputs, IDSSE vs BAR-ATH.")
    parser.add_argument("--degrade", nargs="+", default=[DEFAULT_DEGRADE],
                        help="degradation names to pool (default: %(default)s)")
    parser.add_argument("--match", nargs="+", default=list(IDSSE_MATCHES),
                        help="external matches to pool (default: all IDSSE matches)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="config with team.kind gbdt (default: %(default)s)")
    parser.add_argument("--master", help="master parquet (default: the BAR-ATH master)")
    parser.add_argument("--verbose", action="store_true",
                        help="also print the numbers behind each check (for the person only)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
        if cfg["team"]["kind"] != "gbdt":
            raise ValueError("config team.kind must be gbdt")
        reference = master_statistics(io.load_master(args.master, n_frames=_n_frames()), cfg,
                                      holdout=None)
        degraded = []
        for match in args.match:
            for degrade in args.degrade:
                master, _ = load_external(match, degrade)
                degraded.append(master_statistics(master, cfg, holdout=match))
    except (OSError, ValueError, KeyError, RuntimeError) as exc:  # ConfigError is a ValueError
        print("FAIL  could not load inputs: {}".format(type(exc).__name__))
        return 2
    results = run_checks(reference, degraded)
    for line in report(results, verbose=args.verbose):
        print(line)
    return 0 if all(r.passed for r in results if r.eliminatory) else 1


if __name__ == "__main__":
    raise SystemExit(main())
