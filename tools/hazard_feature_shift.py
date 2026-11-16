"""Shift of the possession-change hazard inputs and outputs between the degraded IDSSE and the BAR-ATH.

Local execution only (a person runs it on the match data; F2-07, decision D-024):

    python tools/hazard_feature_shift.py
    python tools/hazard_feature_shift.py --degrade default hidelo hidehi --verbose

By default it prints only PASS/FAIL per check, plus the *names* of the features that fail (a)
(names, never values). ``--verbose`` adds the numbers behind each check, for the person at the
terminal only: they never go to issues, PRs or agents. No labels are read.

Why: the hazard (``configs/calib/hazard.yaml``, F1-08) is a logistic model fitted on the degraded
IDSSE only, and with ``hazard.kind: calibrated`` it changes the decoder transitions (F2-07). A
feature whose range or meaning differs on the BAR-ATH (ball noise, camera coverage; the lesson of
D-022) would move the hazard everywhere. All statistics use frames inside a half. On the IDSSE
every match uses its leave-one-match-out fold, so the IDSSE side is out of sample like the BAR-ATH;
the BAR-ATH uses the ``all`` model.

Checks:

  (a)  every raw hazard feature: share of BAR-ATH frames outside the pooled IDSSE
       [``Q_LO``, ``Q_HI``] quantile range at most ``OUT_SHARE_MAX``, and NaN share within
       ``NAN_SHARE_TOL_ABS`` (absolute) of the IDSSE. Eliminatory.
  (b)  mean predicted change probability of the model within ``MEAN_PROB_TOL_REL`` (relative) of
       the IDSSE mean. The hazard is a likelihood ratio against the IDSSE base rate, so a large
       shift means it would push every frame the same way. Eliminatory.
  (c)  share of frames with hazard ``h >= STRONG_H`` within ``STRONG_SHARE_TOL_REL`` (relative)
       of the IDSSE mean. Informative (play styles differ).

If an eliminatory check FAILs, do not submit a config with ``hazard.kind: calibrated``: use the
same config with the hazard off.

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
from modeling.evidence.hazard import (  # noqa: E402
    build_hazard_features,
    design_columns,
    design_matrix,
    hazard_from_model,
    load_hazard_file,
    model_logit,
    select_model,
)
from modeling.external.files import IDSSE_MATCHES, load_external  # noqa: E402

DEFAULT_HAZARD_FILE = REPO_ROOT / "configs" / "calib" / "hazard.yaml"
DEFAULT_DEGRADE = "default"

# Fixed by this check (see the module docstring), not tuned.
Q_LO, Q_HI = 0.005, 0.995
OUT_SHARE_MAX = 0.05
NAN_SHARE_TOL_ABS = 0.10
MEAN_PROB_TOL_REL = 0.25
STRONG_H = 1.0
STRONG_SHARE_TOL_REL = 0.35
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


def master_statistics(master: pd.DataFrame, calib: dict, *, holdout: str | None) -> dict:
    """Raw features inside the halves (1 frame in ``SUBSAMPLE``) and the hazard statistics."""
    model = select_model(calib, holdout)
    settings = {**calib["settings"], "window_frames": int(model["window_frames"])}
    inside = master["half"].to_numpy() != 0
    raw = build_hazard_features(master, settings)
    x = design_matrix(raw, calib["features"])
    z = model_logit(x, model, design_columns(calib["features"]))[inside]
    h = hazard_from_model(master, calib, model).astype(np.float64)[inside]
    return {
        "features": raw[calib["features"]].to_numpy(dtype=np.float64)[inside][::SUBSAMPLE],
        "names": list(calib["features"]),
        "mean_prob": float(np.mean(1.0 / (1.0 + np.exp(-z)))) if z.size else None,
        "strong": float(np.mean(h >= STRONG_H)) if h.size else None,
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
    return CheckResult("a", "every hazard feature inside the IDSSE range", not failing, True,
                       "\n        ".join(details), tuple(failing))


def _mean(stats: list[dict], key: str) -> float | None:
    vals = [s[key] for s in stats if s[key] is not None]
    return float(np.mean(vals)) if vals else None


def _relative_check(key: str, title: str, tested, ref, tol: float,
                    eliminatory: bool) -> CheckResult:
    if tested is None or ref is None or ref <= 0:
        return CheckResult(key, title, False, eliminatory, "undefined")
    rel = tested / ref - 1.0
    return CheckResult(key, title, abs(rel) <= tol, eliminatory,
                       "{:.4f} vs {:.4f} ({:+.1%})".format(tested, ref, rel))


def run_checks(reference: dict, degraded: list[dict]) -> list[CheckResult]:
    return [
        feature_check(reference, degraded),
        _relative_check("b", "mean predicted change probability within tolerance",
                        reference["mean_prob"], _mean(degraded, "mean_prob"),
                        MEAN_PROB_TOL_REL, True),
        _relative_check("c", "strong hazard share within tolerance (informative)",
                        reference["strong"], _mean(degraded, "strong"),
                        STRONG_SHARE_TOL_REL, False),
    ]


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
        description="PASS/FAIL shift of the hazard inputs/outputs, IDSSE vs BAR-ATH.")
    parser.add_argument("--degrade", nargs="+", default=[DEFAULT_DEGRADE],
                        help="degradation names to pool (default: %(default)s)")
    parser.add_argument("--match", nargs="+", default=list(IDSSE_MATCHES),
                        help="external matches to pool (default: all IDSSE matches)")
    parser.add_argument("--hazard-file", default=str(DEFAULT_HAZARD_FILE),
                        help="fitted hazard file (default: %(default)s)")
    parser.add_argument("--master", help="master parquet (default: the BAR-ATH master)")
    parser.add_argument("--verbose", action="store_true",
                        help="also print the numbers behind each check (for the person only)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        calib = load_hazard_file(args.hazard_file)
        reference = master_statistics(io.load_master(args.master, n_frames=_n_frames()), calib,
                                      holdout=None)
        degraded = []
        for match in args.match:
            for degrade in args.degrade:
                master, _ = load_external(match, degrade)
                degraded.append(master_statistics(master, calib, holdout=match))
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print("FAIL  could not load inputs: {}".format(type(exc).__name__))
        return 2
    results = run_checks(reference, degraded)
    for line in report(results, verbose=args.verbose):
        print(line)
    return 0 if all(r.passed for r in results if r.eliminatory) else 1


if __name__ == "__main__":
    raise SystemExit(main())
