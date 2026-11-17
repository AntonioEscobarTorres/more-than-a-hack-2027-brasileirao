"""Shift of the GBDT hazard and of the stacked team_logit between the degraded IDSSE and the BAR-ATH.

Local execution only (a person runs it on the match data; F2-09, decision D-029):

    python tools/stack_feature_shift.py --master <data>/processed_fused/master_features.parquet
    python tools/stack_feature_shift.py --master <fused master> --degrade fused hidelo hidehi --verbose

By default it prints only PASS/FAIL per check, plus the *names* of the features that fail (a) or (b)
(names, never values). ``--verbose`` adds the numbers behind each check, for the person at the
terminal only: they never go to issues, PRs or agents. No labels are read.

Why: the GBDT hazard (``configs/calib/hazard_gbdt.yaml``) and the second stage
(``configs/calib/stack.yaml``) are trained on the degraded IDSSE only. Trees do not extrapolate, so
a BAR-ATH input outside the IDSSE range gets the value of the last leaf. Their inputs are master
columns already checked by ``tools/hazard_feature_shift.py`` and ``tools/gbdt_feature_shift.py``;
this tool checks the derived features and the outputs. All statistics use frames inside a half
(1 every ``SUBSAMPLE``). On the IDSSE every match uses its leave-one-match-out models, so the IDSSE
side is out of sample like the BAR-ATH; the BAR-ATH uses the ``all`` models.

Checks:

  (a)  every GBDT hazard feature: share of BAR-ATH frames outside the pooled IDSSE
       [``Q_LO``, ``Q_HI``] quantile range at most ``OUT_SHARE_MAX``, and NaN share within
       ``NAN_SHARE_TOL_ABS`` (absolute) of the IDSSE. Eliminatory.
  (b)  every second-stage feature (first-stage ``team_logit`` and hazard windows), same rule.
       Eliminatory.
  (c)  sign agreement between the stacked and the first-stage ``team_logit`` (frames where the
       first stage is at least ``AGREE_MIN_LOGIT`` in absolute value) within ``AGREE_TOL_ABS``
       (absolute) of the IDSSE mean. The second stage must sharpen the first, not overturn it more
       often than on the IDSSE. Eliminatory.
  (d)  mean predicted change probability of the hazard within ``MEAN_PROB_TOL_REL`` (relative) of
       the IDSSE mean (the hazard is a likelihood ratio against the IDSSE base rate). Eliminatory.
  (e)  share of frames with ``|stacked team_logit| >= STRONG_LOGIT`` within
       ``STRONG_SHARE_TOL_REL`` (relative) of the IDSSE mean. Informative (play styles differ).
  (f)  only with a ``dead_stack`` section: every feature of the dead_logit second stage (``d_fit``
       of the dead GBDT, before the tag prior, and the hazard windows), the rule of (a).
       Eliminatory.

If an eliminatory check FAILs, do not submit the stacked config (``configs/final.yaml``): submit
the D-028 config ``configs/final_d028.yaml`` instead.

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
from modeling import io, variants  # noqa: E402
from modeling.config import load_config  # noqa: E402
from modeling.evidence.hazard_gbdt import (  # noqa: E402
    build_hazard_gbdt_features,
    load_hazard_gbdt_calibration,
)
from modeling.evidence.stack import (  # noqa: E402
    build_dead_stack_features,
    build_stack_features,
    dead_stack_feature_names,
    stack_feature_names,
    stack_team_logit,
)
from modeling.external.files import IDSSE_MATCHES, load_external  # noqa: E402
from modeling.pipeline import hazard_values, variant_params  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "final.yaml"
DEFAULT_DEGRADE = "fused"

# Fixed by this check (the thresholds of tools/gbdt_feature_shift.py and
# tools/hazard_feature_shift.py), not tuned.
Q_LO, Q_HI = 0.005, 0.995
OUT_SHARE_MAX = 0.05
NAN_SHARE_TOL_ABS = 0.10
AGREE_MIN_LOGIT = 1.0
AGREE_TOL_ABS = 0.05
MEAN_PROB_TOL_REL = 0.25
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


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def master_statistics(master: pd.DataFrame, cfg: dict, *, holdout: str | None) -> dict:
    """Hazard and second-stage features (inside the halves, 1 every ``SUBSAMPLE``) and outputs."""
    variants.load_kinds()
    half = master["half"].to_numpy()
    inside = np.flatnonzero(half != 0)[::SUBSAMPLE]
    calib = load_hazard_gbdt_calibration(cfg["hazard"]["calibrated_file"])
    hf = build_hazard_gbdt_features(master, calib["hazard_settings"], calib["control_params"])
    t1 = np.asarray(variants.TEAM_KINDS[cfg["team"]["kind"]](
        master, variant_params(cfg["team"]), holdout=holdout), dtype=np.float64)
    h = hazard_values(master, cfg, holdout=holdout)
    t2 = np.asarray(stack_team_logit(t1, h, half, cfg["stack"], holdout=holdout),
                    dtype=np.float64)
    sx = build_stack_features(t1, h, half)
    a, b = t1[inside], t2[inside]
    sure = np.abs(a) >= AGREE_MIN_LOGIT
    # Predicted change probability: the likelihood ratio h back on the training base rate.
    base = float(calib["meta"]["pi_all"])
    prob = _sigmoid(h[inside] + np.log(base / (1.0 - base)))
    dead_stack = cfg.get("dead_stack")
    dsx = None
    if dead_stack is not None and dead_stack["kind"] != "none":
        from modeling.deadball.dead_gbdt import gbdt_d_fit
        d_fit = gbdt_d_fit(master, variant_params(cfg["dead"]), holdout)
        dsx = build_dead_stack_features(d_fit, h, half)[inside]
    return {
        "dead_stack_features": dsx,
        "dead_stack_names": dead_stack_feature_names(),
        "hazard_features": hf.to_numpy(dtype=np.float64)[inside],
        "hazard_names": list(hf.columns),
        "stack_features": sx[inside],
        "stack_names": stack_feature_names(),
        "agree": float(np.mean(np.sign(a[sure]) == np.sign(b[sure]))) if sure.any() else None,
        "mean_prob": float(np.mean(prob)) if prob.size else None,
        "strong": float(np.mean(np.abs(b) >= STRONG_LOGIT)) if b.size else None,
    }


def range_check(key: str, title: str, X: np.ndarray, pooled: np.ndarray,
                names: list[str]) -> CheckResult:
    failing, details = [], []
    for j, name in enumerate(names):
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
    return CheckResult(key, title, not failing, True, "\n        ".join(details), tuple(failing))


def _mean(stats: list[dict], key: str) -> float | None:
    vals = [s[key] for s in stats if s[key] is not None]
    return float(np.mean(vals)) if vals else None


def run_checks(reference: dict, degraded: list[dict]) -> list[CheckResult]:
    results = [
        range_check("a", "every GBDT hazard feature inside the IDSSE range",
                    reference["hazard_features"],
                    np.vstack([d["hazard_features"] for d in degraded]), reference["hazard_names"]),
        range_check("b", "every second-stage feature inside the IDSSE range",
                    reference["stack_features"],
                    np.vstack([d["stack_features"] for d in degraded]), reference["stack_names"]),
    ]
    tested, ref = reference["agree"], _mean(degraded, "agree")
    title = "stacked/first-stage team_logit sign agreement within tolerance"
    if tested is None or ref is None:
        results.append(CheckResult("c", title, False, True, "undefined"))
    else:
        results.append(CheckResult("c", title, abs(tested - ref) <= AGREE_TOL_ABS, True,
                                   "{:.4f} vs {:.4f} ({:+.4f})".format(tested, ref, tested - ref)))
    tested, ref = reference["mean_prob"], _mean(degraded, "mean_prob")
    title = "mean hazard change probability within tolerance"
    if tested is None or ref is None or ref <= 0:
        results.append(CheckResult("d", title, False, True, "undefined"))
    else:
        rel = tested / ref - 1.0
        results.append(CheckResult("d", title, abs(rel) <= MEAN_PROB_TOL_REL, True,
                                   "{:.4f} vs {:.4f} ({:+.1%})".format(tested, ref, rel)))
    tested, ref = reference["strong"], _mean(degraded, "strong")
    title = "strong stacked team_logit share within tolerance (informative)"
    if tested is None or ref is None or ref <= 0:
        results.append(CheckResult("e", title, False, False, "undefined"))
    else:
        rel = tested / ref - 1.0
        results.append(CheckResult("e", title, abs(rel) <= STRONG_SHARE_TOL_REL, False,
                                   "{:.4f} vs {:.4f} ({:+.1%})".format(tested, ref, rel)))
    if reference.get("dead_stack_features") is not None:
        results.append(range_check(
            "f", "every dead_logit second-stage feature inside the IDSSE range",
            reference["dead_stack_features"],
            np.vstack([d["dead_stack_features"] for d in degraded]),
            reference["dead_stack_names"]))
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
        description="PASS/FAIL shift of the GBDT hazard and stacked team_logit, IDSSE vs BAR-ATH.")
    parser.add_argument("--degrade", nargs="+", default=[DEFAULT_DEGRADE],
                        help="degradation names to pool (default: %(default)s)")
    parser.add_argument("--match", nargs="+", default=list(IDSSE_MATCHES),
                        help="external matches to pool (default: all IDSSE matches)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="config with hazard.kind gbdt and a stack section "
                             "(default: %(default)s)")
    parser.add_argument("--master", help="master parquet (default: the BAR-ATH master)")
    parser.add_argument("--verbose", action="store_true",
                        help="also print the numbers behind each check (for the person only)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
        if cfg["hazard"]["kind"] != "gbdt" or cfg.get("stack", {"kind": "none"})["kind"] != "gbdt":
            raise ValueError("config must have hazard.kind gbdt and stack.kind gbdt")
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
