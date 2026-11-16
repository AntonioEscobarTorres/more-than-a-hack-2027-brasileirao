"""Shift of the calibrated team_logit touch inputs between the degraded IDSSE and the BAR-ATH.

Local execution only (a person runs it on the match data; F1-05 review, decision D-020):

    python tools/touch_feature_shift.py
    python tools/touch_feature_shift.py --degrade default hidelo hidehi --verbose

By default it prints only PASS/FAIL per check. ``--verbose`` adds the numbers behind each check,
for the person at the terminal only: they never go to issues, PRs or agents. No labels are read:
every statistic comes from master features or from the calibrated ``team_logit`` (model ``all``).

Why: the touch feature ``f9`` of ``configs/calib/team.yaml`` needs an abrupt ball change
(``ball_acc`` or ``ball_dir_change`` above a threshold) inside a centered window. On the degraded
IDSSE the ball position noise makes that condition true on about 99% of the frames with a valid
ball, so the fitted weight of ``f9`` in practice rewards "same team close to the ball on most of
the window". If the BAR-ATH ball is smoother, the condition fires less often, ``f9`` is weaker
than in training and the calibrated ``team_logit`` loses part of its evidence. All statistics use
frames with a valid ball inside a half. The tested value is the BAR-ATH master; the reference is
the mean over the degraded external masters (every selected match x degradation).

Checks (which ones are eliminatory depends on the calibration file):

  (a)  share of valid-ball frames where the abrupt-change condition holds is within
       ``ABRUPT_SHARE_TOL_ABS`` (absolute) of the reference. Eliminatory when the calibration
       requires the condition (``touch_require_abrupt``, default true); otherwise reported as
       PASS "not used".
  (b)  share of valid-ball frames where ``f9 != 0`` within ``TOUCH_SHARE_TOL_REL`` (relative).
       Eliminatory when the touch uses persistence only (``touch_require_abrupt: false``, D-021):
       ``f9`` then depends on nothing else; informative otherwise (it follows (a)).
  (c)  share of half frames with ``|team_logit| >= STRONG_LOGIT`` within
       ``STRONG_SHARE_TOL_REL`` (relative). Always informative (play styles differ).

Without the touch feature (``f9`` absent, for example ``configs/calib/team_base.yaml``) (a) and
(b) are reported as PASS "not used".

If an eliminatory check FAILs, use a calibration without the touch (``configs/v2_base.yaml``),
fitted on the IDSSE only.

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
from modeling.evidence.team import (  # noqa: E402
    build_touch_feature,
    load_team_calibration,
    team_logit_calibrated,
    touch_components,
)
from modeling.external.files import IDSSE_MATCHES, load_external  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "v2.yaml"
DEFAULT_CALIBRATION = REPO_ROOT / "configs" / "calib" / "team.yaml"
DEFAULT_DEGRADE = "default"

# Fixed by this check (see the module docstring), not tuned.
ABRUPT_SHARE_TOL_ABS = 0.10
TOUCH_SHARE_TOL_REL = 0.30
STRONG_LOGIT = 2.0
STRONG_SHARE_TOL_REL = 0.20


@dataclass(frozen=True)
class CheckResult:
    key: str
    title: str
    passed: bool
    eliminatory: bool
    detail: str = ""


def _n_frames() -> int:
    return int(games.GAMES[games.GAME]["n_video_frames"])


def master_statistics(master: pd.DataFrame, params: dict, calibration_file: str) -> dict:
    """Statistics of one master (no labels). ``params`` are the config's ``team.params``."""
    calibration = load_team_calibration(calibration_file)
    f_params = {**params, **calibration.get("feature_params", {})}
    comp = touch_components(master, f_params)
    valid = comp["valid"]
    n_valid = int(valid.sum())
    f9 = build_touch_feature(master, f_params).to_numpy(dtype="float64")
    touch = np.isfinite(f9) & (f9 != 0.0)
    in_half = master["half"].to_numpy() != 0
    t = team_logit_calibrated(master, {**params, "calibrated_file": str(calibration_file)},
                              holdout=None).astype("float64")[in_half]
    return {
        "abrupt_share": float(comp["abrupt"][valid].mean()) if n_valid else None,
        "touch_share": float(touch[valid].mean()) if n_valid else None,
        "strong_share": float(np.mean(np.abs(t) >= STRONG_LOGIT)) if t.size else None,
    }


def _mean(stats: list[dict], key: str) -> float | None:
    vals = [s[key] for s in stats if s[key] is not None]
    return float(np.mean(vals)) if vals else None


def _relative(key, title, tested, ref, tol, eliminatory) -> CheckResult:
    if tested is None or ref is None or ref <= 0:
        return CheckResult(key, title, False, eliminatory, "undefined")
    rel = tested / ref - 1.0
    return CheckResult(key, title, abs(rel) <= tol, eliminatory,
                       "{:.4f} vs {:.4f} ({:+.1%})".format(tested, ref, rel))


def calibration_mode(calibration_file: str) -> tuple[bool, bool]:
    """``(uses_touch, require_abrupt)`` of a team calibration file."""
    calibration = load_team_calibration(calibration_file)
    uses_touch = "f9" in calibration["ball"]["features"]
    require_abrupt = bool(calibration.get("feature_params", {}).get("touch_require_abrupt", True))
    return uses_touch, require_abrupt


def run_checks(reference: dict, degraded: list[dict], *, uses_touch: bool = True,
               require_abrupt: bool = True) -> list[CheckResult]:
    """``reference`` is the BAR-ATH; ``degraded`` holds one statistics dict per external master."""
    if not uses_touch:
        results = [CheckResult("a", "abrupt-change share (not used)", True, False),
                   CheckResult("b", "touch share (not used)", True, False)]
    else:
        if not require_abrupt:
            results = [CheckResult("a", "abrupt-change share (not used)", True, False)]
        else:
            tested, ref = reference["abrupt_share"], _mean(degraded, "abrupt_share")
            if tested is None or ref is None:
                results = [CheckResult("a", "abrupt-change share within tolerance", False, True,
                                       "no valid-ball frame")]
            else:
                diff = tested - ref
                results = [CheckResult("a", "abrupt-change share within tolerance",
                                       abs(diff) <= ABRUPT_SHARE_TOL_ABS, True,
                                       "{:.4f} vs {:.4f} ({:+.4f})".format(tested, ref, diff))]
        title = "touch share within tolerance" + ("" if not require_abrupt else " (informative)")
        results.append(_relative("b", title, reference["touch_share"],
                                 _mean(degraded, "touch_share"), TOUCH_SHARE_TOL_REL,
                                 not require_abrupt))
    results.append(_relative("c", "strong team_logit share within tolerance (informative)",
                             reference["strong_share"], _mean(degraded, "strong_share"),
                             STRONG_SHARE_TOL_REL, False))
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
        description="PASS/FAIL shift of the calibrated team_logit touch inputs, IDSSE vs BAR-ATH.")
    parser.add_argument("--degrade", nargs="+", default=[DEFAULT_DEGRADE],
                        help="degradation names to average (default: %(default)s)")
    parser.add_argument("--match", nargs="+", default=list(IDSSE_MATCHES),
                        help="external matches to average (default: all IDSSE matches)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="config whose team.params build the features (default: %(default)s)")
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION),
                        help="calibrated team file (default: %(default)s)")
    parser.add_argument("--master", help="master parquet (default: the BAR-ATH master)")
    parser.add_argument("--verbose", action="store_true",
                        help="also print the numbers behind each check (for the person only)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        params = load_config(args.config)["team"]["params"]
        reference = master_statistics(io.load_master(args.master, n_frames=_n_frames()), params,
                                      args.calibration)
        degraded = []
        for match in args.match:
            for degrade in args.degrade:
                master, _ = load_external(match, degrade)
                degraded.append(master_statistics(master, params, args.calibration))
    except (OSError, ValueError, KeyError) as exc:  # ConfigError is a ValueError
        print("FAIL  could not load inputs: {}".format(type(exc).__name__))
        return 2
    uses_touch, require_abrupt = calibration_mode(args.calibration)
    results = run_checks(reference, degraded, uses_touch=uses_touch,
                         require_abrupt=require_abrupt)
    for line in report(results, verbose=args.verbose):
        print(line)
    return 0 if all(r.passed for r in results if r.eliminatory) else 1


if __name__ == "__main__":
    raise SystemExit(main())
