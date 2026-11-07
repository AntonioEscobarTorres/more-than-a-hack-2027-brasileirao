"""Sanity checks of a submission and of the master against the public ground-truth window (F5-02).

Local execution only (a person runs it on the match data):

    python tools/check_public_window.py --pred outputs/baseline.csv
    python tools/check_public_window.py --pred outputs/baseline.csv --verbose

By default it prints only PASS/FAIL per check. ``--verbose`` adds the numbers behind each check,
for the person at the terminal only: they never go to issues, PRs or agents. The public window is
never used to tune or choose anything, and this tool does not compute spellF1 or macro-F1.

Checks:
  (i)   the CSV is a valid submission (``pm.validate_submission``);
  (ii)  no X on graded frames (``half != 0``) and only X outside them;
  (iii) mean ``d_diff`` is lower on GT-H frames than on GT-A frames;
  (iv)  mean ``n_tags_favor_home - n_tags_favor_away`` is higher on GT-H than on GT-A frames;
  (v)   ``centroid_home_x < centroid_away_x`` on average over the 25 frames from each kickoff
        (``preprocessing.params.KICKOFF_FALLBACK``, the published kickoff frames);
  (vi)  on frames where the public GT and the prediction are both alive, the predicted team is
        right more than half of the time (catches swapped H/A);
  (vii) the predicted share of D is higher on GT-D frames than on GT-alive frames.

Exit code 0 if every check passes, 1 otherwise.
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
from evaluation import possession_metrics as pm  # noqa: E402
from modeling import io  # noqa: E402
from preprocessing.params import KICKOFF_FALLBACK  # noqa: E402

DEFAULT_GT = REPO_ROOT / "evaluation" / "gt" / "{}_possession_gt_public.csv".format(games.GAME)

# Fixed by the task specification (F5-02), not tuned.
KICKOFF_WINDOW_FRAMES = 25
MIN_TEAM_ACCURACY = 0.5


@dataclass
class CheckResult:
    key: str
    title: str
    passed: bool
    detail: str = ""


def _n_frames() -> int:
    return int(games.GAMES[games.GAME]["n_video_frames"])


def _mean(values: np.ndarray, mask: np.ndarray) -> float | None:
    """Mean of the finite ``values`` under ``mask``; None if there is none."""
    sel = values[mask]
    sel = sel[np.isfinite(sel)]
    return float(sel.mean()) if sel.size else None


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else "{:.4f}".format(x)


def read_prediction(path, n_frames: int) -> tuple[np.ndarray | None, str | None]:
    """(codes, None) for a valid submission, else (None, reason)."""
    try:
        table = pm.read_label_csv(path)
    except Exception as exc:  # unreadable file, bad CSV, missing columns
        return None, "could not read: {}".format(exc)
    reason = pm.validate_submission(table, n_frames)
    if reason is not None:
        return None, reason
    return pm.label_codes(table), None


def read_ground_truth(path, n_frames: int) -> np.ndarray:
    table = pm.read_label_csv(path)
    reason = pm.validate_submission(table, n_frames)
    if reason is not None:
        raise ValueError("ground truth {} is not valid: {}".format(path, reason))
    return pm.label_codes(table)


def run_checks(pred_codes: np.ndarray | None, pred_error: str | None, gt_codes: np.ndarray,
               master: pd.DataFrame) -> list[CheckResult]:
    """All seven checks. ``pred_codes`` is None when the submission is invalid."""
    n = len(master)
    if gt_codes.shape != (n,):
        raise ValueError("ground truth has {} frames, master has {}".format(gt_codes.size, n))
    half = master["half"].to_numpy()
    graded = half != 0
    gt_h, gt_a, gt_d = gt_codes == pm.H, gt_codes == pm.A, gt_codes == pm.D
    gt_live = gt_h | gt_a
    valid = pred_codes is not None
    results = []

    # (i)
    results.append(CheckResult("i", "submission is valid", valid,
                               "ok" if valid else str(pred_error)))

    # (ii)
    if valid:
        is_x = pred_codes == pm.X
        x_in = int(np.sum(is_x & graded))
        non_x_out = int(np.sum(~is_x & ~graded))
        results.append(CheckResult(
            "ii", "X exactly on the non-graded frames", x_in == 0 and non_x_out == 0,
            "X on graded frames: {}; non-X outside graded frames: {}".format(x_in, non_x_out)))
    else:
        results.append(CheckResult("ii", "X exactly on the non-graded frames", False,
                                   "invalid submission"))

    # (iii)
    d_diff = master["d_diff"].to_numpy(dtype=np.float64)
    m_h, m_a = _mean(d_diff, gt_h), _mean(d_diff, gt_a)
    results.append(CheckResult(
        "iii", "mean d_diff lower on GT-H than on GT-A",
        m_h is not None and m_a is not None and m_h < m_a,
        "GT-H {}; GT-A {}".format(_fmt(m_h), _fmt(m_a))))

    # (iv)
    balance = (master["n_tags_favor_home"].to_numpy(dtype=np.float64)
               - master["n_tags_favor_away"].to_numpy(dtype=np.float64))
    b_h, b_a = _mean(balance, gt_h), _mean(balance, gt_a)
    results.append(CheckResult(
        "iv", "tag balance (home - away) higher on GT-H than on GT-A",
        b_h is not None and b_a is not None and b_h > b_a,
        "GT-H {}; GT-A {}".format(_fmt(b_h), _fmt(b_a))))

    # (v)
    home_x = master["centroid_home_x"].to_numpy(dtype=np.float64)
    away_x = master["centroid_away_x"].to_numpy(dtype=np.float64)
    frames = np.arange(n)
    ok_v, parts = True, []
    for h, kick in sorted(KICKOFF_FALLBACK.items()):
        window = (frames >= kick) & (frames < kick + KICKOFF_WINDOW_FRAMES) & (half == h)
        c_home, c_away = _mean(home_x, window), _mean(away_x, window)
        ok = c_home is not None and c_away is not None and c_home < c_away
        ok_v &= ok
        parts.append("half {} kickoff {}: home {} away {}".format(
            h, kick, _fmt(c_home), _fmt(c_away)))
    results.append(CheckResult("v", "home centroid left of away centroid after each kickoff",
                               ok_v, "; ".join(parts)))

    # (vi)
    if valid:
        cm3 = pm.scored_confusion(pm.confusion(gt_codes, pred_codes))
        n_both = int(cm3[:pm.D, :pm.D].sum())
        accuracy = pm.team_accuracy(cm3)[1]
        results.append(CheckResult(
            "vi", "predicted team right on most frames where both are alive",
            n_both > 0 and accuracy > MIN_TEAM_ACCURACY,
            "frames alive in both: {}; team accuracy {:.4f}".format(n_both, accuracy)))
    else:
        results.append(CheckResult("vi", "predicted team right on most frames where both are "
                                   "alive", False, "invalid submission"))

    # (vii)
    if valid:
        pred_d = (pred_codes == pm.D).astype(np.float64)
        f_dead = float(pred_d[gt_d].mean()) if gt_d.any() else None
        f_live = float(pred_d[gt_live].mean()) if gt_live.any() else None
        results.append(CheckResult(
            "vii", "predicted D share higher on GT-D than on GT-alive",
            f_dead is not None and f_live is not None and f_dead > f_live,
            "D share on GT-D {}; on GT-alive {}".format(_fmt(f_dead), _fmt(f_live))))
    else:
        results.append(CheckResult("vii", "predicted D share higher on GT-D than on GT-alive",
                                   False, "invalid submission"))
    return results


def report(results: list[CheckResult], verbose: bool = False) -> list[str]:
    lines = []
    for r in results:
        lines.append("{}  ({}) {}".format("PASS" if r.passed else "FAIL", r.key, r.title))
        if verbose and r.detail:
            lines.append("        {}".format(r.detail))
    n_fail = sum(not r.passed for r in results)
    lines.append("PASS" if n_fail == 0 else "FAIL ({} of {} checks)".format(n_fail, len(results)))
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PASS/FAIL sanity checks against the public ground-truth window.")
    parser.add_argument("--pred", required=True, help="submission CSV to check")
    parser.add_argument("--verbose", action="store_true",
                        help="also print the numbers behind each check (for the person only)")
    parser.add_argument("--gt", default=str(DEFAULT_GT),
                        help="public ground truth CSV (default: %(default)s)")
    parser.add_argument("--master",
                        help="master parquet (default: the BAR-ATH master in MTAH_DATA_DIR)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    n = _n_frames()
    try:
        master = io.load_master(args.master, n_frames=n)
        gt_codes = read_ground_truth(args.gt, n)
    except (OSError, ValueError) as exc:
        print("FAIL  could not load inputs: {}".format(exc))
        return 1
    pred_codes, pred_error = read_prediction(args.pred, n)
    results = run_checks(pred_codes, pred_error, gt_codes, master)
    for line in report(results, verbose=args.verbose):
        print(line)
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
