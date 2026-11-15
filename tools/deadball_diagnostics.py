"""Dead-ball and possession plausibility of a submission on the provided match (F3-06).

Local execution only (a person runs it on the match data):

    python tools/deadball_diagnostics.py --pred outputs/v1.csv
    python tools/deadball_diagnostics.py --pred outputs/v1.csv --verbose

By default it prints only PASS/FAIL per check. ``--verbose`` adds the numbers behind each check,
for the person at the terminal only: they never go to issues, PRs or agents. No ground truth is
read: every check uses the predicted labels, the master and bands taken from external data.

Eliminatory checks (the plausibility filter of the selection protocol, F2-06). Each statistic is
computed per half, with the same function that produced the bands from the IDSSE labels
(``modeling.scoring.label_stats.half_stats``), and must lie inside the band of
``configs/plausibility.yaml`` widened by ``BAND_SLACK`` (40%): ``[min * 0.6, max * 1.4]``.

  (i)   D fraction of each half;
  (ii)  possessions per minute of each half;
  (iii) mean possession duration of each half.

Informative checks (fixed thresholds of the task; printed as PASS/FAIL but they never change the
overall result or the exit code):

  (iv)  among graded frames with ``out_of_pitch == 1``, the fraction predicted as not D is below
        ``MAX_OUT_OF_PITCH_NOT_D``;
  (v)   among predicted D segments, the fraction with no dead-ball evidence at all is below
        ``MAX_D_SEGMENTS_WITHOUT_EVIDENCE``. A segment has evidence if, on any of its frames, the
        ball is out of the pitch, the frame is in a ball gap classified as an exit, the players are
        slow (the "still" or "slow players" term of the rules_v1 dead logit is on), a set-piece tag
        is active, or the frame is before the kickoff of its half (forced D).

Overall PASS if the three eliminatory checks pass. Exit code 0 on PASS, 1 on FAIL, 2 if the
inputs could not be loaded.
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
from modeling.config import load_config  # noqa: E402
from modeling.contracts import LABEL_CODES  # noqa: E402
from modeling.deadball.dead import SPEED_LOW_RAMP, STILL_RAMP  # noqa: E402
from modeling.deadball.features import GAP_EXIT, build_deadball_features  # noqa: E402
from modeling.scoring.label_stats import half_stats, segments_from_labels  # noqa: E402

DEFAULT_PLAUSIBILITY = REPO_ROOT / "configs" / "plausibility.yaml"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "v1.yaml"

# Fixed by the task specification (F3-06) and by Section 6 of the plan, not tuned.
BAND_SLACK = 0.4                          # bands widened by 40% on each side
MAX_OUT_OF_PITCH_NOT_D = 0.2              # (iv), informative
MAX_D_SEGMENTS_WITHOUT_EVIDENCE = 0.3     # (v), informative

# Statistic name in configs/plausibility.yaml -> (check key, title, HalfStats attribute).
BAND_CHECKS = (
    ("d_fraction", "i", "D fraction of each half inside the IDSSE band", "d_fraction"),
    ("possessions_per_min", "ii", "possessions per minute of each half inside the IDSSE band",
     "possessions_per_min"),
    ("mean_possession_duration_s", "iii", "mean possession duration of each half inside the "
     "IDSSE band", "mean_possession_duration_s"),
)
MASTER_COLUMNS = ("half", "out_of_pitch", "is_set_piece", "pre_kickoff")


@dataclass
class CheckResult:
    key: str
    title: str
    passed: bool
    eliminatory: bool
    detail: str = ""


def _fps() -> float:
    return float(games.GAMES[games.GAME]["fps"])


def _n_frames() -> int:
    return int(games.GAMES[games.GAME]["n_video_frames"])


def _fmt(x) -> str:
    return "n/a" if x is None or not np.isfinite(x) else "{:.4f}".format(x)


def load_bands(path=None) -> dict:
    """``{statistic: (low, high)}`` from ``configs/plausibility.yaml``, before the slack."""
    path = DEFAULT_PLAUSIBILITY if path is None else Path(path)
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError("{}: expected a mapping".format(path))
    bands = {}
    for name, *_ in BAND_CHECKS:
        entry = raw.get(name)
        if not isinstance(entry, dict) or set(entry) != {"min", "max"}:
            raise ValueError("{}: {} must have exactly the keys min and max".format(path, name))
        low, high = entry["min"], entry["max"]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in (low, high)):
            raise ValueError("{}: {} min/max must be numbers".format(path, name))
        low, high = float(low), float(high)
        if not (np.isfinite(low) and np.isfinite(high) and 0.0 <= low <= high):
            raise ValueError("{}: {} needs 0 <= min <= max, got {} and {}".format(
                path, name, low, high))
        bands[name] = (low, high)
    return bands


def widened(band: tuple[float, float]) -> tuple[float, float]:
    low, high = band
    return low * (1.0 - BAND_SLACK), high * (1.0 + BAND_SLACK)


def read_prediction(path, n_frames: int) -> np.ndarray:
    """C0 codes of a valid submission; ``ValueError`` with the reason otherwise."""
    try:
        table = pm.read_label_csv(path)
    except Exception as exc:  # unreadable file, bad CSV, missing columns
        raise ValueError("could not read {}: {}".format(path, exc)) from exc
    reason = pm.validate_submission(table, n_frames)
    if reason is not None:
        raise ValueError("invalid submission {}: {}".format(path, reason))
    return np.asarray(pm.label_codes(table), dtype=np.int64)


def _halves(half: np.ndarray) -> list[int]:
    return [int(h) for h in np.unique(half) if h != 0]


def band_checks(codes: np.ndarray, half: np.ndarray, bands: dict, fps: float) -> list[CheckResult]:
    """(i)-(iii): every half inside the widened band of each statistic."""
    stats = {h: half_stats(codes, half, h, fps=fps) for h in _halves(half)}
    results = []
    for name, key, title, attr in BAND_CHECKS:
        low, high = widened(bands[name])
        ok, parts = bool(stats), []
        for h, s in stats.items():
            value = float(getattr(s, attr))
            inside = bool(np.isfinite(value) and low <= value <= high)        # NaN -> outside
            ok &= inside
            parts.append("half {}: {} ({})".format(h, _fmt(value), "in" if inside else "OUT"))
        parts.append("band {}..{} after +-{:.0%}".format(_fmt(low), _fmt(high), BAND_SLACK))
        results.append(CheckResult(key, title, ok, True, "; ".join(parts)))
    return results


def evidence_mask(master: pd.DataFrame, dead_params: dict) -> np.ndarray:
    """Frames with any dead-ball evidence (see check (v) in the module docstring)."""
    db = build_deadball_features(master, dead_params)
    still = db["db_still_frac"].to_numpy(dtype="float64")
    mean_speed = db["db_mean_speed"].to_numpy(dtype="float64")
    set_piece = master["is_set_piece"].to_numpy(dtype="float64", na_value=np.nan)
    with np.errstate(invalid="ignore"):                                     # NaN -> False
        slow_players = (still > STILL_RAMP[0]) | (mean_speed < SPEED_LOW_RAMP[0])
        evidence = ((master["out_of_pitch"].to_numpy(dtype="float64") == 1)
                    | (db["db_gap_kind"].to_numpy() == GAP_EXIT)
                    | slow_players
                    | (set_piece == 1)
                    | master["pre_kickoff"].to_numpy(dtype=bool, na_value=False))
    return evidence & (master["half"].to_numpy() != 0)


def evidence_checks(codes: np.ndarray, master: pd.DataFrame, dead_params: dict) -> list[CheckResult]:
    """(iv) and (v), informative."""
    half = master["half"].to_numpy()
    graded = half != 0
    is_d = codes == LABEL_CODES["D"]

    out = (master["out_of_pitch"].to_numpy(dtype="float64") == 1) & graded
    n_out = int(out.sum())
    frac_out_alive = float((out & ~is_d).sum() / n_out) if n_out else 0.0
    result_iv = CheckResult(
        "iv", "ball out of the pitch is predicted as D", frac_out_alive < MAX_OUT_OF_PITCH_NOT_D,
        False, "{} of {} out-of-pitch frames predicted alive (fraction {}, limit < {})".format(
            int((out & ~is_d).sum()), n_out, _fmt(frac_out_alive), MAX_OUT_OF_PITCH_NOT_D))

    evidence = evidence_mask(master, dead_params)
    n_seg = n_bare = 0
    for h in _halves(half):
        for seg in segments_from_labels(codes, half, h):
            if seg.state != "D":
                continue
            n_seg += 1
            n_bare += int(not evidence[seg.start:seg.stop].any())
    frac_bare = n_bare / n_seg if n_seg else 0.0
    result_v = CheckResult(
        "v", "predicted D segments have dead-ball evidence",
        frac_bare < MAX_D_SEGMENTS_WITHOUT_EVIDENCE, False,
        "{} of {} D segments without evidence (fraction {}, limit < {})".format(
            n_bare, n_seg, _fmt(frac_bare), MAX_D_SEGMENTS_WITHOUT_EVIDENCE))
    return [result_iv, result_v]


def run_checks(codes: np.ndarray, master: pd.DataFrame, bands: dict, dead_params: dict,
               fps: float | None = None) -> list[CheckResult]:
    """All five checks. Raises ``ValueError`` if the inputs do not fit together."""
    missing = [c for c in MASTER_COLUMNS if c not in master.columns]
    if missing:
        raise ValueError("master is missing columns {}".format(missing))
    codes = np.asarray(codes)
    if codes.shape != (len(master),):
        raise ValueError("prediction has {} frames, master has {}".format(codes.size, len(master)))
    half = master["half"].to_numpy()
    if not (half != 0).any():
        raise ValueError("master has no frames inside a half")
    fps = _fps() if fps is None else float(fps)
    return band_checks(codes, half, bands, fps) + evidence_checks(codes, master, dead_params)


def overall_passed(results: list[CheckResult]) -> bool:
    return all(r.passed for r in results if r.eliminatory)


def report(results: list[CheckResult], verbose: bool = False) -> list[str]:
    lines = []
    for r in results:
        tag = "" if r.eliminatory else " [informative]"
        lines.append("{}  ({}) {}{}".format("PASS" if r.passed else "FAIL", r.key, r.title, tag))
        if verbose and r.detail:
            lines.extend("        {}".format(part) for part in r.detail.split("; "))
    failed = [r for r in results if r.eliminatory and not r.passed]
    n_elim = sum(r.eliminatory for r in results)
    lines.append("PASS" if not failed else "FAIL ({} of {} eliminatory checks)".format(
        len(failed), n_elim))
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PASS/FAIL dead-ball and possession plausibility of a submission.")
    parser.add_argument("--pred", required=True, help="submission CSV to check")
    parser.add_argument("--verbose", action="store_true",
                        help="also print the numbers behind each check (for the person only)")
    parser.add_argument("--master",
                        help="master parquet (default: the BAR-ATH master in MTAH_DATA_DIR)")
    parser.add_argument("--plausibility", default=str(DEFAULT_PLAUSIBILITY),
                        help="IDSSE bands (default: %(default)s)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="config whose dead.params define the dead-ball features "
                             "(default: %(default)s)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        bands = load_bands(args.plausibility)
        cfg = load_config(args.config)
        if "dead" not in cfg:
            raise ValueError("config {} has no dead section".format(args.config))
        master = io.load_master(args.master, n_frames=_n_frames())
        codes = read_prediction(args.pred, len(master))
        results = run_checks(codes, master, bands, cfg["dead"]["params"])
    except (OSError, ValueError, KeyError) as exc:
        print("ERROR  could not load inputs: {}".format(exc))
        return 2
    for line in report(results, verbose=args.verbose):
        print(line)
    return 0 if overall_passed(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
