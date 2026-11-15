"""Boundary-latency calibration on external IDSSE matches (task F2-05).

For each match, pair predicted transitions with ground-truth ones of the same type
(``pm.match_transitions(..., typed="pair")``) and record latency
``frame_pred - frame_gt``. Write medians per type for ``all`` and leave-one-out
``folds[<match>]`` into ``configs/calib/boundary.yaml``. Types with fewer than
``MIN_PAIRS`` pairs get median 0.

It then evaluates the config with and without ``boundary_shift_file`` (step 5 of F2-05);
each external match uses ``folds[<match>]``, so its own latencies never correct it.

Usage::

    python -m modeling.calibration.boundary --config configs/v1s.yaml \\
        --degrade default --out configs/calib/boundary.yaml \\
        --report reports/sensitivity_decoder.md
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import yaml

from evaluation import possession_metrics as pm

from modeling import paths
from modeling.config import load_config
from modeling.contracts import LABEL_CODES
from modeling.decode.postprocess import BOUNDARY_TYPES
from modeling.external.files import IDSSE_MATCHES, load_external
from modeling.pipeline import predict_labels
from modeling.scoring.external_eval import evaluate_config, to_markdown

# F2-05: fewer than this many pairs → median forced to 0.
MIN_PAIRS = 20
MATCH_TOL = 50
CODE_TO_LABEL = {code: lab for lab, code in LABEL_CODES.items() if lab != "X"}


def _type_key(from_code: int, to_code: int) -> str | None:
    a = CODE_TO_LABEL.get(int(from_code))
    b = CODE_TO_LABEL.get(int(to_code))
    if a is None or b is None:
        return None
    return "{}>{}".format(a, b)


def latencies_for_match(
    gt_codes: np.ndarray,
    pred_codes: np.ndarray,
    *,
    tol: int = MATCH_TOL,
) -> dict[str, list[float]]:
    """Typed latencies (pred − gt) for one match, keyed by ``H>A`` etc."""
    runs = pm.valid_runs(gt_codes)
    filled = pm.fill_abstentions(pred_codes, runs)
    gt_trans = pm.transitions(gt_codes, runs)
    pred_trans = pm.transitions(filled, runs)
    pairs, _missed, _spurious = pm.match_transitions(
        gt_trans, pred_trans, tol=tol, typed="pair")
    out = {key: [] for key in BOUNDARY_TYPES}
    for gt_t, pred_t in pairs:
        key = _type_key(gt_t[1], gt_t[2])
        if key is None or key not in out:
            continue
        out[key].append(float(pred_t[0] - gt_t[0]))
    return out


def round_half_away(value: float) -> int:
    """Nearest integer, halves away from zero (7.5 -> 8, -2.5 -> -3), symmetric in sign."""
    return int(np.sign(value) * np.floor(abs(value) + 0.5))


def median_table(latencies: Mapping[str, Sequence[float]], *,
                 min_pairs: int = MIN_PAIRS) -> dict[str, int]:
    """Median per type rounded to whole frames; 0 when fewer than ``min_pairs`` samples."""
    table = {}
    for key in BOUNDARY_TYPES:
        samples = list(latencies.get(key, []))
        if len(samples) < min_pairs:
            table[key] = 0
        else:
            table[key] = round_half_away(float(np.median(np.asarray(samples, dtype=np.float64))))
    return table


def _merge_latencies(per_match: Mapping[str, Mapping[str, Sequence[float]]],
                     matches: Sequence[str]) -> dict[str, list[float]]:
    merged = {key: [] for key in BOUNDARY_TYPES}
    for match in matches:
        for key in BOUNDARY_TYPES:
            merged[key].extend(per_match[match].get(key, []))
    return merged


def calibrate_boundary(
    cfg: dict,
    *,
    degrade: str = "default",
    matches: Sequence[str] | None = None,
    tol: int = MATCH_TOL,
    min_pairs: int = MIN_PAIRS,
) -> dict:
    """Fit ``all`` and leave-one-out ``folds`` median latency tables.

    Returns a JSON-serialisable mapping ready for ``configs/calib/boundary.yaml``.
    """
    matches = list(IDSSE_MATCHES if matches is None else matches)
    if not matches:
        raise ValueError("no matches to calibrate")
    if cfg["postprocess"]["boundary_shift_file"] is not None:
        raise ValueError("calibrate on a config without boundary_shift_file; latencies measured "
                         "after a shift would correct the correction")

    per_match: dict[str, dict[str, list[float]]] = {}
    for match in matches:
        master, gt = load_external(match, degrade)
        pred = predict_labels(master, cfg, holdout=match)
        pred_codes = pm.label_codes(pred)
        per_match[match] = latencies_for_match(gt, pred_codes, tol=tol)

    result = {
        "all": median_table(_merge_latencies(per_match, matches), min_pairs=min_pairs),
        "folds": {},
        "meta": {
            "degrade": degrade,
            "tol": int(tol),
            "min_pairs": int(min_pairs),
            "types": list(BOUNDARY_TYPES),
            "n_pairs_all": {k: len(v) for k, v in _merge_latencies(per_match, matches).items()},
        },
    }
    for holdout in matches:
        train = [m for m in matches if m != holdout]
        result["folds"][holdout] = median_table(
            _merge_latencies(per_match, train), min_pairs=min_pairs)
    return result


def write_boundary_yaml(table: Mapping, path: str | Path) -> Path:
    """Write the calibration table (UTF-8 YAML)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep a stable key order: all, folds, meta.
    payload = {
        "all": dict(table["all"]),
        "folds": {k: dict(table["folds"][k]) for k in table["folds"]},
    }
    if "meta" in table:
        payload["meta"] = copy.deepcopy(table["meta"])
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def compare_boundary_shift(
    cfg: dict,
    boundary_file: str | Path,
    *,
    degrade: str = "default",
    matches: Sequence[str] | None = None,
    evaluate_fn=None,
) -> dict:
    """F2-05 step 5: evaluate ``cfg`` without and with ``boundary_shift_file``.

    ``evaluate_fn(cfg, degrade, matches) -> DataFrame`` defaults to ``evaluate_config``; each
    match is predicted with ``holdout=<match>``, so the shift it gets is ``folds[<match>]``.
    Returns ``{"off": df, "on": df}``.
    """
    if evaluate_fn is None:
        evaluate_fn = evaluate_config
    off = copy.deepcopy(cfg)
    off["postprocess"]["boundary_shift_file"] = None
    on = copy.deepcopy(cfg)
    on["postprocess"]["boundary_shift_file"] = str(boundary_file)
    return {"off": evaluate_fn(off, degrade, matches), "on": evaluate_fn(on, degrade, matches)}


def comparison_markdown(result: Mapping, metric: str = "spell_f1_mean") -> str:
    """Report section for :func:`compare_boundary_shift`."""
    lines = ["## Step 5: boundary shift (with / without `boundary_shift_file`)", ""]
    for key in ("off", "on"):
        df = result[key]
        lines.append("- {}: {} {:.4f}±{:.4f}".format(
            key, metric, float(df.loc["mean", metric]), float(df.loc["se", metric])))
    lines.append("")
    for key in ("off", "on"):
        lines.extend(["### {}".format(key), "", to_markdown(result[key]), ""])
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Boundary latency calibration (F2-05)")
    parser.add_argument("--config", required=True, help="config used for predictions")
    parser.add_argument("--degrade", default="default")
    parser.add_argument("--out", default="configs/calib/boundary.yaml")
    parser.add_argument("--tol", type=int, default=MATCH_TOL)
    parser.add_argument("--report", default=None,
                        help="append the with/without comparison to this markdown file")
    parser.add_argument("--no-compare", action="store_true",
                        help="only write the calibration file (skip step 5)")
    args = parser.parse_args(argv)

    root = paths.repo_root()
    cfg = load_config(root / args.config)
    table = calibrate_boundary(cfg, degrade=args.degrade, tol=args.tol)
    out = write_boundary_yaml(table, root / args.out)
    print("wrote {}".format(out))
    for key in BOUNDARY_TYPES:
        print("all {}: median={} (n={})".format(
            key, table["all"][key], table["meta"]["n_pairs_all"][key]))
    if args.no_compare:
        return 0
    section = comparison_markdown(compare_boundary_shift(cfg, out, degrade=args.degrade))
    if args.report is not None:
        report = root / args.report
        report.parent.mkdir(parents=True, exist_ok=True)
        previous = report.read_text(encoding="utf-8") if report.is_file() else ""
        report.write_text(previous.rstrip("\n") + "\n\n" + section + "\n" if previous
                          else section + "\n", encoding="utf-8", newline="\n")
        print("appended step 5 to {}".format(report))
    print(section)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BOUNDARY_TYPES",
    "MATCH_TOL",
    "MIN_PAIRS",
    "calibrate_boundary",
    "compare_boundary_shift",
    "comparison_markdown",
    "latencies_for_match",
    "median_table",
    "round_half_away",
    "write_boundary_yaml",
]
