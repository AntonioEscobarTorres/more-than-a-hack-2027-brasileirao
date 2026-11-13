"""Boundary-latency calibration on external IDSSE matches (task F2-05).

For each match, pair predicted transitions with ground-truth ones of the same type
(``pm.match_transitions(..., typed="pair")``) and record latency
``frame_pred - frame_gt``. Write medians per type for ``all`` and leave-one-out
``folds[<match>]`` into ``configs/calib/boundary.yaml``. Types with fewer than
``MIN_PAIRS`` pairs get median 0.

Usage::

    python -m modeling.calibration.boundary --config configs/v1s.yaml \\
        --degrade default --out configs/calib/boundary.yaml
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


def median_table(latencies: Mapping[str, Sequence[float]], *,
                 min_pairs: int = MIN_PAIRS) -> dict[str, int]:
    """Integer median per type; 0 when fewer than ``min_pairs`` samples."""
    table = {}
    for key in BOUNDARY_TYPES:
        samples = list(latencies.get(key, []))
        if len(samples) < min_pairs:
            table[key] = 0
        else:
            table[key] = int(np.median(np.asarray(samples, dtype=np.float64)))
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Boundary latency calibration (F2-05)")
    parser.add_argument("--config", required=True, help="config used for predictions")
    parser.add_argument("--degrade", default="default")
    parser.add_argument("--out", default="configs/calib/boundary.yaml")
    parser.add_argument("--tol", type=int, default=MATCH_TOL)
    args = parser.parse_args(argv)

    root = paths.repo_root()
    cfg = load_config(root / args.config)
    table = calibrate_boundary(cfg, degrade=args.degrade, tol=args.tol)
    out = write_boundary_yaml(table, root / args.out)
    print("wrote {}".format(out))
    for key in BOUNDARY_TYPES:
        print("all {}: median={} (n={})".format(
            key, table["all"][key], table["meta"]["n_pairs_all"][key]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BOUNDARY_TYPES",
    "MATCH_TOL",
    "MIN_PAIRS",
    "calibrate_boundary",
    "latencies_for_match",
    "median_table",
    "write_boundary_yaml",
]
