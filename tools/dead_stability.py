"""Stability of the selected config to the degradation the dead_logit was fitted on (F2-06).

Local execution only (a person runs it on the match data; F2-06, update of 11/11, item 5):

    python tools/dead_stability.py --config configs/final.yaml
    python tools/dead_stability.py --config configs/final.yaml --verbose

The BAR-ATH is predicted three times with the given config, changing only
``dead.calibrated_file``: the dead_logit fitted on ``default`` (``configs/calib/dead.yaml``), on
``hidelo`` and on ``hidehi`` (F3-04, item 7; with ``dead.kind: gbdt`` the F3-08 GBDT fits
``configs/calib/dead_gbdt{,_hidelo,_hidehi}.yaml``). For every pair of runs the share of frames inside
the halves (``half != 0``) whose label changes must be below ``MAX_CHANGED_SHARE`` (5%).

By default it prints only PASS/FAIL per pair. ``--verbose`` adds the shares, for the person at
the terminal only: they never go to issues, PRs or agents. No labels of the match are read. The
check only measures how much the result depends on the degradation; it never ranks candidates.

If it FAILs: prefer the ``pooled`` dead (``configs/calib/dead_pooled.yaml``) if the winner does
not use it already, or the previous candidate of the ladder that passes.

The master follows ``ball_source`` like ``modeling.make_submission``: ``fused`` reads
``<data root>/processed_fused``, so run ``make_submission`` with that config first.

Exit code 0 on PASS, 1 on FAIL, 2 if the inputs could not be loaded.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import sys
from pathlib import Path
from typing import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from modeling import io  # noqa: E402
from modeling.config import ConfigError, load_config, validate_config  # noqa: E402
from modeling.make_submission import _master_location  # noqa: E402
from modeling.pipeline import predict_labels  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "final.yaml"
# dead.kind -> the dead_logit fitted on each degradation (F3-04 logistic; F3-08 GBDT).
DEAD_VARIANTS = {
    "calibrated": {
        "default": "configs/calib/dead.yaml",
        "hidelo": "configs/calib/dead_hidelo.yaml",
        "hidehi": "configs/calib/dead_hidehi.yaml",
    },
    "gbdt": {
        "default": "configs/calib/dead_gbdt.yaml",
        "hidelo": "configs/calib/dead_gbdt_hidelo.yaml",
        "hidehi": "configs/calib/dead_gbdt_hidehi.yaml",
    },
}
# Fixed by the F2-06 specification (update of 11/11, item 5), not tuned.
MAX_CHANGED_SHARE = 0.05


def variant_configs(cfg: dict, variants: Mapping[str, str] | None = None) -> dict[str, dict]:
    """One config per dead calibration; only ``dead.calibrated_file`` changes."""
    kind = cfg["dead"]["kind"] if cfg.get("pipeline") == "hmm" else None
    if kind not in DEAD_VARIANTS:
        raise ValueError("the stability check needs an hmm config with dead.kind in {}".format(
            sorted(DEAD_VARIANTS)))
    if variants is None:
        variants = DEAD_VARIANTS[kind]
    out = {}
    for key, path in variants.items():
        v = copy.deepcopy(cfg)
        v["dead"]["calibrated_file"] = path
        out[key] = validate_config(v)
    return out


def changed_shares(labels: Mapping[str, np.ndarray], half: np.ndarray) -> dict[tuple, float]:
    """Share of in-half frames whose label differs, for every pair of runs."""
    inside = np.asarray(half) != 0
    n = int(inside.sum())
    if n == 0:
        raise ValueError("no frame inside the halves")
    shares = {}
    for a, b in itertools.combinations(list(labels), 2):
        la, lb = np.asarray(labels[a]), np.asarray(labels[b])
        if la.shape != lb.shape or la.shape != inside.shape:
            raise ValueError("label arrays and half must have the same length")
        shares[(a, b)] = float((la[inside] != lb[inside]).sum()) / n
    return shares


def report(shares: Mapping[tuple, float], verbose: bool = False) -> tuple[bool, list[str]]:
    lines = []
    ok_all = True
    for (a, b), share in shares.items():
        ok = share < MAX_CHANGED_SHARE
        ok_all &= ok
        line = "{} vs {}: {}".format(a, b, "PASS" if ok else "FAIL")
        if verbose:
            line += " (changed share {:.4f}, max {:.2f})".format(share, MAX_CHANGED_SHARE)
        lines.append(line)
    lines.append("overall: {}".format("PASS" if ok_all else "FAIL"))
    return ok_all, lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python tools/dead_stability.py",
        description="Label stability across the dead_logit fitted on default/hidelo/hidehi.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="config YAML")
    parser.add_argument("--master", help="master parquet (default: follows ball_source)")
    parser.add_argument("--data-dir", help="data bundle root (default: MTAH_DATA_DIR or ./data)")
    parser.add_argument("--verbose", action="store_true",
                        help="print the shares (local terminal only)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
        configs = variant_configs(cfg)
        default_path, n_frames = _master_location(args.data_dir, cfg["ball_source"])
        master_path = Path(args.master) if args.master else default_path
        master = io.load_master(master_path, n_frames=n_frames)
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2

    labels = {key: predict_labels(master, c) for key, c in configs.items()}
    ok, lines = report(changed_shares(labels, master["half"].to_numpy()), verbose=args.verbose)
    for line in lines:
        print(line)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
