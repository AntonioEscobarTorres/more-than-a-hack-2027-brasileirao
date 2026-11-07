"""Single command that builds a submission CSV from a config (task F5-02).

    python -m modeling.make_submission --config configs/baseline.yaml
    python -m modeling.make_submission --config configs/baseline.yaml --skip-preprocessing
    python -m modeling.make_submission --config <yaml> [--out <csv>] [--data-dir <dir>]

Steps: (1) preprocessing (``preprocessing.run_all.run_pipeline``) unless ``--skip-preprocessing``;
(2) load the master (C1); (3) ``predict_labels`` (C8); (4) ``write_submission`` (C6), by default to
``outputs/<name>.csv``; (5) print the config name and hash, the CSV SHA-256 and a summary of the
*predicted* labels only (H/A/D shares on the graded frames, possessions per half).

The thread limits below are set before numpy is imported, so inference runs single-threaded
and the CSV is byte-reproducible on any machine.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import games  # noqa: E402
from evaluation import possession_metrics as pm  # noqa: E402

from . import io, paths  # noqa: E402
from .baseline import half_blocks  # noqa: E402
from .config import BALL_SOURCES, ConfigError, config_hash, load_config  # noqa: E402
from .contracts import STATES  # noqa: E402
from .pipeline import predict_labels  # noqa: E402

SUPPORTED_BALL_SOURCES = ("tactical",)   # "fused" arrives with F4-06
assert set(SUPPORTED_BALL_SOURCES) <= set(BALL_SOURCES)


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_summary(labels: np.ndarray, half: np.ndarray) -> dict:
    """Statistics of the predicted labels only: shares of H/A/D on graded frames and the number
    of possessions (maximal H or A runs, never across a half) per half."""
    half = np.asarray(half)
    graded = half != 0
    n_graded = int(graded.sum())
    shares = {s: (float(np.sum(labels[graded] == s)) / n_graded if n_graded else 0.0)
              for s in STATES}
    codes = pm.label_codes(labels)
    possessions: dict[int, int] = {}
    for h, start, stop in half_blocks(half):
        possessions[h] = possessions.get(h, 0) + len(pm.possession_spells(codes, [(start, stop)]))
    return {"n_graded": n_graded, "shares": shares, "possessions": possessions}


def format_summary(summary: dict) -> list[str]:
    shares = "  ".join("{} {:.4f}".format(s, summary["shares"][s]) for s in STATES)
    possessions = ", ".join("half {} = {}".format(h, n)
                            for h, n in sorted(summary["possessions"].items()))
    return ["labels on {} graded frames: {}".format(summary["n_graded"], shares),
            "possessions: {}".format(possessions or "none")]


def _master_location(data_dir):
    """(path, n_frames) for load_master: the default location, or <data-dir>/processed."""
    if data_dir is None:
        return None, None
    path = Path(data_dir).expanduser() / "processed" / io.MASTER_FILENAME
    return path, int(games.GAMES[games.GAME]["n_video_frames"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m modeling.make_submission",
        description="Build a submission CSV from a modelling config.")
    parser.add_argument("--config", required=True, help="config YAML (configs/<name>.yaml)")
    parser.add_argument("--out", help="output CSV (default: outputs/<name>.csv)")
    parser.add_argument("--skip-preprocessing", action="store_true",
                        help="use the existing master instead of running the preprocessing")
    parser.add_argument("--data-dir",
                        help="data bundle root (default: MTAH_DATA_DIR or ./data)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        cfg = load_config(args.config)
    except (ConfigError, FileNotFoundError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2
    if cfg["ball_source"] not in SUPPORTED_BALL_SOURCES:
        print("error: ball_source {!r} is not supported yet (F4-06)".format(cfg["ball_source"]),
              file=sys.stderr)
        return 2
    print("config: {} (sha256 {})".format(cfg["name"], config_hash(cfg)))

    if not args.skip_preprocessing:
        from preprocessing.run_all import run_pipeline
        rc = run_pipeline(data_dir=args.data_dir)
        if rc != 0:
            print("error: preprocessing failed (exit code {}); no submission written".format(rc),
                  file=sys.stderr)
            return rc if isinstance(rc, int) and rc > 0 else 1

    master_path, n_frames = _master_location(args.data_dir)
    try:
        master = io.load_master(master_path, n_frames=n_frames)
    except FileNotFoundError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 1

    labels = predict_labels(master, cfg)
    out = Path(args.out) if args.out else paths.outputs_dir() / "{}.csv".format(cfg["name"])
    out = io.write_submission(labels, out)

    print("csv: {} (sha256 {})".format(out, file_sha256(out)))
    for line in format_summary(label_summary(labels, master["half"].to_numpy())):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
