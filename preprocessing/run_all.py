"""Single entry point for the preprocessing pipeline (task E1).

Runs the intermediate builders in dependency order and writes
``data/processed/master_features.parquet`` from a fresh data bundle:

1. **C** — Smart Tagging + kickoffs (``export_tagging``) → ``tagging.parquet``,
   ``kickoffs.json``. Must run before A-V1 because ``prep_tracking --stage V1``
   requires a validated ``kickoffs.json``.
2. **C5** — Homography (optional) → ``homography.parquet``.
3. **A** — Tracking with ``--stage V1`` (CLI default is V0) →
   ``tracking_clean.parquet``, ``slot_teams.parquet``.
4. **B** — Proximities → ``proximities.parquet``.
5. **D** — Master table → ``master_features.parquet``.

Usage::

    python -m preprocessing.run_all
    python -m preprocessing.run_all --check-determinism
    python -m preprocessing.run_all --skip-homography
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def _data_root(data_dir=None):
    if data_dir:
        return str(data_dir)
    env = os.environ.get("MTAH_DATA_DIR")
    if env:
        return env
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def run_pipeline(
    data_dir=None,
    processed_dir=None,
    skip_homography=False,
    check_determinism=False,
):
    """Execute C → C5 → A(V1) → B → D. Returns a process exit code (0 = ok)."""
    from games import GAME, half_video_blocks
    from preprocessing.export_tagging import export_tagging_data
    from preprocessing.prep_homography import export_homography_data
    from preprocessing import prep_tracking, prep_proximities, build_master_table

    root = Path(_data_root(data_dir))
    processed = Path(processed_dir) if processed_dir else root / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    blocks = half_video_blocks(GAME)
    data_dir_str = str(root)

    LOGGER.info("C: export tagging → %s", processed)
    export_tagging_data(processed, blocks, data_dir=data_dir_str)

    if skip_homography:
        LOGGER.info("C5: skipped (--skip-homography)")
    else:
        LOGGER.info("C5: export homography → %s", processed)
        try:
            export_homography_data(processed, blocks, data_dir=data_dir_str)
        except Exception as exc:
            # Homography is optional in the master join; missing assets must not
            # block tracking / proximities / master.
            LOGGER.warning("C5 skipped (homography unavailable): %s", exc)

    LOGGER.info("A: prep_tracking --stage V1 → %s", processed)
    rc = prep_tracking.main([
        "--data-dir", data_dir_str,
        "--output-dir", str(processed),
        "--stage", "V1",
    ])
    if rc:
        return rc

    LOGGER.info("B: prep_proximities → %s", processed)
    rc = prep_proximities.main(["--processed-dir", str(processed)])
    if rc:
        return rc

    LOGGER.info("D: build_master_table → %s", processed)
    d_args = ["--processed-dir", str(processed)]
    if check_determinism:
        d_args.append("--check-determinism")
    return build_master_table.main(d_args)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Run the full preprocessing pipeline (C → C5 → A-V1 → B → D).")
    parser.add_argument(
        "--data-dir",
        help="data bundle root (defaults to MTAH_DATA_DIR or ./data)")
    parser.add_argument(
        "--processed-dir",
        help="output directory (defaults to <data-dir>/processed)")
    parser.add_argument(
        "--skip-homography", action="store_true",
        help="skip optional C5 homography step")
    parser.add_argument(
        "--check-determinism", action="store_true",
        help="rebuild the master a second time and compare sha256")
    args = parser.parse_args(argv)
    return run_pipeline(
        data_dir=args.data_dir,
        processed_dir=args.processed_dir,
        skip_homography=args.skip_homography,
        check_determinism=args.check_determinism,
    )


if __name__ == "__main__":
    raise SystemExit(main())
