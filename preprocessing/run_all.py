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
import tempfile
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def _data_root(data_dir=None):
    if data_dir:
        return str(data_dir)
    env = os.environ.get("MTAH_DATA_DIR")
    if env:
        return env
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _run_steps(root, processed, skip_homography):
    """One complete run C → C5 → A(V1) → B → D into ``processed``. Returns an exit code."""
    from games import GAME, half_video_blocks
    from preprocessing.export_tagging import export_tagging_data
    from preprocessing.prep_homography import export_homography_data
    from preprocessing import prep_tracking, prep_proximities, build_master_table

    processed.mkdir(parents=True, exist_ok=True)
    blocks = half_video_blocks(GAME)
    data_dir_str = str(root)

    LOGGER.info("C: export tagging → %s", processed)
    export_tagging_data(processed, blocks, data_dir=data_dir_str)

    homography = processed / build_master_table.HOMOGRAPHY_FILE
    if skip_homography:
        LOGGER.info("C5: skipped (--skip-homography)")
        _remove_stale(homography)
    else:
        LOGGER.info("C5: export homography → %s", processed)
        try:
            export_homography_data(processed, blocks, data_dir=data_dir_str)
        except Exception as exc:
            # Homography is optional in the master join; missing assets must not
            # block tracking / proximities / master.
            LOGGER.warning("C5 skipped (homography unavailable): %s", exc)
            _remove_stale(homography)

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
    return build_master_table.main(["--processed-dir", str(processed)])


def _remove_stale(path):
    """build_master_table includes homography.parquet whenever it exists, so a file left by an
    earlier run must not survive a run where C5 was skipped or failed."""
    if path.exists():
        LOGGER.warning("C5: removing stale %s", path)
        path.unlink()


def run_pipeline(
    data_dir=None,
    processed_dir=None,
    skip_homography=False,
    check_determinism=False,
):
    """Execute C → C5 → A(V1) → B → D. Returns a process exit code (0 = ok).

    With ``check_determinism`` the whole pipeline runs a second time from the raw data into a
    temporary directory and the SHA-256 of both masters must match (issue #26).
    """
    from preprocessing.build_master_table import OUTPUT_FILE, file_sha256

    root = Path(_data_root(data_dir))
    processed = Path(processed_dir) if processed_dir else root / "processed"
    rc = _run_steps(root, processed, skip_homography)
    if rc or not check_determinism:
        return rc

    LOGGER.info("determinism: second complete run")
    with tempfile.TemporaryDirectory() as tmp:
        second = Path(tmp) / "processed"
        rc = _run_steps(root, second, skip_homography)
        if rc:
            return rc
        first_hash = file_sha256(processed / OUTPUT_FILE)
        second_hash = file_sha256(second / OUTPUT_FILE)
    if first_hash != second_hash:
        LOGGER.error("determinism: FAILED (%s != %s)", first_hash, second_hash)
        return 1
    LOGGER.info("determinism: OK (sha256 %s)", first_hash)
    return 0


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
        help="run the whole pipeline a second time and compare the master sha256")
    args = parser.parse_args(argv)
    return run_pipeline(
        data_dir=args.data_dir,
        processed_dir=args.processed_dir,
        skip_homography=args.skip_homography,
        check_determinism=args.check_determinism,
    )


if __name__ == "__main__":
    raise SystemExit(main())
