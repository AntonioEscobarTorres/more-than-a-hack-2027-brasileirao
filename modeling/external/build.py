"""Build external (IDSSE) masters and ground-truth labels across degradation modes (F4-04).

CLI usage:
    python -m modeling.external.build --matches all --degrade none --seed 0
    python -m modeling.external.build --matches all --degrade default --seed 0
    python -m modeling.external.build --matches all --degrade harder --seed 0
    python -m modeling.external.build --matches all --degrade easier --seed 0
    python -m modeling.external.build --matches all --degrade hidelo --seed 0
    python -m modeling.external.build --matches all --degrade hidehi --seed 0
    python -m modeling.external.build --verify
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from preprocessing.pipeline_api import build_master_from_arrays
from ..io import validate_master

from .. import paths
from ..config import load_config
from ..contracts import LABEL_CODES
from . import degrade as degrade_mod
from .files import (
    IDSSE_MATCHES,
    NO_DEGRADE,
    X_CODE,
    ExternalMatch,
    arrays_path,
    labels_path,
    master_path,
)

BUILD_DEGRADE_CHOICES = ("none", "default", "harder", "easier", "fused", "hidelo", "hidehi")
D_CODE = LABEL_CODES["D"]
H_CODE = LABEL_CODES["H"]
A_CODE = LABEL_CODES["A"]

DEFAULT_REPORT_PATH = paths.repo_root() / "reports" / "external_masters.md"
DEFAULT_EXTERNAL_DATA_DOC = paths.repo_root() / "docs" / "external_data.md"

# A hash row is a Markdown table row holding artifact file names and SHA-256 digests. Columns
# may sit between them (sizes, notes): the i-th file name on the row pairs with the i-th digest.
_HASH_FILE_RE = re.compile(r"(?<![A-Za-z0-9_.-])([A-Za-z0-9_-][A-Za-z0-9_.-]*\.(?:npz|parquet|npy))\b")
_HASH_DIGEST_RE = re.compile(r"(?<![0-9a-fA-F])([0-9a-fA-F]{64})(?![0-9a-fA-F])")


def _resolve_matches(matches_arg: list[str]) -> list[str]:
    if len(matches_arg) == 1 and matches_arg[0] == "all":
        return list(IDSSE_MATCHES)
    for m in matches_arg:
        if m not in IDSSE_MATCHES:
            raise ValueError(
                "unknown match {!r}: expected 'all' or one of {}".format(m, IDSSE_MATCHES)
            )
    return list(matches_arg)


def _seed_for_match(base_seed: int, match_id: str) -> int:
    if match_id in IDSSE_MATCHES:
        return int(base_seed) + int(IDSSE_MATCHES.index(match_id))
    return int(base_seed)


def _load_degrade_config(mode: str) -> dict:
    cfg_path = paths.repo_root() / "configs" / "degrade_{}.yaml".format(mode)
    return load_config(cfg_path)  # validates the keys and that ``name`` matches the file


def _call_build_master(match: ExternalMatch) -> pd.DataFrame:
    """Build the master via ``build_master_from_arrays(players, ball, slot_team_ids, blocks, pitch)``."""
    pitch = (float(match.pitch[0]), float(match.pitch[1]))
    return build_master_from_arrays(
        match.players,
        match.ball,
        match.slot_team_ids,
        match.blocks(),
        pitch=pitch,
    )


def summarize_master(
    match_id: str,
    mode: str,
    master: pd.DataFrame,
    labels: np.ndarray,
    bg_hazard: float | None = None,
) -> dict[str, Any]:
    """Compute public summary metrics for ``reports/external_masters.md`` on half frames.

    Empty denominators (no graded, D or in-play frames) are reported as 0.0.
    """
    half = master["half"].to_numpy()
    graded = half != 0
    n_graded = int(np.count_nonzero(graded))

    obs = (master["ball_observed"].to_numpy() == 1) & graded
    filled = (master["ball_filled"].to_numpy() == 1) & graded
    jump = (master["ball_jump_removed"].to_numpy() == 1) & graded

    tracked = (
        master["n_tracked_home"].to_numpy(dtype=np.float64)[graded]
        + master["n_tracked_away"].to_numpy(dtype=np.float64)[graded]
    )

    labels_arr = np.asarray(labels)
    if labels_arr.dtype.kind in ("U", "S", "O"):
        is_d = (labels_arr == "D") & graded
        is_play = ((labels_arr == "H") | (labels_arr == "A")) & graded
    else:
        is_d = (labels_arr == D_CODE) & graded
        is_play = ((labels_arr == H_CODE) | (labels_arr == A_CODE)) & graded

    n_d = int(np.count_nonzero(is_d))
    n_play = int(np.count_nonzero(is_play))

    pct_obs = 100.0 * float(np.count_nonzero(obs)) / n_graded if n_graded else 0.0
    pct_filled = 100.0 * float(np.count_nonzero(filled)) / n_graded if n_graded else 0.0
    pct_jump = 100.0 * float(np.count_nonzero(jump)) / n_graded if n_graded else 0.0
    mean_tracked = float(np.mean(tracked)) if tracked.size else 0.0
    pct_d = 100.0 * n_d / n_graded if n_graded else 0.0
    pct_vis_d = 100.0 * float(np.count_nonzero(obs & is_d)) / n_d if n_d else 0.0
    pct_vis_play = 100.0 * float(np.count_nonzero(obs & is_play)) / n_play if n_play else 0.0

    return {
        "match": match_id,
        "degrade": mode,
        "pct_ball_observed": pct_obs,
        "pct_ball_filled": pct_filled,
        "pct_ball_jump_removed": pct_jump,
        "mean_n_tracked": mean_tracked,
        "pct_d": pct_d,
        "pct_ball_observed_d": pct_vis_d,
        "pct_ball_observed_play": pct_vis_play,
        "bg_hazard": None if mode == NO_DEGRADE else bg_hazard,
    }


def format_report_row(row: dict[str, Any]) -> str:
    bg_str = (
        "n/a"
        if row.get("bg_hazard") is None
        else "{:.6f}".format(float(row["bg_hazard"]))
    )
    return (
        "| {match} | {degrade} | {pct_ball_observed:.2f}% | {pct_ball_filled:.2f}% "
        "| {pct_ball_jump_removed:.2f}% | {mean_n_tracked:.2f} | {pct_d:.2f}% "
        "| {pct_ball_observed_d:.2f}% | {pct_ball_observed_play:.2f}% | {bg} |"
    ).format(bg=bg_str, **row)


def update_report(rows: list[dict[str, Any]], report_path: Path | None = None) -> Path:
    """Write or update ``reports/external_masters.md`` preserving rows from earlier runs."""
    target = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[tuple[str, str], str] = {}
    if target.is_file():
        for line in target.read_text(encoding="utf-8").splitlines():
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            if len(parts) >= 2 and parts[0] in IDSSE_MATCHES:
                existing[(parts[0], parts[1])] = line

    for row in rows:
        existing[(row["match"], row["degrade"])] = format_report_row(row)

    mode_order = {m: i for i, m in enumerate(BUILD_DEGRADE_CHOICES)}
    match_order = {m: i for i, m in enumerate(IDSSE_MATCHES)}
    sorted_keys = sorted(
        existing.keys(),
        key=lambda k: (mode_order.get(k[1], 99), match_order.get(k[0], 99), k[0], k[1]),
    )

    lines = [
        "# External Masters Overview (IDSSE)",
        "",
        "| Match | Degrade | % Ball Observed | % Ball Filled | % Ball Jump Removed "
        "| Mean Tracked (H+A) | % D | % Ball Vis in D | % Ball Vis in Play | bg_hazard |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(existing[k] for k in sorted_keys)
    lines.append("")
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def _atomic_write(path: Path, writer: Callable[[Path], None]) -> None:
    """Write through a temp file in the same directory, then rename.

    A crash mid-write leaves no partial file that ``--verify`` would later flag.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        writer(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _save_npy(path: Path, arr: np.ndarray) -> None:
    # Write through a file handle so numpy does not append ".npy" to the temp name.
    with open(path, "wb") as handle:
        np.save(handle, arr, allow_pickle=False)


def build_one(match_id: str, mode: str, seed: int = 0) -> dict[str, Any]:
    """Build master and labels for a single ``(match_id, mode)`` pair and return its summary."""
    raw_match = ExternalMatch.load(arrays_path(match_id))
    bg_hazard: float | None = None
    if mode == NO_DEGRADE:
        work_match = raw_match
    else:
        cfg = _load_degrade_config(mode)
        match_seed = _seed_for_match(seed, match_id)
        work_match, info = degrade_mod.degrade_with_info(raw_match, cfg, seed=match_seed)
        bg_hazard = float(info["bg_hazard"])

    master = _call_build_master(work_match)
    validate_master(master, n_frames=work_match.n_frames)

    labels = np.array(work_match.labels, dtype=np.int8, copy=True)
    if labels.shape[0] != len(master):
        raise ValueError(
            "labels length {} != master length {} for {} / {}".format(
                labels.shape[0], len(master), match_id, mode
            )
        )
    labels[master["half"].to_numpy() == 0] = X_CODE

    _atomic_write(
        master_path(match_id, mode), lambda p: master.to_parquet(p, index=False)
    )
    _atomic_write(labels_path(match_id, mode), lambda p: _save_npy(p, labels))

    return summarize_master(match_id, mode, master, labels, bg_hazard=bg_hazard)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_expected_hashes(doc_path: Path | None = None) -> dict[str, str]:
    """Parse ``filename -> sha256`` entries from ``docs/external_data.md``."""
    target = Path(doc_path) if doc_path is not None else DEFAULT_EXTERNAL_DATA_DOC
    if not target.is_file():
        raise FileNotFoundError("hash specification not found: {}".format(target))
    text = target.read_text(encoding="utf-8")
    hashes: dict[str, str] = {}
    for line in text.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        names = _HASH_FILE_RE.findall(line)
        digests = _HASH_DIGEST_RE.findall(line)
        if not names or not digests:
            continue  # not a hash row (prose or a table without digests)
        if len(names) != len(digests):
            raise ValueError(
                "hash row in {} has {} file name(s) but {} SHA-256 value(s): {}".format(
                    target, len(names), len(digests), line.strip()
                )
            )
        for name, digest in zip(names, digests):
            _add_expected_hash(hashes, name, digest.lower(), target)
    if not hashes:
        raise ValueError("no SHA-256 entries found in {}".format(target))
    return hashes


def _add_expected_hash(hashes: dict[str, str], name: str, digest: str, target: Path) -> None:
    if name in hashes and hashes[name] != digest:
        raise ValueError("conflicting SHA-256 entries for {} in {}".format(name, target))
    hashes[name] = digest


def _resolve_local_artifact(filename: str) -> Path:
    base = paths.ext_dir()
    if filename.endswith(".npz"):
        return base / "arrays" / filename
    return base / "processed" / filename


def verify_external_files(doc_path: Path | None = None) -> int:
    """Compare local external files with hashes in ``docs/external_data.md``."""
    expected = parse_expected_hashes(doc_path)
    all_ok = True
    for filename, want_hash in sorted(expected.items()):
        local_path = _resolve_local_artifact(filename)
        if not local_path.is_file():
            print("DIFFERENT  {} (missing)".format(filename))
            all_ok = False
            continue
        got_hash = _sha256_file(local_path)
        if got_hash == want_hash:
            print("IDENTICAL  {}".format(filename))
        else:
            print("DIFFERENT  {} (expected {}, got {})".format(filename, want_hash, got_hash))
            all_ok = False
    return 0 if all_ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or verify external (IDSSE) masters and labels (F4-04)."
    )
    parser.add_argument(
        "--matches",
        nargs="+",
        default=["all"],
        help="match IDs to build, or 'all' (default: %(default)s)",
    )
    parser.add_argument(
        "--degrade",
        choices=BUILD_DEGRADE_CHOICES,
        default="default",
        help="degradation mode to apply (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="base deterministic seed (default: %(default)s)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify local files against SHA-256 hashes in docs/external_data.md",
    )
    parser.add_argument(
        "--doc",
        default=None,
        help="override path to docs/external_data.md when running --verify",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="override path to reports/external_masters.md",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verify:
        try:
            return verify_external_files(args.doc)
        except (OSError, ValueError) as exc:
            print("DIFFERENT  cannot read the expected hashes: {}".format(exc))
            return 1

    matches = _resolve_matches(args.matches)
    for match_id in matches:
        row = build_one(match_id, args.degrade, seed=args.seed)
        # Update the report after each match so a later failure keeps earlier rows.
        update_report([row], report_path=args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())