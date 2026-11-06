"""Filesystem locations used by the modelling code.

Every function resolves its path at call time, so tests can redirect them with
``monkeypatch.setattr(games, "DATA_DIR", ...)`` and ``monkeypatch.setenv("MTAH_EXT_DIR", ...)``.
"""
from __future__ import annotations

import os
from pathlib import Path

import games

EXT_DIR_ENV = "MTAH_EXT_DIR"
EXT_DIR_DEFAULT = "~/mtah-external"


def repo_root() -> Path:
    """Root of this repository (the folder that holds games.py)."""
    return Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    """Match data bundle: games.DATA_DIR, which honours MTAH_DATA_DIR."""
    return Path(games.DATA_DIR)


def processed_dir() -> Path:
    """Outputs of the preprocessing pipeline (master_features.parquet, persisted logits)."""
    return data_dir() / "processed"


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def ext_dir() -> Path:
    """External (IDSSE) data root: $MTAH_EXT_DIR, default ~/mtah-external.

    It must live outside the repository: the external files are large and must never be
    committed by accident.
    """
    path = Path(os.environ.get(EXT_DIR_ENV, EXT_DIR_DEFAULT)).expanduser().resolve()
    if _is_within(path, repo_root()):
        raise ValueError(
            "{} resolves to {}, which is inside the repository ({}); point it to a folder "
            "outside the repository".format(EXT_DIR_ENV, path, repo_root()))
    return path


def outputs_dir() -> Path:
    """Generated submissions (git-ignored)."""
    return repo_root() / "outputs"


__all__ = ["EXT_DIR_DEFAULT", "EXT_DIR_ENV", "data_dir", "ext_dir", "outputs_dir",
           "processed_dir", "repo_root"]
