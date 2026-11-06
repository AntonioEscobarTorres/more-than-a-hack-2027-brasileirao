"""Contracts C1 (master table), C6 (submission CSV) and C2 persistence (logits parquet).

Nothing here fills, imputes or reorders data: a table that does not meet its contract is
rejected with a ``ValueError``.
"""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import games
from evaluation import possession_metrics as pm
from preprocessing.schema import master_columns

from . import paths
from .contracts import validate_logits

MASTER_FILENAME = "master_features.parquet"
LOGITS_COLUMNS = ("video_frame", "team_logit", "dead_logit")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _default_n_frames() -> int:
    return int(games.GAMES[games.GAME]["n_video_frames"])


# -- C1: master ---------------------------------------------------------------------------------

def validate_master(df: pd.DataFrame, n_frames: int | None = None) -> None:
    """C1: required columns and dtypes, optional ``reproj_error``, ``video_frame == arange(n)``.

    Columns are matched by name (order is not checked). Columns outside the schema are rejected.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("master must be a DataFrame, got {}".format(type(df).__name__))
    required = master_columns(include_optional=False)
    allowed = dict(master_columns(include_optional=True))

    missing = [c for c, _ in required if c not in df.columns]
    if missing:
        raise ValueError("master is missing {} columns: {}".format(len(missing), missing[:10]))
    extra = [c for c in df.columns if c not in allowed]
    if extra:
        raise ValueError("master has columns outside the schema: {}".format(extra[:10]))
    if df.columns.duplicated().any():
        raise ValueError("master has duplicated columns: {}".format(
            sorted(set(df.columns[df.columns.duplicated()]))))

    wrong = [(c, df[c].dtype.name, dtype) for c, dtype in allowed.items()
             if c in df.columns and df[c].dtype.name != dtype]
    if wrong:
        raise ValueError("master columns with wrong dtype (column, got, expected): {}".format(
            wrong[:10]))

    if n_frames is not None and len(df) != n_frames:
        raise ValueError("master has {} rows, expected {}".format(len(df), n_frames))
    if not np.array_equal(df["video_frame"].to_numpy(), np.arange(len(df))):
        raise ValueError("master video_frame must be 0..{} in order".format(len(df) - 1))


def load_master(path=None, n_frames: int | None = None) -> pd.DataFrame:
    """C1: read and validate a master table.

    ``path=None`` loads the BAR-ATH master from ``processed_dir()`` and, if ``n_frames`` is also
    None, checks the length against ``games``. With an explicit ``path`` and ``n_frames=None``
    the length is not checked (external masters).
    """
    if path is None:
        path = paths.processed_dir() / MASTER_FILENAME
        if n_frames is None:
            n_frames = _default_n_frames()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("master not found: {}".format(path))
    df = pd.read_parquet(path)
    validate_master(df, n_frames)
    return df


# -- C6: submission -----------------------------------------------------------------------------

def write_submission(labels, path, n_expected: int | None = None) -> Path:
    """C6: write ``frame,label`` (UTF-8, ``\\n``) and validate it with the official checker.

    The file is written to a temporary sibling and only moved into place once it passes
    ``pm.validate_submission``, so an invalid submission never lands at ``path``.
    """
    if n_expected is None:
        n_expected = _default_n_frames()
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be 1-D, got shape {}".format(labels.shape))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("frame,label\n")
            handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels.tolist()))
        message = pm.validate_submission(pm.read_label_csv(tmp), n_expected)
        if message is not None:
            raise ValueError("invalid submission: {}".format(message))
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


# -- C2 persistence: logits ---------------------------------------------------------------------

def _check_name(name: str) -> str:
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ValueError("invalid name {!r}: use letters, digits, '_', '.' or '-'".format(name))
    return name


def logits_path(name: str) -> Path:
    """``processed_dir()/logits_{name}.parquet``."""
    return paths.processed_dir() / "logits_{}.parquet".format(_check_name(name))


def save_logits(team, dead, name: str) -> Path:
    """Persist C2 evidence as ``video_frame, team_logit, dead_logit``. Returns the file path."""
    n = np.shape(team)[0] if np.ndim(team) == 1 else -1
    validate_logits(team, dead, n)
    path = logits_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({
        "video_frame": np.arange(n, dtype=np.int32),
        "team_logit": team,
        "dead_logit": dead,
    })
    frame.to_parquet(path, index=False)
    return path


def load_logits(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Read logits saved by :func:`save_logits` and return ``(team_logit, dead_logit)``."""
    path = logits_path(name)
    if not path.is_file():
        raise FileNotFoundError("logits not found: {}".format(path))
    frame = pd.read_parquet(path)
    if tuple(frame.columns) != LOGITS_COLUMNS:
        raise ValueError("{} must have columns {}, got {}".format(
            path, list(LOGITS_COLUMNS), list(frame.columns)))
    if not np.array_equal(frame["video_frame"].to_numpy(), np.arange(len(frame))):
        raise ValueError("{}: video_frame must be 0..{} in order".format(path, len(frame) - 1))
    team = frame["team_logit"].to_numpy()
    dead = frame["dead_logit"].to_numpy()
    validate_logits(team, dead, len(frame))
    return team, dead


__all__ = ["LOGITS_COLUMNS", "MASTER_FILENAME", "load_logits", "load_master", "logits_path",
           "save_logits", "validate_master", "write_submission"]
