"""Contract C9: on-disk format of the external (IDSSE) matches.

An :class:`ExternalMatch` is stored as ``ext_dir()/"arrays"/f"{match_id}.npz"``. The time axis
holds the first half, then ``GAP_FRAMES`` empty frames (NaN, ``half == 0``, label X), then the
second half, so that no possession can cross halftime in the metric.

Derived files (F4-04) live in ``ext_dir()/"processed"``: a 59-column master per degradation and
its int8 label vector (C0 codes).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.schema import N_SLOTS

from .. import paths
from ..contracts import LABEL_CODES
from ..io import load_master

IDSSE_MATCHES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
GAP_FRAMES = 250
N_HOME_SLOTS = 24
X_CODE = LABEL_CODES["X"]
NO_DEGRADE = "none"

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

# field -> (dtype, shape with "T" for the time axis)
ARRAY_FIELDS = {
    "players": (np.float32, ("T", N_SLOTS, 3)),
    "ball": (np.float32, ("T", 3)),
    "ball_z": (np.float32, ("T",)),
    "labels": (np.int8, ("T",)),
    "half": (np.int8, ("T",)),
    "slot_team_ids": (np.int8, (N_SLOTS,)),
    "pitch": (np.float64, (2,)),
}
NPZ_KEYS = frozenset(("match_id",) + tuple(ARRAY_FIELDS))


def _check_name(kind: str, value: str) -> str:
    if not isinstance(value, str) or not _NAME_RE.match(value):
        raise ValueError("invalid {} {!r}: use letters, digits, '_' or '-'".format(kind, value))
    return value


def _check_degrade(degrade: str) -> str:
    _check_name("degrade", degrade)
    if degrade == NO_DEGRADE:
        return degrade
    cfg = paths.repo_root() / "configs" / "degrade_{}.yaml".format(degrade)
    if not cfg.is_file():
        raise ValueError("unknown degrade {!r}: expected {!r} or an existing {}".format(
            degrade, NO_DEGRADE, cfg))
    return degrade


def expected_half(n_first: int, n_second: int) -> np.ndarray:
    """The C9 ``half`` vector: n_first ones, GAP_FRAMES zeros, n_second twos (int8)."""
    return np.concatenate([np.full(n_first, 1, np.int8), np.zeros(GAP_FRAMES, np.int8),
                           np.full(n_second, 2, np.int8)])


@dataclass(eq=False)
class ExternalMatch:
    match_id: str
    players: np.ndarray        # float32 (T, 46, 3): x_n, y_n in [0, 1], speed in m/s
    ball: np.ndarray           # float32 (T, 3): x_n, y_n, speed (m/s or NaN)
    ball_z: np.ndarray         # float32 (T,): height in metres (used only by degradation)
    labels: np.ndarray         # int8 (T,): C0 codes
    half: np.ndarray           # int8 (T,): 1, 2; 0 on the artificial halftime gap
    slot_team_ids: np.ndarray  # int8 (46,): 0 = home (slots 0-23), 1 = away (slots 24-45)
    pitch: np.ndarray          # float64 (2,): (L, W) in metres

    def __post_init__(self):
        self.validate()

    @property
    def n_frames(self) -> int:
        return int(self.half.shape[0])

    def validate(self) -> None:
        """Check names, dtypes, shapes and the C9 layout; raise ValueError on any mismatch."""
        _check_name("match_id", self.match_id)
        for field, (dtype, shape) in ARRAY_FIELDS.items():
            arr = getattr(self, field)
            if not isinstance(arr, np.ndarray):
                raise ValueError("{}: must be a numpy array, got {}".format(
                    field, type(arr).__name__))
            if arr.dtype != dtype:
                raise ValueError("{}: dtype must be {}, got {}".format(
                    field, np.dtype(dtype), arr.dtype))
        n = self.half.shape[0] if self.half.ndim == 1 else -1
        for field, (_, shape) in ARRAY_FIELDS.items():
            want = tuple(n if s == "T" else s for s in shape)
            got = getattr(self, field).shape
            if got != want:
                raise ValueError("{}: shape must be {}, got {}".format(field, want, got))

        n_first = int(np.count_nonzero(self.half == 1))
        n_second = int(np.count_nonzero(self.half == 2))
        if n_first == 0 or n_second == 0 or not np.array_equal(
                self.half, expected_half(n_first, n_second)):
            raise ValueError("half must be 1s, then {} zeros, then 2s".format(GAP_FRAMES))

        if not np.isin(self.labels, list(LABEL_CODES.values())).all():
            raise ValueError("labels must be C0 codes {}".format(sorted(LABEL_CODES.values())))
        gap = self.half == 0
        if not np.array_equal(self.labels == X_CODE, gap):
            raise ValueError("labels must be X ({}) exactly on the halftime gap".format(X_CODE))
        for field in ("players", "ball", "ball_z"):
            if not np.isnan(getattr(self, field)[gap]).all():
                raise ValueError("{} must be NaN on the halftime gap".format(field))

        home = np.r_[np.zeros(N_HOME_SLOTS, np.int8), np.ones(N_SLOTS - N_HOME_SLOTS, np.int8)]
        if not np.array_equal(self.slot_team_ids, home):
            raise ValueError("slot_team_ids must be 0 for slots 0-{} and 1 for slots {}-{}".format(
                N_HOME_SLOTS - 1, N_HOME_SLOTS, N_SLOTS - 1))
        if not (np.isfinite(self.pitch).all() and (self.pitch > 0).all()):
            raise ValueError("pitch must be two positive finite lengths, got {}".format(
                self.pitch))

    def blocks(self) -> list[tuple[int, int, int]]:
        """``[(1, f0, f1), (2, f0, f1)]`` with inclusive frame limits."""
        out = []
        for h in (1, 2):
            idx = np.flatnonzero(self.half == h)
            out.append((h, int(idx[0]), int(idx[-1])))
        return out

    def save(self, path=None) -> Path:
        """Write the match with ``np.savez_compressed``; default path is ``arrays_path``."""
        self.validate()
        path = Path(path) if path is not None else arrays_path(self.match_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {field: getattr(self, field) for field in ARRAY_FIELDS}
        with open(path, "wb") as handle:
            np.savez_compressed(handle, match_id=np.array(self.match_id), **arrays)
        return path

    @classmethod
    def load(cls, path) -> "ExternalMatch":
        """Read a file written by :meth:`save`, checking keys, dtypes and shapes."""
        with np.load(Path(path), allow_pickle=False) as data:
            keys = set(data.files)
            if keys != NPZ_KEYS:
                raise ValueError("{}: keys must be {}, got {}".format(
                    path, sorted(NPZ_KEYS), sorted(keys)))
            match_id = data["match_id"]
            if match_id.shape != () or match_id.dtype.kind != "U":
                raise ValueError("{}: match_id must be a scalar string".format(path))
            arrays = {field: data[field] for field in ARRAY_FIELDS}
        return cls(match_id=str(match_id), **arrays)


# -- paths and loaders --------------------------------------------------------------------------

def arrays_path(match: str) -> Path:
    return paths.ext_dir() / "arrays" / "{}.npz".format(_check_name("match", match))


def master_path(match: str, degrade: str) -> Path:
    return paths.ext_dir() / "processed" / "{}__{}.parquet".format(
        _check_name("match", match), _check_degrade(degrade))


def labels_path(match: str, degrade: str) -> Path:
    return paths.ext_dir() / "processed" / "{}__{}_labels.npy".format(
        _check_name("match", match), _check_degrade(degrade))


def load_external(match: str, degrade: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Load an external master (C1, length not fixed) and its int8 C0 label vector."""
    master = load_master(master_path(match, degrade))
    lpath = labels_path(match, degrade)
    if not lpath.is_file():
        raise FileNotFoundError("labels not found: {}".format(lpath))
    labels = np.load(lpath, allow_pickle=False)
    if labels.dtype != np.int8 or labels.shape != (len(master),):
        raise ValueError("{}: labels must be int8 ({},), got {} {}".format(
            lpath, len(master), labels.dtype, labels.shape))
    if not np.isin(labels, list(LABEL_CODES.values())).all():
        raise ValueError("{}: labels must be C0 codes".format(lpath))
    gap = master["half"].to_numpy() == 0
    if not np.array_equal(labels == X_CODE, gap):
        raise ValueError("{}: labels must be X exactly where half == 0".format(lpath))
    return master, labels


__all__ = ["ARRAY_FIELDS", "ExternalMatch", "GAP_FRAMES", "IDSSE_MATCHES", "NO_DEGRADE",
           "arrays_path", "expected_half", "labels_path", "load_external", "master_path"]
