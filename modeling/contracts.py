"""Contracts C0 (label codes), C2 (per-frame evidence) and C4 (forced-D mask).

See docs/contracts.md. Every validator raises ``ValueError`` with a message that names the
offending argument; none of them modifies its input.
"""
from __future__ import annotations

import numpy as np

from evaluation import possession_metrics as pm

# -- C0: label codes ----------------------------------------------------------------------------
STATES = ("H", "A", "D")
LABEL_CODES = {"H": 0, "A": 1, "D": 2, "X": 3}
LABEL_DTYPE = np.dtype("<U1")

assert tuple(LABEL_CODES) == tuple(pm.LABELS), "LABEL_CODES must follow pm.LABELS order"
assert all(LABEL_CODES[s] == i for i, s in enumerate(pm.LABELS)), "codes must be 0..3"
assert STATES == tuple(pm.LABELS[:3]), "STATES must be the first three pm.LABELS"

# -- C2: evidence bounds ------------------------------------------------------------------------
TEAM_LOGIT_MAX = 8.0
DEAD_LOGIT_MAX = 12.0


def _check_1d(name: str, arr, n: int) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        raise ValueError("{} must be a numpy array, got {}".format(name, type(arr).__name__))
    if arr.shape != (n,):
        raise ValueError("{} must have shape ({},), got {}".format(name, n, arr.shape))
    return arr


def _check_half(half) -> np.ndarray:
    half = np.asarray(half)
    if half.ndim != 1:
        raise ValueError("half must be 1-D, got shape {}".format(half.shape))
    if not np.issubdtype(half.dtype, np.integer):
        raise ValueError("half must be an integer array, got dtype {}".format(half.dtype))
    return half


def _check_logit(name: str, arr, n: int, bound: float) -> None:
    _check_1d(name, arr, n)
    if arr.dtype != np.float32:
        raise ValueError("{} must be float32, got {}".format(name, arr.dtype))
    finite = np.isfinite(arr)
    if not finite.all():
        bad = np.flatnonzero(~finite)
        raise ValueError("{} has {} non-finite values (first at index {})".format(
            name, bad.size, int(bad[0])))
    over = np.abs(arr) > bound
    if over.any():
        bad = np.flatnonzero(over)
        raise ValueError("{} has {} values with |x| > {} (first at index {})".format(
            name, bad.size, bound, int(bad[0])))


def validate_logits(team, dead, n: int, *, half=None) -> None:
    """C2: ``team`` and ``dead`` are float32[n], finite and within their bounds.

    If ``half`` is given, also check that both are exactly 0 where ``half == 0``.
    """
    _check_logit("team_logit", team, n, TEAM_LOGIT_MAX)
    _check_logit("dead_logit", dead, n, DEAD_LOGIT_MAX)
    if half is not None:
        half = _check_half(half)
        if half.shape != (n,):
            raise ValueError("half must have shape ({},), got {}".format(n, half.shape))
        gap = half == 0
        for name, arr in (("team_logit", team), ("dead_logit", dead)):
            if (arr[gap] != 0).any():
                raise ValueError("{} must be 0 where half == 0".format(name))


def validate_force_d(mask, half) -> None:
    """C4: ``mask`` is bool[n] (n = len(half)) and never True where ``half == 0``."""
    half = _check_half(half)
    _check_1d("force_d", mask, half.shape[0])
    if mask.dtype != np.bool_:
        raise ValueError("force_d must be bool, got {}".format(mask.dtype))
    if (mask & (half == 0)).any():
        raise ValueError("force_d is True on frames where half == 0")


def validate_labels(labels, half) -> None:
    """C0/C5: ``labels`` is ``<U1``[n] over {H, A, D, X}, with X if and only if ``half == 0``."""
    half = _check_half(half)
    _check_1d("labels", labels, half.shape[0])
    if labels.dtype != LABEL_DTYPE:
        raise ValueError("labels must have dtype <U1, got {}".format(labels.dtype))
    unknown = sorted(set(np.unique(labels).tolist()) - set(LABEL_CODES))
    if unknown:
        raise ValueError("labels outside {}: {}".format(list(LABEL_CODES), unknown[:5]))
    is_x = labels == "X"
    gap = half == 0
    if (is_x & ~gap).any():
        bad = np.flatnonzero(is_x & ~gap)
        raise ValueError("label X inside a half on {} frames (first at {})".format(
            bad.size, int(bad[0])))
    if (gap & ~is_x).any():
        bad = np.flatnonzero(gap & ~is_x)
        raise ValueError("label other than X where half == 0 on {} frames (first at {})".format(
            bad.size, int(bad[0])))


__all__ = ["DEAD_LOGIT_MAX", "LABEL_CODES", "LABEL_DTYPE", "STATES", "TEAM_LOGIT_MAX",
           "validate_force_d", "validate_labels", "validate_logits"]
