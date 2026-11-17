"""Stacked ``team_logit`` (F2-09, decision D-029): a second GBDT over the first-stage evidence.

The first stage is the configured team evidence ``t`` (the F1-07 GBDT); the possession-change
hazard ``h`` is the configured hazard (the GBDT hazard of :mod:`modeling.evidence.hazard_gbdt`,
before ``beta``). The second stage sees both in time windows and returns a new ``team_logit``::

    t2 = clip((g(X) - g(S X)) / 2, +-TEAM_LOGIT_MAX);  t2 = 0 where half == 0;  float32

Features (:func:`stack_feature_names`), per ``half`` block, NaN on ``half == 0``:

- ``t`` and its past/future nanmeans over ``W`` frames (``t_past_W``, ``t_fut_W``) for ``W`` in
  :data:`T_WINDOWS` (all odd under a team swap);
- ``h`` and its past/future maxima over ``W`` frames (``h_past_max_W``, ``h_fut_max_W``) for ``W``
  in :data:`H_WINDOWS` (all even).

``S`` flips the sign of the odd columns, so ``t2`` is exactly antisymmetric. Why it helps: the HMM
treats frames as independent given the state and the hazard only changes the transition costs; the
second stage learns from IDSSE where, between a past that says one team and a future that says the
other, the switch actually happens, and sharpens ``t`` there (D-029).

Training (``python -m modeling.calibration.stack_gbdt``) uses out-of-sample first-stage inputs only:
the ``all`` model is trained on every IDSSE match with that match's leave-one-match-out ``t`` and
``h``; the fold model of match ``m`` on the other matches ``j`` with ``t`` and ``h`` from models
trained without both ``m`` and ``j`` (nested), so a fold never sees its own match.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml

from .. import paths
from ..contracts import DEAD_LOGIT_MAX, TEAM_LOGIT_MAX
from .windows import rolling_window_stats

VARIANT = "gbdt"
STACK_KIND_NAMES = ("none", "gbdt")
T_WINDOWS = (6, 12, 25, 50, 100, 200)
H_WINDOWS = (12, 50)
D_WINDOWS = (6, 12, 25, 50, 100, 200)
CALIB_KEYS = ("variant", "sklearn_version", "model_file", "fold_dir", "features")


def stack_feature_names() -> list[str]:
    names = ["t"]
    for w in T_WINDOWS:
        names.extend(["t_past_{}".format(w), "t_fut_{}".format(w)])
    for w in H_WINDOWS:
        names.extend(["h_past_max_{}".format(w), "h_fut_max_{}".format(w)])
    names.append("h")
    return names


def stack_sign() -> np.ndarray:
    """Team-swap sign of every feature: -1 for the ``t`` columns (odd), +1 for ``h`` (even)."""
    return np.array([-1.0 if n.startswith("t") else 1.0 for n in stack_feature_names()])


def build_stack_features(t, h, half) -> np.ndarray:
    """Feature matrix ``(n, len(stack_feature_names()))`` in float64, NaN on ``half == 0``."""
    t = np.asarray(t, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    half = np.asarray(half)
    if t.shape != h.shape or t.shape != half.shape or t.ndim != 1:
        raise ValueError("t, h and half must be 1-D arrays of the same length")
    cols = [t]
    for w in T_WINDOWS:
        r = rolling_window_stats(t, half, w, ("past_mean", "fut_mean"))
        cols.extend([r["past_mean"], r["fut_mean"]])
    for w in H_WINDOWS:
        r = rolling_window_stats(h, half, w, ("past_max", "fut_max"))
        cols.extend([r["past_max"], r["fut_max"]])
    cols.append(h)
    X = np.column_stack(cols)
    X[half == 0, :] = np.nan
    return X


def antisymmetric_stack_logit(model, X: np.ndarray) -> np.ndarray:
    """``(g(X) - g(S X)) / 2`` in float64, single-threaded (rule 6)."""
    from threadpoolctl import threadpool_limits

    if X.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    Xs = X * stack_sign()
    with threadpool_limits(limits=1):
        g = model.decision_function(X).astype(np.float64)
        gs = model.decision_function(Xs).astype(np.float64)
    return 0.5 * (g - gs)


def stack_from_model(model, t, h, half) -> np.ndarray:
    """Second-stage ``team_logit`` (float32) from a fitted model."""
    half = np.asarray(half)
    X = build_stack_features(t, h, half)
    t2 = np.zeros(len(half), dtype=np.float64)
    inside = half != 0
    t2[inside] = antisymmetric_stack_logit(model, X[inside])
    t2 = np.clip(t2, -TEAM_LOGIT_MAX, TEAM_LOGIT_MAX)
    t2 = np.where(inside & np.isfinite(t2), t2, 0.0) + 0.0
    return t2.astype(np.float32)


# -- dead_logit second stage (D-029) ----------------------------------------------------------------
#
# Same idea for the dead ball: a second GBDT over ``d_fit`` (the F3-08 dead GBDT likelihood ratio,
# before the Smart Tagging prior) and the hazard ``h`` in time windows. Every feature is even under
# a team swap. The output is a likelihood ratio against the D share of its training rows, and the
# fixed tag prior is added *after* it (the IDSSE has no tags), then clipped as the dead kinds are:
#
#     d2 = clip(g(X) - logit(pi_D) + tag_prior, +-DEAD_LOGIT_MAX);  d2 = 0 where half == 0

def dead_stack_feature_names() -> list[str]:
    names = ["d", "h"]
    for w in D_WINDOWS:
        names.extend(["d_past_{}".format(w), "d_fut_{}".format(w)])
    for w in H_WINDOWS:
        names.extend(["h_past_max_{}".format(w), "h_fut_max_{}".format(w)])
    return names


def build_dead_stack_features(d_fit, h, half) -> np.ndarray:
    """Feature matrix ``(n, len(dead_stack_feature_names()))`` in float64, NaN on ``half == 0``."""
    d = np.asarray(d_fit, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    half = np.asarray(half)
    if d.shape != h.shape or d.shape != half.shape or d.ndim != 1:
        raise ValueError("d_fit, h and half must be 1-D arrays of the same length")
    cols = [d, h]
    for w in D_WINDOWS:
        r = rolling_window_stats(d, half, w, ("past_mean", "fut_mean"))
        cols.extend([r["past_mean"], r["fut_mean"]])
    for w in H_WINDOWS:
        r = rolling_window_stats(h, half, w, ("past_max", "fut_max"))
        cols.extend([r["past_max"], r["fut_max"]])
    X = np.column_stack(cols)
    X[half == 0, :] = np.nan
    return X


def dead_stack_from_model(model, pi_d: float, d_fit, h, half, tag_prior=None) -> np.ndarray:
    """Second-stage ``dead_logit`` (float32) from a fitted model and its training D share."""
    from threadpoolctl import threadpool_limits

    pi_d = float(pi_d)
    if not 0.0 < pi_d < 1.0:
        raise ValueError("pi_D must be in (0, 1), got {}".format(pi_d))
    half = np.asarray(half)
    X = build_dead_stack_features(d_fit, h, half)
    out = np.zeros(len(half), dtype=np.float64)
    inside = half != 0
    if inside.any():
        with threadpool_limits(limits=1):
            g = model.decision_function(X[inside]).astype(np.float64)
        out[inside] = g - np.log(pi_d / (1.0 - pi_d))
    if tag_prior is not None:
        out = out + np.asarray(tag_prior, dtype=np.float64)
    out = np.clip(out, -DEAD_LOGIT_MAX, DEAD_LOGIT_MAX)
    out = np.where(inside & np.isfinite(out), out, 0.0) + 0.0
    return out.astype(np.float32)


# -- calibration file and models -----------------------------------------------------------------

def resolve_repo_path(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else paths.repo_root() / p


@lru_cache(maxsize=8)
def _load_yaml(path_str: str, mtime_ns: int, target: str) -> dict:
    del mtime_ns
    with open(path_str, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or data.get("variant") != VARIANT:
        raise ValueError("invalid stack calibration file: {}".format(path_str))
    missing = [k for k in CALIB_KEYS if k not in data]
    if missing:
        raise ValueError("stack calibration file {} misses {}".format(path_str, missing))
    if data.get("target", "team") != target:
        raise ValueError("{} is a {!r} stack file, expected {!r}".format(
            path_str, data.get("target", "team"), target))
    expected = stack_feature_names() if target == "team" else dead_stack_feature_names()
    if list(data["features"]) != expected:
        raise ValueError("feature list of {} does not match the code".format(path_str))
    return data


def load_stack_calibration(path, target: str = "team") -> dict:
    """Read ``configs/calib/stack.yaml`` (``target`` team) or ``dead_stack.yaml`` (``dead``)."""
    if target not in ("team", "dead"):
        raise ValueError("target must be 'team' or 'dead', got {!r}".format(target))
    p = resolve_repo_path(path)
    if not p.is_file():
        raise FileNotFoundError("stack calibration file not found: {}".format(p))
    return _load_yaml(str(p.resolve()), p.stat().st_mtime_ns, target)


def fold_model_path(calib: Mapping, holdout: str) -> Path:
    fold_dir = Path(calib["fold_dir"])
    if not fold_dir.is_absolute():
        fold_dir = paths.ext_dir() / fold_dir
    return fold_dir / "{}.joblib".format(holdout)


@lru_cache(maxsize=16)
def _load_bundle(path_str: str, mtime_ns: int) -> dict:
    del mtime_ns
    import joblib
    return joblib.load(path_str)


def load_stack_bundle(calib: Mapping, holdout: str | None, features) -> dict:
    import sklearn

    version = str(calib["sklearn_version"])
    if sklearn.__version__ != version:
        raise RuntimeError("stack model needs scikit-learn {}, found {}".format(
            version, sklearn.__version__))
    p = resolve_repo_path(calib["model_file"]) if holdout is None else fold_model_path(calib, holdout)
    if not p.is_file():
        raise FileNotFoundError(
            "stack model not found: {} (run python -m modeling.calibration.stack_gbdt)".format(p))
    bundle = _load_bundle(str(p.resolve()), p.stat().st_mtime_ns)
    if bundle.get("sklearn_version") != version:
        raise RuntimeError("model {} was saved with scikit-learn {}, expected {}".format(
            p, bundle.get("sklearn_version"), version))
    if list(bundle.get("features", [])) != list(features):
        raise ValueError("model {} has a different feature list".format(p))
    return bundle


def load_stack_model(calib: Mapping, holdout: str | None):
    return load_stack_bundle(calib, holdout, stack_feature_names())["model"]


def stack_team_logit(t, h, half, section: Mapping, *, holdout: str | None = None) -> np.ndarray:
    """Second-stage ``team_logit`` for the config section ``stack`` (``kind`` gbdt)."""
    if section["kind"] != VARIANT:
        raise ValueError("stack kind must be {!r}, got {!r}".format(VARIANT, section["kind"]))
    calib = load_stack_calibration(section["calibrated_file"], "team")
    return stack_from_model(load_stack_model(calib, holdout), t, h, half)


def dead_stack_logit(d_fit, h, half, section: Mapping, *, tag_prior=None,
                     holdout: str | None = None) -> np.ndarray:
    """Second-stage ``dead_logit`` for the config section ``dead_stack`` (``kind`` gbdt)."""
    if section["kind"] != VARIANT:
        raise ValueError("dead_stack kind must be {!r}, got {!r}".format(VARIANT, section["kind"]))
    calib = load_stack_calibration(section["calibrated_file"], "dead")
    bundle = load_stack_bundle(calib, holdout, dead_stack_feature_names())
    return dead_stack_from_model(bundle["model"], bundle["pi_d"], d_fit, h, half, tag_prior)


__all__ = ["D_WINDOWS", "H_WINDOWS", "STACK_KIND_NAMES", "T_WINDOWS", "antisymmetric_stack_logit",
           "build_dead_stack_features", "build_stack_features", "dead_stack_feature_names",
           "dead_stack_from_model", "dead_stack_logit", "load_stack_bundle",
           "load_stack_calibration", "load_stack_model", "stack_feature_names",
           "stack_from_model", "stack_sign", "stack_team_logit"]
