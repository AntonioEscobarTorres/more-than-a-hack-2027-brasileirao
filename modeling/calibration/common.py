"""Shared calibration utilities for L2 logistic regression on external IDSSE matches (F1-05, F3-04).

Rules enforced:
- Uncentred scaling (divide by training std without subtracting the mean) to preserve antisymmetry.
- Inference dot products use explicit ``(X * w).sum(axis=1)`` in float64, never ``@``/BLAS (rule 6).
- Leave-one-match-out cross-validation over ``C in {0.01, 0.1, 1.0}``.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    _HAS_SKLEARN = True
except ModuleNotFoundError:  # pragma: no cover - fallback when sklearn is not installed
    LogisticRegression = None
    roc_auc_score = None
    _HAS_SKLEARN = False

from evaluation import possession_metrics as pm

SUBSAMPLE_STEP = 5
DEFAULT_C_GRID = (0.01, 0.1, 1.0)
SCALE_EPS = 1e-12


def alive_frames_mask(gt_codes: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Boolean mask of graded alive frames (H or A, inside a half)."""
    gt_codes = np.asarray(gt_codes)
    half = np.asarray(half)
    return (half != 0) & ((gt_codes == pm.H) | (gt_codes == pm.A))


def subsample_indices(mask: np.ndarray, step: int = SUBSAMPLE_STEP) -> np.ndarray:
    """Deterministic 1-every-``step`` indices among frames where ``mask`` is True."""
    if step < 1:
        raise ValueError("step must be >= 1, got {}".format(step))
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    return idx[::step]


def compute_uncentered_scales(X: np.ndarray) -> np.ndarray:
    """Training standard deviation per feature without centering (float64, guarded > 0)."""
    X64 = np.asarray(X, dtype=np.float64)
    if X64.ndim != 2 or len(X64) == 0:
        raise ValueError("X must be a non-empty 2-D array, got shape {}".format(X64.shape))
    scales = np.std(X64, axis=0, ddof=0)
    scales = np.where(np.isfinite(scales) & (scales > SCALE_EPS), scales, 1.0)
    return scales.astype(np.float64)


def linear_logit(
    X: np.ndarray,
    weights: np.ndarray,
    scales: np.ndarray,
    intercept: float = 0.0,
) -> np.ndarray:
    """Deterministic float64 linear score ``(X / scales * weights).sum(axis=1) + intercept``."""
    X64 = np.asarray(X, dtype=np.float64)
    w64 = np.asarray(weights, dtype=np.float64)
    s64 = np.asarray(scales, dtype=np.float64)
    eff_w = np.where(s64 > 0.0, w64 / s64, 0.0)
    return (X64 * eff_w).sum(axis=1) + float(intercept)


def binary_log_loss(y_true: np.ndarray, logit: np.ndarray) -> float:
    """Stable binary cross-entropy from raw logits for ``y_true in {0, 1}``."""
    y = np.asarray(y_true, dtype=np.float64)
    z = np.asarray(logit, dtype=np.float64)
    if len(y) == 0:
        return float("nan")
    signed = (2.0 * y - 1.0) * z
    return float(np.mean(np.logaddexp(0.0, -signed)))


def _numpy_roc_auc(y: np.ndarray, z: np.ndarray) -> float:
    """Exact ROC AUC via average-tie ranks (Mann-Whitney U), identical to sklearn."""
    order = np.argsort(z, kind="mergesort")
    z_sorted = z[order]
    n = len(z_sorted)
    ranks = np.empty(n, dtype=np.float64)
    edges = np.flatnonzero(np.r_[True, z_sorted[1:] != z_sorted[:-1], True])
    for a, b in zip(edges[:-1], edges[1:]):
        ranks[order[a:b]] = 0.5 * (a + 1 + b)
    pos = y == 1
    n1 = float(np.sum(pos))
    n0 = float(n - n1)
    return float((np.sum(ranks[pos]) - 0.5 * n1 * (n1 + 1.0)) / (n1 * n0))


def binary_auc(y_true: np.ndarray, logit: np.ndarray) -> float:
    """ROC AUC from raw logits; returns NaN if only one class is present."""
    y = np.asarray(y_true, dtype=np.int8)
    z = np.asarray(logit, dtype=np.float64)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    if _HAS_SKLEARN:
        return float(roc_auc_score(y, z))
    return _numpy_roc_auc(y, z)


def _fit_logistic_l2_numpy(
    X_scaled: np.ndarray,
    y_bin: np.ndarray,
    c_value: float,
    *,
    fit_intercept: bool = False,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> tuple[np.ndarray, float]:
    """Damped Newton-Raphson (IRLS) minimizing sklearn's exact L2 logistic objective."""
    n, d = X_scaled.shape
    y = y_bin.astype(np.float64)
    reg = 1.0 / float(c_value)

    if fit_intercept:
        Xd = np.column_stack([X_scaled, np.ones(n, dtype=np.float64)])
        reg_vec = np.r_[np.full(d, reg, dtype=np.float64), 0.0]
    else:
        Xd = X_scaled
        reg_vec = np.full(d, reg, dtype=np.float64)

    theta = np.zeros(Xd.shape[1], dtype=np.float64)

    def _obj(th: np.ndarray) -> float:
        z = (Xd * th).sum(axis=1)
        signed = (2.0 * y - 1.0) * z
        return float(np.sum(np.logaddexp(0.0, -signed)) + 0.5 * np.sum(reg_vec * th * th))

    cur_loss = _obj(theta)
    for _ in range(max_iter):
        z = (Xd * theta).sum(axis=1)
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))
        diff = p - y
        grad = (Xd * diff[:, None]).sum(axis=0) + reg_vec * theta
        if float(np.max(np.abs(grad))) < tol:
            break
        w_diag = np.maximum(p * (1.0 - p), 1e-12)
        H = (Xd.T * w_diag) @ Xd + np.diag(reg_vec)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]

        alpha = 1.0
        while alpha > 1e-6:
            cand = theta - alpha * step
            cand_loss = _obj(cand)
            if cand_loss <= cur_loss + 1e-12:
                theta = cand
                cur_loss = cand_loss
                break
            alpha *= 0.5

    if fit_intercept:
        return theta[:d] + 0.0, float(theta[d])
    return theta + 0.0, 0.0


def fit_logistic_l2(
    X: np.ndarray,
    y: np.ndarray,
    c_value: float,
    *,
    fit_intercept: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit L2 logistic regression on ``X / scales`` and return ``(weights, scales, intercept)``."""
    X64 = np.asarray(X, dtype=np.float64)
    y_bin = np.asarray(y, dtype=np.int8)
    scales = compute_uncentered_scales(X64)
    X_scaled = X64 / scales

    if len(np.unique(y_bin)) < 2:
        return np.zeros(X64.shape[1], dtype=np.float64), scales, 0.0

    if _HAS_SKLEARN:
        clf = LogisticRegression(
            C=float(c_value),
            solver="lbfgs",
            fit_intercept=fit_intercept,
            max_iter=1000,
        )
        clf.fit(X_scaled, y_bin)
        weights = clf.coef_.reshape(-1).astype(np.float64) + 0.0
        intercept = float(clf.intercept_[0]) if fit_intercept else 0.0
        return weights, scales, intercept

    weights, intercept = _fit_logistic_l2_numpy(
        X_scaled, y_bin, c_value, fit_intercept=fit_intercept)
    return weights, scales, intercept


def lomo_log_loss_by_c(
    match_data: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    fit_intercept: bool = False,
) -> dict[float, float]:
    """Mean leave-one-match-out log-loss for each ``C`` of ``c_grid`` (ascending ``C``)."""
    matches = list(match_data.keys())
    if len(matches) < 2:
        raise ValueError("need at least 2 matches for leave-one-out calibration")
    c_losses = {}
    for c_val in sorted(float(c) for c in c_grid):
        fold_losses = []
        for holdout in matches:
            X_tr = np.vstack([match_data[m][0] for m in matches if m != holdout])
            y_tr = np.concatenate([match_data[m][1] for m in matches if m != holdout])
            X_va, y_va = match_data[holdout]
            if len(X_tr) == 0 or len(X_va) == 0:
                continue
            w, s, b = fit_logistic_l2(X_tr, y_tr, c_val, fit_intercept=fit_intercept)
            pred_logit = linear_logit(X_va, w, s, b)
            fold_losses.append(binary_log_loss(y_va, pred_logit))
        c_losses[float(c_val)] = float(np.nanmean(fold_losses)) if fold_losses else float("inf")
    return c_losses


def select_c_and_fit_folds(
    match_data: Mapping[str, tuple[np.ndarray, np.ndarray]],
    feature_names: Sequence[str],
    *,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    fit_intercept: bool = False,
) -> dict:
    """Select ``C`` by mean leave-one-match-out log-loss and fit ``all`` and ``folds``."""
    matches = list(match_data.keys())
    feat_names = list(feature_names)
    c_losses = lomo_log_loss_by_c(match_data, c_grid=c_grid, fit_intercept=fit_intercept)

    # Strictly lower loss wins; ties keep the smaller C (stronger regularisation).
    best_c = None
    best_loss = float("inf")
    for c_val, mean_loss in c_losses.items():
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_c = c_val

    assert best_c is not None

    def _pack(w: np.ndarray, s: np.ndarray, b: float) -> dict:
        out = {
            "scales": {name: float(s[i]) for i, name in enumerate(feat_names)},
            "weights": {name: float(w[i]) for i, name in enumerate(feat_names)},
        }
        if fit_intercept:
            out["intercept"] = float(b)
        return out

    X_all = np.vstack([match_data[m][0] for m in matches])
    y_all = np.concatenate([match_data[m][1] for m in matches])
    w_all, s_all, b_all = fit_logistic_l2(X_all, y_all, best_c, fit_intercept=fit_intercept)

    folds = {}
    fold_metrics = {}
    for holdout in matches:
        X_tr = np.vstack([match_data[m][0] for m in matches if m != holdout])
        y_tr = np.concatenate([match_data[m][1] for m in matches if m != holdout])
        X_va, y_va = match_data[holdout]
        w_f, s_f, b_f = fit_logistic_l2(X_tr, y_tr, best_c, fit_intercept=fit_intercept)
        folds[holdout] = _pack(w_f, s_f, b_f)
        val_logit = linear_logit(X_va, w_f, s_f, b_f)
        fold_metrics[holdout] = {
            "log_loss": binary_log_loss(y_va, val_logit),
            "auc": binary_auc(y_va, val_logit),
            "n_samples": int(len(y_va)),
        }

    return {
        "C": float(best_c),
        "c_grid_log_loss": c_losses,
        "features": feat_names,
        "all": _pack(w_all, s_all, b_all),
        "folds": folds,
        "fold_metrics": fold_metrics,
    }


def possession_boundary_shift(
    gt_codes: np.ndarray,
    pred_codes: np.ndarray,
    *,
    iou_threshold: float = pm.SPELL_IOU_HEADLINE,
) -> float:
    """Mean boundary shift (pred - gt, in frames) across matched possessions."""
    runs = pm.valid_runs(gt_codes)
    filled = pm.fill_abstentions(pred_codes, runs)
    gt_spells = pm.possession_spells(gt_codes, runs)
    pred_spells = pm.possession_spells(filled, runs)
    overlaps = pm.spell_overlaps(gt_spells, pred_spells)

    used_gt = set()
    used_pred = set()
    shifts = []
    for iou, gt_idx, pred_idx in overlaps:
        if iou < iou_threshold:
            break
        if gt_idx in used_gt or pred_idx in used_pred:
            continue
        used_gt.add(gt_idx)
        used_pred.add(pred_idx)
        gt_a, gt_b, _ = gt_spells[gt_idx]
        pr_a, pr_b, _ = pred_spells[pred_idx]
        shifts.append(0.5 * ((pr_a - gt_a) + (pr_b - gt_b)))
    return float(np.mean(shifts)) if shifts else float("nan")


__all__ = [
    "DEFAULT_C_GRID",
    "SUBSAMPLE_STEP",
    "alive_frames_mask",
    "binary_auc",
    "binary_log_loss",
    "compute_uncentered_scales",
    "fit_logistic_l2",
    "linear_logit",
    "lomo_log_loss_by_c",
    "possession_boundary_shift",
    "select_c_and_fit_folds",
    "subsample_indices",
]