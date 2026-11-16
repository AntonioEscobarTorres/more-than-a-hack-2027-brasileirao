"""Fit of the possession-change hazard on the external IDSSE matches (task F1-08).

Target (per half): frames less than ``target_w_frames`` from a direct H<->A change, or from a
D run shorter than ``bridge_d_frames`` between different teams. Frames labelled D or X are left
out of the fit.

Fit: standardised design matrix (training mean/std), ``LogisticRegression(solver="lbfgs")``
with intercept, 1 frame in ``subsample_every``. The pair (``window_frames``, ``C``) is chosen in
``window_grid`` x ``c_grid`` by pooled leave-one-match-out log-loss: over the 7 matches for
``all``, and over the 6 training matches of each fold for ``folds[<match>]`` (nested, so the
held-out match never picks its own parameters). ``window_frames`` does not change the target,
so the log-losses are comparable; ``target_w_frames`` and ``bridge_d_frames`` define the target
and are fixed in the fit config.

Complementarity with v1 (:func:`v1_complementarity`): every true change is classed by what
``v1_config`` predicts around it (on time, late/early, missed), and the out-of-fold hazard peak
around it is compared with the peak on control windows away from any change.

Artifact checks (pre-registered in ``configs/hazard_fit.yaml``):

* **still ball** (D-009): out-of-fold AUC on frames with the ball visible and stopped for more
  than ``still_min_frames`` versus the other frames. If the first exceeds the second by more than
  ``artifact_auc_margin``, ``still_ball_features`` leave the fit.
* **ball-gap cue** (D-016: the degraded ball is hidden much more often in D): if the single-
  feature AUC of any of ``gap_features`` moves by more than ``artifact_auc_margin`` between the
  ``hidelo`` and ``hidehi`` masters, ``gap_features`` leave the fit.

When a check fires, the model is refitted once without that group and the report shows both
rounds.

Usage::

    python -m modeling.calibration.hazard --fit-config configs/hazard_fit.yaml \\
        --out configs/calib/hazard.yaml --report reports/calibration_hazard.md
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn
import yaml
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from modeling import paths
from modeling.config import Float, Int, Str, _check_mapping, load_config
from modeling.contracts import LABEL_CODES
from modeling.deadball.features import _true_runs, half_blocks
from evaluation import possession_metrics as pm
from modeling.evidence.hazard import (RAW_FEATURES, build_hazard_features, design_columns,
                                      design_matrix, hazard_from_model, model_logit)
from modeling.external.files import IDSSE_MATCHES, load_external
from modeling.pipeline import predict_labels

H, A, D = LABEL_CODES["H"], LABEL_CODES["A"], LABEL_CODES["D"]
STD_FLOOR = 1e-12


# -- fit config ---------------------------------------------------------------------------------

class StrList:
    """A non-empty list of non-empty strings, optionally restricted to ``choices``."""

    def __init__(self, choices=None):
        self.choices = choices

    def check(self, value, where):
        if not isinstance(value, list) or not value:
            raise ValueError("{}: expected a non-empty list, got {!r}".format(where, value))
        for item in value:
            Str(choices=self.choices).check(item, where)
        if len(set(value)) != len(value):
            raise ValueError("{}: duplicated entries in {!r}".format(where, value))
        return list(value)


class IntList:
    def check(self, value, where):
        if not isinstance(value, list) or not value:
            raise ValueError("{}: expected a non-empty list, got {!r}".format(where, value))
        out = [Int(ge=1).check(v, where) for v in value]
        if len(set(out)) != len(out):
            raise ValueError("{}: duplicated entries in {!r}".format(where, value))
        return sorted(out)


class FloatList:
    def check(self, value, where):
        if not isinstance(value, list) or not value:
            raise ValueError("{}: expected a non-empty list, got {!r}".format(where, value))
        out = [Float(gt=0.0).check(v, where) for v in value]
        if len(set(out)) != len(out):
            raise ValueError("{}: duplicated entries in {!r}".format(where, value))
        return sorted(out)


_FEATURE_NAMES = tuple(RAW_FEATURES)
FIT_SCHEMA = {
    "team_config": Str(),
    "degrade_fit": Str(),
    "degrade_eval": StrList(),
    "target_w_frames": Int(ge=1),
    "bridge_d_frames": Int(ge=0),
    "window_grid": IntList(),
    "acc_clip_mps2": Float(gt=0.0),
    "dist_clip_m": Float(gt=0.0),
    "gap_clip_frames": Float(gt=0.0),
    "h_clip": Float(gt=0.0),
    "subsample_every": Int(ge=1),
    "c_grid": FloatList(),
    "max_iter": Int(ge=1),
    "still_speed_mps": Float(gt=0.0),
    "still_min_frames": Int(ge=1),
    "min_slice_positives": Int(ge=1),
    "artifact_auc_margin": Float(gt=0.0, le=1.0),
    "features": StrList(_FEATURE_NAMES),
    "still_ball_features": StrList(_FEATURE_NAMES),
    "gap_features": StrList(_FEATURE_NAMES),
    "v1_config": Str(),
    "on_time_frames": Int(ge=0),
    "match_tol_frames": Int(ge=1),
}


def load_fit_config(path) -> dict:
    """Strict reader of ``configs/hazard_fit.yaml``; ``team_params`` is added from ``team_config``."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    fit = _check_mapping(FIT_SCHEMA, raw, "")
    team_cfg = Path(fit["team_config"])
    if not team_cfg.is_absolute():
        team_cfg = paths.repo_root() / team_cfg
    cfg = load_config(team_cfg)
    if fit["on_time_frames"] > fit["match_tol_frames"]:
        raise ValueError("on_time_frames must be <= match_tol_frames")
    if cfg["team"]["kind"] != "rules_v1":
        raise ValueError("team_config must use team.kind rules_v1 (dlogit feature)")
    fit["team_params"] = cfg["team"]["params"]
    return fit


def feature_settings(fit: Mapping) -> dict:
    """Settings shared by every model of the calibrated file (``window_frames`` is per model)."""
    return {
        "target_w_frames": int(fit["target_w_frames"]),
        "bridge_d_frames": int(fit["bridge_d_frames"]),
        "acc_clip_mps2": float(fit["acc_clip_mps2"]),
        "dist_clip_m": float(fit["dist_clip_m"]),
        "gap_clip_frames": float(fit["gap_clip_frames"]),
        "h_clip": float(fit["h_clip"]),
        "team_params": _plain(fit["team_params"]),
    }


def _plain(obj):
    """Nested dict/list with plain Python scalars (safe for yaml.safe_dump)."""
    if isinstance(obj, Mapping):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    return obj


# -- target and slices --------------------------------------------------------------------------

def _label_runs(codes: np.ndarray) -> list[tuple[int, int, int]]:
    """``(code, start, stop)`` of every maximal run of equal codes."""
    n = len(codes)
    if n == 0:
        return []
    cuts = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    starts = np.r_[0, cuts]
    stops = np.r_[cuts, n]
    return [(int(codes[a]), int(a), int(b)) for a, b in zip(starts, stops)]


def change_targets(labels: np.ndarray, half: np.ndarray, target_w: int,
                   bridge_d: int) -> tuple[np.ndarray, np.ndarray]:
    """``(y, eligible)``: y = 1 within ``target_w`` frames of a team change; eligible = H or A.

    Direct change at frame c (first frame of the new team): positives in [c - w, c + w).
    Change through a D run [s, e) shorter than ``bridge_d``: positives in [s - w, e + w).
    Nothing crosses a half; any other label between two teams (X, long D) is not a change.
    """
    labels = np.asarray(labels)
    y = np.zeros(len(labels), dtype=np.int8)
    for start, stop in half_blocks(np.asarray(half)):
        runs = _label_runs(labels[start:stop])
        for i, (code, _, _) in enumerate(runs):
            if code not in (H, A) or i + 1 >= len(runs):
                continue
            nxt, s, e = runs[i + 1]
            if nxt in (H, A) and nxt != code:
                lo, hi = s - target_w, s + target_w
            elif (nxt == D and (e - s) < bridge_d and i + 2 < len(runs)
                  and runs[i + 2][0] in (H, A) and runs[i + 2][0] != code):
                lo, hi = s - target_w, e + target_w
            else:
                continue
            n_block = stop - start
            y[start + max(lo, 0):start + min(hi, n_block)] = 1
    eligible = np.isin(labels, (H, A)) & (np.asarray(half) != 0)
    return y, eligible


def still_ball_mask(master: pd.DataFrame, speed_mps: float, min_frames: int) -> np.ndarray:
    """Ball observed with speed < ``speed_mps`` for more than ``min_frames`` frames in a row."""
    speed = master["ball_speed"].to_numpy(dtype=np.float64)
    observed = master["ball_observed"].to_numpy(dtype=np.float64) == 1
    with np.errstate(invalid="ignore"):
        still = observed & np.isfinite(speed) & (speed < speed_mps)
    out = np.zeros(len(master), dtype=bool)
    half = master["half"].to_numpy()
    for start, stop in half_blocks(half):
        for a, b in _true_runs(still[start:stop]):
            if b - a > min_frames:
                out[start + a:start + b] = True
    return out


# -- data ---------------------------------------------------------------------------------------

def match_data(master: pd.DataFrame, labels: np.ndarray, fit: Mapping,
               features: Sequence[str], window: int) -> dict:
    """Eligible rows of one match: design matrix, target, still-ball slice, subsample mask."""
    settings = {**feature_settings(fit), "window_frames": int(window)}
    raw = build_hazard_features(master, settings)
    y, eligible = change_targets(labels, master["half"].to_numpy(),
                                 settings["target_w_frames"], settings["bridge_d_frames"])
    still = still_ball_mask(master, fit["still_speed_mps"], fit["still_min_frames"])
    idx = np.flatnonzero(eligible)
    sub = np.zeros(len(idx), dtype=bool)
    sub[::int(fit["subsample_every"])] = True
    return {
        "x": design_matrix(raw.iloc[idx], features),
        "y": y[idx].astype(np.int64),
        "still": still[idx],
        "sub": sub,
    }


def load_mode(fit: Mapping, degrade: str, features: Sequence[str],
              matches: Sequence[str], windows: Sequence[int]) -> dict:
    """``{window: {match: match_data}}`` for one degradation."""
    out = {int(w): {} for w in windows}
    for match in matches:
        master, labels = load_external(match, degrade)
        for w in out:
            out[w][match] = match_data(master, labels, fit, features, w)
    return out


def _stack(data: Mapping, matches: Sequence[str], *, sub: bool) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for m in matches:
        d = data[m]
        mask = d["sub"] if sub else slice(None)
        xs.append(d["x"][mask])
        ys.append(d["y"][mask])
    return np.concatenate(xs), np.concatenate(ys)


# -- fit ----------------------------------------------------------------------------------------

def fit_model(x: np.ndarray, y: np.ndarray, columns: Sequence[str], c: float,
              max_iter: int) -> dict:
    """Standardise with the training mean/std and fit the logistic model."""
    y = np.asarray(y)
    if y.min() == y.max():
        raise ValueError("the training target has a single class")
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std > STD_FLOOR, std, 1.0)
    z = (x - mean) / std
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        lr = LogisticRegression(C=c, solver="lbfgs", max_iter=max_iter).fit(z, y)
    return {
        "C": float(c),
        "pi": float(y.mean()),
        "intercept": float(lr.intercept_[0]),
        "mean": {col: float(v) for col, v in zip(columns, mean)},
        "std": {col: float(v) for col, v in zip(columns, std)},
        "weight": {col: float(v) for col, v in zip(columns, lr.coef_[0])},
        "n_train": int(len(y)),
        "converged": bool(lr.n_iter_[0] < max_iter),
    }


def predict_proba(x: np.ndarray, model: Mapping, columns: Sequence[str]) -> np.ndarray:
    """Calibrated probability of the model (before the prior correction)."""
    return expit(model_logit(x, model, columns))


def choose_params(data: Mapping, matches: Sequence[str], columns: Sequence[str],
                  c_grid: Sequence[float], max_iter: int) -> tuple[int, float, dict]:
    """(window, C) with the smallest pooled leave-one-match-out log-loss.

    ``data`` is ``{window: {match: match_data}}``. Ties go to the smaller window, then the
    smaller C. Returns ``(window, C, {"W=<w> C=<c>": log-loss})``.
    """
    if len(matches) < 2:
        raise ValueError("choosing parameters by leave-one-match-out needs at least 2 matches")
    scores = {}
    for w in sorted(data):
        for c in sorted(c_grid):
            total, count = 0.0, 0
            for held in matches:
                train = [m for m in matches if m != held]
                model = fit_model(*_stack(data[w], train, sub=True), columns, c, max_iter)
                x, y = _stack(data[w], [held], sub=True)
                total += log_loss(y, predict_proba(x, model, columns), labels=[0, 1]) * len(y)
                count += len(y)
            scores[(w, c)] = total / count
    best = min(sorted(scores), key=lambda k: scores[k])
    return best[0], best[1], {"W={} C={}".format(w, c): v for (w, c), v in scores.items()}


def _fit_final(data: Mapping, train: Sequence[str], columns, w: int, c: float,
               max_iter: int) -> dict:
    model = fit_model(*_stack(data[w], train, sub=True), columns, c, max_iter)
    return {"window_frames": int(w), **model}


def fit_all(data: Mapping, matches: Sequence[str], columns: Sequence[str], fit: Mapping) -> dict:
    """``all`` (parameters by LOMO over every match) and nested ``folds``."""
    w_all, c_all, scores_all = choose_params(data, matches, columns, fit["c_grid"],
                                             fit["max_iter"])
    result = {"all": _fit_final(data, matches, columns, w_all, c_all, fit["max_iter"]),
              "folds": {}, "scores": {"all": scores_all}}
    for held in matches:
        train = [m for m in matches if m != held]
        w, c, scores = choose_params(data, train, columns, fit["c_grid"], fit["max_iter"])
        result["folds"][held] = _fit_final(data, train, columns, w, c, fit["max_iter"])
        result["scores"][held] = scores
    return result


# -- evaluation ---------------------------------------------------------------------------------

def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score)) if 0 < y.sum() < len(y) else float("nan")


def evaluate_mode(data: Mapping, models: Mapping, columns: Sequence[str],
                  min_pos: int) -> dict:
    """Out-of-fold log-loss and AUC per match, pooled AUC in/out of the still-ball slice.

    ``data`` is ``{window: {match: match_data}}``; each match uses its fold's window.
    """
    per_match, ys, ps, stills = {}, [], [], []
    for match, model in models.items():
        d = data[model["window_frames"]][match]
        p = predict_proba(d["x"], model, columns)
        per_match[match] = {"log_loss": float(log_loss(d["y"], p, labels=[0, 1])),
                            "auc": safe_auc(d["y"], p)}
        ys.append(d["y"])
        ps.append(p)
        stills.append(d["still"])
    y, p, still = np.concatenate(ys), np.concatenate(ps), np.concatenate(stills)
    pooled = {"log_loss": float(log_loss(y, p, labels=[0, 1])), "auc": safe_auc(y, p)}
    return {"per_match": per_match, "pooled": pooled,
            "still": slice_auc(y, p, still, min_pos), "n": int(len(y)), "pos": int(y.sum())}


def slice_auc(y: np.ndarray, score: np.ndarray, mask: np.ndarray, min_pos: int) -> dict:
    """AUC inside and outside ``mask``; NaN where the slice has fewer than ``min_pos`` positives."""
    out = {}
    for name, sel in (("in", mask), ("out", ~mask)):
        ok = y[sel].sum() >= min_pos and (len(y[sel]) - y[sel].sum()) >= min_pos
        out[name] = safe_auc(y[sel], score[sel]) if ok else float("nan")
        out["n_" + name] = int(sel.sum())
        out["pos_" + name] = int(y[sel].sum())
    return out


def feature_aucs(data: Mapping, columns: Sequence[str], min_pos: int) -> dict:
    """Single-column AUC on the pooled eligible frames, overall and in/out of the still slice."""
    x = np.concatenate([d["x"] for d in data.values()])
    y = np.concatenate([d["y"] for d in data.values()])
    still = np.concatenate([d["still"] for d in data.values()])
    out = {}
    for j, col in enumerate(columns):
        s = slice_auc(y, x[:, j], still, min_pos)
        out[col] = {"all": safe_auc(y, x[:, j]), "in": s["in"], "out": s["out"]}
    return out


def _bigger(a: float, b: float, margin: float) -> bool:
    return bool(np.isfinite(a) and np.isfinite(b) and a - b > margin)


def artifact_checks(evals: Mapping, feat_auc_by_mode: Mapping, fit: Mapping,
                    features: Sequence[str]) -> dict:
    """Which pre-registered groups leave the fit, and why."""
    margin = fit["artifact_auc_margin"]
    still = evals[fit["degrade_fit"]]["still"]
    still_fires = _bigger(still["in"], still["out"], margin)

    gap_moves = {}
    if "hidelo" in feat_auc_by_mode and "hidehi" in feat_auc_by_mode:
        for name in fit["gap_features"]:
            if name not in features:
                continue
            for col in design_columns([name]):
                lo = feat_auc_by_mode["hidelo"][col]["all"]
                hi = feat_auc_by_mode["hidehi"][col]["all"]
                gap_moves[col] = float(abs(hi - lo)) if np.isfinite(lo) and np.isfinite(hi) \
                    else float("nan")
    gap_fires = any(np.isfinite(v) and v > margin for v in gap_moves.values())
    finite_moves = [v for v in gap_moves.values() if np.isfinite(v)]
    drop = []
    if still_fires:
        drop += [f for f in fit["still_ball_features"] if f in features]
    if gap_fires:
        drop += [f for f in fit["gap_features"] if f in features and f not in drop]
    still_gap = still["in"] - still["out"] if np.isfinite(still["in"]) and np.isfinite(
        still["out"]) else float("nan")
    return {"still_fires": still_fires, "gap_fires": gap_fires, "gap_moves": gap_moves,
            "drop": drop, "still_gap": float(still_gap),
            "gap_max": float(max(finite_moves)) if finite_moves else float("nan")}


def run_round(fit: Mapping, features: Sequence[str], matches: Sequence[str],
              log=print, complementarity: bool = True) -> dict:
    """Fit on ``degrade_fit`` and evaluate on every ``degrade_eval`` mode."""
    columns = design_columns(features)
    log("fit: loading {} masters ({})".format(len(matches), fit["degrade_fit"]))
    fit_data = load_mode(fit, fit["degrade_fit"], features, matches, fit["window_grid"])
    log("fit: choosing (window, C) and fitting {} folds".format(len(matches)))
    models = fit_all(fit_data, matches, columns, fit)
    w_all = models["all"]["window_frames"]
    needed = sorted({m["window_frames"] for m in models["folds"].values()} | {w_all})
    evals, feat_auc = {}, {}
    for degrade in fit["degrade_eval"]:
        log("eval: {}".format(degrade))
        data = fit_data if degrade == fit["degrade_fit"] else load_mode(
            fit, degrade, features, matches, needed)
        evals[degrade] = evaluate_mode(data, models["folds"], columns,
                                       fit["min_slice_positives"])
        feat_auc[degrade] = feature_aucs(data[w_all], columns, fit["min_slice_positives"])
    checks = artifact_checks(evals, feat_auc, fit, features)
    out = {"features": list(features), "columns": columns, "models": models,
           "evals": evals, "feature_auc": feat_auc, "checks": checks}
    if complementarity:
        out["complementarity"] = v1_complementarity(fit, features, models["folds"], matches, log)
    return out


def calibrate_hazard(fit: Mapping, matches: Sequence[str] | None = None, log=print) -> list:
    """Round 1 with every feature; round 2 (once) without the groups the checks drop."""
    matches = list(IDSSE_MATCHES if matches is None else matches)
    rounds = [run_round(fit, fit["features"], matches, log)]
    drop = rounds[0]["checks"]["drop"]
    # v1 complementarity is only reported for the final round
    if drop:
        rounds[0].pop("complementarity", None)
    if drop:
        kept = [f for f in fit["features"] if f not in drop]
        if not kept:
            raise ValueError("the artifact checks dropped every feature")
        log("checks dropped {}; refitting".format(drop))
        rounds.append(run_round(fit, kept, matches, log))
    return rounds


# -- complementarity with v1 ------------------------------------------------------------------

COMP_CLASSES = ("on_time", "off_time", "missed", "control")


def change_events(codes: np.ndarray, half: np.ndarray,
                  bridge_d: int) -> list[tuple[int, int, int, int, int]]:
    """Team changes ``(frame, from, to, block_start, block_stop)``, per half.

    Direct H<->A: the first frame of the new team. Through a D run [s, e) shorter than
    ``bridge_d`` between different teams: the D midpoint ``(s + e) // 2``.
    """
    out = []
    for start, stop in half_blocks(np.asarray(half)):
        runs = _label_runs(np.asarray(codes)[start:stop])
        for i, (code, _, _) in enumerate(runs):
            if code not in (H, A) or i + 1 >= len(runs):
                continue
            nxt, s, e = runs[i + 1]
            if nxt in (H, A) and nxt != code:
                out.append((start + s, code, nxt, start, stop))
            elif (nxt == D and (e - s) < bridge_d and i + 2 < len(runs)
                  and runs[i + 2][0] in (H, A) and runs[i + 2][0] != code):
                out.append((start + (s + e) // 2, code, runs[i + 2][0], start, stop))
    return out


def classify_changes(true_events, pred_events, on_time: int, tol: int) -> list[str]:
    """Class of each true change: nearest unused predicted change of the same direction.

    ``on_time`` if |delta| <= ``on_time``, ``off_time`` if <= ``tol``, else ``missed``.
    """
    used = np.zeros(len(pred_events), dtype=bool)
    out = []
    for frame, a, b, _, _ in true_events:
        best, best_d = -1, tol + 1
        for j, (pf, pa, pb, _, _) in enumerate(pred_events):
            d = abs(pf - frame)
            if not used[j] and pa == a and pb == b and d < best_d:
                best, best_d = j, d
        if best < 0:
            out.append("missed")
        else:
            used[best] = True
            out.append("on_time" if best_d <= on_time else "off_time")
    return out


def window_peak(h: np.ndarray, frame: int, w: int, start: int, stop: int) -> float:
    """max h over [frame - w, frame + w), inside the half block."""
    lo, hi = max(frame - w, start), min(frame + w, stop)
    return float(h[lo:hi].max())


def control_centres(y: np.ndarray, eligible: np.ndarray, half: np.ndarray, w: int) -> list:
    """Centres every 2w frames whose whole window [c - w, c + w) is eligible with y == 0."""
    out = []
    for start, stop in half_blocks(half):
        for c in range(start + w, stop - w, 2 * w):
            if eligible[c - w:c + w].all() and not y[c - w:c + w].any():
                out.append((c, start, stop))
    return out


def v1_complementarity(fit: Mapping, features: Sequence[str], models: Mapping,
                       matches: Sequence[str], log=print) -> dict:
    """Out-of-fold hazard peak around true changes, split by what v1 predicts there."""
    v1_path = Path(fit["v1_config"])
    if not v1_path.is_absolute():
        v1_path = paths.repo_root() / v1_path
    v1_cfg = load_config(v1_path)
    calib = {"features": list(features), "settings": feature_settings(fit)}
    w, tol = int(fit["target_w_frames"]), int(fit["match_tol_frames"])
    peaks = {k: [] for k in COMP_CLASSES}
    for match in matches:
        log("complementarity: {}".format(match))
        master, labels = load_external(match, fit["degrade_fit"])
        half = master["half"].to_numpy()
        h = hazard_from_model(master, calib, models[match]).astype(np.float64)
        pred = pm.label_codes(predict_labels(master, v1_cfg, holdout=match))
        true_ev = change_events(labels, half, int(fit["bridge_d_frames"]))
        pred_ev = change_events(pred, half, tol)
        for ev, cls in zip(true_ev, classify_changes(true_ev, pred_ev,
                                                      int(fit["on_time_frames"]), tol)):
            peaks[cls].append(window_peak(h, ev[0], w, ev[3], ev[4]))
        y, eligible = change_targets(labels, half, w, int(fit["bridge_d_frames"]))
        for c, start, stop in control_centres(y, eligible, half, w):
            peaks["control"].append(window_peak(h, c, w, start, stop))
    return summarize_peaks(peaks)


def summarize_peaks(peaks: Mapping[str, Sequence[float]]) -> dict:
    """Per class: n, median peak, share of peaks > 0 and > 1, AUC of the class vs control."""
    control = np.asarray(peaks["control"], dtype=np.float64)
    summary = {}
    for k in COMP_CLASSES:
        v = np.asarray(peaks[k], dtype=np.float64)
        auc = float("nan")
        if k != "control" and v.size and control.size:
            y = np.r_[np.ones(v.size), np.zeros(control.size)]
            auc = safe_auc(y, np.r_[v, control])
        summary[k] = {"n": int(v.size),
                      "median": float(np.median(v)) if v.size else float("nan"),
                      "frac_pos": float((v > 0).mean()) if v.size else float("nan"),
                      "frac_gt1": float((v > 1).mean()) if v.size else float("nan"),
                      "auc_vs_control": auc}
    return summary


def d_run_stats(matches: Sequence[str], degrade: str) -> dict:
    """Durations (frames) of D runs between two team runs: different teams vs the same team."""
    between = {"different": [], "same": []}
    for match in matches:
        master, labels = load_external(match, degrade)
        for start, stop in half_blocks(master["half"].to_numpy()):
            runs = _label_runs(labels[start:stop])
            for i in range(len(runs) - 2):
                a, (d, s, e), b = runs[i][0], runs[i + 1], runs[i + 2][0]
                if a in (H, A) and d == D and b in (H, A):
                    between["different" if a != b else "same"].append(e - s)
    out = {}
    for k, v in between.items():
        v = np.asarray(v, dtype=np.float64)
        out[k] = {"n": int(v.size), "min": float(v.min()) if v.size else float("nan"),
                  "p10": float(np.percentile(v, 10)) if v.size else float("nan"),
                  "median": float(np.median(v)) if v.size else float("nan")}
    return out


def complementarity_markdown(comp: Mapping, fit: Mapping) -> list[str]:
    lines = ["### Complementarity with v1 (`{}`, `{}` masters, out-of-fold)".format(
        fit["v1_config"], fit["degrade_fit"]), "",
        "Each true H<->A change is classed by the nearest v1 change of the same direction "
        "(v1 changes through a D shorter than {} frames count): **on time** within {} frames, "
        "**off time** within {}, otherwise **missed**. *Peak* = max `h` within {} frames of the "
        "change. *Control* = windows of the same size with no change and no D/X.".format(
            fit["match_tol_frames"], fit["on_time_frames"], fit["match_tol_frames"],
            fit["target_w_frames"]), "",
        "| class | n | median peak h | share peak h > 0 | share peak h > 1 | AUC vs control |",
        "|:---|---:|---:|---:|---:|---:|"]
    for k in COMP_CLASSES:
        c = comp[k]
        lines.append("| {} | {} | {} | {} | {} | {} |".format(
            k, c["n"], _f(c["median"], 3), _f(c["frac_pos"], 3), _f(c["frac_gt1"], 3),
            _f(c["auc_vs_control"], 3)))
    return lines + [""]


# -- outputs ------------------------------------------------------------------------------------

def _model_payload(model: Mapping) -> dict:
    return {k: _plain(model[k]) for k in ("window_frames", "C", "pi", "intercept", "mean", "std",
                                          "weight")}


def calib_payload(fit: Mapping, rounds: Sequence[Mapping], matches: Sequence[str]) -> dict:
    final = rounds[-1]
    return {
        "features": list(final["features"]),
        "settings": feature_settings(fit),
        "all": _model_payload(final["models"]["all"]),
        "folds": {m: _model_payload(final["models"]["folds"][m]) for m in matches},
        "meta": {
            "task": "F1-08",
            "degrade_fit": fit["degrade_fit"],
            "matches": list(matches),
            "subsample_every": int(fit["subsample_every"]),
            "c_grid": [float(c) for c in fit["c_grid"]],
            "window_grid": [int(w) for w in fit["window_grid"]],
            "dropped_features": [f for f in fit["features"] if f not in final["features"]],
            "sklearn": sklearn.__version__,
        },
    }


def write_calib(payload: Mapping, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ("# F1-08 possession-change hazard, fitted on IDSSE only. Generated by\n"
              "# python -m modeling.calibration.hazard; do not edit by hand.\n")
    path.write_text(header + yaml.safe_dump(_plain(payload), sort_keys=False),
                    encoding="utf-8", newline="\n")
    return path


def _f(v, nd=4) -> str:
    return "n/a" if v is None or not np.isfinite(v) else "{:.{}f}".format(v, nd)


def _round_markdown(r: Mapping, fit: Mapping, title: str) -> list[str]:
    lines = ["## {}".format(title), "",
             "Features: {}".format(", ".join("`{}`".format(f) for f in r["features"])), ""]
    m = r["models"]
    lines += ["### Model and C", "",
              "| model | W | C | pi | n_train | converged | best LOMO log-loss | worst |",
              "|:---|---:|---:|---:|---:|:---|---:|---:|"]
    for name in ["all"] + list(m["folds"]):
        model = m["all"] if name == "all" else m["folds"][name]
        sc = m["scores"][name].values()
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            name, model["window_frames"], model["C"], _f(model["pi"]), model["n_train"],
            model["converged"], _f(min(sc)), _f(max(sc))))
    lines += ["", "LOMO log-loss of `all` per (W, C):", "", "| W, C | log-loss |", "|:---|---:|"]
    for key, v in m["scores"]["all"].items():
        lines.append("| {} | {} |".format(key, _f(v)))
    lines += ["", "Weights of `all` (standardised features):", "",
              "| column | weight |", "|:---|---:|"]
    for col in r["columns"]:
        lines.append("| `{}` | {} |".format(col, _f(m["all"]["weight"][col])))

    modes = list(r["evals"])
    lines += ["", "### Out-of-fold log-loss and AUC per match and degradation", "",
              "Each match is predicted by `folds[<match>]` (fitted without it).", "",
              "| match | " + " | ".join("{} log-loss | {} AUC".format(d, d) for d in modes) + " |",
              "|:---|" + "---:|" * (2 * len(modes))]
    for match in r["evals"][modes[0]]["per_match"]:
        cells = []
        for d in modes:
            pm_ = r["evals"][d]["per_match"][match]
            cells += [_f(pm_["log_loss"]), _f(pm_["auc"])]
        lines.append("| {} | {} |".format(match, " | ".join(cells)))
    cells = []
    for d in modes:
        cells += [_f(r["evals"][d]["pooled"]["log_loss"]), _f(r["evals"][d]["pooled"]["auc"])]
    lines.append("| **pooled** | {} |".format(" | ".join(cells)))

    lines += ["", "### Still-ball slice (ball visible, speed < {} m/s for more than {} frames)"
              .format(fit["still_speed_mps"], fit["still_min_frames"]), "",
              "| degrade | AUC in slice | AUC outside | frames in (pos) | frames out (pos) |",
              "|:---|---:|---:|---:|---:|"]
    for d in modes:
        s = r["evals"][d]["still"]
        lines.append("| {} | {} | {} | {} ({}) | {} ({}) |".format(
            d, _f(s["in"]), _f(s["out"]), s["n_in"], s["pos_in"], s["n_out"], s["pos_out"]))

    lines += ["", "### Single-feature AUC ({}; pooled eligible frames; W = {})".format(
        fit["degrade_fit"], m["all"]["window_frames"]), "",
        "AUC < 0.5 means the column is larger away from changes.", "",
        "| column | AUC all | AUC in still slice | AUC outside | AUC hidelo | AUC hidehi |",
        "|:---|---:|---:|---:|---:|---:|"]
    fa = r["feature_auc"]
    for col in r["columns"]:
        base = fa[fit["degrade_fit"]][col]
        lo = fa.get("hidelo", {}).get(col, {}).get("all")
        hi = fa.get("hidehi", {}).get(col, {}).get("all")
        lines.append("| `{}` | {} | {} | {} | {} | {} |".format(
            col, _f(base["all"]), _f(base["in"]), _f(base["out"]), _f(lo), _f(hi)))

    c = r["checks"]
    lines += ["", "### Artifact checks (margin {})".format(fit["artifact_auc_margin"]), "",
              "- still ball (D-009): {}".format(
                  "**fires**, drop {}".format(fit["still_ball_features"]) if c["still_fires"]
                  else "does not fire"),
              "- ball-gap cue (D-016), |AUC hidehi - AUC hidelo|: {}; {}".format(
                  ", ".join("`{}` {}".format(k, _f(v)) for k, v in c["gap_moves"].items())
                  or "n/a",
                  "**fires**, drop {}".format(fit["gap_features"]) if c["gap_fires"]
                  else "does not fire"),
              "- sensitivity to the margin: largest observed gaps are {} (still ball, in - out) "
              "and {} (ball gap, hidehi vs hidelo); a check fires only for a margin below its "
              "gap".format(_f(c["still_gap"]), _f(c["gap_max"])),
              ""]
    if "complementarity" in r:
        lines += complementarity_markdown(r["complementarity"], fit)
    return lines


def report_markdown(fit: Mapping, rounds: Sequence[Mapping], fit_path: str,
                    d_stats: Mapping | None = None) -> str:
    final = rounds[-1]
    dropped = [f for f in fit["features"] if f not in final["features"]]
    lines = [
        "# F1-08: possession-change hazard, calibration on IDSSE",
        "",
        "Generated by `python -m modeling.calibration.hazard` from `{}`. IDSSE masters only "
        "(public data); nothing from the match data.".format(fit_path),
        "",
        "- target: frames less than {} frames from an H<->A change ({}); D and X frames are out "
        "of the fit".format(
            fit["target_w_frames"],
            "bridging through D is off" if fit["bridge_d_frames"] == 0 else
            "or from a D shorter than {} frames between different teams".format(
                fit["bridge_d_frames"])),
        "- fit: `{}` masters, 1 frame in {}, (W, C) in {} x {} by pooled leave-one-match-out "
        "log-loss (nested inside each fold)".format(fit["degrade_fit"], fit["subsample_every"],
                                                     fit["window_grid"], fit["c_grid"]),
        "- final features: {}".format(", ".join("`{}`".format(f) for f in final["features"])),
        "- dropped by the artifact checks: {}".format(
            ", ".join("`{}`".format(f) for f in dropped) if dropped else "none"),
        "- adoption is decided only by F2-06, through F2-07",
        "",
    ]
    if d_stats is not None:
        lines += ["## D runs between two team runs (`{}` labels, frames)".format(
            fit["degrade_fit"]), "",
            "Basis of `bridge_d_frames`: a short D between different teams would be a handover "
            "worth counting as a change. In IDSSE no such short D exists, and D between different "
            "teams lasts as long as D within a team (a real stoppage), so bridging is off.", "",
            "| between | n | min | p10 | median |", "|:---|---:|---:|---:|---:|"]
        for k in ("different", "same"):
            st = d_stats[k]
            lines.append("| {} teams | {} | {} | {} | {} |".format(
                k, st["n"], _f(st["min"], 0), _f(st["p10"], 0), _f(st["median"], 0)))
        lines.append("")
    edge = [name for name, model in [("all", final["models"]["all"])]
            + list(final["models"]["folds"].items())
            if model["window_frames"] == max(fit["window_grid"])]
    if edge:
        scores = final["models"]["scores"]["all"]
        by_w = {w: min(v for k, v in scores.items() if k.startswith("W={} ".format(w)))
                for w in fit["window_grid"]}
        lines += ["- note: W = {} (the largest in the grid) is chosen by {} of {} models; the "
                  "best LOMO log-loss of `all` per W is {}, so a larger W could still help a "
                  "little".format(max(fit["window_grid"]), len(edge),
                                  1 + len(final["models"]["folds"]),
                                  ", ".join("{}: {}".format(w, _f(v)) for w, v in by_w.items())),
                  ""]
    for i, r in enumerate(rounds, 1):
        title = "Round {}{}".format(i, " (final)" if i == len(rounds) else "")
        lines += _round_markdown(r, fit, title)
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Possession-change hazard calibration (F1-08)")
    parser.add_argument("--fit-config", default="configs/hazard_fit.yaml")
    parser.add_argument("--out", default="configs/calib/hazard.yaml")
    parser.add_argument("--report", default="reports/calibration_hazard.md")
    args = parser.parse_args(argv)

    root = paths.repo_root()
    fit = load_fit_config(root / args.fit_config)
    rounds = calibrate_hazard(fit)
    d_stats = d_run_stats(list(IDSSE_MATCHES), fit["degrade_fit"])
    out = write_calib(calib_payload(fit, rounds, list(IDSSE_MATCHES)), root / args.out)
    print("wrote {}".format(out))
    report = root / args.report
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(report_markdown(fit, rounds, args.fit_config, d_stats), encoding="utf-8",
                      newline="\n")
    print("wrote {}".format(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["FIT_SCHEMA", "artifact_checks", "calib_payload", "calibrate_hazard",
           "change_events", "change_targets", "choose_params", "classify_changes",
           "complementarity_markdown", "control_centres", "summarize_peaks", "v1_complementarity", "window_peak", "evaluate_mode", "feature_aucs", "fit_all",
           "fit_model", "load_fit_config", "match_data", "predict_proba", "report_markdown",
           "still_ball_mask", "write_calib"]
