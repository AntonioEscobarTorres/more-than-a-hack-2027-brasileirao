"""Calibration of the dead-ball evidence ``dead_logit`` on the external IDSSE masters (task F3-04).

A logistic regression (dead ball vs live ball) is fitted on the degraded IDSSE masters, with the
features below (no Smart Tagging: the external matches have no tags). Its logit is turned into a
likelihood ratio by subtracting the prior log-odds of the training data,

    d_fit = logit_model - logit(pi_D)

so that the share of D is not counted twice (once in the logit, once in the HMM transitions). The
``calibrated`` dead variant (``modeling.deadball.dead``) then adds the fixed Smart Tagging prior of
Appendix A.3 (``dead_tag_prior``) and clips to +-12.

Features (``FEATURE_SETS["full"]``). Every missing value is imputed inside the design matrix only,
always with an indicator (rule 4):

    has_ball            ball position available (observed or interpolated)
    out_of_pitch        out_of_pitch == 1; 0 without a ball (has_ball is its indicator)
    gap_exit, gap_air, gap_long
                        one-hot of db_gap_kind (F3-01)
    still_frac          db_still_frac; 0 and still_frac_missing = 1 where it is NaN
    mean_speed          db_mean_speed; 0 and mean_speed_missing = 1 where it is NaN
    fast_in_pitch       db_fast_in_pitch
    log_gap             log1p(min(gap_length, gap_log_clip_frames)); 0 and log_gap_missing = 1
                        where gap_length is NaN

``db_ball_rest`` and ``ball_speed`` are never used (update of 11/11, item 3): on the external
matches the ball freezes during dead balls, so they leak the label. The ``reduced`` set drops
``has_ball`` and ``log_gap`` (with its indicator) and is used when the live-play gap scenario R1
fails (item 4).

The ``interactions`` set adds the pairwise products of the three drivers of a ball gap: its kind
(``gap_exit``, ``gap_long``), its length (``log_gap``) and the player speed (``mean_speed``),

    exit_x_log_gap, long_x_log_gap, exit_x_speed, long_x_speed, speed_x_log_gap

Without them the model has one slope in gap length for every gap kind and speed. On IDSSE the
sign of that slope depends on both: a long gap with slow players is mostly a dead ball, a long
exit gap with running players is mostly live play after the restart (the gap spans it), and a
short exit gap with running players is mostly the dead ball that has just started. Air gaps are
rare and never longer than ``air_max_frames``, so they get no interaction. The products are taken
after the imputation, so a missing input gives 0 and its own indicator still flags it.

Fitting: each column is standardised with the mean and standard deviation of the training rows,
then ``LogisticRegression`` (lbfgs, with intercept) is fitted for each ``C`` of ``C_GRID``; ``C``
is chosen by the mean log-loss with one match left out at a time. The chosen ``C`` gives the model
``all`` (every match) and one model per fold (``folds[<match>]``, fitted without that match).

Feature set: every set of ``CANDIDATE_SETS`` is fitted, and the one with the lowest
leave-one-match-out log-loss among those where R1 holds on every model is kept (ties go to the
earlier, simpler set). R2 is never a selection criterion, only reported. If R1 fails for every
candidate, the ``reduced`` set is fitted (item 4).

Usage (external data only; the provided match is never read)::

    python -m modeling.calibration.dead --config configs/v1.yaml

writes ``configs/calib/dead.yaml`` (``default`` masters), ``dead_pooled.yaml`` (``default``,
``hidelo`` and ``hidehi`` together), ``dead_hidelo.yaml`` and ``dead_hidehi.yaml`` (only for the
stability check of F2-06), and ``reports/calibration_dead.md``.
"""
from __future__ import annotations

import argparse
import copy
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from modeling import paths
from modeling.deadball.features import GAP_AIR, GAP_EXIT, GAP_LONG, build_deadball_features

FORMAT_VERSION = 1

# Fixed by the task specification (F3-04), not tuned.
C_GRID = (0.01, 0.1, 1.0)
SUBSAMPLE = 5                      # 1 frame in 5 for the single-mode fits
POOLED_SUBSAMPLE = 15              # 1 frame in 15 of each mode for the pooled fit
POOLED_MODES = ("default", "hidelo", "hidehi")
GAP_LOG_CLIP_FRAMES = 1500         # log1p(min(gap_length, 1500))
MAX_ITER = 1000
WORSE_LOG_LOSS = 0.10              # item 5: report a mode whose log-loss is > 10% worse
STD_FLOOR = 1e-12                  # a constant column is not scaled (std -> 1)

FULL_FEATURES = ("has_ball", "out_of_pitch", "gap_exit", "gap_air", "gap_long", "still_frac",
                 "still_frac_missing", "mean_speed", "mean_speed_missing", "fast_in_pitch",
                 "log_gap", "log_gap_missing")
REDUCED_FEATURES = tuple(f for f in FULL_FEATURES
                         if f not in ("has_ball", "log_gap", "log_gap_missing"))
# Pairwise products of the gap drivers (see the module docstring): name -> (factor, factor).
INTERACTIONS = {
    "exit_x_log_gap": ("gap_exit", "log_gap"),
    "long_x_log_gap": ("gap_long", "log_gap"),
    "exit_x_speed": ("gap_exit", "mean_speed"),
    "long_x_speed": ("gap_long", "mean_speed"),
    "speed_x_log_gap": ("mean_speed", "log_gap"),
}
INTERACTION_FEATURES = tuple(INTERACTIONS)
ALL_FEATURES = FULL_FEATURES + INTERACTION_FEATURES
FEATURE_SETS = {"full": FULL_FEATURES, "interactions": ALL_FEATURES,
                "reduced": REDUCED_FEATURES}
CANDIDATE_SETS = ("full", "interactions")      # chosen by log-loss, simpler first (ties)
FALLBACK_SET = "reduced"                        # item 4, when R1 fails for every candidate

# Expected sign of the effect of each feature in the model ``all`` (task specification). With
# interactions the weight alone is not the effect, so the check uses the slope of ``d_fit`` in
# that feature (``effective_slopes``) at the training means and at every scenario.
EXPECTED_SIGNS = {"out_of_pitch": 1, "fast_in_pitch": -1, "mean_speed": -1}

# Raw per-frame inputs of the design matrix.
RAW_COLUMNS = ("has_ball", "out_of_pitch", "db_gap_kind", "gap_length", "db_still_frac",
               "db_mean_speed", "db_fast_in_pitch")

# Robustness scenarios of item 4. The values are inputs of the check, not match data.
_R1 = {"has_ball": 0.0, "out_of_pitch": np.nan, "db_gap_kind": GAP_LONG, "gap_length": 50.0,
       "db_still_frac": 0.2, "db_mean_speed": 2.5, "db_fast_in_pitch": 0.0}
SCENARIOS = {
    "R1": dict(_R1),                                              # live play, long gap: d < 0
    "R2": {**_R1, "db_gap_kind": GAP_EXIT},                       # ball out: d > 0
    "R3": {**_R1, "db_still_frac": 0.45, "db_mean_speed": 1.8},   # slow possession: reported
}

MODEL_KEYS = ("mean", "std", "weights", "intercept", "pi_d", "n_train")

# Report only: upper edges of the mean-player-speed bins (m/s) of the empirical P(D) table. The
# first bin also holds 0; the last bin is everything above the last edge. Never used to fit.
SPEED_BIN_EDGES_MPS = (1.0, 1.5, 1.9, 2.3, 2.7, 3.5)
GAP_KIND_COLUMNS = (("gap_exit", "exit gap"), ("gap_long", "long gap"), ("gap_air", "air gap"))
# Report only: the empirical cell of a scenario holds the frames of its gap kind with a gap of at
# most SCENARIO_CELL_MAX_GAP_FRAMES frames (R1 and R2 have 50) and a mean player speed within
# SCENARIO_CELL_SPEED_HALF_WIDTH_MPS of the scenario's. Its interval is a bootstrap over whole
# gaps (frames of one gap are not independent). Never used to fit or to choose anything.
SCENARIO_CELL_MAX_GAP_FRAMES = 100
SCENARIO_CELL_SPEED_HALF_WIDTH_MPS = 0.4
SCENARIO_CELL_BOOTSTRAP_DRAWS = 2000
SCENARIO_CELL_BOOTSTRAP_SEED = 0
SCENARIO_CELL_INTERVAL = (5.0, 95.0)           # percentiles: a 90% interval


# -- features ------------------------------------------------------------------------------------
def raw_features(master: pd.DataFrame, params: dict) -> pd.DataFrame:
    """The raw inputs of the design matrix, one row per master frame (NaN kept)."""
    missing = [c for c in ("half", "ball_x", "ball_y", "out_of_pitch", "gap_length")
               if c not in master.columns]
    if missing:
        raise ValueError("master is missing columns {}".format(missing))
    db = build_deadball_features(master, params)
    has_ball = (np.isfinite(master["ball_x"].to_numpy(dtype="float64"))
                & np.isfinite(master["ball_y"].to_numpy(dtype="float64")))
    return pd.DataFrame({
        "has_ball": has_ball.astype("float64"),
        "out_of_pitch": master["out_of_pitch"].to_numpy(dtype="float64"),
        "db_gap_kind": db["db_gap_kind"].to_numpy(dtype="float64"),
        "gap_length": master["gap_length"].to_numpy(dtype="float64"),
        "db_still_frac": db["db_still_frac"].to_numpy(dtype="float64"),
        "db_mean_speed": db["db_mean_speed"].to_numpy(dtype="float64"),
        "db_fast_in_pitch": db["db_fast_in_pitch"].to_numpy(dtype="float64"),
    }, index=master.index)


def _fill(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(x)
    return np.where(ok, x, 0.0), (~ok).astype("float64")


def design_matrix(raw: pd.DataFrame, features: Sequence[str],
                  gap_log_clip_frames: float = GAP_LOG_CLIP_FRAMES) -> np.ndarray:
    """float64 (n, len(features)) design matrix, no NaN, columns in the order of ``features``."""
    unknown = [f for f in features if f not in ALL_FEATURES]
    if unknown:
        raise ValueError("unknown features {}".format(unknown))
    missing = [c for c in RAW_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError("raw features are missing columns {}".format(missing))

    def col(name):
        return raw[name].to_numpy(dtype="float64")

    has_ball = np.where(col("has_ball") == 1, 1.0, 0.0)
    out_of_pitch = np.where(col("out_of_pitch") == 1, 1.0, 0.0)          # NaN -> 0
    kind = col("db_gap_kind")
    still, still_missing = _fill(col("db_still_frac"))
    speed, speed_missing = _fill(col("db_mean_speed"))
    with np.errstate(invalid="ignore"):
        log_gap = np.log1p(np.minimum(col("gap_length"), gap_log_clip_frames))
    log_gap, log_gap_missing = _fill(log_gap)
    columns = {
        "has_ball": has_ball, "out_of_pitch": out_of_pitch,
        "gap_exit": (kind == GAP_EXIT).astype("float64"),
        "gap_air": (kind == GAP_AIR).astype("float64"),
        "gap_long": (kind == GAP_LONG).astype("float64"),
        "still_frac": still, "still_frac_missing": still_missing,
        "mean_speed": speed, "mean_speed_missing": speed_missing,
        "fast_in_pitch": np.where(col("db_fast_in_pitch") == 1, 1.0, 0.0),
        "log_gap": log_gap, "log_gap_missing": log_gap_missing,
    }
    for name, (a, b) in INTERACTIONS.items():
        columns[name] = columns[a] * columns[b]          # imputed factors: missing -> 0
    if not len(features):
        return np.zeros((len(raw), 0))
    return np.column_stack([columns[f] for f in features]).astype("float64")


# -- model ---------------------------------------------------------------------------------------
def _vector(model: Mapping, key: str, features: Sequence[str]) -> np.ndarray:
    return np.asarray([float(model[key][f]) for f in features], dtype="float64")


def model_logit(X: np.ndarray, model: Mapping, features: Sequence[str]) -> np.ndarray:
    """Logit of the fitted model, P(D) on the log-odds scale (prior included). float64[n]."""
    mean, std = _vector(model, "mean", features), _vector(model, "std", features)
    weights = _vector(model, "weights", features)
    z = (X - mean) / std
    return (z * weights).sum(axis=1) + float(model["intercept"])     # explicit sum, rule 6


def prior_logit(model: Mapping) -> float:
    pi = float(model["pi_d"])
    return float(np.log(pi) - np.log1p(-pi))


def d_fit(X: np.ndarray, model: Mapping, features: Sequence[str]) -> np.ndarray:
    """Likelihood ratio log p(obs | D) - log p(obs | alive): the model logit minus logit(pi_D)."""
    return model_logit(X, model, features) - prior_logit(model)


def fit_model(X: np.ndarray, y: np.ndarray, features: Sequence[str], C: float) -> dict:
    """Standardise with the training rows and fit the logistic regression with intercept."""
    from sklearn.linear_model import LogisticRegression

    X = np.asarray(X, dtype="float64")
    y = np.asarray(y, dtype=bool)
    if X.ndim != 2 or X.shape != (len(y), len(features)):
        raise ValueError("X must have shape ({}, {}), got {}".format(len(y), len(features),
                                                                  X.shape))
    pi = float(y.mean()) if len(y) else float("nan")
    if not 0.0 < pi < 1.0:
        raise ValueError("training rows need both classes, got P(D) = {}".format(pi))
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std > STD_FLOOR, std, 1.0)
    clf = LogisticRegression(C=C, solver="lbfgs", max_iter=MAX_ITER, fit_intercept=True)
    clf.fit((X - mean) / std, y)
    return {
        "mean": {f: float(v) for f, v in zip(features, mean)},
        "std": {f: float(v) for f, v in zip(features, std)},
        "weights": {f: float(v) for f, v in zip(features, clf.coef_[0])},
        "intercept": float(clf.intercept_[0]),
        "pi_d": pi,
        "n_train": int(len(y)),
    }


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -x))


def log_loss(y: np.ndarray, logit: np.ndarray) -> float:
    """Mean binary log-loss from logits (stable for large |logit|)."""
    y = np.asarray(y, dtype=bool)
    return float(np.mean(np.where(y, np.logaddexp(0.0, -logit), np.logaddexp(0.0, logit))))


def auc(y: np.ndarray, score: np.ndarray) -> float:
    """ROC AUC of ``score`` for ``y`` (ties count 1/2); NaN without both classes."""
    y = np.asarray(y, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(np.asarray(score, dtype="float64")).rank(method="average").to_numpy()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# -- data ----------------------------------------------------------------------------------------
def usable_rows(master: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    """Indices of frames inside a half with a label other than X."""
    from modeling.contracts import LABEL_CODES

    labels = np.asarray(labels)
    if labels.shape != (len(master),):
        raise ValueError("labels must have one value per master row")
    return np.flatnonzero((master["half"].to_numpy() != 0) & (labels != LABEL_CODES["X"]))


def match_design(master: pd.DataFrame, labels: np.ndarray, params: dict,
                 features: Sequence[str] = ALL_FEATURES,
                 gap_log_clip_frames: float = GAP_LOG_CLIP_FRAMES) -> tuple[np.ndarray, np.ndarray]:
    """``(X, y)`` over the usable frames of one match, in frame order (not subsampled)."""
    from modeling.contracts import LABEL_CODES

    rows = usable_rows(master, labels)
    raw = raw_features(master, params).iloc[rows]
    X = design_matrix(raw, features, gap_log_clip_frames)
    y = np.asarray(labels)[rows] == LABEL_CODES["D"]
    return X, y


def _subsample(X: np.ndarray, y: np.ndarray, step: int) -> tuple[np.ndarray, np.ndarray]:
    return X[::step], y[::step]


def _stack(parts):
    Xs = [p[0] for p in parts]
    ys = [p[1] for p in parts]
    return np.vstack(Xs), np.concatenate(ys)


def training_rows(data: Mapping[str, Sequence[tuple[np.ndarray, np.ndarray]]],
                  matches: Sequence[str], step: int, columns: Sequence[int]):
    """Stack the subsampled rows of ``matches``. ``data[m]`` is a list of ``(X, y)``, one per mode."""
    parts = []
    for m in matches:
        for X, y in data[m]:
            Xs, ys = _subsample(X, y, step)
            parts.append((Xs[:, columns], ys))
    return _stack(parts)


def _columns(features: Sequence[str]) -> list[int]:
    """Positions of ``features`` in a design built with ``ALL_FEATURES`` (or a prefix of it)."""
    return [ALL_FEATURES.index(f) for f in features]


def calibrate(data: Mapping[str, Sequence[tuple[np.ndarray, np.ndarray]]],
              features: Sequence[str], step: int, c_grid: Sequence[float] = C_GRID) -> dict:
    """Choose ``C`` by leave-one-match-out log-loss, then fit ``all`` and every fold.

    ``data[m]`` holds the ``(X, y)`` of match ``m`` (one entry per degradation mode), with the
    columns of ``ALL_FEATURES`` (or of a prefix of it that covers ``features``). The held-out
    log-loss is computed on the subsampled rows of the held-out match. Ties in the mean log-loss
    go to the smaller ``C``. ``cv_by_match`` is the held-out log-loss of each match at that ``C``.
    """
    matches = sorted(data)
    if len(matches) < 2:
        raise ValueError("need at least two matches to calibrate, got {}".format(matches))
    cols = _columns(features)
    cv, by_match = {}, {}
    for C in sorted(c_grid):
        losses = {}
        for m in matches:
            X_tr, y_tr = training_rows(data, [o for o in matches if o != m], step, cols)
            X_te, y_te = training_rows(data, [m], step, cols)
            model = fit_model(X_tr, y_tr, features, C)
            losses[m] = log_loss(y_te, model_logit(X_te, model, features))
        cv[float(C)] = float(np.mean(list(losses.values())))
        by_match[float(C)] = losses
    best = None
    for C in sorted(cv):
        if best is None or cv[C] < cv[best]:
            best = C
    X_all, y_all = training_rows(data, matches, step, cols)
    folds = {}
    for m in matches:
        X_tr, y_tr = training_rows(data, [o for o in matches if o != m], step, cols)
        folds[m] = fit_model(X_tr, y_tr, features, best)
    return {"C": best, "cv_log_loss": cv, "cv_by_match": by_match[best],
            "all": fit_model(X_all, y_all, features, best), "folds": folds}


# -- robustness scenarios ------------------------------------------------------------------------
def scenario_design(features: Sequence[str],
                    gap_log_clip_frames: float = GAP_LOG_CLIP_FRAMES) -> pd.DataFrame:
    """One design row per scenario of ``SCENARIOS`` (index = scenario name)."""
    raw = pd.DataFrame([{c: float(SCENARIOS[s][c]) for c in RAW_COLUMNS} for s in SCENARIOS],
                       index=list(SCENARIOS))
    return pd.DataFrame(design_matrix(raw, features, gap_log_clip_frames), index=raw.index,
                        columns=list(features))


def scenario_values(model: Mapping, features: Sequence[str],
                    gap_log_clip_frames: float = GAP_LOG_CLIP_FRAMES) -> dict:
    X = scenario_design(features, gap_log_clip_frames)
    values = d_fit(X.to_numpy(), model, features)
    return {s: float(v) for s, v in zip(X.index, values)}


def robustness(calib: Mapping, features: Sequence[str],
               gap_log_clip_frames: float = GAP_LOG_CLIP_FRAMES) -> dict:
    """R1-R3 for ``all`` and every fold, and whether R1 (< 0) and R2 (> 0) hold everywhere."""
    models = {"all": calib["all"], **{"fold " + m: f for m, f in calib["folds"].items()}}
    values = {name: scenario_values(model, features, gap_log_clip_frames)
              for name, model in models.items()}
    return {
        "values": values,
        "r1_pass": all(v["R1"] < 0 for v in values.values()),
        "r2_pass": all(v["R2"] > 0 for v in values.values()),
    }


def _attempt(data, name: str, step: int, c_grid: Sequence[float]) -> tuple[dict, dict, dict]:
    features = FEATURE_SETS[name]
    calib = calibrate(data, features, step, c_grid)
    check = robustness(calib, features)
    summary = {"feature_set": name, "C": float(calib["C"]),
               "cv_log_loss": float(calib["cv_log_loss"][calib["C"]]),
               "cv_log_loss_by_match": {m: float(v) for m, v in calib["cv_by_match"].items()},
               "r1_pass": bool(check["r1_pass"]), "r2_pass": bool(check["r2_pass"]),
               "scenarios_all": dict(check["values"]["all"])}
    return calib, check, summary


def calibrate_with_fallback(data, step: int, c_grid: Sequence[float] = C_GRID) -> dict:
    """Fit every candidate set and keep the best one where R1 holds; else the reduced set.

    "Best" is the lowest leave-one-match-out log-loss at the chosen ``C``; ties go to the earlier
    set of ``CANDIDATE_SETS``. R2 is reported for every attempt but never used to choose. If R1
    fails for every candidate, the ``reduced`` set is fitted and kept (item 4), whatever its R1.
    """
    attempts = {name: _attempt(data, name, step, c_grid) for name in CANDIDATE_SETS}
    tried = [attempts[name][2] for name in CANDIDATE_SETS]
    passing = [name for name in CANDIDATE_SETS if attempts[name][1]["r1_pass"]]
    if passing:
        name = min(passing, key=lambda n: (attempts[n][2]["cv_log_loss"],
                                           CANDIDATE_SETS.index(n)))
        calib, check, _ = attempts[name]
    else:
        name = FALLBACK_SET
        calib, check, summary = _attempt(data, name, step, c_grid)
        tried.append(summary)
    return {**calib, "feature_set": name, "features": list(FEATURE_SETS[name]),
            "robustness": check, "tried": tried}


def effective_slopes(model: Mapping, features: Sequence[str],
                     rows: Mapping[str, Mapping[str, float]]) -> dict:
    """Slope of ``d_fit`` in each feature of ``EXPECTED_SIGNS``, per unit, at each design row.

    ``rows`` maps a row name to design values (``{feature: value}``, as built by
    :func:`design_matrix`). The slope of ``x`` is ``w_x / std_x`` plus, for every interaction
    ``x * o`` of ``features``, ``w / std * o``. Without interactions it is ``w_x / std_x`` at
    every row. Returns ``{feature: {row: slope}}`` for the features of ``EXPECTED_SIGNS`` that
    are in ``features``.
    """
    def raw_weight(f):
        return float(model["weights"][f]) / float(model["std"][f])

    out = {}
    for f in EXPECTED_SIGNS:
        if f not in features:
            continue
        out[f] = {}
        for row_name, row in rows.items():
            slope = raw_weight(f)
            for name, (a, b) in INTERACTIONS.items():
                if name not in features:
                    continue
                if a == f:
                    slope += raw_weight(name) * float(row[b])
                if b == f:
                    slope += raw_weight(name) * float(row[a])
            out[f][row_name] = slope
    return out


def sign_rows(model: Mapping, features: Sequence[str],
              gap_log_clip_frames: float = GAP_LOG_CLIP_FRAMES) -> dict:
    """The rows of the sign check: the training means of ``model`` and every scenario."""
    scen = scenario_design(features, gap_log_clip_frames)
    rows = {"training mean": {f: float(model["mean"][f]) for f in features}}
    rows.update({s: {f: float(scen.loc[s, f]) for f in features} for s in scen.index})
    return rows


def signs_ok(model: Mapping, features: Sequence[str],
             gap_log_clip_frames: float = GAP_LOG_CLIP_FRAMES) -> bool:
    """Every slope of :func:`effective_slopes` has its expected sign at every sign row."""
    slopes = effective_slopes(model, features, sign_rows(model, features, gap_log_clip_frames))
    return all(np.sign(v) == EXPECTED_SIGNS[f] for f, per_row in slopes.items()
               for v in per_row.values())


# -- calibration file ----------------------------------------------------------------------------
def feature_params(params: Mapping) -> dict:
    """The ``dead.params`` that define the features: everything but the weights."""
    return {k: v for k, v in params.items() if k not in ("weights", "calibrated_file")}


def calibration_document(result: Mapping, params: Mapping, *, train_modes: Sequence[str],
                         step: int, description: str) -> dict:
    return {
        "format_version": FORMAT_VERSION,
        "description": description,
        "feature_set": result["feature_set"],
        "features": list(result["features"]),
        "gap_log_clip_frames": GAP_LOG_CLIP_FRAMES,
        "train_modes": list(train_modes),
        "subsample": int(step),
        "c_grid": [float(c) for c in C_GRID],
        "C": float(result["C"]),
        "cv_log_loss": {float(k): float(v) for k, v in result["cv_log_loss"].items()},
        "feature_params": copy.deepcopy(feature_params(params)),
        "robustness": {
            "r1_pass": bool(result["robustness"]["r1_pass"]),
            "r2_pass": bool(result["robustness"]["r2_pass"]),
            "values": result["robustness"]["values"],
            "tried": result["tried"],
        },
        "all": result["all"],
        "folds": dict(sorted(result["folds"].items())),
    }


def write_calibration(document: Mapping, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ("# Generated by modeling.calibration.dead (F3-04) from the degraded IDSSE masters.\n"
              "# External data only; the provided match is never read. Do not edit by hand.\n")
    path.write_text(header + yaml.safe_dump(dict(document), sort_keys=False,
                                            default_flow_style=False), encoding="utf-8")
    return path


def validate_calibration(doc, source: str = "calibration") -> dict:
    """Check the structure of a calibration document; returns it unchanged."""
    if not isinstance(doc, dict):
        raise ValueError("{}: expected a mapping".format(source))
    for key in ("format_version", "features", "gap_log_clip_frames", "feature_params", "all",
                "folds"):
        if key not in doc:
            raise ValueError("{}: missing key {!r}".format(source, key))
    if doc["format_version"] != FORMAT_VERSION:
        raise ValueError("{}: format_version {} is not {}".format(
            source, doc["format_version"], FORMAT_VERSION))
    features = doc["features"]
    if (not isinstance(features, list) or len(set(features)) != len(features)
            or any(f not in ALL_FEATURES for f in features)):
        raise ValueError("{}: bad feature list {}".format(source, features))
    models = {"all": doc["all"], **{"folds." + str(k): v for k, v in dict(doc["folds"]).items()}}
    for name, model in models.items():
        if not isinstance(model, dict) or any(k not in model for k in MODEL_KEYS):
            raise ValueError("{}: model {} needs keys {}".format(source, name, MODEL_KEYS))
        for key in ("mean", "std", "weights"):
            if not isinstance(model[key], dict) or set(model[key]) != set(features):
                raise ValueError("{}: {}.{} must have one value per feature".format(
                    source, name, key))
            if not all(np.isfinite(float(v)) for v in model[key].values()):
                raise ValueError("{}: {}.{} has non-finite values".format(source, name, key))
        if not all(float(v) > 0 for v in model["std"].values()):
            raise ValueError("{}: {}.std must be positive".format(source, name))
        if not 0.0 < float(model["pi_d"]) < 1.0:
            raise ValueError("{}: {}.pi_d must be in (0, 1)".format(source, name))
        if not np.isfinite(float(model["intercept"])):
            raise ValueError("{}: {}.intercept must be finite".format(source, name))
    return doc


@lru_cache(maxsize=16)
def _load_cached(path: str, mtime_ns: int, size: int) -> dict:
    with open(path, encoding="utf-8") as fh:
        return validate_calibration(yaml.safe_load(fh), path)


def load_calibration(path) -> dict:
    """Read and validate a calibration file (cached by path, modification time and size)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("calibration file not found: {}".format(path))
    stat = path.stat()
    return _load_cached(str(path.resolve()), stat.st_mtime_ns, stat.st_size)


def select_model(doc: Mapping, holdout: str | None) -> dict:
    """``folds[holdout]`` when a holdout is given, ``all`` otherwise."""
    if holdout is None:
        return doc["all"]
    folds = doc["folds"]
    if holdout not in folds:
        raise ValueError("no fold for holdout {!r}; folds are {}".format(holdout, sorted(folds)))
    return folds[holdout]


def check_feature_params(doc: Mapping, params: Mapping) -> None:
    """The features at inference must be built exactly as in the fit."""
    fitted = doc["feature_params"]
    used = feature_params(params)
    diff = sorted(k for k in set(fitted) | set(used) if fitted.get(k) != used.get(k))
    if diff:
        raise ValueError("dead.params differ from the calibration on {}".format(diff))


def calibrated_d_fit(master: pd.DataFrame, params: Mapping, doc: Mapping,
                     holdout: str | None) -> np.ndarray:
    """``d_fit`` of every master frame (float64), exactly 0 where ``half == 0``."""
    check_feature_params(doc, params)
    model = select_model(doc, holdout)
    features = doc["features"]
    X = design_matrix(raw_features(master, dict(params)), features, doc["gap_log_clip_frames"])
    values = d_fit(X, model, features)
    return np.where(master["half"].to_numpy() != 0, values, 0.0) + 0.0


# -- report --------------------------------------------------------------------------------------
def _fmt(x) -> str:
    return "n/a" if x is None or not np.isfinite(x) else "{:.4f}".format(x)


def evaluate_folds(result: Mapping, eval_data: Mapping[str, Mapping[str, tuple]]) -> dict:
    """``{mode: {match: (log_loss, auc)}}`` of each fold model on its held-out match (all frames)."""
    cols = _columns(result["features"])
    out = {}
    for mode, per_match in eval_data.items():
        out[mode] = {}
        for m, (X, y) in sorted(per_match.items()):
            model = result["folds"][m]
            logit = model_logit(X[:, cols], model, result["features"])
            out[mode][m] = (log_loss(y, logit), auc(y, logit))
    return out


def single_feature_auc(eval_data: Mapping[str, tuple]) -> dict:
    """``{match: {feature: AUC}}`` of each column alone (D vs alive), held-out frames."""
    out = {}
    for m, (X, y) in sorted(eval_data.items()):
        out[m] = {f: auc(y, X[:, i]) for i, f in enumerate(ALL_FEATURES[:X.shape[1]])}
    return out


def speed_bin_labels() -> list[str]:
    edges = SPEED_BIN_EDGES_MPS
    labels = ["<= {}".format(edges[0])]
    labels += ["{}-{}".format(a, b) for a, b in zip(edges[:-1], edges[1:])]
    labels.append("> {}".format(edges[-1]))
    return labels


def _gap_length_at_most(X: np.ndarray, max_gap_frames: float | None) -> np.ndarray:
    """Rows inside a long gap of at most ``max_gap_frames`` frames, or every row for ``None``.

    ``log_gap`` is ``log1p(min(gap_length, clip))``, so the bound is applied on that scale; it
    must be below ``GAP_LOG_CLIP_FRAMES``.
    """
    if max_gap_frames is None:
        return np.ones(len(X), dtype=bool)
    if not max_gap_frames < GAP_LOG_CLIP_FRAMES:
        raise ValueError("max_gap_frames must be below {}".format(GAP_LOG_CLIP_FRAMES))
    known = X[:, ALL_FEATURES.index("log_gap_missing")] == 0
    return known & (X[:, ALL_FEATURES.index("log_gap")] <= np.log1p(max_gap_frames))


def dead_rate_by_speed(eval_data: Mapping[str, tuple],
                       max_gap_frames: float | None = None) -> dict:
    """Empirical P(D) and frame count by gap kind and mean-player-speed bin, all matches pooled.

    ``eval_data[m]`` is the ``(X, y)`` of match ``m``. Frames whose mean player speed is missing
    are left out. Rows are the gap kinds of ``GAP_KIND_COLUMNS`` plus ``no long gap``. With
    ``max_gap_frames``, the gap rows keep only gaps of at most that many frames (``no long gap``
    is unchanged). Returns ``{row: [(P(D) or NaN, n_frames) per bin]}``.
    """
    X = np.vstack([v[0] for v in eval_data.values()])
    y = np.concatenate([v[1] for v in eval_data.values()])
    speed = X[:, ALL_FEATURES.index("mean_speed")]
    known = X[:, ALL_FEATURES.index("mean_speed_missing")] == 0
    bins = np.searchsorted(np.asarray(SPEED_BIN_EDGES_MPS), speed, side="left")
    short = _gap_length_at_most(X, max_gap_frames)
    gap_any = np.zeros(len(y), dtype=bool)
    rows = {}
    for column, label in GAP_KIND_COLUMNS:
        in_kind = X[:, ALL_FEATURES.index(column)] == 1
        gap_any |= in_kind
        rows[label] = in_kind & short
    rows["no long gap"] = ~gap_any
    out = {}
    for label, mask in rows.items():
        cells = []
        for b in range(len(SPEED_BIN_EDGES_MPS) + 1):
            sel = mask & known & (bins == b)
            n = int(sel.sum())
            cells.append((float(y[sel].mean()) if n else float("nan"), n))
        out[label] = cells
    return out


def _gap_runs(X: np.ndarray) -> np.ndarray:
    """Id of the long gap of each row (consecutive rows with the same ``log_gap`` > 0), -1 off it.

    ``X`` holds the usable frames of one match in frame order; two different gaps are always
    separated by observed frames, so a change of ``log_gap`` or a row without a gap ends a run.
    """
    lg = X[:, ALL_FEATURES.index("log_gap")]
    in_gap = (lg > 0) & (X[:, ALL_FEATURES.index("log_gap_missing")] == 0)
    start = in_gap & ~np.r_[False, in_gap[:-1] & (lg[1:] == lg[:-1])]
    ids = np.cumsum(start) - 1
    return np.where(in_gap, ids, -1)


def scenario_cells(eval_data: Mapping[str, tuple], pi_d: float,
                   scenarios: Sequence[str] = ("R1", "R2")) -> dict:
    """Empirical ``d`` of the frames that look like each scenario, with a bootstrap interval.

    The cell of a scenario holds the frames of its gap kind, with a gap of at most
    ``SCENARIO_CELL_MAX_GAP_FRAMES`` frames and a mean player speed within
    ``SCENARIO_CELL_SPEED_HALF_WIDTH_MPS`` of the scenario's. ``d = logit(P(D | cell)) -
    logit(pi_d)``, the quantity ``d_fit`` estimates there. The interval resamples whole gaps
    (``SCENARIO_CELL_BOOTSTRAP_DRAWS`` draws, seed ``SCENARIO_CELL_BOOTSTRAP_SEED``); P(D) is
    kept inside ``[1 / (n + 1), n / (n + 1)]`` so the logit stays finite. Report only.
    Returns ``{scenario: {"n_frames", "n_gaps", "p_dead", "d", "low", "high"}}``.
    """
    prior = float(np.log(pi_d) - np.log1p(-pi_d))
    kind_column = {GAP_EXIT: "gap_exit", GAP_LONG: "gap_long", GAP_AIR: "gap_air"}
    out = {}
    for s in scenarios:
        spec = SCENARIOS[s]
        sums, sizes = [], []
        for m, (X, y) in sorted(eval_data.items()):
            speed = X[:, ALL_FEATURES.index("mean_speed")]
            sel = ((X[:, ALL_FEATURES.index(kind_column[int(spec["db_gap_kind"])])] == 1)
                   & _gap_length_at_most(X, SCENARIO_CELL_MAX_GAP_FRAMES)
                   & (X[:, ALL_FEATURES.index("mean_speed_missing")] == 0)
                   & (np.abs(speed - spec["db_mean_speed"]) <= SCENARIO_CELL_SPEED_HALF_WIDTH_MPS))
            runs = _gap_runs(X)[sel]
            ys = np.asarray(y)[sel]
            for r in np.unique(runs):
                in_run = runs == r
                sums.append(float(ys[in_run].sum()))
                sizes.append(float(in_run.sum()))
        sums, sizes = np.asarray(sums), np.asarray(sizes)
        n = float(sizes.sum())
        if n == 0:
            out[s] = {"n_frames": 0, "n_gaps": 0, "p_dead": float("nan"), "d": float("nan"),
                      "low": float("nan"), "high": float("nan")}
            continue

        def d_of(k, total):
            p = min(max(k / total, 1.0 / (total + 1.0)), total / (total + 1.0))
            return float(np.log(p) - np.log1p(-p) - prior)

        rng = np.random.default_rng(SCENARIO_CELL_BOOTSTRAP_SEED)
        draws = []
        for _ in range(SCENARIO_CELL_BOOTSTRAP_DRAWS):
            pick = rng.integers(0, len(sizes), len(sizes))
            draws.append(d_of(sums[pick].sum(), sizes[pick].sum()))
        low, high = np.percentile(draws, SCENARIO_CELL_INTERVAL)
        out[s] = {"n_frames": int(n), "n_gaps": int(len(sizes)), "p_dead": float(sums.sum() / n),
                  "d": d_of(sums.sum(), n), "low": float(low), "high": float(high)}
    return out


def _pass(ok: bool) -> str:
    return "PASS" if ok else "**FAIL**"


def _rates_table(rates: Mapping) -> list[str]:
    labels = speed_bin_labels()
    lines = ["| Gap kind | " + " | ".join("{} m/s".format(b) for b in labels) + " |",
             "|---|" + "---:|" * len(labels)]
    for row, cells in rates.items():
        lines.append("| {} | {} |".format(row, " | ".join(
            "n/a (0)" if n == 0 else "{:.3f} ({})".format(p, n) for p, n in cells)))
    return lines


def _choice_lines(doc: Mapping) -> list[str]:
    tried = doc["robustness"]["tried"]
    lines = ["Feature-set choice (lowest leave-one-match-out log-loss among the sets where R1 "
             "holds; R2 is not a criterion). R1–R3 are the `d_fit` of the model `all`:", "",
             "| Set | C | Log-loss | R1 | R2 | R3 | R1 on every model | R2 on every model | Kept |",
             "|---|---:|---:|---:|---:|---:|:---:|:---:|:---:|"]
    for t in tried:
        sc = t["scenarios_all"]
        lines.append("| `{}` | {} | {} | {:+.3f} | {:+.3f} | {:+.3f} | {} | {} | {} |".format(
            t["feature_set"], t["C"], _fmt(t["cv_log_loss"]), sc["R1"], sc["R2"], sc["R3"],
            _pass(t["r1_pass"]), _pass(t["r2_pass"]),
            "yes" if t["feature_set"] == doc["feature_set"] else ""))
    if len(tried) > 1:
        names = [t["feature_set"] for t in tried]
        base = tried[0]["cv_log_loss_by_match"]
        lines += ["", "Held-out log-loss of each set by match (change against `{}`):".format(
            names[0]), "",
            "| Match | " + " | ".join("`{}`".format(n) for n in names) + " |",
            "|---|" + "---:|" * len(names)]
        for m in sorted(base):
            cells = [_fmt(base[m])]
            cells += ["{} ({:+.4f})".format(_fmt(t["cv_log_loss_by_match"][m]),
                                            t["cv_log_loss_by_match"][m] - base[m])
                      for t in tried[1:]]
            lines.append("| {} | {} |".format(m, " | ".join(cells)))
        for t in tried[1:]:
            better = sum(t["cv_log_loss_by_match"][m] < base[m] for m in base)
            lines += ["", "`{}` has a lower held-out log-loss than `{}` in {} of {} "
                      "matches.".format(t["feature_set"], names[0], better, len(base))]
    return lines


def report_markdown(results: Mapping[str, Mapping], fold_eval: Mapping[str, Mapping],
                    feature_auc: Mapping, dead_rates: Mapping | None = None,
                    short_dead_rates: Mapping | None = None,
                    cells: Mapping | None = None) -> str:
    lines = ["# Calibration of the dead-ball evidence (F3-04)", "",
             "Generated by `modeling.calibration.dead` from the degraded IDSSE masters (external "
             "data only). Log-loss and AUC are computed with each fold model on its held-out match "
             "(every usable frame). `d_fit` is the model logit minus `logit(pi_D)`.", "",
             "## Summary", "",
             "| File | Feature set | C | R1 (< 0) | R2 (> 0) | R2 `d_fit` (`all`) | "
             "Expected signs |",
             "|---|---|---:|:---:|:---:|---:|:---:|"]
    for name, res in results.items():
        doc = res["document"]
        ok = signs_ok(doc["all"], doc["features"], doc["gap_log_clip_frames"])
        lines.append("| `{}.yaml` | `{}` | {} | {} | {} | {:+.3f} | {} |".format(
            name, doc["feature_set"], doc["C"], _pass(doc["robustness"]["r1_pass"]),
            _pass(doc["robustness"]["r2_pass"]), doc["robustness"]["values"]["all"]["R2"],
            "ok" if ok else "**WRONG**"))
    lines.append("")
    if cells:
        lines += ["## The scenarios against the data (all matches)", "",
                  "Frames that look like each scenario: its gap kind, a gap of at most {} frames "
                  "and a mean player speed within ±{} m/s of the scenario's. `d (data)` is "
                  "`logit(P(D | cell)) − logit(pi_D)`, the value a faithful `d_fit` has "
                  "there; the interval is a {:.0f}% bootstrap over whole gaps. The last columns "
                  "are the "
                  "`d_fit` of each model `all`; compare each mode with the model fitted on it "
                  "(`dead` for `default`).".format(
                      SCENARIO_CELL_MAX_GAP_FRAMES, SCENARIO_CELL_SPEED_HALF_WIDTH_MPS,
                      SCENARIO_CELL_INTERVAL[1] - SCENARIO_CELL_INTERVAL[0]), "",
                  "| Mode | Scenario | Frames | Gaps | P(D) | d (data) | Interval | "
                  + " | ".join("`{}`".format(n) for n in results) + " |",
                  "|---|---|---:|---:|---:|---:|---|" + "---:|" * len(results)]
        for mode, per_scenario in cells.items():
            for s, c in per_scenario.items():
                row = "| {} | {} | {} | {} | {} | {:+.3f} | [{:+.3f}, {:+.3f}] | {} |"
                lines.append(row.format(
                    mode, s, c["n_frames"], c["n_gaps"], _fmt(c["p_dead"]), c["d"], c["low"],
                    c["high"], " | ".join(
                        "{:+.3f}".format(res["document"]["robustness"]["values"]["all"][s])
                        for res in results.values())))
        lines.append("")
    for name, res in results.items():
        doc = res["document"]
        lines += ["## `{}`".format(name), "",
                  "- training modes: {}; 1 frame in {}".format(", ".join(doc["train_modes"]),
                                                               doc["subsample"]),
                  "- feature set: `{}` ({})".format(doc["feature_set"], ", ".join(
                      "`{}`".format(f) for f in doc["features"])),
                  "- C = {} (leave-one-match-out log-loss: {})".format(doc["C"], ", ".join(
                      "{}: {}".format(c, _fmt(v)) for c, v in doc["cv_log_loss"].items())),
                  "- pi_D (model `all`): {}".format(_fmt(doc["all"]["pi_d"])), ""]
        lines += _choice_lines(doc) + [""]
        lines += ["Standardised weights of the model `all`:", "",
                  "| Feature | Weight |", "|---|---:|"]
        for f in doc["features"]:
            lines.append("| `{}` | {:+.4f} |".format(f, doc["all"]["weights"][f]))
        rows = sign_rows(doc["all"], doc["features"], doc["gap_log_clip_frames"])
        slopes = effective_slopes(doc["all"], doc["features"], rows)
        lines += ["", "Expected signs: slope of `d_fit` per unit of the feature (with "
                  "interactions it depends on the other features, so it is shown at the "
                  "training means and at each scenario):", "",
                  "| Feature | Expected | " + " | ".join(rows) + " | Check |",
                  "|---|:---:|" + "---:|" * len(rows) + ":---:|"]
        for f, per_row in slopes.items():
            good = all(np.sign(v) == EXPECTED_SIGNS[f] for v in per_row.values())
            lines.append("| `{}` | {} | {} | {} |".format(
                f, "+" if EXPECTED_SIGNS[f] > 0 else "−",
                " | ".join("{:+.3f}".format(per_row[r]) for r in rows),
                "ok" if good else "**WRONG SIGN**"))
        lines += ["", "Robustness scenarios (`d_fit`; R1 must be < 0, R2 > 0, R3 is reported):", "",
                  "| Model | R1 | R2 | R3 |", "|---|---:|---:|---:|"]
        for model_name, vals in doc["robustness"]["values"].items():
            lines.append("| {} | {:+.4f} | {:+.4f} | {:+.4f} |".format(
                model_name, vals["R1"], vals["R2"], vals["R3"]))
        evals = fold_eval.get(name)
        if evals:
            modes = list(evals)
            lines += ["", "Held-out log-loss / AUC by degradation mode:", "",
                      "| Match | " + " | ".join("{} log-loss | {} AUC".format(m, m) for m in modes)
                      + " |", "|---|" + "---:|---:|" * len(modes)]
            for match in sorted(next(iter(evals.values()))):
                row_cells = []
                for m in modes:
                    ll, a = evals[m][match]
                    row_cells += [_fmt(ll), _fmt(a)]
                lines.append("| {} | {} |".format(match, " | ".join(row_cells)))
            means = {m: float(np.mean([v[0] for v in evals[m].values()])) for m in modes}
            lines.append("| **mean** | {} |".format(" | ".join(
                "{} | {}".format(_fmt(means[m]), _fmt(float(np.mean(
                    [v[1] for v in evals[m].values()])))) for m in modes)))
            base = means.get("default")
            if base is not None:
                changes = ["`{}` {:+.1%}".format(m, means[m] / base - 1.0) for m in modes
                           if m != "default"]
                lines += ["", "Mean log-loss relative to `default`: {} (item 5 flags more than "
                          "{:+.0%}).".format(", ".join(changes), WORSE_LOG_LOSS)]
                for m in modes:
                    if m != "default" and means[m] > base * (1.0 + WORSE_LOG_LOSS):
                        lines.append("")
                        lines.append("**Note:** mean log-loss on `{}` is more than {:.0%} worse "
                                     "than on `default`.".format(m, WORSE_LOG_LOSS))
        lines.append("")
    if dead_rates:
        lines += ["## Empirical P(D) by gap kind and mean player speed (`default`, all matches)", "",
                  "Share of dead-ball frames, with the frame count in brackets. Every gap length "
                  "is pooled, so the gap rows are dominated by gaps of hundreds of frames.", ""]
        lines += _rates_table(dead_rates) + [""]
    if short_dead_rates:
        lines += ["### Gaps of at most {} frames".format(SCENARIO_CELL_MAX_GAP_FRAMES), "",
                  "Same table, keeping only the gaps as short as the scenarios' (R1 and R2 have "
                  "50 frames); `no long gap` is unchanged. A long exit gap with running players "
                  "is mostly live play after the restart, still inside the gap; a short one is "
                  "mostly the dead ball that has just started. That is why the gap length and "
                  "the player speed change the effect of an exit, and why the `interactions` "
                  "set exists.", ""]
        lines += _rates_table(short_dead_rates) + [""]
    auc_features = list(next(iter(feature_auc.values()))) if feature_auc else []
    lines += ["## AUC of each feature alone (D vs alive, `default`, held-out match)", "",
              "A very high AUC on a ball-related feature would point to label leakage. An AUC "
              "below 0.5 means the feature points to live play; its strength is the distance "
              "from 0.5. Columns that are constant on IDSSE (no missing values) show 0.5.", "",
              "| Match | " + " | ".join("`{}`".format(f) for f in auc_features) + " |",
              "|---|" + "---:|" * len(auc_features)]
    for m, row in feature_auc.items():
        lines.append("| {} | {} |".format(m, " | ".join(_fmt(row[f]) for f in auc_features)))
    lines.append("")
    return "\n".join(lines)


# -- command line --------------------------------------------------------------------------------
def load_mode(matches: Sequence[str], mode: str, params: dict) -> dict:
    """``{match: (X, y)}`` for one degradation mode, with the columns of ``ALL_FEATURES``."""
    from modeling.external.files import load_external

    out = {}
    for m in matches:
        master, labels = load_external(m, mode)
        out[m] = match_design(master, labels, params)
    return out


def run(params: dict, matches: Sequence[str], out_dir: Path, report_path: Path) -> dict:
    """Every fit of the task, the four calibration files and the report."""
    by_mode = {mode: load_mode(matches, mode, params) for mode in POOLED_MODES}
    variants = {
        "dead": (("default",), SUBSAMPLE, "dead_logit calibrated on the default masters"),
        "dead_pooled": (POOLED_MODES, POOLED_SUBSAMPLE,
                        "dead_logit calibrated on default, hidelo and hidehi together"),
        "dead_hidelo": (("hidelo",), SUBSAMPLE,
                        "dead_logit calibrated on hidelo only (F2-06 stability check only)"),
        "dead_hidehi": (("hidehi",), SUBSAMPLE,
                        "dead_logit calibrated on hidehi only (F2-06 stability check only)"),
    }
    results, fold_eval, written = {}, {}, {}
    for name, (modes, step, description) in variants.items():
        data = {m: [by_mode[mode][m] for mode in modes] for m in matches}
        result = calibrate_with_fallback(data, step)
        doc = calibration_document(result, params, train_modes=modes, step=step,
                                   description=description)
        written[name] = write_calibration(doc, out_dir / "{}.yaml".format(name))
        results[name] = {"document": doc}
        if name in ("dead", "dead_pooled"):
            fold_eval[name] = evaluate_folds(result, by_mode)
    feature_auc = single_feature_auc(by_mode["default"])
    dead_rates = dead_rate_by_speed(by_mode["default"])
    short_rates = dead_rate_by_speed(by_mode["default"], SCENARIO_CELL_MAX_GAP_FRAMES)
    # The labels do not depend on the mode, so pi_D is the same in every mode.
    pi_d = results["dead"]["document"]["all"]["pi_d"]
    cells = {mode: scenario_cells(by_mode[mode], pi_d) for mode in POOLED_MODES}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_markdown(results, fold_eval, feature_auc, dead_rates,
                                           short_rates, cells), encoding="utf-8")
    written["report"] = report_path
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/v1.yaml",
                        help="config whose dead.params build the features (default: %(default)s)")
    parser.add_argument("--matches", default="all",
                        help="all or comma-separated IDSSE match ids (default: all)")
    parser.add_argument("--out-dir", default="configs/calib",
                        help="directory of the calibration files (default: %(default)s)")
    parser.add_argument("--report", default="reports/calibration_dead.md",
                        help="report path (default: %(default)s)")
    return parser


def main(argv=None) -> int:
    from modeling.config import load_config
    from modeling.external.files import IDSSE_MATCHES

    args = build_parser().parse_args(argv)
    root = paths.repo_root()

    def resolve(p):
        p = Path(p)
        return p if p.is_absolute() else root / p

    cfg = load_config(resolve(args.config))
    if "dead" not in cfg:
        raise SystemExit("config {} has no dead section".format(args.config))
    matches = (list(IDSSE_MATCHES) if args.matches == "all"
               else [m.strip() for m in args.matches.split(",") if m.strip()])
    written = run(cfg["dead"]["params"], matches, resolve(args.out_dir), resolve(args.report))
    for name, path in written.items():
        print("{}: {}".format(name, path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
