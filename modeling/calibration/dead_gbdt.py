"""GBDT ``dead_logit`` calibration on degraded external IDSSE matches (F3-08).

The F1-07 model (:data:`MODEL_PARAMS`, unchanged) fitted on the F3-04 data: frames inside the halves
with a label other than X, 1 every :data:`SUBSAMPLE`, target D vs alive. Features of
:mod:`modeling.deadball.dead_gbdt` (frame features plus windows of ``W`` in
:data:`modeling.deadball.dead_gbdt.DEFAULT_WINDOW_SIZES`), NaN kept (the model routes it natively).

The likelihood ratio is ``d_fit = g(x) - logit(pi_D)``, with ``pi_D`` the share of D in the
training rows of that model. The registered ``gbdt`` dead kind adds the fixed Smart Tagging prior and
clips to +-12 (:func:`modeling.deadball.dead.dead_logit_gbdt`).

One model per leave-one-match-out fold, saved under ``ext_dir()/<fold_dir>/<match>.joblib`` (not
committed), and one with every match, ``models/dead_gbdt/all.joblib`` (committed, the only one used
for a CSV). The calibration file records the scikit-learn version; inference refuses any other.

Candidates (update of 11/11, item 3):

- ``dead_gbdt``: ``default`` masters, 1 frame in 5; reference F3-04 file ``configs/calib/dead.yaml``;
- ``dead_gbdt_pooled`` (optional): ``default``, ``hidelo`` and ``hidehi`` together, 1 frame in 15 of
  each, the held-out match left out of the three; reference ``configs/calib/dead_pooled.yaml``.

Every fold model is also evaluated on its held-out match in ``hidelo`` and ``hidehi`` (log-loss and
AUC side by side with the F3-04 fold model on the same rows), and on the robustness scenarios R1-R3
of F3-04 with coherent window values (:data:`SCENARIOS`). Nothing is chosen here: the adoption is
decided by F2-06. Never calibrates on ``degrade="none"``. Nothing here reads BAR-ATH.

Usage (external data only)::

    python -m modeling.calibration.dead_gbdt --config configs/v2.yaml
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from evaluation import possession_metrics as pm
from modeling import paths
from modeling.calibration import dead as cd
from modeling.calibration.team_gbdt import MODEL_PARAMS, _memoized_evidence
from modeling.contracts import LABEL_CODES
from modeling.deadball.dead_gbdt import (
    DEFAULT_WINDOW_SIZES,
    FORMAT_VERSION,
    VARIANT,
    build_features,
    d_fit_values,
    feature_names,
    prior_logit,
    feature_params,
    model_logit,
    resolve_repo_path,
    save_bundle,
)
from modeling.deadball.features import GAP_EXIT, GAP_LONG

SUBSAMPLE = cd.SUBSAMPLE                       # 5, as F3-04
POOLED_SUBSAMPLE = cd.POOLED_SUBSAMPLE         # 15 of each mode, as F3-04
POOLED_MODES = cd.POOLED_MODES                 # default, hidelo, hidehi
EVAL_MODES = cd.POOLED_MODES                   # cross-evaluation (item 3)
DEFAULT_LAM_GRID = (0.1, 0.2, 0.3, 0.5)        # report only: lam_dead of the decoder table

# name -> (train modes, subsample, F3-04 reference file, description)
CANDIDATES = {
    "dead_gbdt": (("default",), SUBSAMPLE, "configs/calib/dead.yaml",
                  "GBDT dead_logit fitted on the default masters"),
    "dead_gbdt_pooled": (POOLED_MODES, POOLED_SUBSAMPLE, "configs/calib/dead_pooled.yaml",
                         "GBDT dead_logit fitted on default, hidelo and hidehi together"),
}
# F2-06 stability check only (update of 11/11, item 5; ``tools/dead_stability.py``), as the
# ``dead_hidelo``/``dead_hidehi`` files of F3-04. Never part of the default run; never used for a CSV.
STABILITY_CANDIDATES = {
    "dead_gbdt_hidelo": (("hidelo",), SUBSAMPLE, "configs/calib/dead_hidelo.yaml",
                         "GBDT dead_logit fitted on hidelo only (F2-06 stability check only)"),
    "dead_gbdt_hidehi": (("hidehi",), SUBSAMPLE, "configs/calib/dead_hidehi.yaml",
                         "GBDT dead_logit fitted on hidehi only (F2-06 stability check only)"),
}
ALL_CANDIDATES = {**CANDIDATES, **STABILITY_CANDIDATES}

# Robustness scenarios of F3-04 (item 4) written on the GBDT features, with window values coherent
# with each story. The values are inputs of the check, not match data. The frame values are those
# of F3-04; the players move at the same pace in the past and in the future windows.
#   R1   live play, 50-frame long gap in midfield (the ball was seen around it, far from the
#        lines) => d_fit < 0 on every model;
#   R2   R1, but the gap is an exit: the ball was seen leaving the pitch over the touchline
#        (centred min of d_touchline below 0, out_of_pitch window max 1) => d_fit > 0 on every model;
#   R2b  R1 with an exit gap but the ball last seen just inside the touchline and never outside
#        (exit rule (a) of F3-01 without the ball seen out); reported only;
#   R3   R1 with a slow possession (more still players, lower speed); reported only.
# Inside a gap the ball is not seen, so the frame values of out_of_pitch and of the line distances
# are NaN; the centred windows (W >= 25 around a frame of a 50-frame gap) reach the frames where the
# ball was seen.
SCENARIO_GAP_FRAMES = 50.0
SCENARIO_LIVE_PLAYERS = {"still": 0.2, "speed": 2.5}          # F3-04 R1
SCENARIO_SLOW_PLAYERS = {"still": 0.45, "speed": 1.8}         # F3-04 R3
SCENARIO_NEAR_LINE_M = 3.0                                    # exit_line_m of the configs
SCENARIO_BALL = {
    "midfield": {"d_touchline": 15.0, "d_goalline": 30.0, "out_of_pitch": 0.0},
    "seen_out": {"d_touchline": -1.0, "d_goalline": 30.0, "out_of_pitch": 1.0},
    "near_inside": {"d_touchline": 1.0, "d_goalline": 30.0, "out_of_pitch": 0.0},
}
SCENARIOS = {
    "R1": {"db_gap_kind": GAP_LONG, "players": SCENARIO_LIVE_PLAYERS, "ball": "midfield"},
    "R2": {"db_gap_kind": GAP_EXIT, "players": SCENARIO_LIVE_PLAYERS, "ball": "seen_out"},
    "R2b": {"db_gap_kind": GAP_EXIT, "players": SCENARIO_LIVE_PLAYERS, "ball": "near_inside"},
    "R3": {"db_gap_kind": GAP_LONG, "players": SCENARIO_SLOW_PLAYERS, "ball": "midfield"},
}
REPORTED_ONLY = ("R2b", "R3")

# Report only: the empirical cell of a scenario holds the usable frames of its gap kind with a gap of
# at most cd.SCENARIO_CELL_MAX_GAP_FRAMES frames, a mean player speed within
# cd.SCENARIO_CELL_SPEED_HALF_WIDTH_MPS of the scenario's, a change of mean speed (future minus past,
# middle window size) of at most SCENARIO_CELL_SPEED_CHANGE_MPS, and the ball condition of the
# scenario on the centred windows of that size (midfield: d_touchline min >= SCENARIO_NEAR_LINE_M;
# seen_out: out_of_pitch max == 1; near_inside: out_of_pitch max == 0 and 0 <= d_touchline min <
# SCENARIO_NEAR_LINE_M). The still fraction is not a condition (F3-04 did not use it either).
SCENARIO_CELL_SPEED_CHANGE_MPS = 0.5


# -- scenarios -----------------------------------------------------------------------------------
def scenario_rows(window_sizes: Sequence[int] = DEFAULT_WINDOW_SIZES) -> pd.DataFrame:
    """One feature row per scenario of :data:`SCENARIOS` (index = scenario name)."""
    rows = {}
    for name, spec in SCENARIOS.items():
        still, speed = spec["players"]["still"], spec["players"]["speed"]
        ball = SCENARIO_BALL[spec["ball"]]
        row = {"out_of_pitch": np.nan, "db_gap_kind": float(spec["db_gap_kind"]),
               "db_still_frac": still, "db_mean_speed": speed, "db_fast_in_pitch": 0.0,
               "gap_length": SCENARIO_GAP_FRAMES, "d_touchline": np.nan, "d_goalline": np.nan}
        for w in window_sizes:
            for short, value in (("still", still), ("speed", speed)):
                row["{}_past_mean_{}".format(short, w)] = value
                row["{}_fut_mean_{}".format(short, w)] = value
                row["{}_fut_minus_past_{}".format(short, w)] = 0.0
            for line in ("d_touchline", "d_goalline"):
                row["{}_cent_min_{}".format(line, w)] = ball[line]
            row["out_of_pitch_cent_max_{}".format(w)] = ball["out_of_pitch"]
        rows[name] = row
    return pd.DataFrame.from_dict(rows, orient="index")[feature_names(window_sizes)].astype(
        np.float64)


def cell_window(window_sizes: Sequence[int]) -> int:
    """Window size of the empirical cells: the middle one of ``window_sizes``."""
    sizes = list(window_sizes)
    return int(sizes[len(sizes) // 2])


def scenario_cell_mask(X: np.ndarray, scenario: str, window_sizes: Sequence[int]) -> np.ndarray:
    """Rows of a feature matrix (columns of ``feature_names(window_sizes)``) in a scenario's cell."""
    names = feature_names(window_sizes)
    w = cell_window(window_sizes)

    def col(name):
        return X[:, names.index(name)]

    spec = SCENARIOS[scenario]
    with np.errstate(invalid="ignore"):
        sel = ((col("db_gap_kind") == spec["db_gap_kind"])
               & (col("gap_length") <= cd.SCENARIO_CELL_MAX_GAP_FRAMES)
               & (np.abs(col("db_mean_speed") - spec["players"]["speed"])
                  <= cd.SCENARIO_CELL_SPEED_HALF_WIDTH_MPS)
               & (np.abs(col("speed_fut_minus_past_{}".format(w)))
                  <= SCENARIO_CELL_SPEED_CHANGE_MPS))
        touch = col("d_touchline_cent_min_{}".format(w))
        out = col("out_of_pitch_cent_max_{}".format(w))
        if spec["ball"] == "midfield":
            sel &= touch >= SCENARIO_NEAR_LINE_M
        elif spec["ball"] == "seen_out":
            sel &= out == 1
        else:
            sel &= (out == 0) & (touch >= 0) & (touch < SCENARIO_NEAR_LINE_M)
    return sel


def gap_runs(X: np.ndarray, window_sizes: Sequence[int]) -> np.ndarray:
    """Id of the long gap of each row (consecutive rows with the same ``gap_length`` > 0), -1 off it.

    ``X`` holds the usable frames of one match in frame order (as :func:`match_data`).
    """
    gl = X[:, feature_names(window_sizes).index("gap_length")]
    in_gap = np.isfinite(gl) & (gl > 0)
    start = in_gap & ~np.r_[False, in_gap[:-1] & (gl[1:] == gl[:-1])]
    return np.where(in_gap, np.cumsum(start) - 1, -1)


def scenario_cells(data: Mapping[str, dict], d_fit: Mapping[str, Mapping[str, np.ndarray]],
                   pi_d: float, window_sizes: Sequence[int]) -> dict:
    """Empirical ``d`` of the frames that look like each scenario, next to the models' ``d_fit``.

    ``data[m]`` is the :func:`match_data` of match ``m``; ``d_fit[name][m]`` is the out-of-fold
    ``d_fit`` of model ``name`` on its rows. ``d = logit(P(D | cell)) - logit(pi_d)``, with a
    bootstrap interval over whole gaps (the F3-04 settings). Report only.
    """
    prior = float(np.log(pi_d) - np.log1p(-pi_d))
    out = {}
    for s in SCENARIOS:
        sums, sizes, model_sum = [], [], {name: 0.0 for name in d_fit}
        for m, d in sorted(data.items()):
            sel = scenario_cell_mask(d["X"], s, window_sizes)
            runs = gap_runs(d["X"], window_sizes)[sel]
            ys = d["y"][sel]
            for r in np.unique(runs):
                in_run = runs == r
                sums.append(float(ys[in_run].sum()))
                sizes.append(float(in_run.sum()))
            for name in d_fit:
                model_sum[name] += float(d_fit[name][m][sel].sum())
        sums, sizes = np.asarray(sums), np.asarray(sizes)
        n = float(sizes.sum())
        if n == 0:
            out[s] = {"n_frames": 0, "n_gaps": 0, "p_dead": float("nan"), "d": float("nan"),
                      "low": float("nan"), "high": float("nan"),
                      "models": {name: float("nan") for name in d_fit}}
            continue

        def d_of(k, total):
            p = min(max(k / total, 1.0 / (total + 1.0)), total / (total + 1.0))
            return float(np.log(p) - np.log1p(-p) - prior)

        rng = np.random.default_rng(cd.SCENARIO_CELL_BOOTSTRAP_SEED)
        draws = []
        for _ in range(cd.SCENARIO_CELL_BOOTSTRAP_DRAWS):
            pick = rng.integers(0, len(sizes), len(sizes))
            draws.append(d_of(sums[pick].sum(), sizes[pick].sum()))
        low, high = np.percentile(draws, cd.SCENARIO_CELL_INTERVAL)
        out[s] = {"n_frames": int(n), "n_gaps": int(len(sizes)), "p_dead": float(sums.sum() / n),
                  "d": d_of(sums.sum(), n), "low": float(low), "high": float(high),
                  "models": {name: model_sum[name] / n for name in d_fit}}
    return out


def scenario_values(model, pi_d: float, window_sizes: Sequence[int]) -> dict:
    """``d_fit`` of each scenario."""
    X = scenario_rows(window_sizes)
    values = d_fit_values(model, X.to_numpy(), pi_d)
    return {s: float(v) for s, v in zip(X.index, values)}


# -- data ----------------------------------------------------------------------------------------
def match_data(master: pd.DataFrame, labels: np.ndarray, params: Mapping,
               window_sizes: Sequence[int] = DEFAULT_WINDOW_SIZES,
               references: Mapping[str, Mapping] | None = None,
               holdout: str | None = None) -> dict:
    """Usable frames of one match, in frame order (not subsampled).

    ``{"X": float64 (n, k), "y": bool (n,), "ref": {name: float64 (n,)}}``. ``ref[name]`` is the
    logit (prior included) of the F3-04 model of ``references[name]`` that left ``holdout`` out
    (model ``all`` for ``None``), on the same rows.
    """
    rows = cd.usable_rows(master, labels)
    X = build_features(master, params, window_sizes).to_numpy(dtype=np.float64)[rows]
    y = np.asarray(labels)[rows] == LABEL_CODES["D"]
    ref = {}
    for name, doc in (references or {}).items():
        model = cd.select_model(doc, holdout)
        logit = cd.calibrated_d_fit(master, params, doc, holdout) + cd.prior_logit(model)
        ref[name] = logit[rows]
    return {"X": X, "y": y, "ref": ref}


def training_rows(data: Mapping[str, Mapping[str, dict]], matches: Sequence[str],
                  modes: Sequence[str], step: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows of ``matches`` in every mode of ``modes``, 1 in ``step`` of each (match, mode)."""
    Xs, ys = [], []
    for m in matches:
        for mode in modes:
            d = data[mode][m]
            Xs.append(d["X"][::step])
            ys.append(d["y"][::step])
    return np.vstack(Xs), np.concatenate(ys)


# -- model ---------------------------------------------------------------------------------------
def fit_model(X: np.ndarray, y: np.ndarray, model_params: Mapping = MODEL_PARAMS):
    """Fit the classifier (single-threaded, deterministic); returns ``(model, pi_D)``."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=bool)
    if X.ndim != 2 or X.shape[0] != len(y):
        raise ValueError("X must have one row per label, got {} and {}".format(X.shape, len(y)))
    pi = float(y.mean()) if len(y) else float("nan")
    if not 0.0 < pi < 1.0:
        raise ValueError("training rows need both classes, got P(D) = {}".format(pi))
    clf = HistGradientBoostingClassifier(**dict(model_params))
    with threadpool_limits(limits=1):
        clf.fit(X, y.astype(np.int8))
    return clf, pi


def calibrate(data: Mapping[str, Mapping[str, dict]], *, matches: Sequence[str],
              train_modes: Sequence[str], step: int, window_sizes: Sequence[int],
              fold_root: Path, all_path: Path, reference: str | None = None,
              eval_modes: Sequence[str] | None = None, model_params: Mapping = MODEL_PARAMS,
              log: Callable = print) -> dict:
    """Fit and save every fold model and ``all``; evaluate the folds and the scenarios.

    ``data[mode][match]`` is the :func:`match_data` of that match. Fold metrics are computed on
    every usable frame of the held-out match, in every mode of ``eval_modes`` present in ``data``;
    ``reference`` names the ``ref`` logit shown next to the GBDT one. ``oof_d_fit[mode][match]`` is
    the ``d_fit`` of the fold model on its held-out match (not written to the file).
    """
    matches = list(matches)
    if len(matches) < 2:
        raise ValueError("need at least two matches to calibrate, got {}".format(matches))
    eval_modes = [m for m in (eval_modes or EVAL_MODES) if m in data]
    features = feature_names(window_sizes)
    pi_folds, n_folds, metrics, scen = {}, {}, {mode: {} for mode in eval_modes}, {}
    oof = {mode: {} for mode in eval_modes}
    for holdout in matches:
        t0 = time.time()
        X, y = training_rows(data, [m for m in matches if m != holdout], train_modes, step)
        model, pi = fit_model(X, y, model_params)
        save_bundle(fold_root / "{}.joblib".format(holdout), model, features, pi)
        pi_folds[holdout], n_folds[holdout] = pi, int(len(y))
        for mode in eval_modes:
            d = data[mode][holdout]
            z = model_logit(model, d["X"])
            oof[mode][holdout] = z - prior_logit(pi)
            row = {"log_loss": cd.log_loss(d["y"], z), "auc": cd.auc(d["y"], z)}
            if reference is not None:
                zr = d["ref"][reference]
                row.update({"log_loss_ref": cd.log_loss(d["y"], zr), "auc_ref": cd.auc(d["y"], zr)})
            metrics[mode][holdout] = {k: float(v) for k, v in row.items()}
        scen["fold " + holdout] = scenario_values(model, pi, window_sizes)
        log("fold {} trained in {:.0f}s".format(holdout, time.time() - t0))
    t0 = time.time()
    X, y = training_rows(data, matches, train_modes, step)
    model, pi_all = fit_model(X, y, model_params)
    save_bundle(all_path, model, features, pi_all)
    log("all trained in {:.0f}s".format(time.time() - t0))
    values = {"all": scenario_values(model, pi_all, window_sizes), **scen}
    return {
        "pi_d": {"all": float(pi_all), "folds": dict(sorted(pi_folds.items()))},
        "n_train": {"all": int(len(y)), "folds": dict(sorted(n_folds.items()))},
        "fold_metrics": metrics,
        "robustness": {"r1_pass": all(v["R1"] < 0 for v in values.values()),
                       "r2_pass": all(v["R2"] > 0 for v in values.values()),
                       "values": values},
        "oof_d_fit": oof,
    }


def calibration_document(result: Mapping, params: Mapping, *, description: str,
                         train_modes: Sequence[str], step: int, window_sizes: Sequence[int],
                         model_file: str, fold_dir: str, reference_file: str | None,
                         reference_doc: Mapping | None = None, cells: Mapping | None = None,
                         model_params: Mapping = MODEL_PARAMS) -> dict:
    """The calibration file (plain Python types only)."""
    import sklearn

    model_path = resolve_repo_path(model_file)
    robustness = {"r1_pass": bool(result["robustness"]["r1_pass"]),
                  "r2_pass": bool(result["robustness"]["r2_pass"]),
                  "values": result["robustness"]["values"]}
    if reference_doc is not None:
        robustness["reference_values"] = reference_doc["robustness"]["values"]
    return {
        "format_version": FORMAT_VERSION,
        "variant": VARIANT,
        "description": description,
        "sklearn_version": sklearn.__version__,
        "model_file": str(model_file),
        "fold_dir": str(fold_dir),
        "model_params": dict(model_params),
        "train_modes": list(train_modes),
        "subsample": int(step),
        "window_sizes": [int(w) for w in window_sizes],
        "feature_params": copy.deepcopy(feature_params(params)),
        "features": feature_names(window_sizes),
        "pi_d": result["pi_d"],
        "reference_file": reference_file,
        "robustness": robustness,
        "meta": {
            "n_train": result["n_train"],
            "fold_metrics": result["fold_metrics"],
            "all_model_bytes": int(model_path.stat().st_size) if model_path.is_file() else None,
            "scenario_cells": dict(cells) if cells is not None else None,
        },
    }


def write_calibration(document: Mapping, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ("# Generated by modeling.calibration.dead_gbdt (F3-08) from the degraded IDSSE "
              "masters.\n# External data only; the provided match is never read. Do not edit by "
              "hand.\n")
    path.write_text(header + yaml.safe_dump(dict(document), sort_keys=False,
                                            default_flow_style=False),
                    encoding="utf-8", newline="\n")
    return path


# -- report helpers ------------------------------------------------------------------------------
def feature_diagnostics(data: Mapping[str, dict], features: Sequence[str]) -> dict:
    """Per feature: mean, min and max over matches of the AUC alone (D vs alive, finite rows) and
    the NaN share among D and among alive frames (all matches pooled). Report only."""
    out = {}
    y_all = np.concatenate([d["y"] for d in data.values()])
    X_all = np.vstack([d["X"] for d in data.values()])
    for i, f in enumerate(features):
        aucs = []
        for d in data.values():
            x = d["X"][:, i]
            ok = np.isfinite(x)
            aucs.append(cd.auc(d["y"][ok], x[ok]))
        aucs = np.asarray(aucs, dtype=np.float64)
        nan = ~np.isfinite(X_all[:, i])
        out[f] = {"auc_mean": float(np.nanmean(aucs)) if np.isfinite(aucs).any() else np.nan,
                  "auc_min": float(np.nanmin(aucs)) if np.isfinite(aucs).any() else np.nan,
                  "auc_max": float(np.nanmax(aucs)) if np.isfinite(aucs).any() else np.nan,
                  "nan_dead": float(nan[y_all].mean()) if y_all.any() else np.nan,
                  "nan_alive": float(nan[~y_all].mean()) if (~y_all).any() else np.nan}
    return out


def evaluate_decoder(cfgs: Mapping[str, dict], lam_grid: Sequence[float], *, degrade: str,
                     matches: Sequence[str], load_fn) -> pd.DataFrame:
    """spellF1 and spell count ratio per (config, lam_dead, match), leave-one-match-out."""
    from modeling.pipeline import predict_labels
    from modeling.scoring.metrics import score

    rows = []
    with _memoized_evidence():
        for match in matches:
            master, gt = load_fn(match, degrade)
            for name, cfg in cfgs.items():
                for lam in lam_grid:
                    c = copy.deepcopy(cfg)
                    c["decoder"]["lam_dead"] = float(lam)
                    m = score(gt, pm.label_codes(predict_labels(master, c, holdout=match)))
                    rows.append({"config": name, "lam_dead": float(lam), "match": match,
                                 "spell_f1_mean": float(m["spell_f1_mean"]),
                                 "count_ratio": float(m["n_pred_spells"]) / float(m["n_gt_spells"])})
            del master, gt
    return pd.DataFrame(rows)


def summarize_decoder(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, standard error and mean count ratio per (config, lam_dead)."""
    g = df.groupby(["config", "lam_dead"], sort=False)
    out = g["spell_f1_mean"].agg(["mean", "std", "count"])
    out["se"] = out["std"] / np.sqrt(out["count"])
    out["count_ratio"] = g["count_ratio"].mean()
    return out[["mean", "se", "count_ratio"]]


def paired_decoder(df: pd.DataFrame, own: Mapping[str, str]) -> pd.DataFrame:
    """Per (config, lam_dead): spellF1 minus that of its base config ``own[config]`` at the same
    ``lam_dead``, match by match. Columns: mean difference, its standard error, matches won."""
    rows = []
    key = ["match", "lam_dead"]
    for name, base in own.items():
        a = df[df["config"] == name].set_index(key)["spell_f1_mean"]
        b = df[df["config"] == base].set_index(key)["spell_f1_mean"]
        diff = (a - b.reindex(a.index)).dropna()
        for lam, d in diff.groupby(level="lam_dead", sort=False):
            rows.append({"config": name, "base": base, "lam_dead": float(lam),
                         "diff": float(d.mean()),
                         "se": float(d.std() / np.sqrt(len(d))) if len(d) > 1 else float("nan"),
                         "wins": int((d > 0).sum()), "n": int(len(d))})
    return pd.DataFrame(rows)


def _fmt(x, spec="{:.4f}") -> str:
    return "n/a" if x is None or not np.isfinite(x) else spec.format(x)


def _pass(ok: bool) -> str:
    return "PASS" if ok else "**FAIL**"


def report_markdown(docs: Mapping[str, Mapping], diagnostics: Mapping | None = None,
                    decoder: Mapping[str, pd.DataFrame] | None = None,
                    decoder_note: str = "", decoder_own: Mapping[str, str] | None = None) -> str:
    lines = ["# Calibration of the dead-ball evidence, GBDT variant (F3-08)", "",
             "Generated by `modeling.calibration.dead_gbdt` from the degraded IDSSE masters "
             "(external data only; the provided match is never read). Every number below comes "
             "from a model that did not see the match it is computed on (leave one match out), "
             "except the scenarios of the model `all`. The F3-04 logistic models of the "
             "reference file are evaluated on exactly the same rows. Nothing is chosen here: "
             "the adoption is decided by F2-06.", ""]
    first = next(iter(docs.values()))
    lines += ["- model: `HistGradientBoostingClassifier` ({}), scikit-learn `{}`".format(
                  ", ".join("`{}={}`".format(k, v) for k, v in first["model_params"].items()),
                  first["sklearn_version"]),
              "- features ({}): {}".format(len(first["features"]), ", ".join(
                  "`{}`".format(f) for f in first["features"])),
              "- no imputation, no indicator: the model routes NaN natively; `db_ball_rest`, "
              "`ball_speed` and the windowed share of `ball_observed` are never features",
              "- `d_fit = decision_function(x) − logit(pi_D)`, `pi_D` of the model's own training "
              "rows; `d = clip(d_fit + dead_tag_prior, ±12)`", ""]

    lines += ["## Summary", "",
              "| File | Training modes | 1 in | `pi_D` (`all`) | Model size | R1 (< 0) | R2 (> 0) "
              "| Reference |",
              "|---|---|---:|---:|---:|:---:|:---:|---|"]
    for name, doc in docs.items():
        size = doc["meta"]["all_model_bytes"]
        lines.append("| `{}.yaml` | {} | {} | {} | {} | {} | {} | `{}` |".format(
            name, ", ".join(doc["train_modes"]), doc["subsample"], _fmt(doc["pi_d"]["all"]),
            "n/a" if size is None else "{:.2f} MB".format(size / 2**20),
            _pass(doc["robustness"]["r1_pass"]), _pass(doc["robustness"]["r2_pass"]),
            doc["reference_file"]))
    lines.append("")

    for name, doc in docs.items():
        fm = doc["meta"]["fold_metrics"]
        lines += ["## `{}`".format(name), "",
                  "{}. Reference (F3-04): `{}`.".format(doc["description"], doc["reference_file"]),
                  "", "### Held-out log-loss and AUC, GBDT vs F3-04 (same rows)", "",
                  "Each fold model on every usable frame of its held-out match, in each "
                  "degradation mode (item 3: cross-evaluation in `hidelo` and `hidehi`). Lower "
                  "log-loss and higher AUC are better.", ""]
        for mode, per_match in fm.items():
            lines += ["#### `{}`".format(mode), "",
                      "| Match | GBDT log-loss | F3-04 log-loss | Change | GBDT AUC | F3-04 AUC |",
                      "|---|---:|---:|---:|---:|---:|"]
            for match, r in per_match.items():
                lines.append("| {} | {} | {} | {:+.1%} | {} | {} |".format(
                    match, _fmt(r["log_loss"]), _fmt(r["log_loss_ref"]),
                    r["log_loss"] / r["log_loss_ref"] - 1.0, _fmt(r["auc"]), _fmt(r["auc_ref"])))
            mean = {k: float(np.mean([r[k] for r in per_match.values()]))
                    for k in ("log_loss", "log_loss_ref", "auc", "auc_ref")}
            better = sum(r["log_loss"] < r["log_loss_ref"] for r in per_match.values())
            lines += ["| **mean** | **{}** | **{}** | {:+.1%} | **{}** | **{}** |".format(
                _fmt(mean["log_loss"]), _fmt(mean["log_loss_ref"]),
                mean["log_loss"] / mean["log_loss_ref"] - 1.0, _fmt(mean["auc"]),
                _fmt(mean["auc_ref"])), "",
                "GBDT log-loss lower than F3-04 in {} of {} matches.".format(better, len(per_match)),
                ""]
        if "default" in fm:
            base = np.mean([r["log_loss"] for r in fm["default"].values()])
            changes = ["`{}` {:+.1%}".format(mode, np.mean([r["log_loss"] for r in
                                                             fm[mode].values()]) / base - 1.0)
                       for mode in fm if mode != "default"]
            if changes:
                lines += ["Mean GBDT log-loss relative to `default`: {} (F3-04 item 5 flags more "
                          "than {:+.0%}).".format(", ".join(changes), cd.WORSE_LOG_LOSS), ""]
        rob = doc["robustness"]
        names = list(next(iter(rob["values"].values())))
        ref_vals = rob.get("reference_values", {})
        lines += ["### Robustness scenarios (`d_fit`)", "",
                  "Feature rows written from the F3-04 stories, with coherent windows (see "
                  "`SCENARIOS` in `modeling/calibration/dead_gbdt.py`): a 50-frame gap, players at "
                  "the same pace before and after. R1: long gap, the ball seen around it in "
                  "midfield; must be < 0. R2: exit gap, the ball seen leaving over the touchline; "
                  "must be > 0. R2b: exit gap, the ball last seen {:g} m inside the touchline and "
                  "never outside (reported). R3: R1 with a slow possession (reported). The F3-04 "
                  "columns are the reference file's values of its own R1-R3 (no windows; its R2 "
                  "has no ball position).".format(SCENARIO_BALL["near_inside"]["d_touchline"]), "",
                  "| Model | " + " | ".join(names) + " | F3-04 R1 | F3-04 R2 | F3-04 R3 |",
                  "|---|" + "---:|" * (len(names) + 3)]
        for model_name, v in rob["values"].items():
            r = ref_vals.get(model_name, {})
            lines.append("| {} | {} | {} |".format(
                model_name, " | ".join("{:+.3f}".format(v[n]) for n in names),
                " | ".join(_fmt(r.get(n), "{:+.3f}") for n in ("R1", "R2", "R3"))))
        lines += ["", "R1 on every model: {}; R2 on every model: {}.".format(
            _pass(rob["r1_pass"]), _pass(rob["r2_pass"])), ""]
        cells = doc["meta"].get("scenario_cells")
        if cells:
            first_cell = next(iter(cells.values()))
            models = list(first_cell["models"])
            lines += ["### The scenarios against the data (`default`, all matches)", "",
                      "Usable frames that look like each scenario: its gap kind, a gap of at most "
                      "{} frames, a mean player speed within ±{} m/s of the scenario's, a change of "
                      "mean speed (future − past, W = {}) of at most {} m/s and the scenario's ball "
                      "condition on the centred windows of the same size. `d (data)` is "
                      "`logit(P(D | cell)) − logit(pi_D)`, with a {:.0f}% bootstrap interval over "
                      "whole gaps; the model columns are the mean out-of-fold `d_fit` on those "
                      "frames (each fold model on its held-out match). The still fraction is not a "
                      "condition.".format(
                          cd.SCENARIO_CELL_MAX_GAP_FRAMES, cd.SCENARIO_CELL_SPEED_HALF_WIDTH_MPS,
                          cell_window(doc["window_sizes"]), SCENARIO_CELL_SPEED_CHANGE_MPS,
                          cd.SCENARIO_CELL_INTERVAL[1] - cd.SCENARIO_CELL_INTERVAL[0]), "",
                      "| Scenario | Frames | Gaps | P(D) | d (data) | Interval | "
                      + " | ".join("{} `d_fit`".format(m) for m in models) + " |",
                      "|---|---:|---:|---:|---:|---|" + "---:|" * len(models)]
            for sname, c in cells.items():
                lines.append("| {} | {} | {} | {} | {} | [{}, {}] | {} |".format(
                    sname, c["n_frames"], c["n_gaps"], _fmt(c["p_dead"], "{:.3f}"),
                    _fmt(c["d"], "{:+.3f}"), _fmt(c["low"], "{:+.3f}"), _fmt(c["high"], "{:+.3f}"),
                    " | ".join(_fmt(c["models"][m], "{:+.3f}") for m in models)))
            lines.append("")

    if diagnostics:
        lines += ["## Features alone (D vs alive, `default`, all usable frames)", "",
                  "AUC of each feature alone on its finite rows (mean, min and max over the 7 "
                  "matches; below 0.5 points to live play) and the share of NaN among dead-ball "
                  "and live frames. A very high AUC on a ball feature, or a large gap between the "
                  "two NaN shares, is the \"ball vanished => D\" channel that depends on the "
                  "degradation (F4-04).", "",
                  "| Feature | AUC mean | AUC min | AUC max | NaN share (D) | NaN share (alive) |",
                  "|---|---:|---:|---:|---:|---:|"]
        for f, r in diagnostics.items():
            lines.append("| `{}` | {} | {} | {} | {} | {} |".format(
                f, _fmt(r["auc_mean"], "{:.3f}"), _fmt(r["auc_min"], "{:.3f}"),
                _fmt(r["auc_max"], "{:.3f}"), _fmt(r["nan_dead"], "{:.3f}"),
                _fmt(r["nan_alive"], "{:.3f}")))
        lines.append("")

    if decoder:
        lines += ["## Decoder: spellF1 on degraded IDSSE by `lam_dead` (mean ± SE over the matches)",
                  "", decoder_note, "",
                  "Every match is decoded with the team and dead models that left it out. Count "
                  "ratio = predicted / true possessions. Report only: no value is chosen here.", ""]
        for deg, df in decoder.items():
            s = summarize_decoder(df)
            lines += ["### `degrade={}`".format(deg), "",
                      "| Config | lam_dead | spellF1 | SE | Count ratio |",
                      "|---|---:|---:|---:|---:|"]
            for (name, lam), r in s.iterrows():
                lines.append("| {} | {} | {:.4f} | {:.4f} | {:.3f} |".format(
                    name, lam, r["mean"], r["se"], r["count_ratio"]))
            lines.append("")
            if decoder_own:
                paired = paired_decoder(df, decoder_own)
                lines += ["Paired by match: spellF1 of each GBDT config minus that of its base "
                          "config at the same `lam_dead` (mean ± SE of the differences, matches "
                          "where the GBDT is higher).", "",
                          "| Config | Base | lam_dead | Difference | SE | Matches higher |",
                          "|---|---|---:|---:|---:|---:|"]
                for _, r in paired.iterrows():
                    lines.append("| {} | {} | {} | {:+.4f} | {} | {} of {} |".format(
                        r["config"], r["base"], r["lam_dead"], r["diff"], _fmt(r["se"]),
                        r["wins"], r["n"]))
                lines.append("")
    return "\n".join(lines)


# -- command line --------------------------------------------------------------------------------
def load_all(matches: Sequence[str], modes: Sequence[str], params: Mapping,
             window_sizes: Sequence[int], references: Mapping[str, Mapping], load_fn) -> dict:
    """``{mode: {match: match_data}}``; the reference logits use the fold of each match."""
    out = {}
    for mode in modes:
        out[mode] = {}
        for m in matches:
            master, labels = load_fn(m, mode)
            out[mode][m] = match_data(master, labels, params, window_sizes, references, holdout=m)
            del master, labels
    return out


def decoder_configs(base_cfgs: Mapping[str, dict], files: Mapping[str, Path]
                    ) -> tuple[dict, dict]:
    """For each base config: itself (its own dead kind) and one copy per GBDT file.

    Returns ``(configs, own)``: ``own`` maps each GBDT copy to the name of its base config.
    """
    out, own = {}, {}
    for base_name, cfg in base_cfgs.items():
        base_label = "{} (dead {})".format(base_name, cfg["dead"]["kind"])
        out[base_label] = cfg
        for name, path in files.items():
            c = copy.deepcopy(cfg)
            c["dead"]["kind"] = "gbdt"
            c["dead"]["calibrated_file"] = str(path)
            label = "{} (dead {})".format(base_name, name)
            out[label] = c
            own[label] = base_label
    return out, own


def run(params: Mapping, matches: Sequence[str], *, out_dir: Path, model_dir: Path,
        report_path: Path, candidates: Sequence[str] = ("dead_gbdt", "dead_gbdt_pooled"),
        window_sizes: Sequence[int] = DEFAULT_WINDOW_SIZES, fold_root: Path | None = None,
        base_cfgs: Mapping[str, dict] | None = None,
        lam_grid: Sequence[float] = DEFAULT_LAM_GRID, decoder_modes: Sequence[str] = EVAL_MODES,
        reference_files: Mapping[str, str] | None = None, load_fn=None,
        model_params: Mapping = MODEL_PARAMS, log: Callable = print) -> dict:
    """Every fit of the task, the calibration files, the models and the report.

    ``model_dir`` holds ``<candidate>/all.joblib``; ``fold_root`` (default ``ext_dir()``) holds the
    fold models under ``models/<candidate>/``. The files store ``model_dir`` and the fold directory
    relative to the repository root and to ``ext_dir()`` when they are inside them.
    ``reference_files`` overrides the F3-04 reference file of a candidate (tests).
    """
    if load_fn is None:
        from modeling.external.files import load_external as load_fn
    unknown = [c for c in candidates if c not in ALL_CANDIDATES]
    if unknown:
        raise ValueError("unknown candidates {}".format(unknown))
    root = paths.repo_root()
    fold_root = paths.ext_dir() if fold_root is None else Path(fold_root)
    ref_files = {name: str((reference_files or {}).get(name, ALL_CANDIDATES[name][2]))
                 for name in candidates}
    references = {f: cd.load_calibration(resolve_repo_path(f)) for f in ref_files.values()}
    modes = list(dict.fromkeys(list(EVAL_MODES) + [m for c in candidates
                                                   for m in ALL_CANDIDATES[c][0]]))
    t0 = time.time()
    data = load_all(matches, modes, params, window_sizes, references, load_fn)
    log("features of {} matches x {} modes in {:.0f}s".format(len(matches), len(modes),
                                                               time.time() - t0))

    def rel(path: Path, base: Path) -> str:
        path = Path(path).resolve()
        try:
            return str(path.relative_to(base.resolve()))
        except ValueError:
            return str(path)

    docs, files = {}, {}
    for name in candidates:
        train_modes, step, _, description = ALL_CANDIDATES[name]
        ref_file = ref_files[name]
        all_path = Path(model_dir) / name / "all.joblib"
        fold_dir = fold_root / "models" / name
        result = calibrate(data, matches=matches, train_modes=train_modes, step=step,
                           window_sizes=window_sizes, fold_root=fold_dir, all_path=all_path,
                           reference=ref_file, model_params=model_params, log=log)
        ref_doc = references[ref_file]
        ref_d_fit = {m: data["default"][m]["ref"][ref_file]
                     - cd.prior_logit(cd.select_model(ref_doc, m)) for m in matches}
        cells = scenario_cells(data["default"], {"GBDT": result["oof_d_fit"]["default"],
                                                 "F3-04": ref_d_fit},
                               result["pi_d"]["all"], window_sizes)
        doc = calibration_document(
            result, params, description=description, train_modes=train_modes, step=step,
            window_sizes=window_sizes, model_file=rel(all_path, root),
            fold_dir=rel(fold_dir, fold_root) if fold_root == paths.ext_dir() else str(fold_dir),
            reference_file=ref_file, reference_doc=ref_doc, cells=cells,
            model_params=model_params)
        files[name] = write_calibration(doc, Path(out_dir) / "{}.yaml".format(name))
        docs[name] = doc

    diagnostics = feature_diagnostics(data["default"], feature_names(window_sizes))
    del data

    decoder, own = {}, {}
    note = ""
    if base_cfgs:
        cfgs, own = decoder_configs(base_cfgs, files)
        for deg in decoder_modes:
            t0 = time.time()
            decoder[deg] = evaluate_decoder(cfgs, lam_grid, degrade=deg, matches=matches,
                                            load_fn=load_fn)
            log("decoder on {} in {:.0f}s".format(deg, time.time() - t0))
        note = "Base configs: {}; each also with `dead.kind: gbdt` and the files above.".format(
            ", ".join("`{}` (`lam_dead` {})".format(n, c["decoder"]["lam_dead"])
                      for n, c in base_cfgs.items()))
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_markdown(docs, diagnostics, decoder, note, own),
                           encoding="utf-8", newline="\n")
    return {"files": files, "docs": docs, "report": report_path, "decoder": decoder}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/v2.yaml",
                        help="config whose dead.params build the features (default: %(default)s)")
    parser.add_argument("--decoder-configs", nargs="*", default=["configs/v2.yaml"],
                        help="base configs of the spellF1 table (default: %(default)s); none to "
                             "skip it")
    parser.add_argument("--matches", default="all",
                        help="all or comma-separated IDSSE match ids (default: all)")
    parser.add_argument("--no-pooled", action="store_true", help="skip the pooled candidate")
    parser.add_argument("--candidates", nargs="+", choices=sorted(ALL_CANDIDATES),
                        help="explicit candidate list (overrides --no-pooled); the "
                             "dead_gbdt_hidelo/hidehi fits serve only the F2-06 stability check")
    parser.add_argument("--out-dir", default="configs/calib")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--report", default="reports/calibration_dead_gbdt.md")
    parser.add_argument("--lam-grid", nargs="*", type=float, default=list(DEFAULT_LAM_GRID))
    return parser


def main(argv=None) -> int:
    from modeling.config import load_config
    from modeling.external.files import IDSSE_MATCHES

    args = build_parser().parse_args(argv)
    cfg = load_config(resolve_repo_path(args.config))
    matches = (list(IDSSE_MATCHES) if args.matches == "all"
               else [m.strip() for m in args.matches.split(",") if m.strip()])
    base_cfgs = {}
    for path in args.decoder_configs:
        c = load_config(resolve_repo_path(path))
        base_cfgs[c.get("name", Path(path).stem)] = c
    if args.candidates:
        candidates = tuple(args.candidates)
    else:
        candidates = ("dead_gbdt",) if args.no_pooled else tuple(CANDIDATES)
    out = run(cfg["dead"]["params"], matches, out_dir=resolve_repo_path(args.out_dir),
              model_dir=resolve_repo_path(args.model_dir),
              report_path=resolve_repo_path(args.report), candidates=candidates,
              base_cfgs=base_cfgs, lam_grid=args.lam_grid)
    for name, path in out["files"].items():
        print("{}: {}".format(name, path))
    print("report: {}".format(out["report"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ALL_CANDIDATES", "CANDIDATES", "STABILITY_CANDIDATES", "DEFAULT_LAM_GRID", "EVAL_MODES", "MODEL_PARAMS", "POOLED_MODES",
           "POOLED_SUBSAMPLE", "REPORTED_ONLY", "SCENARIOS", "SUBSAMPLE", "calibrate",
           "calibration_document", "cell_window", "gap_runs", "scenario_cell_mask",
           "scenario_cells",
           "decoder_configs", "evaluate_decoder", "feature_diagnostics", "fit_model", "load_all",
           "match_data", "paired_decoder", "report_markdown", "run", "scenario_rows", "scenario_values",
           "summarize_decoder", "training_rows", "write_calibration"]
