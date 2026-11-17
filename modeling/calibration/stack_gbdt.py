"""Calibration of the GBDT hazard and of the stacked team_logit on degraded IDSSE (F2-09, D-029).

Usage::

    python -m modeling.calibration.stack_gbdt --config configs/final_d028.yaml --jobs 7

Trains, on the ``default`` masters only (D-016; never ``none``, never BAR-ATH):

1. the GBDT hazard (:mod:`modeling.evidence.hazard_gbdt`): one model per leave-one-match-out fold
   (``ext_dir()/models/hazard_gbdt/<match>.joblib``, not committed), one per pair of matches left
   out (``ext_dir()/models/stack_pairs/``, for the nested inputs below) and ``all``
   (``models/hazard_gbdt/all.joblib``, committed, the one used for the CSV);
2. the first-stage team GBDT of the reference config once per pair of matches left out (the same
   features, parameters and samples as F1-07, ``ball`` model only: the reference config must use
   ``gap_model: logistic``), for the nested inputs below;
3. the second stage (:mod:`modeling.evidence.stack`): the fold model of match ``m`` on the other
   matches ``j``, with ``t`` and ``h`` of ``j`` from the models trained without ``m`` and ``j``
   (the gap frames of ``t`` keep the leave-one-match-out logistic of ``j``, a linear model of 6
   matches); ``all`` on every match with its leave-one-match-out ``t`` and ``h``
   (``models/team_stack/all.joblib``, committed).

Every fit is single-threaded and seeded (``random_state=0``, no early stopping): the same inputs
give the same models. Writes ``configs/calib/hazard_gbdt.yaml`` and ``configs/calib/stack.yaml``.
"""
from __future__ import annotations

import argparse
import itertools
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import yaml

from evaluation import possession_metrics as pm
from modeling import paths
from modeling.calibration.common import binary_auc, binary_log_loss
from modeling.calibration.hazard import change_targets, feature_settings, load_fit_config
from modeling.config import load_config
from modeling.contracts import TEAM_LOGIT_MAX
from modeling.evidence.hazard_gbdt import (
    build_hazard_gbdt_features,
    hazard_gbdt_feature_names,
)
from modeling.evidence.stack import build_stack_features, stack_feature_names, stack_sign
from modeling.evidence.team import team_logit_gbdt
from modeling.evidence.team_gbdt import (
    build_gbdt_features,
    gbdt_feature_layout,
    load_gbdt_calibration,
    resolve_repo_path,
    swap_matrix,
    swap_permutation,
)
from modeling.external.files import IDSSE_MATCHES, load_external
from modeling.calibration.common import SUBSAMPLE_STEP, alive_frames_mask, subsample_indices
from modeling.calibration.team_gbdt import MODEL_PARAMS as TEAM_MODEL_PARAMS

DEGRADE = "default"
HAZARD_FIT = "configs/hazard_fit_nodir.yaml"
# Same capacity as F1-07 / F3-08 (fixed, not tuned).
GBDT_PARAMS = {
    "max_iter": 300,
    "learning_rate": 0.05,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 500,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": 0,
}
HAZARD_TARGET_W = 12
HAZARD_BRIDGE_D = 0
HAZARD_SUBSAMPLE = 5
HAZARD_DIR = "models/hazard_gbdt"
STACK_DIR = "models/team_stack"
PAIR_DIR = "models/stack_pairs"


# -- fitting (module level: run in worker processes) --------------------------------------------

def _fit_classifier(X: np.ndarray, y: np.ndarray, params: Mapping):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits

    clf = HistGradientBoostingClassifier(**dict(params))
    with threadpool_limits(limits=1):
        clf.fit(X, y)
    return clf


def fit_hazard(X: np.ndarray, y: np.ndarray) -> dict:
    """``{"model", "pi"}``: the GBDT hazard on rows ``X`` with target ``y``."""
    y = np.asarray(y, dtype=np.int8)
    pi = float(y.mean())
    if not 0.0 < pi < 1.0:
        raise ValueError("hazard target has a single class")
    return {"model": _fit_classifier(X, y, GBDT_PARAMS), "pi": pi}


def fit_antisymmetric(X: np.ndarray, y: np.ndarray, perm: np.ndarray | None,
                      sign: np.ndarray, params: Mapping):
    """Fit on ``X`` plus its team swap with flipped labels (``perm`` None: sign flip only)."""
    Xs = X * sign if perm is None else swap_matrix(X, perm, sign)
    X_aug = np.vstack([X, Xs])
    y_aug = np.concatenate([y, 1 - np.asarray(y)]).astype(np.int8)
    return _fit_classifier(X_aug, y_aug, params)


def _job(task):
    kind, args, params = task
    if kind == "hazard":
        X, y = args
        y = np.asarray(y, dtype=np.int8)
        pi = float(y.mean())
        if not 0.0 < pi < 1.0:
            raise ValueError("hazard target has a single class")
        return {"model": _fit_classifier(X, y, params), "pi": pi}
    if kind == "team":
        X, y, perm, sign = args
        return fit_antisymmetric(X, y, perm, sign, params)
    if kind == "stack":
        X, y = args
        return fit_antisymmetric(X, y, None, stack_sign(), params)
    raise ValueError(kind)


def run_jobs(tasks, jobs: int) -> list:
    """``[_job(t) for t in tasks]`` in ``jobs`` processes, at most ``2 * jobs`` tasks in flight.

    ``tasks`` may be a generator: its arrays are built only when a slot is free.
    """
    from concurrent.futures import FIRST_COMPLETED, wait

    jobs = max(1, int(jobs))
    if jobs == 1:
        return [_job(task) for task in tasks]
    results, pending = {}, {}
    it = enumerate(tasks)
    with ProcessPoolExecutor(jobs) as ex:
        for i, task in it:
            pending[ex.submit(_job, task)] = i
            if len(pending) >= 2 * jobs:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for f in done:
                    results[pending.pop(f)] = f.result()
        for f in list(pending):
            results[pending.pop(f)] = f.result()
    return [results[i] for i in range(len(results))]


# -- inference helpers ---------------------------------------------------------------------------

def hazard_output(model: Mapping, feats: np.ndarray, half: np.ndarray, h_clip: float) -> np.ndarray:
    """Same arithmetic as :func:`modeling.evidence.hazard_gbdt.hazard_from_bundle` (float32)."""
    from modeling.evidence.hazard_gbdt import hazard_from_bundle
    import pandas as pd

    return hazard_from_bundle(pd.DataFrame(feats), half, model, h_clip)


def team_output(model, t_lomo: np.ndarray, X_full: np.ndarray, valid: np.ndarray,
                perm: np.ndarray, sign: np.ndarray) -> np.ndarray:
    """First-stage ``team_logit`` with ``model`` as the ``ball`` model (float32).

    The gap frames keep ``t_lomo`` (the leave-one-match-out logistic); same arithmetic as
    :func:`modeling.evidence.team_gbdt.team_logit_gbdt_impl` without the tag prior (0 on IDSSE).
    """
    from modeling.evidence.team_gbdt import antisymmetric_logit

    t = np.asarray(t_lomo, dtype=np.float64).copy()
    t[valid] = antisymmetric_logit(model, X_full[valid], perm, sign)
    t = np.clip(t, -TEAM_LOGIT_MAX, TEAM_LOGIT_MAX)
    return t.astype(np.float32)


def stack_rows(t: np.ndarray, h: np.ndarray, half: np.ndarray, gt: np.ndarray,
               step: int = SUBSAMPLE_STEP) -> tuple[np.ndarray, np.ndarray]:
    """Second-stage training rows: alive frames, 1 every ``step``, y = 1 for H."""
    X = build_stack_features(t, h, half)
    idx = subsample_indices(alive_frames_mask(gt, half), step=step)
    return X[idx], (np.asarray(gt)[idx] == pm.H).astype(np.int8)


# -- main routine --------------------------------------------------------------------------------

def _pairs(matches: Sequence[str]) -> list[tuple[str, str]]:
    return list(itertools.combinations(matches, 2))


def _save(path: Path, payload: dict) -> Path:
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path, compress=3)
    return path


def calibrate(cfg: dict, *, matches: Sequence[str] | None = None, jobs: int = 1,
              hazard_fit_path: str = HAZARD_FIT, load_fn=None,
              gbdt_params: Mapping = GBDT_PARAMS, team_model_params: Mapping = TEAM_MODEL_PARAMS,
              hazard_dir: str = HAZARD_DIR, stack_dir: str = STACK_DIR,
              pair_dir: str = PAIR_DIR, hazard_pair_dir: str | None = None,
              reuse_hazard: bool = False, log=print) -> tuple[dict, dict]:
    """Train and save every model; return the two calibration tables (hazard, stack).

    ``hazard_dir`` and ``stack_dir`` hold the committed ``all.joblib`` (relative to the repository)
    and name the fold folders (relative to ``ext_dir()``); ``pair_dir`` (first-stage pair models)
    and ``hazard_pair_dir`` (hazard pair models, default ``pair_dir``) are under ``ext_dir()``.
    ``reuse_hazard`` loads the hazard models already saved there (same data, same seed: the same
    models) instead of fitting them again, e.g. for a new first stage.
    """
    import sklearn

    if load_fn is None:
        load_fn = load_external
    matches = list(IDSSE_MATCHES if matches is None else matches)
    if len(matches) < 3:
        raise ValueError("need at least 3 matches (nested folds)")
    if cfg["team"]["kind"] != "gbdt":
        raise ValueError("the reference config must use team.kind gbdt")
    team_params = cfg["team"]["params"]
    team_file = str(resolve_repo_path(cfg["team"]["calibrated_file"]))
    tcal = load_gbdt_calibration(team_file)
    if tcal["gap_model"] != "logistic":
        raise ValueError("the reference team GBDT must use gap_model logistic")
    f_params = dict(tcal["feature_params"])
    layout = gbdt_feature_layout(f_params)
    perm, sign = swap_permutation(layout)

    fit = load_fit_config(resolve_repo_path(hazard_fit_path))
    h_settings = feature_settings(fit)
    h_clip = float(h_settings["h_clip"])
    control_params = {**dict(team_params), **f_params}
    names = hazard_gbdt_feature_names()

    t0 = time.time()
    data = {}
    for m in matches:
        master, gt = load_fn(m, DEGRADE)
        gt = np.asarray(gt)
        half = master["half"].to_numpy()
        hf = build_hazard_gbdt_features(master, h_settings, control_params).to_numpy(np.float64)
        y_h, elig = change_targets(gt, half, HAZARD_TARGET_W, HAZARD_BRIDGE_D)
        h_idx = np.flatnonzero(elig)[::HAZARD_SUBSAMPLE]
        feats, valid = build_gbdt_features(master, team_params, f_params)
        X_full = feats.to_numpy(dtype=np.float64)
        t_idx = subsample_indices(alive_frames_mask(gt, half), step=SUBSAMPLE_STEP)
        t_idx = t_idx[valid[t_idx]]
        t_lomo = team_logit_gbdt(master, {**team_params, "calibrated_file": team_file},
                                 holdout=m)
        data[m] = {"half": half, "gt": gt, "hf": hf, "y_h": y_h, "elig": elig, "h_idx": h_idx,
                   "X_full": X_full, "valid": valid, "t_idx": t_idx,
                   "y_t": (gt[t_idx] == pm.H).astype(np.int8), "t_lomo": t_lomo}
        del master
        log("loaded {} ({:.0f}s)".format(m, time.time() - t0))

    def hz_rows(keep):
        X = np.vstack([data[m]["hf"][data[m]["h_idx"]] for m in keep])
        y = np.concatenate([data[m]["y_h"][data[m]["h_idx"]] for m in keep])
        return X, y

    def team_rows(keep):
        X = np.vstack([data[m]["X_full"][data[m]["t_idx"]] for m in keep])
        y = np.concatenate([data[m]["y_t"] for m in keep])
        return X, y, perm, sign

    pairs = _pairs(matches)
    ext = paths.ext_dir()
    hazard_pair_dir = pair_dir if hazard_pair_dir is None else hazard_pair_dir

    def hz_path(key, which):
        if key == "hz":
            if which == "all":
                return resolve_repo_path(hazard_dir) / "all.joblib"
            return ext / hazard_dir / "{}.joblib".format(which)
        return ext / hazard_pair_dir / "hazard_{}_{}.joblib".format(*which)

    specs = []
    if not reuse_hazard:
        specs = [("hz", m, "hazard", [x for x in matches if x != m]) for m in matches]
        specs.append(("hz", "all", "hazard", list(matches)))
    for p in pairs:
        keep = [x for x in matches if x not in p]
        if not reuse_hazard:
            specs.append(("hzp", p, "hazard", keep))
        specs.append(("tp", p, "team", keep))
    keys = [(a, b) for a, b, _, _ in specs]
    tasks = ((kind, hz_rows(keep) if kind == "hazard" else team_rows(keep),
              gbdt_params if kind == "hazard" else team_model_params)
             for _, _, kind, keep in specs)
    log("fitting {} first-stage and hazard models with {} jobs".format(len(specs), jobs))
    fitted = dict(zip(keys, run_jobs(tasks, jobs)))
    log("first stage and hazards fitted ({:.0f}s)".format(time.time() - t0))

    version = sklearn.__version__
    if reuse_hazard:
        from modeling.evidence.hazard_gbdt import load_model_bundle
        for key, which in [("hz", m) for m in matches] + [("hz", "all")] + [("hzp", p)
                                                                             for p in pairs]:
            b = load_model_bundle(hz_path(key, which), sklearn_version=version, features=names)
            fitted[(key, which)] = {"model": b["model"], "pi": float(b["pi"])}
        log("hazard models reused")
    else:
        for m in matches:
            _save(hz_path("hz", m),
                  {"sklearn_version": version, "features": names, **fitted[("hz", m)]})
        _save(hz_path("hz", "all"),
              {"sklearn_version": version, "features": names, **fitted[("hz", "all")]})
        for p in pairs:
            _save(hz_path("hzp", p),
                  {"sklearn_version": version, "features": names, **fitted[("hzp", p)]})
    for p in pairs:
        _save(ext / pair_dir / "team_{}_{}.joblib".format(*p),
              {"sklearn_version": version, "features": list(layout["features"]),
               "ball": fitted[("tp", p)]})

    # Hazard out-of-fold quality (eligible frames).
    hz_metrics, h_lomo = {}, {}
    for m in matches:
        d = data[m]
        h_lomo[m] = hazard_output(fitted[("hz", m)], d["hf"], d["half"], h_clip)
        e = d["elig"]
        hz_metrics[m] = {"auc": float(binary_auc(d["y_h"][e], h_lomo[m][e])),
                         "log_loss": float(binary_log_loss(
                             d["y_h"][e], h_lomo[m][e] + np.log(
                                 fitted[("hz", m)]["pi"] / (1 - fitted[("hz", m)]["pi"]))))}

    # Second stage: nested rows for the folds, leave-one-match-out rows for ``all``.
    def pair_inputs(j, m):
        p = tuple(sorted((j, m), key=matches.index))
        d = data[j]
        t = team_output(fitted[("tp", p)], d["t_lomo"], d["X_full"], d["valid"], perm, sign)
        h = hazard_output(fitted[("hzp", p)], d["hf"], d["half"], h_clip)
        return t, h

    s_tasks, s_keys = [], []
    for m in matches:
        Xs, ys = [], []
        for j in matches:
            if j == m:
                continue
            t, h = pair_inputs(j, m)
            X, y = stack_rows(t, h, data[j]["half"], data[j]["gt"])
            Xs.append(X), ys.append(y)
        s_tasks.append(("stack", (np.vstack(Xs), np.concatenate(ys)), gbdt_params))
        s_keys.append(m)
    Xs, ys = [], []
    for j in matches:
        X, y = stack_rows(data[j]["t_lomo"], h_lomo[j], data[j]["half"], data[j]["gt"])
        Xs.append(X), ys.append(y)
    s_tasks.append(("stack", (np.vstack(Xs), np.concatenate(ys)), gbdt_params))
    s_keys.append("all")
    s_fitted = dict(zip(s_keys, run_jobs(s_tasks, jobs)))
    log("second stage fitted ({:.0f}s)".format(time.time() - t0))
    for m in matches:
        _save(ext / stack_dir / "{}.joblib".format(m),
              {"sklearn_version": version, "features": stack_feature_names(),
               "model": s_fitted[m]})
    all_stack = _save(resolve_repo_path(stack_dir) / "all.joblib",
                      {"sklearn_version": version, "features": stack_feature_names(),
                       "model": s_fitted["all"]})
    all_hz = resolve_repo_path(hazard_dir) / "all.joblib"

    hazard_table = {
        "variant": "gbdt",
        "sklearn_version": version,
        "model_file": str(hazard_dir) + "/all.joblib",
        "fold_dir": str(hazard_dir),
        "h_clip": h_clip,
        "hazard_settings": h_settings,
        "control_params": _plain(control_params),
        "model_params": dict(gbdt_params),
        "features": names,
        "meta": {
            "degrade": DEGRADE,
            "target_w_frames": HAZARD_TARGET_W,
            "bridge_d_frames": HAZARD_BRIDGE_D,
            "subsample_every": HAZARD_SUBSAMPLE,
            "hazard_fit": hazard_fit_path,
            "matches": matches,
            "pi_all": float(fitted[("hz", "all")]["pi"]),
            "all_model_bytes": int(all_hz.stat().st_size),
            "fold_metrics": hz_metrics,
        },
    }
    stack_table = {
        "variant": "gbdt",
        "sklearn_version": version,
        "model_file": str(stack_dir) + "/all.joblib",
        "fold_dir": str(stack_dir),
        "model_params": dict(gbdt_params),
        "features": stack_feature_names(),
        "meta": {
            "degrade": DEGRADE,
            "subsample_step": int(SUBSAMPLE_STEP),
            "matches": matches,
            "first_stage": str(cfg["team"]["calibrated_file"]),
            "hazard": str(hazard_dir),
            "nested": "fold m: rows of j != m with t and h from models trained without m and j",
            "all_model_bytes": int(all_stack.stat().st_size),
        },
    }
    return hazard_table, stack_table


def _plain(obj):
    if isinstance(obj, Mapping):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def write_yaml(table: Mapping, path: str | Path, header: str) -> Path:
    path = resolve_repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = header + yaml.safe_dump(_plain(table), sort_keys=False)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate the GBDT hazard and the stacked "
                                                 "team_logit (F2-09)")
    parser.add_argument("--config", default="configs/final_d028.yaml",
                        help="reference config (its team section is the first stage)")
    parser.add_argument("--hazard-out", default="configs/calib/hazard_gbdt.yaml")
    parser.add_argument("--stack-out", default="configs/calib/stack.yaml")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--stack-dir", default=STACK_DIR)
    parser.add_argument("--pair-dir", default=PAIR_DIR,
                        help="first-stage pair models (under ext_dir())")
    parser.add_argument("--reuse-hazard", action="store_true",
                        help="load the hazard models saved by an earlier run instead of fitting "
                             "them (the hazard file is not rewritten)")
    args = parser.parse_args(argv)
    cfg = load_config(resolve_repo_path(args.config))
    hazard_table, stack_table = calibrate(cfg, jobs=args.jobs, stack_dir=args.stack_dir,
                                          pair_dir=args.pair_dir, hazard_pair_dir=PAIR_DIR,
                                          reuse_hazard=args.reuse_hazard)
    header = ("# F2-09 (D-029), fitted on IDSSE only by python -m modeling.calibration.stack_gbdt;"
              " do not edit by hand.\n")
    if not args.reuse_hazard:
        print("wrote", write_yaml(hazard_table, args.hazard_out, header))
    print("wrote", write_yaml(stack_table, args.stack_out, header))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["GBDT_PARAMS", "calibrate", "run_jobs", "fit_antisymmetric", "fit_hazard", "hazard_output",
           "stack_rows", "team_output", "write_yaml"]
