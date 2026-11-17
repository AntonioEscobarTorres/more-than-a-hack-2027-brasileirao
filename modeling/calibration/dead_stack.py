"""Calibration of the stacked dead_logit on degraded IDSSE (F2-09, decision D-029).

Usage (after ``python -m modeling.calibration.stack_gbdt``, whose hazard models it reuses)::

    python -m modeling.calibration.dead_stack --config configs/final.yaml --jobs 7

The second stage (:func:`modeling.evidence.stack.dead_stack_from_model`) sees ``d_fit`` (the F3-08
dead GBDT likelihood ratio, before the tag prior) and the GBDT hazard ``h`` in time windows.
Trained on the ``default`` masters only (never ``none``, never BAR-ATH), on the usable frames of
F3-04 (inside a half, label not X), 1 every ``subsample`` of the dead GBDT file; target D.

- ``all`` (``models/dead_stack/all.joblib``, committed): every match, with its leave-one-match-out
  ``d_fit`` and ``h``;
- fold of match ``m`` (``ext_dir()/models/dead_stack/<m>.joblib``): the other matches ``j``, with
  ``d_fit`` and ``h`` of ``j`` from models trained without ``m`` and ``j`` (nested): the dead GBDT
  pair models are trained here with the F3-08 recipe (``ext_dir()/models/stack_pairs/dead_*``);
  the hazard pair models are the ones of ``stack_gbdt``.

Every fit is single-threaded and seeded. Writes ``configs/calib/dead_stack.yaml``.
"""
from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from evaluation import possession_metrics as pm
from modeling import paths
from modeling.calibration.common import binary_auc, binary_log_loss
from modeling.calibration.dead import usable_rows
from modeling.calibration.stack_gbdt import (
    GBDT_PARAMS,
    PAIR_DIR,
    _plain,
    _save,
    write_yaml,
)
from modeling.config import load_config
from modeling.deadball.dead_gbdt import (
    build_features,
    d_fit_values,
    gbdt_d_fit,
    load_calibration,
    resolve_repo_path,
)
from modeling.evidence.hazard_gbdt import (
    build_hazard_gbdt_features,
    hazard_from_bundle,
    load_hazard_gbdt_calibration,
    load_model_bundle,
)
from modeling.evidence.stack import build_dead_stack_features, dead_stack_feature_names
from modeling.external.files import IDSSE_MATCHES, load_external
from modeling.pipeline import hazard_values, variant_params

DEGRADE = "default"
DEAD_STACK_DIR = "models/dead_stack"


def _fit(X: np.ndarray, y: np.ndarray, params: Mapping) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits

    y = np.asarray(y, dtype=np.int8)
    pi = float(y.mean())
    if not 0.0 < pi < 1.0:
        raise ValueError("training rows need both classes, got P(D) = {}".format(pi))
    clf = HistGradientBoostingClassifier(**dict(params))
    with threadpool_limits(limits=1):
        clf.fit(np.asarray(X, dtype=np.float64), y)
    return {"model": clf, "pi_d": pi}


def _job(task):
    X, y, params = task
    return _fit(X, y, params)


def run_jobs(tasks, jobs: int) -> list:
    """``[_job(t) for t in tasks]`` in ``jobs`` processes, at most ``2 * jobs`` tasks in flight."""
    if max(1, int(jobs)) == 1:
        return [_job(t) for t in tasks]
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    jobs = int(jobs)
    results, pending = {}, {}
    with ProcessPoolExecutor(jobs) as ex:
        for i, task in enumerate(tasks):
            pending[ex.submit(_job, task)] = i
            if len(pending) >= 2 * jobs:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for f in done:
                    results[pending.pop(f)] = f.result()
        for f in list(pending):
            results[pending.pop(f)] = f.result()
    return [results[i] for i in range(len(results))]


def calibrate(cfg: dict, *, matches: Sequence[str] | None = None, jobs: int = 1, load_fn=None,
              gbdt_params: Mapping = GBDT_PARAMS, dead_model_params: Mapping | None = None,
              out_dir: str = DEAD_STACK_DIR, pair_dir: str = PAIR_DIR, log=print) -> dict:
    """Train and save the dead-stack models (and the dead GBDT pair models); return the table."""
    import sklearn

    if load_fn is None:
        load_fn = load_external
    matches = list(IDSSE_MATCHES if matches is None else matches)
    if len(matches) < 3:
        raise ValueError("need at least 3 matches (nested folds)")
    if cfg["dead"]["kind"] != "gbdt" or cfg["hazard"]["kind"] != "gbdt":
        raise ValueError("the reference config must use dead.kind gbdt and hazard.kind gbdt")
    dparams = variant_params(cfg["dead"])
    dcal = load_calibration(dparams["calibrated_file"])
    if list(dcal["train_modes"]) != [DEGRADE]:
        raise ValueError("the dead GBDT must be trained on the default masters only")
    step = int(dcal["subsample"])
    windows = list(dcal["window_sizes"])
    if dead_model_params is None:
        dead_model_params = dict(dcal["model_params"])
    hcal = load_hazard_gbdt_calibration(cfg["hazard"]["calibrated_file"])
    h_clip = float(hcal["h_clip"])
    ext = paths.ext_dir()
    pair_root = Path(pair_dir) if Path(pair_dir).is_absolute() else ext / pair_dir

    t0 = time.time()
    data = {}
    for m in matches:
        master, gt = load_fn(m, DEGRADE)
        gt = np.asarray(gt)
        half = master["half"].to_numpy()
        rows = usable_rows(master, gt)
        Xd = build_features(master, dparams, windows).to_numpy(dtype=np.float64)
        hf = build_hazard_gbdt_features(master, hcal["hazard_settings"], hcal["control_params"])
        data[m] = {"half": half, "gt": gt, "rows": rows, "Xd": Xd, "hf": hf,
                   "d_lomo": gbdt_d_fit(master, dparams, m),
                   "h_lomo": hazard_values(master, cfg, holdout=m)}
        del master
        log("loaded {} ({:.0f}s)".format(m, time.time() - t0))

    def dead_rows(keep):
        X = np.vstack([data[m]["Xd"][data[m]["rows"]][::step] for m in keep])
        y = np.concatenate([(data[m]["gt"][data[m]["rows"]] == pm.D)[::step] for m in keep])
        return X, y

    pairs = list(itertools.combinations(matches, 2))
    tasks = ((*dead_rows([x for x in matches if x not in p]), dead_model_params) for p in pairs)
    log("fitting {} dead GBDT pair models with {} jobs".format(len(pairs), jobs))
    dead_pairs = dict(zip(pairs, run_jobs(tasks, jobs)))
    version = sklearn.__version__
    for p, fitted in dead_pairs.items():
        _save(pair_root / "dead_{}_{}.joblib".format(*p),
              {"sklearn_version": version, "pi_d": fitted["pi_d"], "model": fitted["model"]})

    def pair_inputs(j, m):
        p = tuple(sorted((j, m), key=matches.index))
        d = data[j]
        fitted = dead_pairs[p]
        d_fit = np.zeros(len(d["half"]), dtype=np.float64)
        inside = d["half"] != 0
        d_fit[inside] = d_fit_values(fitted["model"], d["Xd"][inside], fitted["pi_d"])
        hb = load_model_bundle(pair_root / "hazard_{}_{}.joblib".format(*p),
                               sklearn_version=version, features=hcal["features"])
        h = hazard_from_bundle(d["hf"], d["half"], hb, h_clip).astype(np.float64)
        return d_fit, h

    def stack_rows(j, d_fit, h):
        d = data[j]
        X = build_dead_stack_features(d_fit, h, d["half"])[d["rows"]][::step]
        y = (d["gt"][d["rows"]] == pm.D)[::step]
        return X, y

    s_tasks, s_keys = [], []
    for m in matches:
        Xs, ys = [], []
        for j in matches:
            if j == m:
                continue
            X, y = stack_rows(j, *pair_inputs(j, m))
            Xs.append(X), ys.append(y)
        s_tasks.append((np.vstack(Xs), np.concatenate(ys), gbdt_params)), s_keys.append(m)
    Xs, ys = [], []
    for j in matches:
        X, y = stack_rows(j, data[j]["d_lomo"], data[j]["h_lomo"])
        Xs.append(X), ys.append(y)
    s_tasks.append((np.vstack(Xs), np.concatenate(ys), gbdt_params)), s_keys.append("all")
    fitted = dict(zip(s_keys, run_jobs(s_tasks, jobs)))
    log("dead stack fitted ({:.0f}s)".format(time.time() - t0))

    names = dead_stack_feature_names()
    root = Path(out_dir) if Path(out_dir).is_absolute() else None
    fold_root = root if root is not None else ext / out_dir
    for m in matches:
        _save(fold_root / "{}.joblib".format(m),
              {"sklearn_version": version, "features": names, **fitted[m]})
    all_path = _save((root if root is not None else resolve_repo_path(out_dir)) / "all.joblib",
                     {"sklearn_version": version, "features": names, **fitted["all"]})

    # Out-of-fold quality on the usable rows: first-stage d_fit vs second stage (both LR).
    from modeling.evidence.stack import dead_stack_from_model
    metrics = {}
    for m in matches:
        d = data[m]
        d2 = dead_stack_from_model(fitted[m]["model"], fitted[m]["pi_d"], d["d_lomo"],
                                   d["h_lomo"], d["half"]).astype(np.float64)
        y = d["gt"][d["rows"]] == pm.D
        metrics[m] = {"auc_first": float(binary_auc(y, d["d_lomo"][d["rows"]])),
                      "auc_stack": float(binary_auc(y, d2[d["rows"]])),
                      "log_loss_stack": float(binary_log_loss(
                          y, d2[d["rows"]] + np.log(fitted[m]["pi_d"] / (1 - fitted[m]["pi_d"]))))}
    return {
        "variant": "gbdt",
        "target": "dead",
        "sklearn_version": version,
        "model_file": str(out_dir) + "/all.joblib",
        "fold_dir": str(out_dir),
        "model_params": dict(gbdt_params),
        "features": names,
        "meta": {
            "degrade": DEGRADE,
            "subsample": step,
            "matches": matches,
            "first_stage": str(cfg["dead"]["calibrated_file"]),
            "hazard": str(cfg["hazard"]["calibrated_file"]),
            "nested": "fold m: rows of j != m with d_fit and h from models trained without m and j",
            "pi_d_all": float(fitted["all"]["pi_d"]),
            "all_model_bytes": int(all_path.stat().st_size),
            "fold_metrics": metrics,
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate the stacked dead_logit (F2-09)")
    parser.add_argument("--config", default="configs/final.yaml",
                        help="reference config (dead GBDT and GBDT hazard sections)")
    parser.add_argument("--out", default="configs/calib/dead_stack.yaml")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args(argv)
    cfg = load_config(resolve_repo_path(args.config))
    table = calibrate(cfg, jobs=args.jobs)
    header = ("# F2-09 (D-029), fitted on IDSSE only by python -m modeling.calibration.dead_stack;"
              " do not edit by hand.\n")
    print("wrote", write_yaml(_plain(table), args.out, header))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["calibrate", "run_jobs"]
