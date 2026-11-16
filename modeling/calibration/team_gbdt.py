"""GBDT team_logit calibration on degraded external IDSSE matches (F1-07).

Trains two ``HistGradientBoostingClassifier`` models (spec of F1-07), one on alive frames with a
valid ball (``ball``) and one on alive frames without it (``gap``), on the same samples as F1-05
(alive frames, 1 every ``SUBSAMPLE_STEP``). Every training sample also enters swapped
(``S x``, label H<->A), and the evidence is ``(g(x) - g(S x)) / 2``
(:mod:`modeling.evidence.team_gbdt`).

One model pair per leave-one-match-out fold, saved under ``ext_dir()/<fold_dir>/<match>.joblib``
(not committed), and one with the 7 matches, ``models/team_gbdt/all.joblib`` (committed, the only
one used for the CSV). The calibration file records the scikit-learn version; inference refuses
any other version.

Never calibrates on ``degrade="none"`` (clean IDSSE freezes the ball on dead balls and leaks the
label). Nothing here reads BAR-ATH.

Usage::

    python -m modeling.calibration.team_gbdt --config configs/v2.yaml \\
        --out configs/calib/team_gbdt.yaml --model models/team_gbdt/all.joblib \\
        --report reports/calibration_team_gbdt.md
"""
from __future__ import annotations

import argparse
import copy
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from evaluation import possession_metrics as pm
from modeling import paths, variants
from modeling.calibration.common import (
    SUBSAMPLE_STEP,
    alive_frames_mask,
    binary_auc,
    binary_log_loss,
    subsample_indices,
)
from modeling.config import load_config
from modeling.evidence.team import load_team_calibration, team_logit_calibrated
from modeling.evidence.team_gbdt import (
    GAP_MODELS,
    build_gbdt_features,
    gbdt_feature_layout,
    resolve_repo_path,
    swap_matrix,
    swap_permutation,
)
from modeling.evidence.windows import WINDOW_SIZES
from modeling.external.files import IDSSE_MATCHES, load_external
from modeling.pipeline import predict_labels
from modeling.scoring.metrics import score

# F1-07 spec, unchanged.
MODEL_PARAMS = {
    "max_iter": 300,
    "learning_rate": 0.05,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 500,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": 0,
}
DEFAULT_FOLD_DIR = "models/team_gbdt"
# D-023: n_tracked and ev_coll_spread fail the BAR-ATH range check (tools/gbdt_feature_shift.py,
# check (a)); n_still, ball_speed and the gap timing depend on camera coverage or ball noise and add
# at most 0.003 spellF1 on IDSSE (ablation), so they are left out as well.
EXCLUDE_FEATURES = ("n_tracked_home", "n_tracked_away", "ev_coll_spread", "n_still_home", "n_still_away",
                    "ball_speed", "since_last_obs", "until_next_obs")
DEFAULT_LAM_GRID = (0.2, 0.25, 0.3, 0.35, 0.4, 0.5)
SECTIONS = ("ball", "gap")


def gbdt_feature_params(logistic_calib: Mapping, *, window_sizes: Sequence[int] = WINDOW_SIZES,
                        include_bridge: bool = True,
                        exclude_features: Sequence[str] = EXCLUDE_FEATURES) -> dict:
    """F1-05 feature parameters (touch thresholds, bridge taus) plus the window sizes."""
    fp = copy.deepcopy(dict(logistic_calib["feature_params"]))
    fp["window_sizes"] = [int(w) for w in window_sizes]
    fp["include_bridge"] = bool(include_bridge)
    fp["exclude_features"] = [str(c) for c in exclude_features]
    return fp


def match_samples(master: pd.DataFrame, gt: np.ndarray, params: Mapping, f_params: Mapping,
                  *, step: int = SUBSAMPLE_STEP, logistic_logit: np.ndarray | None = None) -> dict:
    """``{"ball": (X, y, zl), "gap": (X, y, zl)}`` on 1-every-``step`` alive frames, y = 1 for H.

    ``zl`` is ``logistic_logit`` on the same rows (the F1-05 reference), or ``None``.
    """
    feats, valid = build_gbdt_features(master, params, f_params)
    half = master["half"].to_numpy()
    idx = subsample_indices(alive_frames_mask(gt, half), step=step)
    X = feats.to_numpy(dtype=np.float64)[idx]
    y = (np.asarray(gt)[idx] == pm.H).astype(np.int8)
    zl = None if logistic_logit is None else np.asarray(logistic_logit, dtype=np.float64)[idx]
    v = valid[idx]
    return {"ball": (X[v], y[v], None if zl is None else zl[v]),
            "gap": (X[~v], y[~v], None if zl is None else zl[~v])}


def fit_gbdt(X: np.ndarray, y: np.ndarray, perm: np.ndarray, sign: np.ndarray,
             model_params: Mapping = MODEL_PARAMS):
    """Fit on ``X`` plus its team swap with flipped labels (single-threaded, deterministic)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits

    X_aug = np.vstack([X, swap_matrix(X, perm, sign)])
    y_aug = np.concatenate([y, 1 - y]).astype(np.int8)
    clf = HistGradientBoostingClassifier(**dict(model_params))
    with threadpool_limits(limits=1):
        clf.fit(X_aug, y_aug)
    return clf


def _antisym(clf, X, perm, sign):
    from modeling.evidence.team_gbdt import antisymmetric_logit
    return antisymmetric_logit(clf, X, perm, sign)


def save_bundle(path: str | Path, models: Mapping, features: Sequence[str]) -> Path:
    import joblib
    import sklearn

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"sklearn_version": sklearn.__version__, "features": list(features),
              "ball": models["ball"], "gap": models["gap"]}
    joblib.dump(bundle, path, compress=3)
    return path


def calibrate_team_gbdt(
    cfg: dict,
    *,
    logistic_file: str | Path = "configs/calib/team.yaml",
    degrade: str = "default",
    matches: Sequence[str] | None = None,
    step: int = SUBSAMPLE_STEP,
    window_sizes: Sequence[int] = WINDOW_SIZES,
    include_bridge: bool = True,
    exclude_features: Sequence[str] = EXCLUDE_FEATURES,
    model_params: Mapping = MODEL_PARAMS,
    all_model_path: str | Path = "models/team_gbdt/all.joblib",
    fold_dir: str | Path = DEFAULT_FOLD_DIR,
    load_fn=None,
    log=print,
) -> dict:
    """Train the fold and ``all`` model pairs, save them and return the calibration table.

    ``meta.fold_metrics`` has the out-of-fold log-loss and AUC of the GBDT and, on the same rows,
    of the F1-05 logistic models of ``logistic_file``.
    """
    import sklearn

    if degrade == "none":
        raise ValueError("calibrating team_logit on degrade='none' is forbidden (label leak)")
    matches = list(IDSSE_MATCHES if matches is None else matches)
    if len(matches) < 2:
        raise ValueError("need at least 2 matches to calibrate team_logit")
    if load_fn is None:
        load_fn = load_external

    params = cfg["team"]["params"]
    logistic_path = str(resolve_repo_path(logistic_file))
    logistic = load_team_calibration(logistic_path)
    f_params = gbdt_feature_params(logistic, window_sizes=window_sizes,
                                   include_bridge=include_bridge,
                                   exclude_features=exclude_features)
    layout = gbdt_feature_layout(f_params)
    features = layout["features"]
    perm, sign = swap_permutation(layout)

    data = {}
    for match in matches:
        master, gt = load_fn(match, degrade)
        # F1-05 reference: the logistic team_logit of the fold that left this match out.
        zl = team_logit_calibrated(master, {**params, "calibrated_file": logistic_path},
                                   holdout=match)
        data[match] = match_samples(master, gt, params, f_params, step=step, logistic_logit=zl)
        del master, gt

    def stack(section, keep):
        X = np.vstack([data[m][section][0] for m in keep])
        y = np.concatenate([data[m][section][1] for m in keep])
        return X, y

    fold_root = Path(fold_dir)
    if not fold_root.is_absolute():
        fold_root = paths.ext_dir() / fold_root
    fold_metrics = {}
    for holdout in matches:
        t0 = time.time()
        train = [m for m in matches if m != holdout]
        models = {s: fit_gbdt(*stack(s, train), perm, sign, model_params) for s in SECTIONS}
        save_bundle(fold_root / "{}.joblib".format(holdout), models, features)
        row = {}
        z_all, zl_all, y_all = [], [], []
        for s in SECTIONS:
            X, y, zl = data[holdout][s]
            z = _antisym(models[s], X, perm, sign)
            row["{}_log_loss".format(s)] = binary_log_loss(y, z)
            row["{}_auc".format(s)] = binary_auc(y, z)
            row["{}_log_loss_logistic".format(s)] = binary_log_loss(y, zl)
            row["{}_auc_logistic".format(s)] = binary_auc(y, zl)
            z_all.append(z), zl_all.append(zl), y_all.append(y)
        y_c, z_c, zl_c = (np.concatenate(a) for a in (y_all, z_all, zl_all))
        row["log_loss"] = binary_log_loss(y_c, z_c)
        row["auc"] = binary_auc(y_c, z_c)
        row["log_loss_logistic"] = binary_log_loss(y_c, zl_c)
        row["auc_logistic"] = binary_auc(y_c, zl_c)
        fold_metrics[holdout] = {k: float(v) for k, v in row.items()}
        log("fold {} trained in {:.0f}s".format(holdout, time.time() - t0))

    t0 = time.time()
    models = {s: fit_gbdt(*stack(s, matches), perm, sign, model_params) for s in SECTIONS}
    all_path = save_bundle(resolve_repo_path(all_model_path), models, features)
    log("all trained in {:.0f}s".format(time.time() - t0))

    return {
        "variant": "gbdt",
        "sklearn_version": sklearn.__version__,
        "gap_model": "gbdt",
        "logistic_file": str(logistic_file),
        "model_file": str(all_model_path),
        "fold_dir": str(fold_dir),
        "model_params": dict(model_params),
        "feature_params": f_params,
        "features": list(features),
        "meta": {
            "degrade": degrade,
            "subsample_step": int(step),
            "matches": matches,
            "n_samples": {s: {m: int(len(data[m][s][1])) for m in matches} for s in SECTIONS},
            "all_model_bytes": int(all_path.stat().st_size),
            "fold_metrics": fold_metrics,
        },
    }


def write_gbdt_yaml(table: Mapping, path: str | Path, *, gap_model: str | None = None) -> Path:
    """Write the calibration file (optionally overriding ``gap_model``)."""
    payload = copy.deepcopy(dict(table))
    if gap_model is not None:
        if gap_model not in GAP_MODELS:
            raise ValueError("gap_model must be one of {}".format(GAP_MODELS))
        payload["gap_model"] = gap_model
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8", newline="\n")
    return path


@contextmanager
def _memoized_evidence():
    """Cache team/dead evidence per (master object, holdout, file) while decoders are swept."""
    variants.load_kinds()
    saved = (dict(variants.TEAM_KINDS), dict(variants.DEAD_KINDS))

    def wrap(fn):
        cache = {}

        def inner(master, params, *, holdout=None):
            key = (id(master), holdout, params.get("calibrated_file"))
            if key not in cache:
                cache[key] = fn(master, params, holdout=holdout)
            return cache[key]
        return inner

    for reg in (variants.TEAM_KINDS, variants.DEAD_KINDS):
        for name in list(reg):
            reg[name] = wrap(reg[name])
    try:
        yield
    finally:
        variants.TEAM_KINDS.clear(), variants.TEAM_KINDS.update(saved[0])
        variants.DEAD_KINDS.clear(), variants.DEAD_KINDS.update(saved[1])


def evaluate_lam_grid(cfgs: Mapping[str, dict], lam_grid: Sequence[float], *, degrade: str,
                      matches: Sequence[str] | None = None, load_fn=None) -> pd.DataFrame:
    """spellF1 and spell count ratio per (config, lam_team, match), LOMO (``holdout=match``)."""
    matches = list(IDSSE_MATCHES if matches is None else matches)
    if load_fn is None:
        load_fn = load_external
    rows = []
    with _memoized_evidence():
        for match in matches:
            master, gt = load_fn(match, degrade)
            for name, cfg in cfgs.items():
                for lam in lam_grid:
                    c = copy.deepcopy(cfg)
                    c["decoder"]["lam_team"] = float(lam)
                    m = score(gt, pm.label_codes(predict_labels(master, c, holdout=match)))
                    rows.append({"config": name, "lam_team": float(lam), "match": match,
                                 "spell_f1_mean": float(m["spell_f1_mean"]),
                                 "count_ratio": float(m["n_pred_spells"]) / float(m["n_gt_spells"])})
            del master, gt
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, standard error and count ratio per (config, lam_team)."""
    g = df.groupby(["config", "lam_team"], sort=True)
    out = g["spell_f1_mean"].agg(["mean", "std", "count"])
    out["se"] = out["std"] / np.sqrt(out["count"])
    out["count_ratio"] = g["count_ratio"].mean()
    return out[["mean", "se", "count_ratio"]]


def render_report(table: Mapping, evals: Mapping[str, pd.DataFrame], *, chosen: Mapping,
                  reference: str) -> str:
    fm = table["meta"]["fold_metrics"]
    lines = [
        "# Team logit GBDT calibration (F1-07)",
        "",
        "- degrade (training): `{}`; samples: alive frames, 1 every {}".format(
            table["meta"]["degrade"], table["meta"]["subsample_step"]),
        "- scikit-learn `{}`; model `{}` ({:.2f} MB); {} features".format(
            table["sklearn_version"], table["model_file"],
            table["meta"]["all_model_bytes"] / 2**20, len(table["features"])),
        "- model params: {}".format(", ".join("`{}={}`".format(k, v)
                                              for k, v in table["model_params"].items())),
        "- logistic reference (F1-05): `{}`".format(table["logistic_file"]),
        "- decoder evaluation: every match predicted with the fold models trained without it",
        "",
        "## Out-of-fold log-loss and AUC (GBDT vs F1-05 logistic, same rows)",
        "",
        "| Match | ball LL | ball LL logistic | ball AUC | ball AUC logistic | gap LL "
        "| gap LL logistic | gap AUC | gap AUC logistic |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    keys = ("ball_log_loss", "ball_log_loss_logistic", "ball_auc", "ball_auc_logistic",
            "gap_log_loss", "gap_log_loss_logistic", "gap_auc", "gap_auc_logistic")
    for match, m in fm.items():
        lines.append("| {} | ".format(match) + " | ".join("{:.4f}".format(m[k]) for k in keys)
                     + " |")
    mean = {k: float(np.mean([m[k] for m in fm.values()])) for k in keys}
    lines.append("| mean | " + " | ".join("{:.4f}".format(mean[k]) for k in keys) + " |")
    lines.append("")
    lines.extend(["## spellF1 on degraded IDSSE by `lam_team` (mean ± SE over 7 matches)", "",
                  "Reference `{}` is the submitted config. Count ratio = predicted / true "
                  "possessions.".format(reference), ""])
    for deg, df in evals.items():
        s = summarize(df)
        lines.extend(["### `degrade={}`".format(deg), "",
                      "| config | lam_team | spellF1 | SE | count ratio |",
                      "|:---|---:|---:|---:|---:|"])
        for (name, lam), r in s.iterrows():
            lines.append("| {} | {} | {:.4f} | {:.4f} | {:.3f} |".format(
                name, lam, r["mean"], r["se"], r["count_ratio"]))
        lines.append("")
    lines.extend(["## Choice", ""])
    for k, v in chosen.items():
        lines.append("- {}: {}".format(k, v))
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate the GBDT team_logit (F1-07)")
    parser.add_argument("--config", default="configs/v2.yaml",
                        help="reference config: team params, dead section and decoder")
    parser.add_argument("--logistic", default="configs/calib/team.yaml")
    parser.add_argument("--degrade", default="default")
    parser.add_argument("--extra-degrades", nargs="*", default=["hidelo", "hidehi"])
    parser.add_argument("--out", default="configs/calib/team_gbdt.yaml")
    parser.add_argument("--model", default="models/team_gbdt/all.joblib")
    parser.add_argument("--fold-dir", default=DEFAULT_FOLD_DIR)
    parser.add_argument("--report", default="reports/calibration_team_gbdt.md")
    parser.add_argument("--lam-grid", nargs="*", type=float, default=list(DEFAULT_LAM_GRID))
    parser.add_argument("--no-bridge", action="store_true")
    args = parser.parse_args(argv)

    root = paths.repo_root()
    cfg = load_config(root / args.config)
    table = calibrate_team_gbdt(cfg, logistic_file=args.logistic, degrade=args.degrade,
                                include_bridge=not args.no_bridge, all_model_path=args.model,
                                fold_dir=args.fold_dir)

    out = root / args.out
    cfgs, files = {}, {}
    for gap_model in GAP_MODELS:
        files[gap_model] = write_gbdt_yaml(
            table, out.with_name("{}.{}.yaml".format(out.stem, gap_model)), gap_model=gap_model)
        c = copy.deepcopy(cfg)
        c["team"]["kind"] = "gbdt"
        c["team"]["calibrated_file"] = str(files[gap_model])
        cfgs["gbdt_gap_" + gap_model] = c
    ref_name = cfg.get("name", "reference")
    evals = {}
    for deg in [args.degrade] + list(args.extra_degrades):
        df = evaluate_lam_grid(cfgs, args.lam_grid, degrade=deg)
        ref = evaluate_lam_grid({ref_name: cfg}, [cfg["decoder"]["lam_team"]], degrade=deg)
        evals[deg] = pd.concat([ref, df], ignore_index=True)
        print(summarize(evals[deg]).to_string())

    # Robust choice: best worst-case mean spellF1 over the degrade modes.
    worst = None
    for deg, df in evals.items():
        s = summarize(df[df["config"] != ref_name])["mean"]
        worst = s if worst is None else np.minimum(worst, s)
    (best_cfg, best_lam), best_val = worst.idxmax(), float(worst.max())
    gap_model = best_cfg.replace("gbdt_gap_", "")
    final = write_gbdt_yaml(table, out, gap_model=gap_model)
    for f in files.values():
        f.unlink()
    chosen = {"gap model": "`{}`".format(gap_model), "lam_team": "`{}`".format(best_lam),
              "worst-case mean spellF1 over the modes": "`{:.4f}`".format(best_val)}
    report = render_report(table, evals, chosen=chosen, reference=ref_name)
    (root / args.report).parent.mkdir(parents=True, exist_ok=True)
    (root / args.report).write_text(report, encoding="utf-8", newline="\n")
    print("wrote {}, {} and {}".format(final, args.model, args.report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DEFAULT_LAM_GRID", "EXCLUDE_FEATURES", "MODEL_PARAMS", "calibrate_team_gbdt", "evaluate_lam_grid",
           "fit_gbdt", "gbdt_feature_params", "match_samples", "render_report", "save_bundle",
           "summarize", "write_gbdt_yaml"]
