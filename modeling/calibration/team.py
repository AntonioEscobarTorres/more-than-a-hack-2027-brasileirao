"""Team-logit calibration on degraded external IDSSE matches.

Fits two antisymmetric L2 logistic regressions without intercept:
- ``ball``: valid-ball alive frames (f1..f8, plus optional f9 touch and f10/f11 pass bridge)
- ``gap``: invalid-ball alive frames (ev_coll_adv, ev_coll_spread, ev_coll_speed, plus optional f10/f11)

Never calibrates on ``degrade="none"`` (clean IDSSE freezes the ball on dead balls and leaks the label).

Touch thresholds (``r_touch_m``, ``touch_min_frames``, ``touch_acc_mps2``,
``touch_dir_change_rad``) are chosen on the same degraded matches by the mean leave-one-match-out
log-loss of the ``touch`` ball model over ``TOUCH_GRID`` (F1-05 Update 4), never on BAR-ATH.

Usage::

    python -m modeling.calibration.team --config configs/v1.yaml \\
        --decoder configs/decoder_idsse.yaml --degrade default --variant touch \\
        --out configs/calib/team.yaml --report reports/calibration_team.md
"""
from __future__ import annotations

import argparse
import copy
import itertools
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from evaluation import possession_metrics as pm
from modeling import paths
from modeling.calibration.common import (
    DEFAULT_C_GRID,
    SUBSAMPLE_STEP,
    alive_frames_mask,
    binary_auc,
    binary_log_loss,
    linear_logit,
    lomo_log_loss_by_c,
    possession_boundary_shift,
    select_c_and_fit_folds,
    subsample_indices,
)
from modeling.config import load_config
from modeling.evidence.team import (
    BALL_BASE_FEATURES,
    DEFAULT_BRIDGE_MEDIAN_FRAMES,
    DEFAULT_BRIDGE_TAUS,
    DEFAULT_R_TOUCH_M,
    DEFAULT_TOUCH_ACC_MPS2,
    DEFAULT_TOUCH_DIR_CHANGE_RAD,
    DEFAULT_TOUCH_MIN_FRAMES,
    DEFAULT_TOUCH_WIN_FRAMES,
    GAP_BASE_FEATURES,
    build_team_calibration_features,
    build_touch_feature,
)
from modeling.external.files import IDSSE_MATCHES, load_external
from modeling.pipeline import predict_labels
from modeling.scoring.metrics import score

VARIANT_NAMES = ("base", "bridge", "touch", "bridge_touch")
DEFAULT_VARIANT = "touch"
EXPECTED_POSITIVE_FEATURES = ("f1", "f2", "f3", "f4", "f5", "f6")

# Touch-threshold grid searched by ``select_touch_params``; the spec start values
# (2.0 m, 3 frames, 8 m/s^2, 0.5 rad) are in the middle of each axis.
TOUCH_GRID = {
    "r_touch_m": (1.5, 2.0, 3.0),
    "touch_min_frames": (2, 3, 5),
    "touch_acc_mps2": (5.0, 8.0, 12.0),
    "touch_dir_change_rad": (0.3, 0.5, 0.8),
}
_TOUCH_INPUT_COLUMNS = ("half", "d_diff", "d_near_home", "d_near_away",
                        "ball_dir_change", "ball_acc")


def variant_feature_names(
    variant: str,
    taus: Sequence[int | float] = DEFAULT_BRIDGE_TAUS,
) -> tuple[list[str], list[str]]:
    """Return ``(ball_features, gap_features)`` for one of the four F1-05 variants."""
    if variant not in VARIANT_NAMES:
        raise ValueError("unknown variant {!r}; expected one of {}".format(variant, VARIANT_NAMES))

    ball_cols = list(BALL_BASE_FEATURES)
    gap_cols = list(GAP_BASE_FEATURES)

    if variant in ("touch", "bridge_touch"):
        ball_cols.append("f9")

    if variant in ("bridge", "bridge_touch"):
        for tau in taus:
            tau_label = int(tau) if float(tau).is_integer() else tau
            ball_cols.extend(["f10_{}".format(tau_label), "f11_{}".format(tau_label)])
            gap_cols.extend(["f10_{}".format(tau_label), "f11_{}".format(tau_label)])

    return ball_cols, gap_cols


def _design_matrix(df: pd.DataFrame, rows: np.ndarray, columns: Sequence[str]) -> np.ndarray:
    """Extract float64 design matrix on ``rows`` with NaN replaced by 0.0 (rule 4)."""
    sub = df.iloc[rows]
    mats = []
    for col in columns:
        arr = sub[col].to_numpy(dtype=np.float64)
        mats.append(np.where(np.isfinite(arr), arr, 0.0))
    return np.column_stack(mats) if mats else np.zeros((len(rows), 0), dtype=np.float64)


def default_feature_params(team_params: Mapping) -> dict:
    """Feature hyperparameters recorded in the calibration table."""
    return {
        "bridge_median_frames": int(team_params.get("bridge_median_frames", DEFAULT_BRIDGE_MEDIAN_FRAMES)),
        "bridge_tau_frames": [int(t) for t in team_params.get("bridge_tau_frames", DEFAULT_BRIDGE_TAUS)],
        "r_touch_m": float(team_params.get("r_touch_m", DEFAULT_R_TOUCH_M)),
        "touch_min_frames": int(team_params.get("touch_min_frames", DEFAULT_TOUCH_MIN_FRAMES)),
        "touch_win_frames": int(team_params.get("touch_win_frames", DEFAULT_TOUCH_WIN_FRAMES)),
        "touch_dir_change_rad": float(team_params.get("touch_dir_change_rad", DEFAULT_TOUCH_DIR_CHANGE_RAD)),
        "touch_acc_mps2": float(team_params.get("touch_acc_mps2", DEFAULT_TOUCH_ACC_MPS2)),
    }


def select_touch_params(
    cfg: dict,
    *,
    degrade: str = "default",
    matches: Sequence[str] | None = None,
    step: int = SUBSAMPLE_STEP,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    grid: Mapping[str, Sequence] | None = None,
    load_fn=None,
) -> dict:
    """Choose the touch thresholds by mean leave-one-match-out log-loss (F1-05 Update 4).

    For every point of ``grid`` the ``touch`` ball model (f1..f8 + f9) is scored by its mean
    leave-one-match-out log-loss, minimised over ``c_grid``. The lowest loss wins; ties keep the
    first point in grid order. Returns ``{"selected", "grid", "results"}``.
    """
    if degrade == "none":
        raise ValueError("selecting touch thresholds on degrade='none' is forbidden (label leak)")
    grid = dict(TOUCH_GRID if grid is None else grid)
    keys = list(TOUCH_GRID)
    if sorted(grid) != sorted(keys):
        raise ValueError("touch grid must have exactly the keys {}".format(keys))
    matches = list(IDSSE_MATCHES if matches is None else matches)
    if len(matches) < 2:
        raise ValueError("need at least 2 matches to select touch thresholds")
    if load_fn is None:
        load_fn = load_external

    team_params = dict(cfg["team"]["params"])
    base_cols = list(BALL_BASE_FEATURES)
    per_match = {}
    for match in matches:
        master, gt = load_fn(match, degrade)
        half = master["half"].to_numpy()
        feats = build_team_calibration_features(
            master, team_params, include_bridge=False, include_touch=False)
        idx = subsample_indices(alive_frames_mask(gt, half), step=step)
        ball_idx = idx[feats["ev_ball_valid"].to_numpy()[idx]]
        y_ball = (gt[ball_idx] == pm.H).astype(np.int8)
        X_base = _design_matrix(feats, ball_idx, base_cols)
        inputs = master[[c for c in _TOUCH_INPUT_COLUMNS if c in master.columns]].copy()
        per_match[match] = (inputs, feats[["ev_ball_valid"]].copy(), ball_idx, X_base, y_ball)
        del master, feats

    results = []
    for values in itertools.product(*(grid[k] for k in keys)):
        combo = dict(zip(keys, values))
        params = {**team_params, **combo}
        data = {}
        for match, (inputs, ev_valid, ball_idx, X_base, y_ball) in per_match.items():
            f9 = build_touch_feature(inputs, params, ev=ev_valid).to_numpy(dtype=np.float64)
            f9 = f9[ball_idx]
            data[match] = (np.column_stack([X_base, np.where(np.isfinite(f9), f9, 0.0)]), y_ball)
        losses = lomo_log_loss_by_c(data, c_grid=c_grid)
        results.append({**combo, "log_loss": float(min(losses.values()))})

    best = min(results, key=lambda r: r["log_loss"])
    selected = {
        "r_touch_m": float(best["r_touch_m"]),
        "touch_min_frames": int(best["touch_min_frames"]),
        "touch_acc_mps2": float(best["touch_acc_mps2"]),
        "touch_dir_change_rad": float(best["touch_dir_change_rad"]),
    }
    return {"selected": selected, "grid": {k: list(grid[k]) for k in keys}, "results": results}


def calibrate_team(
    cfg: dict,
    *,
    variant: str = DEFAULT_VARIANT,
    degrade: str = "default",
    matches: Sequence[str] | None = None,
    step: int = SUBSAMPLE_STEP,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    touch_params: Mapping | None = None,
    load_fn=None,
) -> dict:
    """Fit ``ball`` and ``gap`` L2 logistic models with leave-one-match-out CV.

    ``touch_params`` (for example ``select_touch_params(...)["selected"]``) overrides the touch
    thresholds of ``cfg`` and is recorded in ``feature_params``.
    """
    if degrade == "none":
        raise ValueError("calibrating team_logit on degrade='none' is forbidden (label leak)")
    matches = list(IDSSE_MATCHES if matches is None else matches)
    if len(matches) < 2:
        raise ValueError("need at least 2 matches to calibrate team_logit")
    if load_fn is None:
        load_fn = load_external

    team_params = cfg["team"]["params"]
    f_params = default_feature_params({**team_params, **dict(touch_params or {})})
    taus = f_params["bridge_tau_frames"]
    ball_cols, gap_cols = variant_feature_names(variant, taus=taus)
    need_bridge = variant in ("bridge", "bridge_touch")
    need_touch = variant in ("touch", "bridge_touch")

    ball_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    gap_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    per_match_eval: dict[str, dict] = {}

    for match in matches:
        master, gt = load_fn(match, degrade)
        half = master["half"].to_numpy()
        feats = build_team_calibration_features(
            master, {**team_params, **f_params},
            include_bridge=need_bridge, include_touch=need_touch, taus=taus)

        alive = alive_frames_mask(gt, half)
        idx = subsample_indices(alive, step=step)
        valid_sub = feats["ev_ball_valid"].to_numpy()[idx]
        y_sub = (gt[idx] == pm.H).astype(np.int8)

        ball_idx = idx[valid_sub]
        gap_idx = idx[~valid_sub]
        y_ball = y_sub[valid_sub]
        y_gap = y_sub[~valid_sub]

        X_ball = _design_matrix(feats, ball_idx, ball_cols)
        X_gap = _design_matrix(feats, gap_idx, gap_cols)
        ball_data[match] = (X_ball, y_ball)
        gap_data[match] = (X_gap, y_gap)
        per_match_eval[match] = {
            "X_ball": X_ball, "y_ball": y_ball,
            "X_gap": X_gap, "y_gap": y_gap,
        }

    ball_fit = select_c_and_fit_folds(ball_data, ball_cols, c_grid=c_grid, fit_intercept=False)
    gap_fit = select_c_and_fit_folds(gap_data, gap_cols, c_grid=c_grid, fit_intercept=False)

    # Combined out-of-fold log-loss and AUC per match
    combined_folds = {}
    for match in matches:
        ev_m = per_match_eval[match]
        b_block = ball_fit["folds"][match]
        g_block = gap_fit["folds"][match]
        wb = np.array([b_block["weights"][c] for c in ball_cols], dtype=np.float64)
        sb = np.array([b_block["scales"][c] for c in ball_cols], dtype=np.float64)
        wg = np.array([g_block["weights"][c] for c in gap_cols], dtype=np.float64)
        sg = np.array([g_block["scales"][c] for c in gap_cols], dtype=np.float64)

        z_ball = linear_logit(ev_m["X_ball"], wb, sb)
        z_gap = linear_logit(ev_m["X_gap"], wg, sg)
        y_comb = np.concatenate([ev_m["y_ball"], ev_m["y_gap"]])
        z_comb = np.concatenate([z_ball, z_gap])
        combined_folds[match] = {
            "log_loss": binary_log_loss(y_comb, z_comb),
            "auc": binary_auc(y_comb, z_comb),
            "ball_log_loss": ball_fit["fold_metrics"][match]["log_loss"],
            "ball_auc": ball_fit["fold_metrics"][match]["auc"],
            "gap_log_loss": gap_fit["fold_metrics"][match]["log_loss"],
            "gap_auc": gap_fit["fold_metrics"][match]["auc"],
        }

    neg_signs = [f for f in EXPECTED_POSITIVE_FEATURES
                 if f in ball_fit["all"]["weights"] and ball_fit["all"]["weights"][f] <= 0.0]

    return {
        "variant": variant,
        "feature_params": f_params,
        "ball": {
            "C": ball_fit["C"],
            "features": ball_fit["features"],
            "all": ball_fit["all"],
            "folds": ball_fit["folds"],
        },
        "gap": {
            "C": gap_fit["C"],
            "features": gap_fit["features"],
            "all": gap_fit["all"],
            "folds": gap_fit["folds"],
        },
        "meta": {
            "degrade": degrade,
            "subsample_step": int(step),
            "matches": matches,
            "negative_expected_positive_features": neg_signs,
            "fold_metrics": combined_folds,
        },
    }


def write_team_yaml(table: Mapping, path: str | Path) -> Path:
    """Write ``configs/calib/team.yaml`` in deterministic key order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "variant": str(table["variant"]),
        "feature_params": copy.deepcopy(dict(table["feature_params"])),
        "ball": copy.deepcopy(dict(table["ball"])),
        "gap": copy.deepcopy(dict(table["gap"])),
    }
    if "meta" in table:
        payload["meta"] = copy.deepcopy(dict(table["meta"]))
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8", newline="\n")
    return path


def evaluate_calibrated_table(
    base_cfg: dict,
    calib_path: str | Path,
    *,
    decoder_cfg: Mapping | None = None,
    degrade: str = "default",
    matches: Sequence[str] | None = None,
    load_fn=None,
) -> pd.DataFrame:
    """Evaluate a fitted calibration table on ``degrade`` with ``predict_labels``."""
    matches = list(IDSSE_MATCHES if matches is None else matches)
    if load_fn is None:
        load_fn = load_external

    cfg = copy.deepcopy(base_cfg)
    cfg["team"]["kind"] = "calibrated"
    cfg["team"]["calibrated_file"] = str(calib_path)
    if decoder_cfg is not None:
        dec = decoder_cfg.get("decoder", decoder_cfg)
        cfg["decoder"] = copy.deepcopy(dict(dec))

    rows = []
    for match in matches:
        master, gt = load_fn(match, degrade)
        pred = predict_labels(master, cfg, holdout=match)
        pred_codes = pm.label_codes(pred)
        m = score(gt, pred_codes)
        n_gt = m["n_gt_spells"]
        m["spell_count_ratio"] = float(m["n_pred_spells"]) / float(n_gt) if n_gt > 0 else float("nan")
        m["boundary_shift_frames"] = possession_boundary_shift(gt, pred_codes)
        rows.append(m)

    df = pd.DataFrame(rows, index=pd.Index(matches, name="match"))
    n = len(df)
    mean_row = df.mean()
    se_row = df.std(ddof=1) / np.sqrt(n) if n > 1 else df.mean() * float("nan")
    summary = pd.DataFrame([mean_row, se_row], index=["mean", "se"])
    return pd.concat([df.astype(np.float64), summary])


def _render_touch_selection(selection: Mapping, top: int = 5) -> list[str]:
    keys = list(selection["grid"])
    ranked = sorted(selection["results"], key=lambda r: r["log_loss"])
    lines = [
        "## Touch thresholds (leave-one-match-out log-loss of the `touch` ball model)",
        "",
        "- grid: {}".format(", ".join("`{}` {}".format(k, selection["grid"][k]) for k in keys)),
        "- selected: {}".format(", ".join(
            "`{}` = `{}`".format(k, selection["selected"][k]) for k in keys)),
        "",
        "| Rank | " + " | ".join("`{}`".format(k) for k in keys) + " | LOMO log-loss |",
        "|---:|" + "---:|" * len(keys) + "---:|",
    ]
    for rank, row in enumerate(ranked[:top], start=1):
        lines.append("| {} | ".format(rank) + " | ".join(str(row[k]) for k in keys)
                     + " | {:.5f} |".format(row["log_loss"]))
    lines.append("")
    return lines


def render_calibration_report(
    tables: Mapping[str, Mapping],
    evals_by_variant: Mapping[str, Mapping[str, pd.DataFrame]] | None = None,
    *,
    touch_selection: Mapping | None = None,
    written_variant: str | None = None,
) -> str:
    """Render ``reports/calibration_team.md`` covering all four F1-05 candidate variants."""
    lines = ["# Team Logit Calibration Report (F1-05)", ""]
    if written_variant is not None:
        lines.extend(["- variant written to the calibration file: `{}`".format(written_variant), ""])
    if touch_selection is not None:
        lines.extend(_render_touch_selection(touch_selection))
    for variant, table in tables.items():
        lines.extend([
            "## Variant: `{}`".format(variant),
            "",
            "- ball `C`: `{}`".format(table["ball"]["C"]),
            "- gap `C`: `{}`".format(table["gap"]["C"]),
            "- negative signs among f1..f6: `{}`".format(
                table["meta"].get("negative_expected_positive_features", [])),
            "",
            "### Coefficients (`all`)",
            "",
            "| Model | Feature | Scale | Weight |",
            "|:---|:---|---:|---:|",
        ])
        for section_name in ("ball", "gap"):
            sec = table[section_name]
            for feat in sec["features"]:
                lines.append("| {} | `{}` | {:.4f} | {:+.4f} |".format(
                    section_name, feat, sec["all"]["scales"][feat], sec["all"]["weights"][feat]))
        lines.extend(["", "### Out-of-fold Log-Loss and AUC", "",
                      "| Match | Log-Loss | AUC | Ball Log-Loss | Ball AUC | Gap Log-Loss | Gap AUC |",
                      "|:---|---:|---:|---:|---:|---:|---:|"])
        fold_m = table["meta"]["fold_metrics"]
        for match, m in fold_m.items():
            lines.append("| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                match, m["log_loss"], m["auc"], m["ball_log_loss"], m["ball_auc"],
                m["gap_log_loss"], m["gap_auc"]))
        lines.append("")

        if evals_by_variant and variant in evals_by_variant:
            for deg, df in evals_by_variant[variant].items():
                lines.extend([
                    "### Decoder Evaluation (`degrade={}`)".format(deg),
                    "",
                    "- mean `spell_f1_mean`: `{:.4f} ± {:.4f}`".format(
                        float(df.loc["mean", "spell_f1_mean"]), float(df.loc["se", "spell_f1_mean"])),
                    "- mean `spell_count_ratio`: `{:.4f}`".format(
                        float(df.loc["mean", "spell_count_ratio"])),
                    "- mean `boundary_shift_frames`: `{:+.2f}`".format(
                        float(df.loc["mean", "boundary_shift_frames"])),
                    "",
                    "| Match | spell_f1_mean | Count Ratio (Pred/GT) | Boundary Shift (frames) |",
                    "|:---|---:|---:|---:|",
                ])
                for match in df.index:
                    if match in ("mean", "se"):
                        continue
                    lines.append("| {} | {:.4f} | {:.4f} | {:+.2f} |".format(
                        match, float(df.loc[match, "spell_f1_mean"]),
                        float(df.loc[match, "spell_count_ratio"]),
                        float(df.loc[match, "boundary_shift_frames"])))
                lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate team_logit on degraded IDSSE (F1-05)")
    parser.add_argument("--config", default="configs/v1.yaml")
    parser.add_argument("--decoder", default="configs/decoder_idsse.yaml")
    parser.add_argument("--degrade", default="default")
    parser.add_argument("--variant", default=DEFAULT_VARIANT, choices=VARIANT_NAMES,
                        help="variant written to --out (all 4 are evaluated for the report)")
    parser.add_argument("--out", default="configs/calib/team.yaml")
    parser.add_argument("--report", default="reports/calibration_team.md")
    parser.add_argument("--extra-degrades", nargs="*", default=["hidelo", "hidehi"])
    parser.add_argument("--no-select-touch", action="store_true",
                        help="keep the touch thresholds of --config instead of searching TOUCH_GRID")
    args = parser.parse_args(argv)

    root = paths.repo_root()
    cfg = load_config(root / args.config)
    with open(root / args.decoder, encoding="utf-8") as handle:
        decoder_cfg = yaml.safe_load(handle)

    touch_selection = None if args.no_select_touch else select_touch_params(
        cfg, degrade=args.degrade)
    touch_params = touch_selection["selected"] if touch_selection is not None else None

    tables = {}
    evals = {}
    with tempfile.TemporaryDirectory() as tmp:
        for var in VARIANT_NAMES:
            table = calibrate_team(cfg, variant=var, degrade=args.degrade,
                                   touch_params=touch_params)
            tables[var] = table
            var_path = write_team_yaml(table, Path(tmp) / "team_{}.yaml".format(var))
            evals[var] = {
                args.degrade: evaluate_calibrated_table(
                    cfg, var_path, decoder_cfg=decoder_cfg, degrade=args.degrade),
            }
            for extra_deg in args.extra_degrades:
                try:
                    evals[var][extra_deg] = evaluate_calibrated_table(
                        cfg, var_path, decoder_cfg=decoder_cfg, degrade=extra_deg)
                except FileNotFoundError:
                    pass

    out_path = write_team_yaml(tables[args.variant], root / args.out)
    report_text = render_calibration_report(
        tables, evals, touch_selection=touch_selection, written_variant=args.variant)
    report_path = root / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8", newline="\n")
    print("wrote {} and {}".format(out_path, report_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_VARIANT",
    "EXPECTED_POSITIVE_FEATURES",
    "TOUCH_GRID",
    "VARIANT_NAMES",
    "calibrate_team",
    "default_feature_params",
    "evaluate_calibrated_table",
    "render_calibration_report",
    "select_touch_params",
    "variant_feature_names",
    "write_team_yaml",
]