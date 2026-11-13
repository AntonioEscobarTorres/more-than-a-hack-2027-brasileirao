"""Decoder sensitivity sweep on external IDSSE masters (task F2-05).

Coordinate-wise grid search over λ, min_dwell, mean-duration scale and post-process
rules, with the pre-registered plateau rule and the robust worst-case check on
``hidelo`` / ``hidehi``. Reused by F2-06.

Usage::

    python -m modeling.scoring.sweep --config configs/v1.yaml \\
        --decoder configs/decoder_idsse.yaml --degrade default
"""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from modeling import paths
from modeling.config import load_config, validate_config
from modeling.scoring.external_eval import evaluate_config, to_markdown

# Grids from F2-05 (no magic numbers outside this module / configs).
LAM_GRID = (0.05, 0.1, 0.2, 0.4, 0.8)
MIN_DWELL_HA = (1, 12, 25)
MIN_DWELL_D = (1, 25, 50)
DURATION_SCALES = (0.5, 1.0, 2.0)
D_BRIDGE_GRID = (25, 50)
MIN_SPELL_GRID = (13, 25)
D_CROSS_GRID = (0, 25, 50, 75)
ROBUST_DEGRADES = ("default", "hidelo", "hidehi")
METRIC = "spell_f1_mean"
SPELL_RATIO = "spell_ratio"


def overlay_decoder(cfg: dict, decoder_section: Mapping) -> dict:
    """Deep copy of ``cfg`` with ``decoder`` replaced by ``decoder_section``."""
    out = copy.deepcopy(cfg)
    out["decoder"] = copy.deepcopy(dict(decoder_section))
    return validate_config(out)


def load_base_config(config_path: str | Path, decoder_path: str | Path | None = None) -> dict:
    """Load an HMM config and optionally overlay a decoder YAML (e.g. ``decoder_idsse``)."""
    cfg = load_config(config_path)
    if decoder_path is None:
        return cfg
    raw = yaml.safe_load(Path(decoder_path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "decoder" not in raw:
        raise ValueError("decoder file must contain a top-level 'decoder' mapping")
    return overlay_decoder(cfg, raw["decoder"])


def with_decoder(cfg: dict, **updates) -> dict:
    """Copy ``cfg`` with selected ``decoder`` fields updated (nested dicts merged)."""
    out = copy.deepcopy(cfg)
    dec = out["decoder"]
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(dec.get(key), dict):
            merged = dict(dec[key])
            merged.update(value)
            dec[key] = merged
        else:
            dec[key] = value
    return validate_config(out)


def with_postprocess(cfg: dict, **updates) -> dict:
    """Copy ``cfg`` with selected ``postprocess`` fields updated."""
    out = copy.deepcopy(cfg)
    out["postprocess"].update(updates)
    return validate_config(out)


def _match_rows(df: pd.DataFrame) -> list:
    return [m for m in df.index if m not in ("mean", "se")]


def spell_ratio_from_frame(df: pd.DataFrame) -> float:
    """Mean over matches of ``n_pred_spells / n_gt_spells`` (NaN-safe)."""
    rows = _match_rows(df)
    if not rows:
        return float("nan")
    gt = df.loc[rows, "n_gt_spells"].to_numpy(np.float64)
    pred = df.loc[rows, "n_pred_spells"].to_numpy(np.float64)
    ratios = np.divide(pred, gt, out=np.full(gt.shape, np.nan), where=gt > 0)
    return float(np.nanmean(ratios))


def summarize(df: pd.DataFrame, metric: str = METRIC) -> dict[str, float]:
    """Mean / SE of ``metric`` plus mean spell ratio."""
    return {
        "mean": float(df.loc["mean", metric]),
        "se": float(df.loc["se", metric]),
        SPELL_RATIO: spell_ratio_from_frame(df),
    }


def evaluate_point(cfg: dict, *, degrade: str = "default", matches=None,
                   metric: str = METRIC) -> dict[str, float]:
    """Run ``evaluate_config`` and return mean / SE / spell ratio."""
    df = evaluate_config(cfg, degrade=degrade, matches=matches)
    return summarize(df, metric=metric)


def _is_close(mean: float, best_mean: float, best_se: float) -> bool:
    """Plateau membership: mean within 1 SE of the best mean (higher is better)."""
    if not math.isfinite(mean) or not math.isfinite(best_mean):
        return False
    se = best_se if math.isfinite(best_se) else 0.0
    return mean >= best_mean - se


def select_plateau(
    points: Sequence[Mapping[str, Any]],
    *,
    mean_key: str = "mean",
    se_key: str = "se",
    neighbour_fn=None,
    tiebreak_keys: Sequence[str] = ("lam_team", "lam_dead", "min_dwell_ha", "min_dwell_d"),
) -> Mapping[str, Any]:
    """Pre-registered plateau rule (F2-05).

    1. Plateau = points with ``mean >= best_mean - best_se``.
    2. Prefer a plateau point whose immediate grid neighbours (via ``neighbour_fn``)
       are also in the plateau.
    3. Else take the best mean.
    4. Ties: smaller values of ``tiebreak_keys`` in order.
    """
    if not points:
        raise ValueError("select_plateau needs at least one point")
    best = max(points, key=lambda p: (p[mean_key],) + tuple(
        -float(p.get(k, 0)) for k in tiebreak_keys))
    best_mean = float(best[mean_key])
    best_se = float(best[se_key])
    plateau = [p for p in points if _is_close(float(p[mean_key]), best_mean, best_se)]
    plateau_ids = {id(p) for p in plateau}

    def sort_key(p):
        return (float(p[mean_key]),) + tuple(-float(p.get(k, 0)) for k in tiebreak_keys)

    if neighbour_fn is not None:
        stable = []
        for p in plateau:
            neighbours = neighbour_fn(p, points)
            if neighbours and all(id(n) in plateau_ids for n in neighbours):
                stable.append(p)
        if stable:
            return max(stable, key=sort_key)
    return max(plateau, key=sort_key)


def select_robust(
    candidates: Sequence[Mapping[str, Any]],
    scores_by_degrade: Mapping[str, Mapping[Any, Mapping[str, float]]],
    *,
    id_key: str = "id",
    mean_key: str = "mean",
    degrades: Sequence[str] = ROBUST_DEGRADES,
) -> Mapping[str, Any]:
    """Among plateau candidates, pick the best worst-case mean across ``degrades``."""
    if not candidates:
        raise ValueError("select_robust needs at least one candidate")

    def worst_case(point):
        pid = point[id_key]
        means = []
        for deg in degrades:
            table = scores_by_degrade[deg]
            means.append(float(table[pid][mean_key]))
        return min(means), -sum(means)  # best worst-case; then better sum as secondary

    return max(candidates, key=worst_case)


def _point_id(tag: str, **coords) -> str:
    parts = [tag] + ["{}={}".format(k, coords[k]) for k in sorted(coords)]
    return "|".join(parts)


def _lam_neighbours(point, points):
    """Immediate neighbours on the square λ grid (same lam_dead or same lam_team)."""
    lt, ld = point["lam_team"], point["lam_dead"]
    out = []
    for p in points:
        if p is point:
            continue
        same_t = p["lam_team"] == lt and p["lam_dead"] != ld
        same_d = p["lam_dead"] == ld and p["lam_team"] != lt
        if not (same_t or same_d):
            continue
        # adjacent in the sorted unique grid along the varying axis
        axis = "lam_dead" if same_t else "lam_team"
        values = sorted({q[axis] for q in points if (
            q["lam_team"] == lt if axis == "lam_dead" else q["lam_dead"] == ld)})
        i = values.index(point[axis])
        if abs(values.index(p[axis]) - i) == 1:
            out.append(p)
    return out


def sweep_lambdas(cfg: dict, *, degrade: str = "default", matches=None,
                  grid: Sequence[float] = LAM_GRID) -> list[dict]:
    """Stage 1: ``lam_team × lam_dead`` grid."""
    points = []
    for lam_team in grid:
        for lam_dead in grid:
            trial = with_decoder(cfg, lam_team=float(lam_team), lam_dead=float(lam_dead))
            stats = evaluate_point(trial, degrade=degrade, matches=matches)
            points.append({
                "id": _point_id("lam", lam_team=lam_team, lam_dead=lam_dead),
                "stage": "lam",
                "lam_team": float(lam_team),
                "lam_dead": float(lam_dead),
                "cfg": trial,
                **stats,
            })
    return points


def sweep_min_dwell(cfg: dict, *, degrade: str = "default", matches=None,
                    ha_grid: Sequence[int] = MIN_DWELL_HA,
                    d_grid: Sequence[int] = MIN_DWELL_D) -> list[dict]:
    """Stage 2: min_dwell H/A × D on a fixed λ config."""
    points = []
    for ha in ha_grid:
        for d in d_grid:
            trial = with_decoder(cfg, min_dwell_frames={"H": int(ha), "A": int(ha), "D": int(d)})
            stats = evaluate_point(trial, degrade=degrade, matches=matches)
            points.append({
                "id": _point_id("dwell", min_dwell_ha=ha, min_dwell_d=d),
                "stage": "min_dwell",
                "lam_team": float(cfg["decoder"]["lam_team"]),
                "lam_dead": float(cfg["decoder"]["lam_dead"]),
                "min_dwell_ha": int(ha),
                "min_dwell_d": int(d),
                "cfg": trial,
                **stats,
            })
    return points


def _dwell_neighbours(point, points):
    ha, d = point["min_dwell_ha"], point["min_dwell_d"]
    out = []
    for p in points:
        if p is point:
            continue
        if p["min_dwell_ha"] == ha and p["min_dwell_d"] != d:
            values = sorted({q["min_dwell_d"] for q in points if q["min_dwell_ha"] == ha})
            if abs(values.index(d) - values.index(p["min_dwell_d"])) == 1:
                out.append(p)
        elif p["min_dwell_d"] == d and p["min_dwell_ha"] != ha:
            values = sorted({q["min_dwell_ha"] for q in points if q["min_dwell_d"] == d})
            if abs(values.index(ha) - values.index(p["min_dwell_ha"])) == 1:
                out.append(p)
    return out


def sweep_duration_scale(cfg: dict, *, degrade: str = "default", matches=None,
                         scales: Sequence[float] = DURATION_SCALES) -> list[dict]:
    """Stage 3: scale ``mean_duration_s`` by a factor (base durations from ``cfg``)."""
    base = cfg["decoder"]["mean_duration_s"]
    points = []
    for scale in scales:
        scaled = {s: float(base[s]) * float(scale) for s in base}
        trial = with_decoder(cfg, mean_duration_s=scaled)
        stats = evaluate_point(trial, degrade=degrade, matches=matches)
        points.append({
            "id": _point_id("scale", duration_scale=scale),
            "stage": "duration_scale",
            "duration_scale": float(scale),
            "lam_team": float(cfg["decoder"]["lam_team"]),
            "lam_dead": float(cfg["decoder"]["lam_dead"]),
            "cfg": trial,
            **stats,
        })
    return points


def _scale_neighbours(point, points):
    values = sorted({p["duration_scale"] for p in points})
    i = values.index(point["duration_scale"])
    out = []
    for p in points:
        if p is point:
            continue
        if abs(values.index(p["duration_scale"]) - i) == 1:
            out.append(p)
    return out


def sweep_postprocess(cfg: dict, *, degrade: str = "default", matches=None,
                      include_d_cross: bool = True) -> list[dict]:
    """Stage 4: each post-process rule on in isolation (others off)."""
    base_post = {
        "midpoint_neutral_switch": False,
        "boundary_shift_file": None,
        "d_bridge_frames": 0,
        "d_cross_frames": 0,
        "min_spell_frames": 0,
    }
    trials = [("post|off", dict(base_post))]
    trials.append(("post|midpoint", {**base_post, "midpoint_neutral_switch": True}))
    for n in D_BRIDGE_GRID:
        trials.append(("post|d_bridge={}".format(n), {**base_post, "d_bridge_frames": int(n)}))
    for n in MIN_SPELL_GRID:
        trials.append(("post|min_spell={}".format(n), {**base_post, "min_spell_frames": int(n)}))
    if include_d_cross:
        for n in D_CROSS_GRID:
            if n == 0:
                continue  # already covered by "off"
            trials.append(("post|d_cross={}".format(n), {**base_post, "d_cross_frames": int(n)}))

    points = []
    for pid, post in trials:
        trial = with_postprocess(cfg, **post)
        stats = evaluate_point(trial, degrade=degrade, matches=matches)
        points.append({
            "id": pid,
            "stage": "postprocess",
            "postprocess": post,
            "lam_team": float(cfg["decoder"]["lam_team"]),
            "lam_dead": float(cfg["decoder"]["lam_dead"]),
            "cfg": trial,
            **stats,
        })
    return points


def plateau_candidates(points: Sequence[Mapping[str, Any]], selected: Mapping[str, Any],
                       *, mean_key: str = "mean", se_key: str = "se") -> list:
    """All points within 1 SE of the stage-best (for the robust cross-check)."""
    best_mean = float(selected[mean_key])
    best_se = float(selected[se_key])
    return [p for p in points if _is_close(float(p[mean_key]), best_mean, best_se)]


def robustify_stage(
    points: Sequence[Mapping[str, Any]],
    selected: Mapping[str, Any],
    *,
    evaluate_fn,
    degrades: Sequence[str] = ROBUST_DEGRADES,
) -> Mapping[str, Any]:
    """Re-score plateau candidates on every degrade and pick best worst-case.

    ``evaluate_fn(cfg, degrade) -> {mean, se, ...}``. The ``default`` scores already
    on ``points`` are reused; other degrades are evaluated on demand.
    """
    candidates = plateau_candidates(points, selected)
    scores: dict[str, dict[Any, dict]] = {"default": {p["id"]: p for p in points}}
    for deg in degrades:
        if deg == "default":
            continue
        scores[deg] = {}
        for p in candidates:
            scores[deg][p["id"]] = evaluate_fn(p["cfg"], deg)
    # Ensure every degrade table has every candidate id for select_robust.
    for deg in degrades:
        scores.setdefault(deg, {})
        for p in candidates:
            if p["id"] not in scores[deg]:
                scores[deg][p["id"]] = {k: p[k] for k in ("mean", "se", SPELL_RATIO)
                                        if k in p}
    chosen = select_robust(candidates, scores, degrades=degrades)
    # Attach worst-case table for the report.
    table = {}
    for deg in degrades:
        table[deg] = scores[deg][chosen["id"]]
    out = dict(chosen)
    out["robust_scores"] = table
    out["worst_case_mean"] = min(float(table[d]["mean"]) for d in degrades)
    return out


def run_staged_sweep(
    cfg: dict,
    *,
    degrade: str = "default",
    matches=None,
    robust: bool = True,
    include_d_cross: bool = True,
    evaluate_fn=None,
) -> dict:
    """Run stages 1–4; return history, the chosen config, and per-stage tables.

    ``evaluate_fn(cfg, degrade)`` defaults to :func:`evaluate_point`.
    """
    if evaluate_fn is None:
        def evaluate_fn(c, deg):  # noqa: F811 — default binding
            return evaluate_point(c, degrade=deg, matches=matches)

    history: dict[str, Any] = {"stages": []}
    current = copy.deepcopy(cfg)

    # Stage 1: λ
    lam_points = sweep_lambdas(current, degrade=degrade, matches=matches)
    lam_pick = select_plateau(lam_points, neighbour_fn=_lam_neighbours)
    if robust:
        lam_pick = robustify_stage(lam_points, lam_pick, evaluate_fn=evaluate_fn)
    current = lam_pick["cfg"]
    history["stages"].append({"name": "lam", "points": lam_points, "selected": lam_pick})

    # Stage 2: min_dwell
    dwell_points = sweep_min_dwell(current, degrade=degrade, matches=matches)
    dwell_pick = select_plateau(
        dwell_points, neighbour_fn=_dwell_neighbours,
        tiebreak_keys=("min_dwell_ha", "min_dwell_d", "lam_team", "lam_dead"))
    if robust:
        dwell_pick = robustify_stage(dwell_points, dwell_pick, evaluate_fn=evaluate_fn)
    current = dwell_pick["cfg"]
    history["stages"].append({"name": "min_dwell", "points": dwell_points, "selected": dwell_pick})

    # Stage 3: duration scale
    scale_points = sweep_duration_scale(current, degrade=degrade, matches=matches)
    scale_pick = select_plateau(
        scale_points, neighbour_fn=_scale_neighbours,
        tiebreak_keys=("duration_scale", "lam_team", "lam_dead"))
    if robust:
        scale_pick = robustify_stage(scale_points, scale_pick, evaluate_fn=evaluate_fn)
    current = scale_pick["cfg"]
    history["stages"].append({
        "name": "duration_scale", "points": scale_points, "selected": scale_pick})

    # Stage 4: post-process (isolated); keep the best isolated rule if it beats "off"
    post_points = sweep_postprocess(
        current, degrade=degrade, matches=matches, include_d_cross=include_d_cross)
    post_pick = select_plateau(
        post_points, neighbour_fn=None,
        tiebreak_keys=("lam_team", "lam_dead"))
    if robust:
        post_pick = robustify_stage(post_points, post_pick, evaluate_fn=evaluate_fn)
    current = post_pick["cfg"]
    history["stages"].append({"name": "postprocess", "points": post_points, "selected": post_pick})

    history["selected_cfg"] = current
    return history


def points_to_frame(points: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Flatten sweep points to a table (drops the heavy ``cfg`` column)."""
    rows = []
    for p in points:
        row = {k: v for k, v in p.items() if k not in ("cfg", "postprocess", "robust_scores")}
        rows.append(row)
    return pd.DataFrame(rows)


def render_stage_markdown(stage: Mapping[str, Any]) -> str:
    """Markdown section for one sweep stage."""
    name = stage["name"]
    selected = stage["selected"]
    df = points_to_frame(stage["points"])
    lines = [
        "## Stage: {}".format(name),
        "",
        "- selected: `{}`".format(selected.get("id")),
        "- spellF1 mean±SE: {:.4f}±{:.4f}".format(
            float(selected["mean"]), float(selected["se"])),
        "- spell ratio (pred/gt): {:.4f}".format(float(selected.get(SPELL_RATIO, float("nan")))),
    ]
    if "worst_case_mean" in selected:
        lines.append("- worst-case mean across {}: {:.4f}".format(
            ",".join(ROBUST_DEGRADES), float(selected["worst_case_mean"])))
        if "robust_scores" in selected:
            for deg, stats in selected["robust_scores"].items():
                lines.append("  - {}: {:.4f}±{:.4f} (ratio {:.4f})".format(
                    deg, float(stats["mean"]), float(stats["se"]),
                    float(stats.get(SPELL_RATIO, float("nan")))))
    lines.extend(["", to_markdown(df.set_index("id")), ""])
    return "\n".join(lines)


def write_v1s(cfg: dict, path: str | Path) -> Path:
    """Write the selected config as ``v1s`` (name forced)."""
    out = copy.deepcopy(cfg)
    out["name"] = "v1s"
    out = validate_config(out)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(out, sort_keys=False), encoding="utf-8")
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Decoder sensitivity sweep (F2-05)")
    parser.add_argument("--config", default="configs/v1.yaml")
    parser.add_argument("--decoder", default="configs/decoder_idsse.yaml",
                        help="decoder section overlay (IDSSE stats)")
    parser.add_argument("--degrade", default="default")
    parser.add_argument("--no-robust", action="store_true",
                        help="skip hidelo/hidehi worst-case selection")
    parser.add_argument("--no-d-cross", action="store_true")
    parser.add_argument("--report", default="reports/sensitivity_decoder.md")
    parser.add_argument("--v1s", default="configs/v1s.yaml")
    parser.add_argument("--skip-write", action="store_true")
    args = parser.parse_args(argv)

    root = paths.repo_root()
    cfg = load_base_config(root / args.config, root / args.decoder)
    history = run_staged_sweep(
        cfg, degrade=args.degrade, robust=not args.no_robust,
        include_d_cross=not args.no_d_cross)

    body = [
        "# Decoder sensitivity sweep (F2-05)",
        "",
        "- base config: `{}`".format(args.config),
        "- decoder overlay: `{}`".format(args.decoder),
        "- degrade (primary): `{}`".format(args.degrade),
        "- robust: {}".format(not args.no_robust),
        "",
    ]
    for stage in history["stages"]:
        body.append(render_stage_markdown(stage))
    report = "\n".join(body)

    if not args.skip_write:
        report_path = root / args.report
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8", newline="\n")
        write_v1s(history["selected_cfg"], root / args.v1s)
        print("wrote {}".format(report_path))
        print("wrote {}".format(root / args.v1s))
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "D_BRIDGE_GRID",
    "D_CROSS_GRID",
    "DURATION_SCALES",
    "LAM_GRID",
    "METRIC",
    "MIN_DWELL_D",
    "MIN_DWELL_HA",
    "MIN_SPELL_GRID",
    "ROBUST_DEGRADES",
    "SPELL_RATIO",
    "evaluate_point",
    "load_base_config",
    "overlay_decoder",
    "plateau_candidates",
    "points_to_frame",
    "render_stage_markdown",
    "robustify_stage",
    "run_staged_sweep",
    "select_plateau",
    "select_robust",
    "spell_ratio_from_frame",
    "summarize",
    "sweep_duration_scale",
    "sweep_lambdas",
    "sweep_min_dwell",
    "sweep_postprocess",
    "with_decoder",
    "with_postprocess",
    "write_v1s",
]
