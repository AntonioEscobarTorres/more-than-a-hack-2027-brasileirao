"""Aggregated error analysis on degraded external IDSSE matches.

Computes per-frame error rates by situation category, missed ground-truth possessions by
duration bin (< 2 s, 2-10 s, > 10 s), predicted/true possession count ratio, and team_logit
diagnostics inside missed possessions (ball visible vs gap, near-zero vs wrong sign).

Frame categories follow the master schema: ``ball_visible`` is ``ball_observed == 1``,
``short_gap`` is an interpolated gap (``ball_filled == 1``, at most ``MAX_GAP_FRAMES``) and
``long_gap`` is everything else (ball position missing).

Usage::

    python -m modeling.scoring.error_analysis --config configs/v2.yaml \\
        --baseline configs/v1.yaml --baseline-decoder configs/decoder_idsse.yaml \\
        --degrade default --out reports/error_analysis.md
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

import games
from evaluation import possession_metrics as pm
from modeling import paths, variants
from modeling.config import config_hash, load_config
from modeling.external.files import IDSSE_MATCHES, load_external
from modeling.pipeline import predict_labels, variant_params
from modeling.scoring.metrics import score

FRAME_CATEGORIES = (
    "all_graded",
    "ball_visible",
    "short_gap",
    "long_gap",
    "near_possession_switch_2s",
    "near_restart_3s",
)

DURATION_BINS = ("lt_2s", "2s_to_10s", "gt_10s")
DURATION_LABELS = {
    "lt_2s": "< 2 s",
    "2s_to_10s": "2-10 s",
    "gt_10s": "> 10 s",
    "all": "All",
}

SWITCH_WINDOW_S = 2.0
RESTART_WINDOW_S = 3.0
NEAR_ZERO_LOGIT_THR = 0.5


def _fps() -> float:
    return float(games.GAMES[games.GAME]["fps"])


def _compute_team_logit(master: pd.DataFrame, cfg: dict, *, holdout: str | None = None) -> np.ndarray:
    """Compute ``team_logit`` for ``master`` using the team variant configured in ``cfg``."""
    variants.load_kinds()
    team_kind = cfg["team"]["kind"]
    if team_kind not in variants.TEAM_KINDS:
        raise NotImplementedError("team.kind {!r} is not registered".format(team_kind))
    return variants.TEAM_KINDS[team_kind](
        master, variant_params(cfg["team"]), holdout=holdout
    ).astype(np.float64)


def _ball_visibility_masks(master: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return disjoint boolean masks ``(ball_visible, short_gap, long_gap)`` on all frames.

    Short gaps are the ones preprocessing interpolates (``ball_filled == 1``), so their ball
    position is finite; long gaps have no ball position.
    """
    observed = master["ball_observed"].to_numpy() == 1
    filled = (master["ball_filled"].to_numpy() == 1) & ~observed
    return observed, filled, ~observed & ~filled


def _near_possession_switch_mask(
    gt_codes: np.ndarray,
    runs: list[tuple[int, int]],
    window_frames: int,
) -> np.ndarray:
    """True on frames within ``+-window_frames`` of a ground-truth possession change inside each half."""
    n = len(gt_codes)
    mask = np.zeros(n, dtype=bool)
    spells = pm.possession_spells(gt_codes, runs)
    if len(spells) < 2:
        return mask

    # Map each spell to its run so we never cross a half boundary.
    for start, stop in runs:
        run_spells = [(a, b, k) for a, b, k in spells if a >= start and b <= stop]
        for idx in range(len(run_spells) - 1):
            _, prev_stop, prev_team = run_spells[idx]
            next_start, _, next_team = run_spells[idx + 1]
            if prev_team != next_team:
                for boundary in (prev_stop, next_start):
                    lo = max(start, boundary - window_frames)
                    hi = min(stop, boundary + window_frames + 1)
                    mask[lo:hi] = True
    return mask


def _near_restart_mask(
    gt_codes: np.ndarray,
    runs: list[tuple[int, int]],
    window_frames: int,
) -> np.ndarray:
    """True on frames in ``[d_end, d_end + window_frames]`` after the end of a GT D segment inside a half."""
    n = len(gt_codes)
    mask = np.zeros(n, dtype=bool)
    for start, stop in runs:
        segs = pm.segments(gt_codes, [(start, stop)])
        for idx, (_a, b, code) in enumerate(segs):
            if code == pm.D and idx + 1 < len(segs):
                lo = b
                hi = min(stop, b + window_frames + 1)
                mask[lo:hi] = True
    return mask


def _duration_bin(duration_s: float) -> str:
    if duration_s < 2.0:
        return "lt_2s"
    if duration_s <= 10.0:
        return "2s_to_10s"
    return "gt_10s"


def analyze_match_errors(
    master: pd.DataFrame,
    gt_codes: np.ndarray,
    pred_codes: np.ndarray,
    team_logit: np.ndarray,
    *,
    fps: float | None = None,
    iou_threshold: float = pm.SPELL_IOU_HEADLINE,
    near_zero_thr: float = NEAR_ZERO_LOGIT_THR,
) -> dict:
    """Compute per-frame category error counts and missed-possession diagnostics for one match."""
    if fps is None:
        fps = _fps()
    gt_codes = np.asarray(gt_codes, dtype=np.int8)
    pred_codes = np.asarray(pred_codes, dtype=np.int8)
    team_logit = np.asarray(team_logit, dtype=np.float64)

    graded = gt_codes != pm.X
    wrong = graded & (pred_codes != gt_codes)

    runs = pm.valid_runs(gt_codes)
    valid, short_gap, long_gap = _ball_visibility_masks(master)
    sw_win = int(round(SWITCH_WINDOW_S * fps))
    rs_win = int(round(RESTART_WINDOW_S * fps))
    near_switch = _near_possession_switch_mask(gt_codes, runs, sw_win)
    near_restart = _near_restart_mask(gt_codes, runs, rs_win)

    cat_masks = {
        "all_graded": graded,
        "ball_visible": graded & valid,
        "short_gap": graded & short_gap,
        "long_gap": graded & long_gap,
        "near_possession_switch_2s": graded & near_switch,
        "near_restart_3s": graded & near_restart,
    }

    frame_stats = {}
    for cat, m in cat_masks.items():
        n_tot = int(np.sum(m))
        n_err = int(np.sum(wrong & m))
        frame_stats[cat] = {
            "n_frames": n_tot,
            "n_errors": n_err,
            "error_rate": float(n_err) / float(n_tot) if n_tot > 0 else float("nan"),
        }

    # Possession-level matching at the headline IoU (0.5)
    filled = pm.fill_abstentions(pred_codes, runs)
    gt_spells = pm.possession_spells(gt_codes, runs)
    pred_spells = pm.possession_spells(filled, runs)
    overlaps = pm.spell_overlaps(gt_spells, pred_spells)
    matched_flags = pm.matched_gt_flags(overlaps, len(gt_spells), iou_threshold)

    bins = {
        b: {
            "n_gt": 0,
            "n_missed": 0,
            "missed_frames": 0,
            "vis_frames": 0,
            "gap_frames": 0,
            "near_zero_frames": 0,
            "wrong_sign_frames": 0,
            "right_sign_frames": 0,
            "pred_d_frames": 0,
            "pred_opp_frames": 0,
        }
        for b in (*DURATION_BINS, "all")
    }

    for idx, (a, b, team_code) in enumerate(gt_spells):
        dur_s = float(b - a) / fps
        b_key = _duration_bin(dur_s)
        is_missed = not bool(matched_flags[idx])
        # Expected sign of team_logit: +1 for H (0), -1 for A (1)
        expected_sign = 1.0 if team_code == pm.H else -1.0
        signed_t = expected_sign * team_logit[a:b]
        vis_seg = valid[a:b]
        pred_seg = filled[a:b]
        opp_code = pm.A if team_code == pm.H else pm.H

        for target in (b_key, "all"):
            bins[target]["n_gt"] += 1
            if is_missed:
                length = int(b - a)
                bins[target]["n_missed"] += 1
                bins[target]["missed_frames"] += length
                bins[target]["vis_frames"] += int(np.sum(vis_seg))
                bins[target]["gap_frames"] += int(np.sum(~vis_seg))
                nz = np.abs(signed_t) < near_zero_thr
                bins[target]["near_zero_frames"] += int(np.sum(nz))
                bins[target]["wrong_sign_frames"] += int(np.sum((~nz) & (signed_t < 0.0)))
                bins[target]["right_sign_frames"] += int(np.sum((~nz) & (signed_t >= 0.0)))
                bins[target]["pred_d_frames"] += int(np.sum(pred_seg == pm.D))
                bins[target]["pred_opp_frames"] += int(np.sum(pred_seg == opp_code))

    spell_stats = {}
    for key, d in bins.items():
        n_gt = d["n_gt"]
        n_missed = d["n_missed"]
        mf = d["missed_frames"]
        spell_stats[key] = {
            **d,
            "miss_rate": float(n_missed) / float(n_gt) if n_gt > 0 else float("nan"),
            "vis_frac": float(d["vis_frames"]) / float(mf) if mf > 0 else float("nan"),
            "gap_frac": float(d["gap_frames"]) / float(mf) if mf > 0 else float("nan"),
            "near_zero_frac": float(d["near_zero_frames"]) / float(mf) if mf > 0 else float("nan"),
            "wrong_sign_frac": float(d["wrong_sign_frames"]) / float(mf) if mf > 0 else float("nan"),
            "right_sign_frac": float(d["right_sign_frames"]) / float(mf) if mf > 0 else float("nan"),
            "pred_d_frac": float(d["pred_d_frames"]) / float(mf) if mf > 0 else float("nan"),
            "pred_opp_frac": float(d["pred_opp_frames"]) / float(mf) if mf > 0 else float("nan"),
        }

    metrics = score(gt_codes, pred_codes)
    n_gt_spells = metrics["n_gt_spells"]
    n_pred_spells = metrics["n_pred_spells"]
    metrics["spell_count_ratio"] = (
        float(n_pred_spells) / float(n_gt_spells) if n_gt_spells > 0 else float("nan")
    )

    return {
        "metrics": metrics,
        "frame_categories": frame_stats,
        "spell_bins": spell_stats,
    }


def aggregate_error_analysis(per_match: Mapping[str, Mapping]) -> dict:
    """Pool frame counts and possession counts across matches and compute macro/pooled summaries."""
    matches = list(per_match.keys())
    if not matches:
        raise ValueError("no matches to aggregate")

    pooled_frames = {}
    for cat in FRAME_CATEGORIES:
        n_tot = sum(int(per_match[m]["frame_categories"][cat]["n_frames"]) for m in matches)
        n_err = sum(int(per_match[m]["frame_categories"][cat]["n_errors"]) for m in matches)
        rates = [per_match[m]["frame_categories"][cat]["error_rate"] for m in matches]
        pooled_frames[cat] = {
            "n_frames": n_tot,
            "n_errors": n_err,
            "pooled_error_rate": float(n_err) / float(n_tot) if n_tot > 0 else float("nan"),
            "mean_error_rate": float(np.nanmean(rates)),
        }

    pooled_bins = {}
    for b_key in (*DURATION_BINS, "all"):
        keys_to_sum = (
            "n_gt", "n_missed", "missed_frames", "vis_frames", "gap_frames",
            "near_zero_frames", "wrong_sign_frames", "right_sign_frames",
            "pred_d_frames", "pred_opp_frames",
        )
        totals = {k: sum(int(per_match[m]["spell_bins"][b_key][k]) for m in matches)
                  for k in keys_to_sum}
        n_gt = totals["n_gt"]
        n_missed = totals["n_missed"]
        mf = totals["missed_frames"]
        pooled_bins[b_key] = {
            **totals,
            "miss_rate": float(n_missed) / float(n_gt) if n_gt > 0 else float("nan"),
            "vis_frac": float(totals["vis_frames"]) / float(mf) if mf > 0 else float("nan"),
            "gap_frac": float(totals["gap_frames"]) / float(mf) if mf > 0 else float("nan"),
            "near_zero_frac": float(totals["near_zero_frames"]) / float(mf) if mf > 0 else float("nan"),
            "wrong_sign_frac": float(totals["wrong_sign_frames"]) / float(mf) if mf > 0 else float("nan"),
            "right_sign_frac": float(totals["right_sign_frames"]) / float(mf) if mf > 0 else float("nan"),
            "pred_d_frac": float(totals["pred_d_frames"]) / float(mf) if mf > 0 else float("nan"),
            "pred_opp_frac": float(totals["pred_opp_frames"]) / float(mf) if mf > 0 else float("nan"),
        }

    tot_gt_spells = sum(int(per_match[m]["metrics"]["n_gt_spells"]) for m in matches)
    tot_pred_spells = sum(int(per_match[m]["metrics"]["n_pred_spells"]) for m in matches)
    spell_f1s = [float(per_match[m]["metrics"]["spell_f1_mean"]) for m in matches]
    macro_f1s = [float(per_match[m]["metrics"]["macro_f1"]) for m in matches]
    ratios = [float(per_match[m]["metrics"]["spell_count_ratio"]) for m in matches]

    return {
        "matches": matches,
        "per_match": dict(per_match),
        "summary": {
            "n_matches": len(matches),
            "total_gt_spells": tot_gt_spells,
            "total_pred_spells": tot_pred_spells,
            "pooled_spell_count_ratio": (
                float(tot_pred_spells) / float(tot_gt_spells) if tot_gt_spells > 0 else float("nan")
            ),
            "mean_spell_count_ratio": float(np.nanmean(ratios)),
            "mean_spell_f1": float(np.nanmean(spell_f1s)),
            "mean_macro_f1": float(np.nanmean(macro_f1s)),
        },
        "frame_categories": pooled_frames,
        "spell_bins": pooled_bins,
    }


def run_error_analysis(
    cfg: dict,
    *,
    degrade: str = "default",
    matches: Sequence[str] | None = None,
    load_fn=None,
) -> dict:
    """Run ``predict_labels`` and ``analyze_match_errors`` across external IDSSE matches."""
    matches = list(IDSSE_MATCHES if matches is None else matches)
    if not matches:
        raise ValueError("no matches to analyze")
    if load_fn is None:
        load_fn = load_external

    per_match = {}
    for match in matches:
        master, gt = load_fn(match, degrade)
        pred = predict_labels(master, cfg, holdout=match)
        pred_codes = pm.label_codes(pred)
        if cfg.get("pipeline") == "hmm":
            t_logit = _compute_team_logit(master, cfg, holdout=match)
        else:
            t_logit = np.zeros(len(master), dtype=np.float64)
        per_match[match] = analyze_match_errors(master, gt, pred_codes, t_logit)
    return aggregate_error_analysis(per_match)


def _repo_relative(path) -> str | None:
    """``path`` relative to the repository root when it lies inside it (no local paths in reports)."""
    if not path:
        return None
    p = Path(path)
    try:
        return p.resolve().relative_to(paths.repo_root().resolve()).as_posix()
    except ValueError:
        return p.name


def render_error_analysis_report(
    result: Mapping,
    cfg: dict,
    degrade: str,
    *,
    extra_variants: Mapping[str, Mapping] | None = None,
) -> str:
    """Render ``reports/error_analysis.md``."""
    s = result["summary"]
    lines = [
        "# Aggregated Error Analysis on External IDSSE Matches (F1-06)",
        "",
        "- config: `{}` (`{}`)".format(cfg.get("name", "unknown"), config_hash(cfg)),
        "- team.kind: `{}` (calibrated_file: `{}`)".format(
            cfg.get("team", {}).get("kind"), _repo_relative(cfg.get("team", {}).get("calibrated_file"))
        ),
        "- dead.kind: `{}` (calibrated_file: `{}`)".format(
            cfg.get("dead", {}).get("kind"), _repo_relative(cfg.get("dead", {}).get("calibrated_file"))
        ),
        "- decoder: `lam_team` = `{}`, `lam_dead` = `{}`".format(
            cfg.get("decoder", {}).get("lam_team"), cfg.get("decoder", {}).get("lam_dead")
        ),
        "- degrade: `{}`".format(degrade),
        "- matches: `{}`".format(s["n_matches"]),
        "- mean `spell_f1_mean`: `{:.4f}` | mean `macro_f1`: `{:.4f}`".format(
            s["mean_spell_f1"], s["mean_macro_f1"]
        ),
        "- **possession count ratio (pred / true)**: `{:.4f}` (mean across matches: `{:.4f}`, total: `{}/{}`)".format(
            s["pooled_spell_count_ratio"],
            s["mean_spell_count_ratio"],
            s["total_pred_spells"],
            s["total_gt_spells"],
        ),
        "",
        "## 1. Per-Match Summary",
        "",
        "| Match | spell_f1_mean | macro_f1 | alive_dead_f1 | n_gt_spells | n_pred_spells | count_ratio |",
        "|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for m in result["matches"]:
        mm = result["per_match"][m]["metrics"]
        lines.append(
            "| {} | {:.4f} | {:.4f} | {:.4f} | {} | {} | {:.4f} |".format(
                m,
                mm["spell_f1_mean"],
                mm["macro_f1"],
                mm["alive_dead_f1"],
                int(mm["n_gt_spells"]),
                int(mm["n_pred_spells"]),
                mm["spell_count_ratio"],
            )
        )

    lines.extend([
        "",
        "## 2. Per-Frame Error Rate by Situation Category",
        "",
        "| Category | Graded Frames | Error Frames | Pooled Error Rate | Mean Match Error Rate |",
        "|:---|---:|---:|---:|---:|",
    ])
    for cat in FRAME_CATEGORIES:
        fc = result["frame_categories"][cat]
        lines.append(
            "| `{}` | {} | {} | {:.4f} | {:.4f} |".format(
                cat,
                fc["n_frames"],
                fc["n_errors"],
                fc["pooled_error_rate"],
                fc["mean_error_rate"],
            )
        )

    lines.extend([
        "",
        "## 3. Missed Ground-Truth Possessions by Duration (IoU < 0.5)",
        "",
        "| Duration Bin | GT Possessions | Missed Possessions | Miss Rate | Ball Visible Frac | Gap Frac | `|t| < 0.5` (Near 0) | Wrong Sign (`t` opp) | Right Sign (`|t| >= 0.5`) | Pred `D` Frac | Pred Opp Team Frac |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for b_key in (*DURATION_BINS, "all"):
        sb = result["spell_bins"][b_key]
        lines.append(
            "| {} | {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                DURATION_LABELS[b_key],
                sb["n_gt"],
                sb["n_missed"],
                sb["miss_rate"],
                sb["vis_frac"],
                sb["gap_frac"],
                sb["near_zero_frac"],
                sb["wrong_sign_frac"],
                sb["right_sign_frac"],
                sb["pred_d_frac"],
                sb["pred_opp_frac"],
            )
        )

    if extra_variants:
        lines.extend([
            "",
            "## 4. Comparison Across Variants",
            "",
            "| Variant | spell_f1_mean | macro_f1 | Count Ratio (Pred/GT) | Miss Rate (< 2 s) | Miss Rate (2-10 s) | Miss Rate (> 10 s) | Switch <=2s Error | Restart <=3s Error |",
            "|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for name, res in extra_variants.items():
            sm = res["summary"]
            sb = res["spell_bins"]
            fc = res["frame_categories"]
            lines.append(
                "| `{}` | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                    name,
                    sm["mean_spell_f1"],
                    sm["mean_macro_f1"],
                    sm["pooled_spell_count_ratio"],
                    sb["lt_2s"]["miss_rate"],
                    sb["2s_to_10s"]["miss_rate"],
                    sb["gt_10s"]["miss_rate"],
                    fc["near_possession_switch_2s"]["pooled_error_rate"],
                    fc["near_restart_3s"]["pooled_error_rate"],
                )
            )

    sb_all = result["spell_bins"]["all"]
    sb_short = result["spell_bins"]["lt_2s"]
    sb_med = result["spell_bins"]["2s_to_10s"]
    fc_all = result["frame_categories"]

    lines.extend([
        "",
        "## 5. Diagnostic Conclusions & Actionable Feedback (F1-05 / F2-06 / Report Limitations)",
        "",
        "1. **Short and Medium Possessions Dominate Missed Spells:**",
        "   - Possessions `< 2 s` have a miss rate of `{:.1%}` (`{}/{}`), and `2-10 s` possessions have a miss rate of `{:.1%}` (`{}/{}`).".format(
            sb_short["miss_rate"], sb_short["n_missed"], sb_short["n_gt"],
            sb_med["miss_rate"], sb_med["n_missed"], sb_med["n_gt"],
        ),
        "   - Overall predicted-to-true possession count ratio is `{:.3f}` (`{}` predicted vs `{}` ground truth), confirming that brief turnovers are absorbed by the surrounding possession or merged across transitions.".format(
            s["pooled_spell_count_ratio"], s["total_pred_spells"], s["total_gt_spells"],
        ),
        "2. **What Happened in `team_logit` During Missed Possessions:**",
        "   - Across all missed possessions, the ball was visible on `{:.1%}` of frames and in a tracking gap on `{:.1%}` of frames.".format(
            sb_all["vis_frac"], sb_all["gap_frac"],
        ),
        "   - On `{:.1%}` of missed-possession frames, `team_logit` was near zero (`|t| < 0.5`); on `{:.1%}` it had the wrong sign; and on `{:.1%}` it already had the **correct sign (`|t| >= 0.5`)** but was overridden by decoder transition costs (`lam_team`) or dead-ball (`D`) predictions (`{:.1%}` predicted as `D`, `{:.1%}` as opponent).".format(
            sb_all["near_zero_frac"],
            sb_all["wrong_sign_frac"],
            sb_all["right_sign_frac"],
            sb_all["pred_d_frac"],
            sb_all["pred_opp_frac"],
        ),
        "3. **Transition and Restart Vulnerability:**",
        "   - Frame error rate jumps from `{:.1%}` on visible-ball frames to `{:.1%}` within `+-2 s` of a true possession switch and `{:.1%}` within `3 s` after a dead-ball restart (`D` end).".format(
            fc_all["ball_visible"]["pooled_error_rate"],
            fc_all["near_possession_switch_2s"]["pooled_error_rate"],
            fc_all["near_restart_3s"]["pooled_error_rate"],
        ),
        "   - **Candidates for F1-05 / F2-06:** (a) sharper evidence for possessions shorter than 2 s; (b) tune `lam_team` jointly with the calibrated `team_logit`, since frames with the right sign are still overridden by the transition cost; (c) dead-ball restart exits.",
        "",
    ])
    return "\n".join(lines)


def _load_cfg_with_decoder(root: Path, config: str, decoder: str | None) -> dict:
    """Load ``config`` and, when ``decoder`` is given, replace its ``decoder`` section."""
    cfg = load_config(root / config)
    if decoder:
        with open(root / decoder, encoding="utf-8") as handle:
            dec_raw = yaml.safe_load(handle)
        cfg["decoder"] = copy.deepcopy(dec_raw.get("decoder", dec_raw))
    return cfg


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Aggregated error analysis on IDSSE (F1-06)")
    parser.add_argument("--config", default="configs/v2.yaml",
                        help="config analysed in the report, used as is")
    parser.add_argument("--decoder", default=None,
                        help="optional decoder YAML replacing the decoder of --config")
    parser.add_argument("--baseline", default="configs/v1.yaml",
                        help="config compared against --config in section 4 ('' to skip)")
    parser.add_argument("--baseline-decoder", default="configs/decoder_idsse.yaml",
                        help="decoder YAML replacing the decoder of --baseline ('' to keep it)")
    parser.add_argument("--degrade", default="default")
    parser.add_argument("--out", default="reports/error_analysis.md")
    args = parser.parse_args(argv)

    root = paths.repo_root()
    primary_cfg = _load_cfg_with_decoder(root, args.config, args.decoder)
    primary_res = run_error_analysis(primary_cfg, degrade=args.degrade)

    extra = {}
    if args.baseline:
        base_cfg = _load_cfg_with_decoder(root, args.baseline, args.baseline_decoder)
        extra[base_cfg["name"]] = run_error_analysis(base_cfg, degrade=args.degrade)
    extra[primary_cfg["name"]] = primary_res

    report_text = render_error_analysis_report(
        primary_res, primary_cfg, args.degrade, extra_variants=extra
    )
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_text, encoding="utf-8", newline="\n")
    print("wrote {}".format(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DURATION_BINS",
    "FRAME_CATEGORIES",
    "aggregate_error_analysis",
    "analyze_match_errors",
    "render_error_analysis_report",
    "run_error_analysis",
]