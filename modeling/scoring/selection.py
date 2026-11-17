"""Section 6 selection protocol (task F2-06).

Walks a ladder of candidates from simple to complex. Each step is accepted only when
(a) the paired spellF1 gain on ``default`` exceeds 1 SE, and (b) on every stress
degrade the step does not worsen spellF1 by more than 1 SE. Writes
``reports/selection.md`` and ``configs/final.yaml``.

The duration-stage (F2-08) and hazard (F2-07) steps re-sweep only λ (stage 1 of F2-05, with the
F2-05 robust rule), as the update of 13/11 asks. ``--jobs N`` scores the external matches in N
worker processes; every match is independent, so the tables are the same as with one process.

Usage::

    python -m modeling.scoring.selection --report reports/selection.md --final configs/final.yaml \
        --jobs 7
"""
from __future__ import annotations

import argparse
import copy
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from modeling import paths
from modeling.config import load_config, validate_config
from modeling.scoring import durations_sweep as dsw
from modeling.scoring import sweep
from modeling.scoring.external_eval import (
    SUMMARY_ROWS,
    evaluate_config,
    paired_delta,
    to_markdown,
)
from modeling.scoring.sweep import SPELL_RATIO, spell_ratio_from_frame

METRIC = "spell_f1_mean"
STRESS_DEGRADES = ("harder", "easier", "hidelo", "hidehi")
PRIMARY_DEGRADE = "default"


EvalFn = Callable[[dict, str], pd.DataFrame]


def _score_one_match(job: tuple[dict, str, str]) -> dict:
    cfg, degrade, match = job
    return evaluate_config(cfg, degrade=degrade, matches=[match]).loc[match].to_dict()


def parallel_evaluate_fn(jobs: int, matches: Sequence[str] | None = None) -> EvalFn:
    """``evaluate_config`` with one worker process per match (same table, same order)."""
    from modeling.external.files import IDSSE_MATCHES

    matches = list(IDSSE_MATCHES if matches is None else matches)
    pool = ProcessPoolExecutor(max_workers=int(jobs))

    def evaluate(cfg: dict, degrade: str) -> pd.DataFrame:
        records = list(pool.map(_score_one_match, [(cfg, degrade, m) for m in matches]))
        df = pd.DataFrame(records, index=pd.Index(matches, name="match"))
        n = len(df)
        summary = pd.DataFrame([df.mean(), df.std(ddof=1) / math.sqrt(n)],
                               index=list(SUMMARY_ROWS))
        return pd.concat([df.astype(np.float64), summary])

    evaluate.pool = pool  # type: ignore[attr-defined]
    return evaluate


def resweep_lambda(cfg: dict, evaluate_fn: EvalFn) -> dict:
    """Stage 1 of F2-05 on ``cfg`` (λ grid, plateau within 1 SE, best worst case)."""
    ev = dsw.Evaluator(frame_fn=evaluate_fn)
    points = dsw.sweep_lambdas(cfg, ev)
    return sweep.robustify_stage(points, evaluate_fn=ev, primary=PRIMARY_DEGRADE)["cfg"]


def calibrated_stages(cfg: dict) -> dict[str, int]:
    """F2-08 stage counts (``all``) for the ``min_dwell``/mean durations of ``cfg`` (IDSSE)."""
    import games
    from modeling.calibration.durations import calibrate_durations, load_lengths
    from modeling.external.files import IDSSE_MATCHES

    fps = float(games.GAMES[games.GAME]["fps"])
    result = calibrate_durations(load_lengths(IDSSE_MATCHES), cfg["decoder"], fps=fps)
    return {s: int(result["all"][s]) for s in ("H", "A", "D")}


def _load_named(name: str) -> dict:
    root = paths.repo_root()
    path = root / "configs" / "{}.yaml".format(name)
    return load_config(path)


def mutate(cfg: dict, *, name: str | None = None, **section_updates) -> dict:
    """Deep-copy ``cfg``, apply nested updates, optionally rename, re-validate."""
    out = copy.deepcopy(cfg)
    if name is not None:
        out["name"] = name
    for section, updates in section_updates.items():
        if section not in out:
            out[section] = {}
        if isinstance(updates, Mapping) and isinstance(out[section], dict):
            node = out[section]
            for key, value in updates.items():
                if isinstance(value, Mapping) and isinstance(node.get(key), dict):
                    node[key] = {**node[key], **value}
                else:
                    node[key] = value
        else:
            out[section] = updates
    return validate_config(out)


def gain_accepted(mean_delta: float, se_delta: float) -> bool:
    """Paired gain on ``default`` must exceed 1 SE."""
    if not math.isfinite(mean_delta) or not math.isfinite(se_delta):
        return False
    return mean_delta > se_delta


def stress_ok(mean_delta: float, se_delta: float) -> bool:
    """Step must not worsen spellF1 by more than 1 SE on a stress degrade."""
    if not math.isfinite(mean_delta):
        return False
    se = se_delta if math.isfinite(se_delta) else 0.0
    return mean_delta >= -se


def evaluate_step(
    current: dict,
    candidate: dict,
    *,
    evaluate_fn: EvalFn,
    primary: str = PRIMARY_DEGRADE,
    stress: Sequence[str] = STRESS_DEGRADES,
    require_stress: bool = True,
) -> dict[str, Any]:
    """Compare ``candidate`` vs ``current``; return decision payload."""
    cur_pri = evaluate_fn(current, primary)
    cand_pri = evaluate_fn(candidate, primary)
    mean_d, se_d = paired_delta(cand_pri, cur_pri, metric=METRIC)
    accepted = gain_accepted(mean_d, se_d)
    stress_rows = []
    if require_stress:
        for deg in stress:
            cur_s = evaluate_fn(current, deg)
            cand_s = evaluate_fn(candidate, deg)
            md, sd = paired_delta(cand_s, cur_s, metric=METRIC)
            ok = stress_ok(md, sd)
            stress_rows.append({
                "degrade": deg,
                "mean_delta": md,
                "se_delta": sd,
                "ok": ok,
                "current_mean": float(cur_s.loc["mean", METRIC]),
                "candidate_mean": float(cand_s.loc["mean", METRIC]),
                "current_ratio": spell_ratio_from_frame(cur_s),
                "candidate_ratio": spell_ratio_from_frame(cand_s),
            })
            if not ok:
                accepted = False
    return {
        "current_name": current.get("name"),
        "candidate_name": candidate.get("name"),
        "primary": {
            "degrade": primary,
            "mean_delta": mean_d,
            "se_delta": se_d,
            "gain_ok": gain_accepted(mean_d, se_d),
            "current_mean": float(cur_pri.loc["mean", METRIC]),
            "candidate_mean": float(cand_pri.loc["mean", METRIC]),
            "current_se": float(cur_pri.loc["se", METRIC]),
            "candidate_se": float(cand_pri.loc["se", METRIC]),
            "current_ratio": spell_ratio_from_frame(cur_pri),
            "candidate_ratio": spell_ratio_from_frame(cand_pri),
            "current_table": cur_pri,
            "candidate_table": cand_pri,
        },
        "stress": stress_rows,
        "accepted": accepted,
        "candidate": candidate if accepted else current,
    }


# F2-07 (D-025, item 8): hazard without ``ball_dir_change`` and ``beta`` 2, the F2-07
# recommendation (``configs/v2_gbdt_hzn2.yaml``); ``lam_team`` 0.3 was kept by its re-sweep of λ.
HAZARD_FILE = "configs/calib/hazard_nodir.yaml"
HAZARD_BETA = 2.0
# F3-08 (D-026): the ``default`` GBDT dead_logit; the pooled fit ties with it (D-026, evidence).
DEAD_GBDT_FILE = "configs/calib/dead_gbdt.yaml"


def default_ladder() -> list[dict[str, Any]]:
    """Ladder steps of Section 6 and its updates of 11/11, 13/11 and 16/11.

    Each item is ``{"id", "label", "build": fn(current_cfg) -> candidate_cfg,
    "require_stress": bool, "force_accept": bool}``.
    ``force_accept`` is for ``team_tag_prior`` (pre-registered, no gain rule).
    """
    def from_file(name: str):
        def build(_current):
            return _load_named(name)
        return build

    def with_post(name_suffix, **post):
        def build(current):
            return mutate(current, name="{}_{}".format(current["name"], name_suffix),
                          postprocess=post)
        return build

    def with_dead_pooled(current):
        return mutate(
            current, name="{}_dead_pooled".format(current["name"]),
            dead={"calibrated_file": "configs/calib/dead_pooled.yaml"})

    def with_team_gbdt(current):
        # Keep the winner's dead/decoder/postprocess; swap team to the F1-07 GBDT bundle.
        gbdt = _load_named("v2_gbdt")
        return mutate(
            current, name="{}_gbdt".format(current["name"]),
            team=gbdt["team"],
            decoder={"lam_team": gbdt["decoder"]["lam_team"],
                     "lam_dead": gbdt["decoder"]["lam_dead"]})

    def with_dead_gbdt(current):
        # F3-08 (D-026): GBDT dead_logit fitted on default; nothing else changes (lam_dead kept).
        return mutate(current, name="{}_deadgbdt".format(current["name"]),
                      dead={"kind": "gbdt", "calibrated_file": DEAD_GBDT_FILE})

    def with_hazard(current):
        # F2-07: hazard-guided H/A exits in the decoder; nothing else changes.
        return mutate(current, name="{}_hzn2".format(current["name"]),
                      hazard={"kind": "calibrated", "calibrated_file": HAZARD_FILE,
                              "beta": HAZARD_BETA})

    def with_hazard_lam(current, evaluate_fn):
        # Update of 13/11: the hazard step re-sweeps only λ.
        return resweep_lambda(with_hazard(current), evaluate_fn)

    def durations_control(current, evaluate_fn):
        # D-027 item 7: the same λ re-sweep with the current stages, so a gain of the
        # re-sweep alone is not credited to the stages.
        return resweep_lambda(mutate(current, name="{}_lamctrl".format(current["name"])),
                              evaluate_fn)

    def with_durations(current, evaluate_fn):
        # F2-08 (D-027): stages fitted for this decoder, then re-sweep only λ.
        stages = calibrated_stages(current)
        if stages == dict(current["decoder"]["duration_stages"]):
            return current  # identity: skipped as a no-op
        staged = mutate(current, name="{}_nb".format(current["name"]),
                        decoder={"duration_stages": stages})
        return resweep_lambda(staged, evaluate_fn)

    def with_fused(current):
        # BAR-ATH uses ball_source=fused; IDSSE gain is measured on degrade=fused masters
        # with the same decoder (D-024 reference reports share one config_hash).
        return mutate(current, name="{}_fused".format(current["name"]),
                      ball_source="fused")

    steps = [
        {"id": "baseline", "label": "baseline", "build": from_file("baseline"),
         "require_stress": True, "force_accept": True, "absolute": True},
        {"id": "v1", "label": "v1 (Appendix A priors)", "build": from_file("v1"),
         "require_stress": True, "force_accept": False},
        {"id": "v1s", "label": "v1s (F2-05 decoder)", "build": from_file("v1s"),
         "require_stress": True, "force_accept": False},
        {"id": "v2_base", "label": "v2_base (calibrated, no touch)",
         "build": from_file("v2_base"), "require_stress": True, "force_accept": False},
        {"id": "v2", "label": "v2 (calibrated + touch persistence, D-022)",
         "build": from_file("v2"), "require_stress": True, "force_accept": False},
        {"id": "dead_pooled", "label": "dead pooled (F3-04)",
         "build": with_dead_pooled, "require_stress": True, "force_accept": False},
        {"id": "team_gbdt", "label": "team GBDT (F1-07, D-023)",
         "build": with_team_gbdt, "require_stress": True, "force_accept": False},
        {"id": "dead_gbdt", "label": "dead GBDT (F3-08, D-026)",
         "build": with_dead_gbdt, "require_stress": True, "force_accept": False},
        {"id": "post_midpoint", "label": "postprocess midpoint (F2-05 isolated)",
         "build": with_post("midpoint", midpoint_neutral_switch=True),
         "require_stress": True, "force_accept": False},
        {"id": "post_d_bridge_25", "label": "postprocess d_bridge=25",
         "build": with_post("d_bridge_25", d_bridge_frames=25),
         "require_stress": True, "force_accept": False},
        {"id": "post_d_bridge_50", "label": "postprocess d_bridge=50",
         "build": with_post("d_bridge_50", d_bridge_frames=50),
         "require_stress": True, "force_accept": False},
        {"id": "post_min_spell_13", "label": "postprocess min_spell=13",
         "build": with_post("min_spell_13", min_spell_frames=13),
         "require_stress": True, "force_accept": False},
        {"id": "post_min_spell_25", "label": "postprocess min_spell=25",
         "build": with_post("min_spell_25", min_spell_frames=25),
         "require_stress": True, "force_accept": False},
        {"id": "post_d_cross_25", "label": "postprocess d_cross=25",
         "build": with_post("d_cross_25", d_cross_frames=25),
         "require_stress": True, "force_accept": False},
        {"id": "post_d_cross_50", "label": "postprocess d_cross=50",
         "build": with_post("d_cross_50", d_cross_frames=50),
         "require_stress": True, "force_accept": False},
        {"id": "post_d_cross_75", "label": "postprocess d_cross=75",
         "build": with_post("d_cross_75", d_cross_frames=75),
         "require_stress": True, "force_accept": False},
        {"id": "boundary_shift", "label": "boundary_shift_file (F2-05)",
         "build": with_post(
             "boundary",
             boundary_shift_file="configs/calib/boundary.yaml"),
         "require_stress": True, "force_accept": False},
        {"id": "durations", "label": "semi-Markov durations (F2-08, D-027), λ re-swept",
         "build": with_durations, "needs_eval": True, "control_build": durations_control,
         "require_stress": True, "force_accept": False},
        {"id": "hazard", "label": "hazard-guided transitions (F2-07, D-025, beta 2), λ re-swept",
         "build": with_hazard_lam, "needs_eval": True,
         "require_stress": True, "force_accept": False},
        {"id": "fused", "label": "ball_source fused (F4-06, D-024)",
         "build": with_fused, "require_stress": False, "force_accept": False,
         "fused_idsse": True},
    ]
    return steps


def cached_evaluate_fn(evaluate_fn: EvalFn | None = None) -> EvalFn:
    """Memoize ``evaluate_config`` by ``(name, config_hash, degrade)``."""
    from modeling.config import config_hash

    base = evaluate_fn if evaluate_fn is not None else (
        lambda cfg, degrade: evaluate_config(cfg, degrade=degrade))
    cache: dict[tuple[str, str, str], pd.DataFrame] = {}

    def wrapped(cfg: dict, degrade: str) -> pd.DataFrame:
        key = (str(cfg.get("name")), config_hash(cfg), str(degrade))
        if key not in cache:
            cache[key] = base(cfg, degrade)
        return cache[key]

    wrapped.cache = cache  # type: ignore[attr-defined]
    return wrapped


def run_ladder(
    steps: Sequence[Mapping[str, Any]] | None = None,
    *,
    evaluate_fn: EvalFn | None = None,
) -> dict[str, Any]:
    """Execute the selection ladder; return history and the winning config."""
    if evaluate_fn is None or not hasattr(evaluate_fn, "cache"):
        # Always wrap the default (and plain callables) so the current winner is
        # not re-scored on every step.
        evaluate_fn = cached_evaluate_fn(evaluate_fn)

    steps = list(default_ladder() if steps is None else steps)
    history = []
    current = None

    for step in steps:
        if step.get("absolute"):
            candidate = step["build"](None)
            # Seed the ladder: accept the first absolute candidate unconditionally.
            table = evaluate_fn(candidate, PRIMARY_DEGRADE)
            decision = {
                "id": step["id"],
                "label": step["label"],
                "accepted": True,
                "forced": True,
                "absolute": True,
                "candidate_name": candidate["name"],
                "primary": {
                    "degrade": PRIMARY_DEGRADE,
                    "candidate_mean": float(table.loc["mean", METRIC]),
                    "candidate_se": float(table.loc["se", METRIC]),
                    "candidate_ratio": spell_ratio_from_frame(table),
                    "candidate_table": table,
                },
                "stress": [],
                "candidate": candidate,
            }
            current = candidate
            history.append(decision)
            continue

        assert current is not None
        if step.get("needs_eval"):
            candidate = step["build"](current, evaluate_fn)
        else:
            candidate = step["build"](current)
        # Skip no-ops (identical to current on the mutated fields).
        if candidate["name"] == current["name"] and candidate == current:
            history.append({
                "id": step["id"], "label": step["label"], "accepted": False,
                "skipped": True, "reason": "identical to current",
                "candidate": current,
            })
            continue

        if step.get("fused_idsse"):
            # D-024: paired gain of degrade=fused vs degrade=default on the same model;
            # if accepted, keep ball_source=fused for the BAR-ATH submission config.
            cur_pri = evaluate_fn(current, PRIMARY_DEGRADE)
            cand_pri = evaluate_fn(current, "fused")
            mean_d, se_d = paired_delta(cand_pri, cur_pri, metric=METRIC)
            accepted = gain_accepted(mean_d, se_d)
            decision = {
                "current_name": current.get("name"),
                "candidate_name": candidate.get("name"),
                "primary": {
                    "degrade": "fused_vs_default",
                    "mean_delta": mean_d,
                    "se_delta": se_d,
                    "gain_ok": accepted,
                    "current_mean": float(cur_pri.loc["mean", METRIC]),
                    "candidate_mean": float(cand_pri.loc["mean", METRIC]),
                    "current_se": float(cur_pri.loc["se", METRIC]),
                    "candidate_se": float(cand_pri.loc["se", METRIC]),
                    "current_ratio": spell_ratio_from_frame(cur_pri),
                    "candidate_ratio": spell_ratio_from_frame(cand_pri),
                    "current_table": cur_pri,
                    "candidate_table": cand_pri,
                },
                "stress": [],
                "accepted": accepted,
                "candidate": candidate if accepted else current,
            }
            if accepted:
                current = candidate
        elif step.get("force_accept"):
            decision = evaluate_step(
                current, candidate, evaluate_fn=evaluate_fn,
                require_stress=step.get("require_stress", True))
            decision["accepted"] = True
            decision["forced"] = True
            decision["candidate"] = candidate
            current = candidate
        else:
            decision = evaluate_step(
                current, candidate, evaluate_fn=evaluate_fn,
                require_stress=step.get("require_stress", True))
            if decision["accepted"] and step.get("control_build") is not None:
                # The step must also beat its control with the same rule (D-027, item 7).
                control = step["control_build"](current, evaluate_fn)
                vs_control = evaluate_step(
                    control, candidate, evaluate_fn=evaluate_fn,
                    require_stress=step.get("require_stress", True))
                decision["control"] = {"name": control.get("name"),
                                       "primary": vs_control["primary"],
                                       "stress": vs_control["stress"],
                                       "accepted": vs_control["accepted"]}
                if not vs_control["accepted"]:
                    decision["accepted"] = False
                    decision["candidate"] = current
            if decision["accepted"]:
                current = decision["candidate"]
        decision["id"] = step["id"]
        decision["label"] = step["label"]
        decision["candidate_cfg"] = candidate
        history.append(decision)

    return {"history": history, "winner": current}


def write_final(cfg: dict, path: str | Path) -> Path:
    out = copy.deepcopy(cfg)
    out["name"] = "final"
    out = validate_config(out)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(out, sort_keys=False), encoding="utf-8")
    return path


def _fmt(x) -> str:
    if isinstance(x, float):
        return "nan" if not math.isfinite(x) else "{:.4f}".format(x)
    return str(x)


def render_report(result: Mapping[str, Any], *,
                  local_checks: Mapping[str, str] | None = None) -> str:
    """Markdown selection report (IDSSE tables + local PASS/FAIL placeholders)."""
    lines = [
        "# Selection protocol (F2-06, Section 6)",
        "",
        "Primary metric: spellF1 on degraded IDSSE (`default`). "
        "Acceptance: paired gain > 1 SE on `default`, and no stress mode "
        "(`harder`/`easier`/`hidelo`/`hidehi`) worsens spellF1 by more than 1 SE. "
        "`fused` is gain-only on `default` (D-024). "
        "BAR-ATH filters are local PASS/FAIL only (never used to rank).",
        "",
    ]
    winner = result["winner"]
    lines.append("**Winner:** `{}` → written as `configs/final.yaml`.".format(
        winner.get("name") if winner else "none"))
    lines.append("")

    for step in result["history"]:
        lines.append("## Step: {} — {}".format(step["id"], step["label"]))
        lines.append("")
        if step.get("skipped"):
            lines.append("- skipped: {}".format(step.get("reason")))
            lines.append("")
            continue
        if step.get("forced") and step.get("absolute"):
            p = step["primary"]
            lines.append("- seeded ladder with `{}`: spellF1 {:.4f}±{:.4f}, "
                         "spell ratio {:.4f}".format(
                             step["candidate_name"], p["candidate_mean"],
                             p["candidate_se"], p["candidate_ratio"]))
            lines.append("")
            lines.append(to_markdown(p["candidate_table"]))
            lines.append("")
            continue
        p = step["primary"]
        lines.append("- candidate: `{}` vs current `{}`".format(
            step.get("candidate_name", p and ""), step.get("current_name", "")))
        if "mean_delta" in p:
            lines.append(
                "- default paired Δ spellF1: {:+.4f}±{:.4f} (gain_ok={}, "
                "current {:.4f}, candidate {:.4f}, ratios {:.4f}→{:.4f})".format(
                    p["mean_delta"], p["se_delta"], p["gain_ok"],
                    p["current_mean"], p["candidate_mean"],
                    p["current_ratio"], p["candidate_ratio"]))
        cand_cfg = step.get("candidate_cfg")
        if cand_cfg is not None and "decoder" in cand_cfg:
            dec = cand_cfg["decoder"]
            lines.append("- candidate decoder: lam_team {}, lam_dead {}, duration_stages {}".format(
                dec.get("lam_team"), dec.get("lam_dead"), dec.get("duration_stages")))
        lines.append("- **accepted:** {}".format(step["accepted"]))
        if step.get("forced"):
            lines.append("- forced accept (pre-registered rule)")
        ctrl = step.get("control")
        if ctrl is not None:
            cp = ctrl["primary"]
            lines.append("- vs control `{}`: default Δ {:+.4f}±{:.4f}, accepted={}".format(
                ctrl["name"], cp["mean_delta"], cp["se_delta"], ctrl["accepted"]))
        for row in step.get("stress", []):
            lines.append(
                "- stress `{}`: Δ {:+.4f}±{:.4f} ok={} "
                "(curr {:.4f} / cand {:.4f}, ratio {:.4f}→{:.4f})".format(
                    row["degrade"], row["mean_delta"], row["se_delta"], row["ok"],
                    row["current_mean"], row["candidate_mean"],
                    row["current_ratio"], row["candidate_ratio"]))
        lines.append("")
        if "candidate_table" in p:
            lines.append("### candidate on default")
            lines.append("")
            lines.append(to_markdown(p["candidate_table"]))
            lines.append("")

    lines.append("## Local BAR-ATH checks (human)")
    lines.append("")
    lines.append("Agents never run these. Record PASS/FAIL only.")
    lines.append("")
    checks = {
        "csv_valid (make_submission, SHA-256 recorded)": "pending",
        "check_public_window": "pending",
        "deadball_diagnostics (±40% plausibility)": "pending",
        "stability dead default/hidelo/hidehi (<5% label change, tools/dead_stability.py)":
            "pending",
        "dead_feature_shift (if dead calibrated/gbdt; --calibration of the winner)": "pending",
        "gbdt_feature_shift (if team gbdt; fused master if fused)": "pending",
        "hazard_feature_shift (if hazard; tactical and fused masters)": "pending",
    }
    if local_checks:
        checks.update(local_checks)
    for name, status in checks.items():
        lines.append("- {}: **{}**".format(name, status))
    lines.append("")
    lines.append("## Skipped optional steps")
    lines.append("")
    lines.append("- `team_tag_prior` (F1-05 item 8): 0 on IDSSE, so not part of the gain rule; "
                 "its pre-registered BAR-ATH agreement check was not run, so it stays off.")
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Section 6 selection protocol (F2-06)")
    parser.add_argument("--report", default="reports/selection.md")
    parser.add_argument("--final", default="configs/final.yaml")
    parser.add_argument("--skip-write", action="store_true")
    parser.add_argument("--jobs", type=int, default=1,
                        help="worker processes over the external matches (same results)")
    args = parser.parse_args(argv)

    root = paths.repo_root()
    evaluate_fn = parallel_evaluate_fn(args.jobs) if args.jobs > 1 else None
    result = run_ladder(evaluate_fn=evaluate_fn)
    report = render_report(result)
    if not args.skip_write:
        report_path = root / args.report
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8", newline="\n")
        final_path = write_final(result["winner"], root / args.final)
        print("wrote {}".format(report_path))
        print("wrote {}".format(final_path))
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "METRIC",
    "PRIMARY_DEGRADE",
    "STRESS_DEGRADES",
    "default_ladder",
    "evaluate_step",
    "gain_accepted",
    "mutate",
    "render_report",
    "run_ladder",
    "stress_ok",
    "write_final",
]
