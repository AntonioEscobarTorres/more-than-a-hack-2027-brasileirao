"""Hazard-weight sweep on external IDSSE masters (task F2-07, decision D-025).

The possession-change hazard of F1-08 enters the decoder as ``exit_logit = beta * h`` on the H↔A
exits (``hazard`` config section). This module

1. sweeps ``hazard.beta`` over :data:`modeling.scoring.sweep.BETA_GRID` on a base config, with the
   F2-05 rule: plateau within 1 SE of the best mean on the primary degrade, then the best worst case
   over ``default`` / ``hidelo`` / ``hidehi`` (:func:`modeling.scoring.sweep.robustify_stage`);
2. re-sweeps only stage 1 of F2-05 (λ) with the chosen ``beta`` (skipped when ``beta == 0``);
3. reports spellF1, its standard error (EP), the predicted / true possession count ratio and the
   median latency of the H↔A boundaries per degrade mode, and the decoder time per match.

Every external match is scored with ``holdout = match`` (C8), so the hazard, like the team and dead
evidence, uses the fold fitted without that match. Nothing reads the match data. Usage::

    python -m modeling.scoring.hazard_sweep --config configs/v1s.yaml \\
        --out-config configs/v1s_hz.yaml --report reports/sensitivity_hazard.md
"""
from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

import games
from evaluation import possession_metrics as pm
from modeling import paths, variants
from modeling.calibration.boundary import MATCH_TOL, latencies_for_match
from modeling.config import config_hash, load_config, validate_config
from modeling.deadball.force import build_force_d
from modeling.decode.emission import logits_to_logscores
from modeling.decode.params import DecoderParams
from modeling.decode.viterbi import decode_match
from modeling.external.files import IDSSE_MATCHES, load_external
from modeling.pipeline import hazard_exit_logit, predict_labels, variant_params
from modeling.scoring import sweep
from modeling.scoring.external_eval import SUMMARY_ROWS, paired_delta, to_markdown
from modeling.scoring.metrics import score

HAZARD_FILE = "configs/calib/hazard.yaml"
TEAM_BOUNDARIES = ("H>A", "A>H")
STRESS_DEGRADES = ("harder", "easier")

EvaluateFn = Callable[[dict, str], Mapping[str, float]]


# -- evaluation with latencies and a cache ------------------------------------------------------

class Evaluator:
    """Scores configs on the external matches, ``holdout = match``; results cached per config.

    ``details(cfg, degrade)`` returns ``{"frame", "lat"}``: the per-match metric table of
    :func:`modeling.scoring.external_eval.evaluate_config` (plus ``mean`` / ``se`` rows) and the
    signed H↔A boundary latencies (predicted − true, frames) pooled over the matches. Calling the
    evaluator returns the summary used by the sweep (mean, SE, spell ratio).
    """

    def __init__(self, matches: Sequence[str] | None = None):
        self.matches = list(IDSSE_MATCHES if matches is None else matches)
        if not self.matches:
            raise ValueError("no matches to evaluate")
        self._cache: dict[tuple[str, str], dict] = {}

    def details(self, cfg: dict, degrade: str) -> dict:
        key = (config_hash(cfg), degrade)
        if key not in self._cache:
            records, lat = [], []
            for match in self.matches:
                master, gt = load_external(match, degrade)
                codes = pm.label_codes(predict_labels(master, cfg, holdout=match))
                records.append(score(gt, codes))
                typed = latencies_for_match(gt, codes, tol=MATCH_TOL)
                for boundary in TEAM_BOUNDARIES:
                    lat.extend(typed[boundary])
            df = pd.DataFrame(records, index=pd.Index(self.matches, name="match"))
            n = len(df)
            summary = pd.DataFrame([df.mean(), df.std(ddof=1) / math.sqrt(n)],
                                   index=list(SUMMARY_ROWS))
            self._cache[key] = {"frame": pd.concat([df.astype(np.float64), summary]),
                                "lat": np.asarray(lat, dtype=np.float64)}
        return self._cache[key]

    def __call__(self, cfg: dict, degrade: str) -> dict[str, float]:
        return sweep.summarize(self.details(cfg, degrade)["frame"])


def latency_summary(lat: np.ndarray) -> dict[str, float]:
    """Median signed and median absolute latency (frames) and the number of matched pairs."""
    lat = np.asarray(lat, dtype=np.float64)
    if lat.size == 0:
        return {"median_latency": float("nan"), "median_abs_latency": float("nan"), "n_pairs": 0}
    return {"median_latency": float(np.median(lat)),
            "median_abs_latency": float(np.median(np.abs(lat))), "n_pairs": int(lat.size)}


# -- the two stages -----------------------------------------------------------------------------

def with_calibrated_hazard(cfg: dict, hazard_file: str) -> dict:
    """``cfg`` with ``hazard.kind: calibrated``, ``hazard_file`` and ``beta`` 0 (validated copy)."""
    out = copy.deepcopy(cfg)
    out["hazard"] = {"kind": "calibrated", "calibrated_file": hazard_file, "beta": 0.0}
    return validate_config(out)


def run_hazard_sweep(cfg: dict, *, evaluate_fn: EvaluateFn, hazard_file: str = HAZARD_FILE,
                     degrade: str = "default", robust: bool = True,
                     beta_grid: Sequence[float] = sweep.BETA_GRID,
                     lam_grid: Sequence[float] = sweep.LAM_GRID,
                     lam_dead_grid: Sequence[float] | None = None) -> dict:
    """Stage ``beta`` then stage ``lam`` (F2-05 stage 1 only); returns the history.

    ``history = {"base", "stages": [...], "beta_cfg", "selected_cfg"}``. ``base`` is ``cfg`` with
    the hazard off, ``beta_cfg`` the chosen ``beta`` with the λ of ``cfg``, ``selected_cfg`` the
    final config. With ``beta == 0`` chosen, the λ stage is skipped and ``selected_cfg`` is ``base``.
    """
    if cfg["hazard"]["kind"] != "none" and cfg["hazard"]["beta"] != 0.0:
        raise ValueError("the base config must have the hazard off")
    if 0.0 not in [float(b) for b in beta_grid]:
        raise ValueError("beta_grid must contain 0 (the base)")
    base = copy.deepcopy(cfg)
    history: dict[str, Any] = {"base": base, "stages": []}

    start = with_calibrated_hazard(cfg, hazard_file)
    beta_points = sweep.sweep_beta(start, degrade=degrade, grid=beta_grid,
                                   evaluate_fn=evaluate_fn)
    beta_pick = sweep.select_plateau(beta_points, neighbour_fn=sweep._beta_neighbours,
                                     tiebreak_keys=("beta",))
    if robust:
        beta_pick = sweep.robustify_stage(beta_points, evaluate_fn=evaluate_fn, primary=degrade)
    history["stages"].append({"name": "beta", "points": beta_points, "selected": beta_pick})

    beta = float(beta_pick["beta"])
    if beta == 0.0:
        history["beta_cfg"] = base
        history["selected_cfg"] = base
        return history
    history["beta_cfg"] = beta_pick["cfg"]

    lam_points = sweep.sweep_lambdas(beta_pick["cfg"], degrade=degrade, grid=lam_grid,
                                     dead_grid=lam_dead_grid, evaluate_fn=evaluate_fn)
    lam_pick = sweep.select_plateau(lam_points, neighbour_fn=sweep._lam_neighbours)
    if robust:
        lam_pick = sweep.robustify_stage(lam_points, evaluate_fn=evaluate_fn, primary=degrade)
    history["stages"].append({"name": "lam", "points": lam_points, "selected": lam_pick})
    history["selected_cfg"] = lam_pick["cfg"]
    return history


# -- decoder time -------------------------------------------------------------------------------

def decoder_timing(cfg: dict, matches: Sequence[str], degrade: str = "default",
                   repeats: int = 3) -> pd.DataFrame:
    """Seconds per match of ``decode_match`` without and with ``exit_logit``, and of the hazard.

    ``cfg`` must have the hazard on. Each time is the minimum over ``repeats`` runs, the two
    decoder variants alternating, so a load spike on the machine does not favour either one.
    Wall-clock times: they still vary between runs and machines.
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if cfg["hazard"]["kind"] == "none" or cfg["hazard"]["beta"] == 0.0:
        raise ValueError("decoder_timing needs a config with the hazard on")
    variants.load_kinds()
    params = DecoderParams.from_config(cfg["decoder"], fps=games.GAMES[games.GAME]["fps"])
    rows = []
    for match in matches:
        master, _ = load_external(match, degrade)
        half = master["half"].to_numpy()
        team = variants.TEAM_KINDS[cfg["team"]["kind"]](
            master, variant_params(cfg["team"]), holdout=match)
        dead = variants.DEAD_KINDS[cfg["dead"]["kind"]](
            master, variant_params(cfg["dead"]), holdout=match)
        force_d = build_force_d(master)
        scores = logits_to_logscores(team, dead, params.lam_team, params.lam_dead, force_d,
                                     neg=params.neg)
        t_hazard = t_const = t_mod = math.inf
        for _ in range(repeats):
            t0 = time.perf_counter()
            exit_logit = hazard_exit_logit(master, cfg, holdout=match)
            t_hazard = min(t_hazard, time.perf_counter() - t0)
            t0 = time.perf_counter()
            decode_match(scores, half, force_d, params)
            t_const = min(t_const, time.perf_counter() - t0)
            t0 = time.perf_counter()
            decode_match(scores, half, force_d, params, exit_logit=exit_logit)
            t_mod = min(t_mod, time.perf_counter() - t0)
        rows.append({"frames in halves": int(np.count_nonzero(half != 0)), "hazard s": t_hazard,
                     "decode s (exit_logit None)": t_const, "decode s (exit_logit)": t_mod})
    df = pd.DataFrame(rows, index=pd.Index(list(matches), name="match"))
    out = pd.concat([df, pd.DataFrame([df.mean()], index=["mean"])])
    out["frames in halves"] = [str(int(round(v))) for v in out["frames in halves"]]
    return out


# -- report -------------------------------------------------------------------------------------

def _stage_markdown(stage: Mapping[str, Any]) -> list[str]:
    selected = stage["selected"]
    lines = ["## Stage: {}".format(stage["name"]), "",
             "- selected: `{}`".format(selected["id"]),
             "- spellF1 mean ± SE (primary degrade): {:.4f} ± {:.4f}".format(
                 float(selected["mean"]), float(selected["se"]))]
    if "robust_scores" in selected:
        lines.append("- worst case over {}: {:.4f}".format(
            ", ".join(sweep.ROBUST_DEGRADES), float(selected["worst_case_mean"])))
        for deg, stats in selected["robust_scores"].items():
            lines.append("  - {}: {:.4f} ± {:.4f} (count ratio {:.4f})".format(
                deg, float(stats["mean"]), float(stats["se"]),
                float(stats.get(sweep.SPELL_RATIO, float("nan")))))
    candidates = {p["id"] for p in sweep.plateau_candidates(stage["points"])}
    df = sweep.points_to_frame(stage["points"]).set_index("id")
    df["plateau"] = ["yes" if i in candidates else "" for i in df.index]
    df = df.drop(columns=[c for c in ("stage",) if c in df.columns])
    lines.extend(["", to_markdown(df), ""])
    return lines


def summary_rows(evaluate: Evaluator, configs: Mapping[str, dict],
                 degrades: Sequence[str]) -> pd.DataFrame:
    """One row per (config label, degrade): metrics, latency and paired delta vs the first config."""
    labels = list(configs)
    rows = []
    for deg in degrades:
        ref = evaluate.details(configs[labels[0]], deg)["frame"]
        for label in labels:
            det = evaluate.details(configs[label], deg)
            df = det["frame"]
            match_rows = [m for m in df.index if m not in SUMMARY_ROWS]
            delta, delta_se = paired_delta(df, ref)
            better = int((df.loc[match_rows, "spell_f1_mean"]
                          > ref.loc[match_rows, "spell_f1_mean"]).sum())
            rows.append({"degrade": deg, "config": label,
                         "spellF1": float(df.loc["mean", "spell_f1_mean"]),
                         "SE": float(df.loc["se", "spell_f1_mean"]),
                         "count ratio": sweep.spell_ratio_from_frame(df),
                         **latency_summary(det["lat"]),
                         "delta vs base": delta, "SE delta": delta_se,
                         "matches better": "{}/{}".format(better, len(match_rows))})
    return pd.DataFrame(rows)


def extra_beta_configs(history: Mapping[str, Any], betas: Sequence[float],
                       hazard_file: str = HAZARD_FILE) -> dict[str, dict]:
    """Configs for ``betas`` outside the swept grid, with the λ of the base (informative only).

    The first entry is the base (hazard off), so :func:`summary_rows` compares against it.
    """
    start = with_calibrated_hazard(history["base"], hazard_file)
    out = {"base (hazard off)": history["base"]}
    for beta in betas:
        if not float(beta) > 0.0:
            raise ValueError("extra beta values must be > 0, got {}".format(beta))
        out["beta = {:g}".format(float(beta))] = sweep.with_hazard(start, beta=float(beta))
    return out


def render_report(history: Mapping[str, Any], summary: pd.DataFrame, timing: pd.DataFrame | None,
                  *, args: Mapping[str, Any], extra: pd.DataFrame | None = None) -> str:
    beta = float(history["stages"][0]["selected"]["beta"])
    final = history["selected_cfg"]
    lines = [
        "# Hazard-guided transitions (F2-07)",
        "",
        "Generated by `python -m modeling.scoring.hazard_sweep` (decision D-025). IDSSE masters only "
        "(public data); every match is scored with `holdout = match`, so the hazard, the team and "
        "the dead evidence use the fold fitted without it. Nothing from the match data.",
        "",
        "- base config: `{}` (`{}`)".format(args["config"], config_hash(history["base"])),
        "- hazard file: `{}`; `exit_logit = beta * h` on the H<->A exits of the terminal H and A "
        "sub-states; the transition into frame t uses the hazard of frame t".format(
            args["hazard_file"]),
        "- beta grid: {}; λ grid: lam_team {} x lam_dead {}".format(
            list(args["beta_grid"]), list(args["lam_grid"]),
            list(args["lam_dead_grid"] or args["lam_grid"])),
        "- rule (F2-05): plateau within 1 SE of the best mean on `{}`, then the best worst case "
        "over {} (robust: {})".format(args["degrade"], ", ".join(sweep.ROBUST_DEGRADES),
                                      args["robust"]),
        "- selected: beta = {}, lam_team = {}, lam_dead = {}{}".format(
            beta, final["decoder"]["lam_team"], final["decoder"]["lam_dead"],
            "" if args["out_config"] is None else " -> `{}`".format(args["out_config"])),
        "- latency: predicted minus true frame of the H>A and A>H boundaries paired by the official "
        "matcher (same type, tolerance {} frames), pooled over the matches".format(MATCH_TOL),
        "",
    ]
    for stage in history["stages"]:
        lines.extend(_stage_markdown(stage))
    if beta == 0.0:
        lines.extend(["beta = 0 was selected, so the hazard does not enter and λ is not re-swept.",
                      ""])
    lines.extend(["## Summary per degrade mode", "",
                  "`delta vs base` is the paired per-match difference of spellF1 against the base "
                  "config (hazard off), with its standard error.", "",
                  to_markdown(summary.set_index("degrade")), ""])
    if extra is not None:
        lines.extend(["## Beyond the pre-registered grid (informative, not used for the selection)",
                      "",
                      "`beta` values outside the swept grid, each with the λ of the base config. "
                      "They are reported to show the shape of the curve past the grid edge; the "
                      "selection above uses only the swept grid, and any other `beta` enters only "
                      "through F2-06.", "",
                      to_markdown(extra.set_index("degrade")), ""])
    if timing is not None:
        lines.extend(["## Decoder time per match (`default`, wall clock, best of 3, varies between "
                      "runs)", "",
                      to_markdown(timing), ""])
    return "\n".join(lines)


def write_config(cfg: dict, path: Path, header: str) -> Path:
    """Write ``cfg`` with ``name`` set to the file stem, after validating it."""
    out = copy.deepcopy(cfg)
    out["name"] = path.stem
    out = validate_config(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + yaml.safe_dump(out, sort_keys=False), encoding="utf-8", newline="\n")
    load_config(path)  # the written file must read back under the strict reader
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Hazard-weight sweep (F2-07)")
    parser.add_argument("--config", default="configs/v1s.yaml")
    parser.add_argument("--hazard-file", default=HAZARD_FILE)
    parser.add_argument("--degrade", default="default")
    parser.add_argument("--no-robust", action="store_true")
    parser.add_argument("--beta-grid", nargs="+", type=float, default=list(sweep.BETA_GRID))
    parser.add_argument("--lam-grid", nargs="+", type=float, default=list(sweep.LAM_GRID),
                        help="lam_team grid (and lam_dead grid unless --lam-dead-grid is given)")
    parser.add_argument("--lam-dead-grid", nargs="+", type=float, default=None)
    parser.add_argument("--stress", nargs="*", default=list(STRESS_DEGRADES),
                        help="extra degrade modes of the summary table")
    parser.add_argument("--matches", nargs="+", default=None)
    parser.add_argument("--report", default="reports/sensitivity_hazard.md")
    parser.add_argument("--out-config", default=None,
                        help="write the selected config there (name = file stem)")
    parser.add_argument("--no-timing", action="store_true")
    parser.add_argument("--extra-beta", nargs="+", type=float, default=None,
                        help="beta values outside the grid, reported only (base λ)")
    args = parser.parse_args(argv)

    root = paths.repo_root()
    cfg = load_config(root / args.config)
    evaluate = Evaluator(args.matches)
    history = run_hazard_sweep(
        cfg, evaluate_fn=evaluate, hazard_file=args.hazard_file, degrade=args.degrade,
        robust=not args.no_robust, beta_grid=args.beta_grid, lam_grid=args.lam_grid,
        lam_dead_grid=args.lam_dead_grid)

    configs = {"base (hazard off)": history["base"]}
    if history["selected_cfg"] is not history["base"]:
        configs["beta, base λ"] = history["beta_cfg"]
        configs["beta + λ re-swept"] = history["selected_cfg"]
    degrades = list(sweep.ROBUST_DEGRADES) + [d for d in args.stress
                                               if d not in sweep.ROBUST_DEGRADES]
    summary = summary_rows(evaluate, configs, degrades)
    timing = None
    if not args.no_timing and history["selected_cfg"] is not history["base"]:
        timing = decoder_timing(history["selected_cfg"], evaluate.matches)

    extra = None
    if args.extra_beta:
        extra = summary_rows(evaluate, extra_beta_configs(history, args.extra_beta,
                                                          args.hazard_file), degrades)

    report = render_report(history, summary, timing, extra=extra, args={
        "config": args.config, "hazard_file": args.hazard_file, "degrade": args.degrade,
        "robust": not args.no_robust, "beta_grid": args.beta_grid, "lam_grid": args.lam_grid,
        "lam_dead_grid": args.lam_dead_grid, "out_config": args.out_config})
    report_path = root / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8", newline="\n")
    print("wrote {}".format(report_path))
    if args.out_config is not None:
        header = ("# Generated by python -m modeling.scoring.hazard_sweep --config {} (F2-07, "
                  "D-025): hazard beta and lam re-swept on degraded IDSSE.\n# Nothing here was "
                  "fitted or chosen on BAR-ATH. Report: {}.\n").format(args.config, args.report)
        print("wrote {}".format(write_config(history["selected_cfg"], root / args.out_config,
                                             header)))
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["Evaluator", "HAZARD_FILE", "decoder_timing", "extra_beta_configs", "latency_summary",
           "render_report",
           "run_hazard_sweep", "summary_rows", "with_calibrated_hazard", "write_config"]
