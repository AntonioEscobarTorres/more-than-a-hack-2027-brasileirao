"""Duration-stage step of the selection ladder on external IDSSE masters (task F2-08, D-025).

Given a base config and the ``all`` stage counts of a ``configs/calib/durations*.yaml`` file
(:mod:`modeling.calibration.durations`), this module

1. scores the base (its own ``duration_stages``) and the base with the calibrated stages;
2. re-sweeps only stage 1 of F2-05 (λ) with the calibrated stages, with the F2-05 rule (plateau
   within 1 SE of the best mean on ``default``, then the best worst case over ``default`` /
   ``hidelo`` / ``hidehi``, :func:`modeling.scoring.sweep.robustify_stage`);
3. as a control, re-sweeps λ the same way with the base stages, so a gain of the λ re-sweep alone
   is not credited to the stages;
4. reports spellF1, its standard error (EP), the predicted / true possession count ratio and the
   paired difference against the base in every degrade mode, and applies the F2-06 acceptance rule
   (paired gain > 1 EP on ``default``; no stress mode worse by more than 1 EP) against the base and,
   for the stages with re-swept λ, against the control.

When the calibrated stages equal the base ones the step is the identity (the decoder is unchanged
bit for bit) and nothing is evaluated. Every external match is scored with ``holdout = match``.
Nothing reads the match data. Usage::

    python -m modeling.scoring.durations_sweep --config configs/v2_gbdt.yaml \\
        --calib configs/calib/durations_v2_gbdt.yaml --report reports/durations_sweep_v2_gbdt.md
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd
import yaml

from modeling.config import config_hash, load_config
from modeling.external.files import IDSSE_MATCHES
from modeling.scoring import sweep
from modeling.scoring.external_eval import evaluate_config, paired_delta

PRIMARY = "default"
STRESS_DEGRADES = ("harder", "easier", "hidelo", "hidehi")
MODES = (PRIMARY,) + STRESS_DEGRADES
METRIC = sweep.METRIC

FrameFn = Callable[[dict, str], pd.DataFrame]


class Evaluator:
    """Per-match metric tables of :func:`evaluate_config`, cached by ``(config hash, degrade)``."""

    def __init__(self, matches: Sequence[str] | None = None, frame_fn: FrameFn | None = None,
                 cache_dir: str | Path | None = None):
        self.matches = list(IDSSE_MATCHES if matches is None else matches)
        self._frame_fn = frame_fn or (lambda cfg, deg: evaluate_config(cfg, deg, self.matches))
        self._cache: dict[tuple[str, str], pd.DataFrame] = {}
        # Optional on-disk copy of the tables (IDSSE metrics only), keyed by config hash and degrade.
        self._cache_dir = None if cache_dir is None else Path(cache_dir)

    def frame(self, cfg: dict, degrade: str) -> pd.DataFrame:
        key = (config_hash(cfg), degrade)
        if key not in self._cache:
            path = None if self._cache_dir is None else \
                self._cache_dir / "{}__{}__{}.csv".format(key[0], degrade, "_".join(self.matches))
            if path is not None and path.is_file():
                self._cache[key] = pd.read_csv(path, index_col=0)
            else:
                self._cache[key] = self._frame_fn(cfg, degrade)
                if path is not None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    self._cache[key].to_csv(path)
        return self._cache[key]

    def __call__(self, cfg: dict, degrade: str) -> dict[str, float]:
        return sweep.summarize(self.frame(cfg, degrade))


def read_stages(calib_path: str | Path) -> dict[str, int]:
    """The ``all`` stage counts of a durations calibration file."""
    raw = yaml.safe_load(Path(calib_path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"all", "folds"}:
        raise ValueError("{}: expected the keys 'all' and 'folds'".format(calib_path))
    return {s: int(raw["all"][s]) for s in ("H", "A", "D")}


def with_stages(cfg: dict, stages: Mapping[str, int]) -> dict:
    return sweep.with_decoder(cfg, duration_stages={s: int(v) for s, v in stages.items()})


def sweep_lambdas(cfg: dict, evaluate: Evaluator, grid: Sequence[float] = sweep.LAM_GRID):
    """Stage 1 of F2-05 (λ grid) through the cached evaluator; same point layout as F2-05."""
    points = []
    for lam_team in grid:
        for lam_dead in grid:
            trial = sweep.with_decoder(cfg, lam_team=float(lam_team), lam_dead=float(lam_dead))
            points.append({
                "id": "lam|lam_dead={}|lam_team={}".format(lam_dead, lam_team),
                "stage": "lam",
                "lam_team": float(lam_team),
                "lam_dead": float(lam_dead),
                "cfg": trial,
                **evaluate(trial, PRIMARY),
            })
    return points


def acceptance(evaluate: Evaluator, candidate: dict, base: dict,
               modes: Sequence[str] = MODES) -> dict[str, Any]:
    """Paired spellF1 difference ``candidate − base`` per mode and the F2-06 verdict."""
    deltas = {}
    for mode in modes:
        deltas[mode] = paired_delta(evaluate.frame(candidate, mode), evaluate.frame(base, mode),
                                    METRIC)
    d0, se0 = deltas[PRIMARY]
    gain = math.isfinite(se0) and d0 > se0
    worse = [m for m in modes if m != PRIMARY and deltas[m][0] < -deltas[m][1]]
    return {"deltas": deltas, "gain": bool(gain), "worse_modes": worse,
            "accepted": bool(gain and not worse)}


def run_durations_step(base: dict, stages: Mapping[str, int], evaluate: Evaluator,
                       modes: Sequence[str] = MODES) -> dict[str, Any]:
    """Steps 1–3 of the module docstring; ``{"identity": True}`` when the stages do not change."""
    if dict(stages) == dict(base["decoder"]["duration_stages"]):
        return {"identity": True, "stages": dict(stages)}
    staged = with_stages(base, stages)
    points = sweep_lambdas(staged, evaluate)
    pick = sweep.robustify_stage(points, evaluate_fn=evaluate, primary=PRIMARY)
    control_points = sweep_lambdas(base, evaluate)
    control = sweep.robustify_stage(control_points, evaluate_fn=evaluate, primary=PRIMARY)
    configs = {"base": base, "stages": staged, "stages_lam": pick["cfg"],
               "base_lam": control["cfg"]}
    accept = {name: acceptance(evaluate, cfg, base, modes)
              for name, cfg in configs.items() if name != "base"}
    accept["stages_lam vs base_lam"] = acceptance(evaluate, pick["cfg"], control["cfg"], modes)
    return {
        "identity": False,
        "stages": dict(stages),
        "lam_points": points,
        "lam_pick": pick,
        "control_points": control_points,
        "control_pick": control,
        "configs": configs,
        "accept": accept,
    }


def summary_frame(evaluate: Evaluator, configs: Mapping[str, dict],
                  modes: Sequence[str] = MODES) -> pd.DataFrame:
    """spellF1 / EP / count ratio / paired Δ against ``configs['base']`` per config and mode."""
    rows = []
    for name, cfg in configs.items():
        for mode in modes:
            frame = evaluate.frame(cfg, mode)
            stats = sweep.summarize(frame)
            d, se = paired_delta(frame, evaluate.frame(configs["base"], mode), METRIC)
            rows.append({"config": name, "mode": mode, "spell_f1": stats["mean"],
                         "ep": stats["se"], "count_ratio": stats[sweep.SPELL_RATIO],
                         "delta_vs_base": d, "delta_ep": se})
    return pd.DataFrame(rows)


def extra_d_rows(evaluate: Evaluator, base: dict, stages: Mapping[str, int],
                 extra_d: Sequence[int]) -> list[dict]:
    """Informational: other D stage counts on ``default`` (never used for a choice)."""
    rows = []
    for r_d in extra_d:
        cfg = with_stages(base, {**stages, "D": int(r_d)})
        frame = evaluate.frame(cfg, PRIMARY)
        stats = sweep.summarize(frame)
        d, se = paired_delta(frame, evaluate.frame(base, PRIMARY), METRIC)
        rows.append({"r_D": int(r_d), "spell_f1": stats["mean"], "ep": stats["se"],
                     "count_ratio": stats[sweep.SPELL_RATIO], "delta": d, "delta_ep": se})
    return rows


def render_section(history: Mapping[str, Any], *, config_name: str, calib: str,
                   summary: pd.DataFrame | None = None, extra: Sequence[dict] = ()) -> str:
    stages = history["stages"]
    lines = ["## Selection step on `configs/{}.yaml`".format(config_name), "",
             "- calibration file: `{}`, `all` = H {H}, A {A}, D {D}".format(calib, **stages)]
    if history["identity"]:
        lines += ["- the calibrated stages equal the config's (`duration_stages` all 1): the "
                  "decoder is unchanged bit for bit (tests/test_modeling_negbin.py), so the step "
                  "is the identity and nothing was evaluated.", ""]
        return "\n".join(lines)
    pick, control = history["lam_pick"], history["control_pick"]
    lines += [
        "- λ re-swept with the stages (F2-05 stage 1, plateau + worst case over default / hidelo "
        "/ hidehi): `lam_team` {}, `lam_dead` {}".format(
            pick["cfg"]["decoder"]["lam_team"], pick["cfg"]["decoder"]["lam_dead"]),
        "- control, λ re-swept the same way with the base stages: `lam_team` {}, `lam_dead` {}".format(
            control["cfg"]["decoder"]["lam_team"], control["cfg"]["decoder"]["lam_dead"]),
        "",
        "Configs: `base` = `{}` as committed; `stages` = base + calibrated stages; `stages_lam` = "
        "`stages` with λ re-swept; `base_lam` = base with λ re-swept (control).".format(config_name),
        "",
        "| Config | Mode | spellF1 | EP | Pred/true possessions | Δ vs base | EP(Δ) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    if summary is not None:
        for row in summary.itertuples(index=False):
            delta = "—" if row.config == "base" else "{:+.4f}".format(row.delta_vs_base)
            dse = "—" if row.config == "base" else "{:.4f}".format(row.delta_ep)
            lines.append("| {} | {} | {:.4f} | {:.4f} | {:.3f} | {} | {} |".format(
                row.config, row.mode, row.spell_f1, row.ep, row.count_ratio, delta, dse))
    lines += ["", "F2-06 rule (paired gain > 1 EP on default; no stress mode worse by > 1 EP); "
              "against `base` unless named:", ""]
    for name, acc in history["accept"].items():
        d0, se0 = acc["deltas"][PRIMARY]
        lines.append("- `{}`: Δ default {:+.4f} ± {:.4f}; gain: {}; stress modes worse by > 1 EP: "
                     "{} → **{}**".format(name, d0, se0, "yes" if acc["gain"] else "no",
                                          ", ".join(acc["worse_modes"]) or "none",
                                          "accepted" if acc["accepted"] else "rejected"))
    acc = history["accept"]["stages_lam vs base_lam"]
    lines += ["", "Paired Δ `stages_lam` − `base_lam` per mode (the stage effect at re-swept λ): "
              + "; ".join("{} {:+.4f} ± {:.4f}".format(m, *acc["deltas"][m]) for m in acc["deltas"]),
              "", "λ grid on default, with the stages and with the base stages (control):", "",
              "| lam_team | lam_dead | spellF1 (stages) | EP | Pred/true | spellF1 (control) | EP "
              "| Pred/true |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    ctrl = {(q["lam_team"], q["lam_dead"]): q for q in history["control_points"]}
    for p in sorted(history["lam_points"], key=lambda q: (q["lam_team"], q["lam_dead"])):
        c = ctrl[(p["lam_team"], p["lam_dead"])]
        lines.append("| {} | {} | {:.4f} | {:.4f} | {:.3f} | {:.4f} | {:.4f} | {:.3f} |".format(
            p["lam_team"], p["lam_dead"], p["mean"], p["se"], p[sweep.SPELL_RATIO],
            c["mean"], c["se"], c[sweep.SPELL_RATIO]))
    if extra:
        lines += ["", "Informational only, never used for a choice: other D stage counts with the "
                  "base λ (default):", "",
                  "| r_D | spellF1 | EP | Pred/true possessions | Δ vs base | EP(Δ) |",
                  "|---:|---:|---:|---:|---:|---:|"]
        for row in extra:
            lines.append("| {} | {:.4f} | {:.4f} | {:.3f} | {:+.4f} | {:.4f} |".format(
                row["r_D"], row["spell_f1"], row["ep"], row["count_ratio"], row["delta"],
                row["delta_ep"]))
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--calib", required=True, help="configs/calib/durations*.yaml")
    parser.add_argument("--report", required=True, help="markdown section to write")
    parser.add_argument("--extra-d", type=int, nargs="*", default=[],
                        help="informational D stage counts to score on default")
    parser.add_argument("--cache-dir", default=None,
                        help="keep the per-match IDSSE metric tables here (outside the repo)")
    args = parser.parse_args(argv)

    base = load_config(args.config)
    stages = read_stages(args.calib)
    evaluate = Evaluator(cache_dir=args.cache_dir)
    history = run_durations_step(base, stages, evaluate)
    summary, extra = None, []
    if not history["identity"]:
        summary = summary_frame(evaluate, history["configs"])
        extra = extra_d_rows(evaluate, base, stages, args.extra_d)
    text = render_section(history, config_name=base["name"], calib=args.calib,
                          summary=summary, extra=extra)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(text, encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["Evaluator", "MODES", "STRESS_DEGRADES", "acceptance", "extra_d_rows", "read_stages",
           "render_section", "run_durations_step", "summary_frame", "sweep_lambdas",
           "with_stages"]
