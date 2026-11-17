"""Decoder sweep for the stacked config (F2-09, decision D-029), on degraded IDSSE only.

The evidence is :func:`modeling.pipeline.evidence_logits` (the code path of the submission).

Usage::

    python -m modeling.scoring.stack_sweep --config configs/final.yaml --jobs 7

The evidence (stacked ``team_logit``, ``dead_logit``, hazard ``h``) does not depend on the decoder,
so it is computed once per (match, degradation) with the leave-one-match-out models, and only the
decoder is re-run over the pre-registered grid :data:`BETA_GRID` x :data:`LAM_TEAM_GRID` x
:data:`LAM_DEAD_GRID` (everything else in the decoder stays as in the config).

Choice rule (the one of the F2-05 / F2-07 re-sweeps, D-018 and the update of 13/11): the plateau
is every grid point whose mean spellF1 on ``default`` is within 1 SE of the best one; among the
plateau, the point with the best worst case over ``default``, ``hidelo`` and ``hidehi`` wins. The
``fused``, ``harder`` and ``easier`` columns are reported, not used to choose.

Writes ``reports/stack_sweep.md`` and, with ``--write``, the chosen ``beta``, ``lam_team`` and
``lam_dead`` back into the config.
"""
from __future__ import annotations

import argparse
import copy
import itertools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

import games
from evaluation import possession_metrics as pm
from modeling import paths
from modeling.config import config_hash, load_config
from modeling.decode.emission import logits_to_logscores
from modeling.decode.params import DecoderParams
from modeling.decode.viterbi import decode_match
from modeling.deadball.force import build_force_d
from modeling.external.files import IDSSE_MATCHES, load_external
from modeling.pipeline import evidence_logits, final_imposition
from modeling.scoring.metrics import score

BETA_GRID = (1.0, 1.25, 1.5, 1.75, 2.0)
LAM_TEAM_GRID = (0.3, 0.4, 0.5, 0.6)
LAM_DEAD_GRID = (0.2, 0.3, 0.4, 0.5)
CHOICE_DEGRADES = ("default", "hidelo", "hidehi")
REPORT_DEGRADES = ("default", "hidelo", "hidehi", "fused", "harder", "easier")


def evidence(match: str, degrade: str, cfg: dict) -> dict:
    """Decoder inputs of one external match under ``cfg`` (leave-one-match-out models)."""
    master, gt = load_external(match, degrade)
    half = master["half"].to_numpy()
    team, dead, h = evidence_logits(master, cfg, holdout=match)
    return {"team": np.asarray(team), "dead": np.asarray(dead), "h": h, "half": half,
            "force": build_force_d(master), "gt": np.asarray(gt)}


def decode_with(ev: dict, decoder: dict, beta: float) -> np.ndarray:
    """Same steps as :func:`modeling.pipeline.hmm_labels` (no post-processing rule is on)."""
    p = DecoderParams.from_config(decoder, fps=games.GAMES[games.GAME]["fps"])
    s = logits_to_logscores(ev["team"], ev["dead"], p.lam_team, p.lam_dead, ev["force"],
                            neg=p.neg)
    exit_logit = None if ev["h"] is None or beta == 0 else beta * ev["h"]
    labels = decode_match(s, ev["half"], ev["force"], p, exit_logit=exit_logit)
    labels, _ = final_imposition(labels, ev["half"], ev["force"])
    return labels


def _grid_job(args):
    match, degrade, cfg, combos = args
    ev = evidence(match, degrade, cfg)
    out = []
    for beta, lt, ld in combos:
        dec = copy.deepcopy(cfg["decoder"])
        dec["lam_team"], dec["lam_dead"] = float(lt), float(ld)
        m = score(ev["gt"], pm.label_codes(decode_with(ev, dec, beta)))
        out.append((float(m["spell_f1_mean"]), float(m["n_pred_spells"]) / float(m["n_gt_spells"])))
    return match, degrade, out


def run_grid(cfg: dict, *, degrades=REPORT_DEGRADES, matches=IDSSE_MATCHES, jobs: int = 1):
    """``{degrade: array (n_matches, n_combos, 2)}`` (spellF1, count ratio) and the combos."""
    if cfg["postprocess"]["boundary_shift_file"] is not None or any(
            cfg["postprocess"][k] for k in ("midpoint_neutral_switch", "d_bridge_frames",
                                            "d_cross_frames", "min_spell_frames")):
        raise ValueError("the sweep assumes every post-processing rule is off")
    combos = list(itertools.product(BETA_GRID, LAM_TEAM_GRID, LAM_DEAD_GRID))
    tasks = [(m, d, cfg, combos) for d in degrades for m in matches]
    with ProcessPoolExecutor(max(1, int(jobs))) as ex:
        res = list(ex.map(_grid_job, tasks))
    out = {d: np.zeros((len(matches), len(combos), 2)) for d in degrades}
    for match, degrade, rows in res:
        out[degrade][list(matches).index(match)] = np.asarray(rows)
    return out, combos


def choose(out: dict, combos: list) -> tuple[int, dict]:
    """Index of the chosen combo (plateau at 1 SE on ``default``, best worst case) and a summary."""
    f = out["default"][:, :, 0]
    mean = f.mean(axis=0)
    se = f.std(axis=0, ddof=1) / np.sqrt(f.shape[0])
    best = int(np.argmax(mean))
    plateau = np.flatnonzero(mean >= mean[best] - se[best])
    worst = np.min(np.stack([out[d][:, :, 0].mean(axis=0) for d in CHOICE_DEGRADES]), axis=0)
    pick = int(plateau[np.argmax(worst[plateau])])
    return pick, {"best_default": best, "plateau": plateau.tolist(), "worst": worst}


def render(out: dict, combos: list, pick: int, info: dict, cfg: dict) -> str:
    lines = ["# Decoder sweep of the stacked config (F2-09, D-029)", "",
             "- config: `{}` (`{}`)".format(cfg["name"], config_hash(cfg)[:8]),
             "- evidence: leave-one-match-out models (stacked team_logit, dead GBDT, GBDT hazard)",
             "- grid: beta {} x lam_team {} x lam_dead {}".format(
                 list(BETA_GRID), list(LAM_TEAM_GRID), list(LAM_DEAD_GRID)),
             "- rule: plateau at 1 SE of the best `default` mean, then the best worst case over "
             "`default`/`hidelo`/`hidehi`", "",
             "| beta | lam_team | lam_dead | " + " | ".join(REPORT_DEGRADES) + " | worst | ratio |",
             "|---:|---:|---:|" + "---:|" * (len(REPORT_DEGRADES) + 2)]
    order = np.argsort(-out["default"][:, :, 0].mean(axis=0))
    for i in order:
        b, lt, ld = combos[i]
        cells = ["{:.4f}".format(out[d][:, i, 0].mean()) for d in REPORT_DEGRADES]
        mark = " **(chosen)**" if i == pick else (" (plateau)" if i in info["plateau"] else "")
        lines.append("| {} | {} | {} | {} | {:.4f} | {:.3f} |{}".format(
            b, lt, ld, " | ".join(cells), info["worst"][i], out["default"][:, i, 1].mean(), mark))
    b, lt, ld = combos[pick]
    f = out["default"][:, pick, 0]
    lines.extend(["", "## Choice", "",
                  "- beta `{}`, lam_team `{}`, lam_dead `{}`: `default` {:.4f} ± {:.4f}, "
                  "worst case {:.4f}".format(b, lt, ld, f.mean(), f.std(ddof=1) / np.sqrt(len(f)),
                                             info["worst"][pick]), ""])
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Decoder sweep of the stacked config (F2-09)")
    parser.add_argument("--config", default="configs/final.yaml")
    parser.add_argument("--report", default="reports/stack_sweep.md")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--write", action="store_true",
                        help="write the chosen beta/lam_team/lam_dead into the config file")
    args = parser.parse_args(argv)
    root = paths.repo_root()
    cfg = load_config(root / args.config)
    out, combos = run_grid(cfg, jobs=args.jobs)
    pick, info = choose(out, combos)
    report = render(out, combos, pick, info, cfg)
    Path(root / args.report).write_text(report, encoding="utf-8", newline="\n")
    print(report.splitlines()[-2])
    if args.write:
        b, lt, ld = combos[pick]
        path = root / args.config
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        raw["hazard"]["beta"] = float(b)
        raw["decoder"]["lam_team"], raw["decoder"]["lam_dead"] = float(lt), float(ld)
        path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8", newline="\n")
        load_config(path)
        print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["BETA_GRID", "LAM_DEAD_GRID", "LAM_TEAM_GRID", "choose", "decode_with", "evidence",
           "run_grid"]
