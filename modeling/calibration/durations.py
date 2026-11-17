"""Number of duration stages ``r_s`` for the semi-Markov decoder (task F2-08).

After the ``min_dwell`` chain of ``m_s − 1`` sub-states, the decoder can hold a label in
``r_s`` stages, each with self-loop ``p_s``. The segment length is then
``L = (m_s − 1) + Y`` with ``Y`` the sum of ``r_s`` geometric variables on ``{1, 2, …}``, i.e. a
shifted negative binomial. The mean is kept at the config's ``mean_s`` with
``p_s = 1 − r_s / (mean_s − m_s + 1)``.

This module picks ``r_s ∈ R_GRID`` by maximum likelihood on the **true** IDSSE segment lengths
(labels, per half, the F2-03 segmentation of ``modeling.scoring.label_stats``), with ``m_s`` and
``mean_s`` taken from the candidate config. ``all`` uses every match; ``folds[<match>]`` leaves
that match out. BAR-ATH is never read.

Support. A segment shorter than ``m_s − 1 + r_s`` has probability zero under ``r_s`` stages, so
the plain likelihoods of different ``r`` would be computed on different segments. Every ``r`` is
therefore scored on the same segments, those with ``L ≥ m_s − 1 + max(R_GRID)``, with the
likelihood truncated to that support (``log P(L) − log P(L ≥ L0)``). The mean constraint stays the
config's (the decoder's) mean, not the mean of the kept segments.

Usage::

    python -m modeling.calibration.durations --config configs/v1s.yaml \\
        --out configs/calib/durations.yaml --report reports/durations.md
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import yaml
from scipy.special import gammaln

import games
from modeling.config import load_config
from modeling.contracts import STATES
from modeling.external.files import IDSSE_MATCHES, ExternalMatch, arrays_path
from modeling.scoring.label_stats import stats_from_match

# F2-08 grid of stage counts; ties go to the smallest r (fewest sub-states).
R_GRID = (1, 2, 3, 4, 6, 8)
# Text histogram of the report, per state: (bin width, last edge) in seconds, then one open bin.
HIST_BINS = {"H": (1, 15), "A": (1, 15), "D": (4, 60)}


def stage_part_mean(mean_frames: float, min_dwell: int) -> float:
    """Mean of the stage part ``Y = L − (m − 1)``: ``mean − m + 1``."""
    return float(mean_frames) - int(min_dwell) + 1.0


def advance_prob(mean_frames: float, min_dwell: int, stages: int) -> float:
    """Per-stage exit probability ``1 − p_s = r / (mean − m + 1)`` that keeps the mean.

    Raises ``ValueError`` when ``mean − m + 1 <= r`` (no valid ``p_s`` in ``[0, 1)``).
    """
    mu = stage_part_mean(mean_frames, min_dwell)
    stages = int(stages)
    if stages < 1:
        raise ValueError("duration stages must be >= 1, got {}".format(stages))
    if not mu > stages:
        raise ValueError(
            "mean_frames − min_dwell + 1 ({}) must exceed duration_stages ({})".format(
                mu, stages))
    return stages / mu


def length_logpmf(lengths, mean_frames: float, min_dwell: int, stages: int) -> np.ndarray:
    """Log-pmf of segment lengths under ``min_dwell`` + ``stages`` geometric stages.

    ``Y = L − (m − 1) ≥ r`` and ``P(Y = y) = C(y − 1, r − 1) q^r (1 − q)^(y − r)`` with
    ``q = r / (mean − m + 1)``. Lengths outside the support get ``-inf``.
    """
    lengths = np.asarray(lengths, dtype=np.int64)
    q = advance_prob(mean_frames, min_dwell, stages)
    r = int(stages)
    y = (lengths - (int(min_dwell) - 1)).astype(np.float64)
    out = np.full(lengths.shape, -np.inf, dtype=np.float64)
    ok = y >= r
    yy = y[ok]
    out[ok] = (gammaln(yy) - gammaln(float(r)) - gammaln(yy - r + 1.0)
               + r * math.log(q) + (yy - r) * math.log1p(-q))
    return out


def support_floor(min_dwell: int, grid: Sequence[int] = R_GRID) -> int:
    """Shortest length every ``r`` in ``grid`` can produce: ``m − 1 + max(grid)``."""
    return int(min_dwell) - 1 + int(max(grid))


def truncated_loglik(lengths, mean_frames: float, min_dwell: int, stages: int,
                     floor: int) -> float:
    """Log-likelihood of ``lengths ≥ floor`` under the length model truncated at ``floor``."""
    lengths = np.asarray(lengths, dtype=np.int64)
    kept = lengths[lengths >= floor]
    if kept.size == 0:
        return 0.0
    below = np.arange(int(min_dwell) - 1 + int(stages), int(floor), dtype=np.int64)
    p_below = float(np.exp(length_logpmf(below, mean_frames, min_dwell, stages)).sum()) \
        if below.size else 0.0
    if not p_below < 1.0:
        return -math.inf
    ll = length_logpmf(kept, mean_frames, min_dwell, stages)
    return float(ll.sum() - kept.size * math.log1p(-p_below))


def choose_stages(lengths: np.ndarray, mean_frames: float, min_dwell: int,
                  grid: Sequence[int] = R_GRID) -> tuple[int, dict[int, float]]:
    """Maximum-likelihood ``r`` in ``grid`` (smallest on ties) and the log-likelihood table.

    Values of ``r`` with ``mean − m + 1 <= r`` are infeasible and get ``-inf``.
    """
    floor = support_floor(min_dwell, grid)
    table: dict[int, float] = {}
    best_r, best_ll = None, -math.inf
    for r in sorted(int(v) for v in grid):
        if not stage_part_mean(mean_frames, min_dwell) > r:
            table[r] = -math.inf
            continue
        ll = truncated_loglik(lengths, mean_frames, min_dwell, r, floor)
        table[r] = ll
        if best_r is None or ll > best_ll:
            best_r, best_ll = r, ll
    if best_r is None:
        raise ValueError("no feasible r in {} for mean {} and min_dwell {}".format(
            list(grid), mean_frames, min_dwell))
    return best_r, table


def dispersion_ratio(lengths: np.ndarray, mean_frames: float, min_dwell: int,
                     floor: int) -> float:
    """Variance of ``Y = L − m + 1`` on ``L ≥ floor`` over the geometric (``r = 1``) variance.

    With the mean held at ``mu = mean − m + 1``, ``r`` stages give ``Var Y = mu (mu − r) / r``,
    which decreases in ``r``: a ratio at or above 1 means the data are at least as dispersed as
    ``r = 1`` allows, so no ``r > 1`` can fit them better on spread.
    """
    lengths = np.asarray(lengths, dtype=np.int64)
    kept = lengths[lengths >= floor]
    if kept.size < 2:
        return float("nan")
    y = (kept - (int(min_dwell) - 1)).astype(np.float64)
    mu = stage_part_mean(mean_frames, min_dwell)
    return float(np.var(y) / (mu * (mu - 1.0)))


# -- IDSSE segments ------------------------------------------------------------------------------

def match_lengths(match: ExternalMatch) -> dict[str, np.ndarray]:
    """True segment lengths (frames) per state for one match, F2-03 segmentation (per half)."""
    out: dict[str, list[int]] = {s: [] for s in STATES}
    for half in stats_from_match(match):
        for seg in half.segments:
            out[seg.state].append(int(seg.length))
    return {s: np.asarray(v, dtype=np.int64) for s, v in out.items()}


def load_lengths(matches: Sequence[str]) -> dict[str, dict[str, np.ndarray]]:
    """``{match: {state: lengths}}`` from ``$MTAH_EXT_DIR/arrays/<match>.npz`` (labels only)."""
    return {m: match_lengths(ExternalMatch.load(arrays_path(m))) for m in matches}


def pool(per_match: Mapping[str, Mapping[str, np.ndarray]],
         matches: Sequence[str]) -> dict[str, np.ndarray]:
    return {s: np.concatenate([np.asarray(per_match[m][s], dtype=np.int64) for m in matches])
            if matches else np.empty(0, dtype=np.int64) for s in STATES}


def decoder_durations(cfg_decoder: Mapping, fps: float) -> tuple[dict, dict]:
    """``(mean_frames, min_dwell)`` per state from a validated ``decoder`` section."""
    mean = {s: float(cfg_decoder["mean_duration_s"][s]) * float(fps) for s in STATES}
    dwell = {s: int(cfg_decoder["min_dwell_frames"][s]) for s in STATES}
    return mean, dwell


def fit_stages(lengths: Mapping[str, np.ndarray], mean: Mapping[str, float],
               dwell: Mapping[str, int], grid: Sequence[int] = R_GRID):
    """``({state: r}, {state: {r: loglik}})`` for one pool of segments."""
    chosen, tables = {}, {}
    for s in STATES:
        chosen[s], tables[s] = choose_stages(lengths[s], mean[s], dwell[s], grid)
    return chosen, tables


def calibrate_durations(per_match: Mapping[str, Mapping[str, np.ndarray]], cfg_decoder: Mapping,
                        *, fps: float, grid: Sequence[int] = R_GRID) -> dict:
    """``all`` (every match) and leave-one-match-out ``folds`` stage counts, plus the tables."""
    matches = list(per_match)
    if not matches:
        raise ValueError("no matches to calibrate")
    mean, dwell = decoder_durations(cfg_decoder, fps)
    chosen_all, tables_all = fit_stages(pool(per_match, matches), mean, dwell, grid)
    folds, fold_tables = {}, {}
    for held in matches:
        rest = [m for m in matches if m != held]
        folds[held], fold_tables[held] = fit_stages(pool(per_match, rest), mean, dwell, grid)
    return {"all": chosen_all, "folds": folds, "tables_all": tables_all,
            "tables_folds": fold_tables, "mean_frames": mean, "min_dwell": dwell}


def calib_payload(result: Mapping) -> dict:
    """The ``configs/calib/durations.yaml`` mapping: ``all`` and ``folds`` only."""
    return {
        "all": {s: int(result["all"][s]) for s in STATES},
        "folds": {m: {s: int(v[s]) for s in STATES} for m, v in result["folds"].items()},
    }


def write_calib(result: Mapping, path: Path, *, config_name: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Generated by modeling.calibration.durations (F2-08) from the true IDSSE segment lengths.\n"
        "# decoder.duration_stages by maximum likelihood over r in {} with m_s and mean_s of\n"
        "# configs/{}.yaml; folds leave one match out. BAR-ATH is never read.\n"
    ).format(list(R_GRID), config_name)
    path.write_text(header + yaml.safe_dump(calib_payload(result), sort_keys=False),
                    encoding="utf-8")
    return path


# -- report --------------------------------------------------------------------------------------

def histogram_lines(lengths: Mapping[str, np.ndarray], fps: float,
                    bins: Mapping[str, tuple[int, int]] | None = None,
                    width: int = 40) -> list[str]:
    """Text histogram of the segment lengths per state: ``bins[s] = (bin_s, max_s)`` gives
    ``bin_s``-second bins up to ``max_s`` seconds, then one open bin."""
    bins = HIST_BINS if bins is None else bins
    lines = []
    for s in STATES:
        step, top_s = bins[s]
        sec = np.asarray(lengths[s], dtype=np.float64) / float(fps)
        edges = list(range(0, top_s, step))
        counts = [int(np.count_nonzero((sec >= b) & (sec < b + step))) for b in edges]
        counts.append(int(np.count_nonzero(sec >= top_s)))
        labels = ["{}–{}".format(b, b + step) for b in edges] + ["≥{}".format(top_s)]
        top = max(counts) or 1
        lines.append("```text")
        lines.append("{} (n = {}, {} s bins, # = {:.1f} segments)".format(
            s, int(sec.size), step, top / width))
        pad = max(len(label) for label in labels)
        for label, c in zip(labels, counts):
            lines.append("{} | {:>4} {}".format(label.rjust(pad), c,
                                                 "#" * int(round(width * c / top))))
        lines.append("```")
        lines.append("")
    return lines


def report_lines(result: Mapping, per_match: Mapping[str, Mapping[str, np.ndarray]], *,
                 config_name: str, fps: float, grid: Sequence[int] = R_GRID) -> list[str]:
    pooled = pool(per_match, list(per_match))
    mean, dwell = result["mean_frames"], result["min_dwell"]
    lines = [
        "## Calibration of `duration_stages` on `configs/{}.yaml`".format(config_name),
        "",
        "True IDSSE segment lengths ({} matches, per half, X never forms a segment). Mean and "
        "`min_dwell` from the config; every `r` is scored on the segments with "
        "`L ≥ m − 1 + {}` and the likelihood truncated there.".format(len(per_match), max(grid)),
        "",
        "| State | m (frames) | mean (frames) | segments | kept (L ≥ floor) | median L | "
        "Var Y / Var geometric | r (all) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in STATES:
        floor = support_floor(dwell[s], grid)
        L = np.asarray(pooled[s])
        lines.append("| {} | {} | {:.1f} | {} | {} | {:.0f} | {:.2f} | {} |".format(
            s, dwell[s], mean[s], L.size, int(np.count_nonzero(L >= floor)),
            float(np.median(L)) if L.size else float("nan"),
            dispersion_ratio(L, mean[s], dwell[s], floor), result["all"][s]))
    lines.extend(["", "Log-likelihood by `r` (all matches; Δ against the best):", "",
                  "| State | " + " | ".join("r = {}".format(r) for r in grid) + " |",
                  "|---|" + "---:|" * len(grid)])
    for s in STATES:
        table = result["tables_all"][s]
        best = max(table.values())
        cells = []
        for r in grid:
            v = table[r]
            cells.append("—" if not math.isfinite(v) else "{:.1f} ({:+.1f})".format(v, v - best))
        lines.append("| {} | {} |".format(s, " | ".join(cells)))
    lines.extend(["", "Leave-one-match-out choice:", "",
                  "| Held-out match | " + " | ".join(STATES) + " |",
                  "|---|" + "---:|" * len(STATES)])
    for m, row in result["folds"].items():
        lines.append("| {} | {} |".format(m, " | ".join(str(row[s]) for s in STATES)))
    lines.append("")
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, help="candidate HMM config (m_s, mean_s)")
    parser.add_argument("--out", default=None, help="write the calib YAML (all / folds)")
    parser.add_argument("--report", default=None, help="write the markdown calibration section")
    parser.add_argument("--histogram", action="store_true",
                        help="append the text histogram of the IDSSE lengths to the report")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if cfg["pipeline"] != "hmm":
        raise SystemExit("--config must be an hmm config")
    fps = float(games.GAMES[games.GAME]["fps"])
    per_match = load_lengths(IDSSE_MATCHES)
    result = calibrate_durations(per_match, cfg["decoder"], fps=fps)
    print("all:", result["all"])
    for m, row in result["folds"].items():
        print("fold", m, row)
    if args.out:
        print("wrote", write_calib(result, Path(args.out), config_name=cfg["name"]))
    if args.report:
        lines = report_lines(result, per_match, config_name=cfg["name"], fps=fps)
        if args.histogram:
            lines.extend(["### IDSSE segment lengths by state", ""]
                         + histogram_lines(pool(per_match, list(per_match)), fps) + [""])
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text("\n".join(lines), encoding="utf-8")
        print("wrote", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
