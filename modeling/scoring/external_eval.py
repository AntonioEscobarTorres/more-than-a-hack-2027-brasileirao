"""External evaluation harness (task F1-03).

Runs a config through ``predict_labels`` (C8) on the external IDSSE matches (C9) and scores
each match with the official evaluator. Usage::

    python -m modeling.scoring.external_eval --config configs/baseline.yaml --degrade default

The report goes to ``reports/external/<config name>__<degrade>.md`` at the repository root.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import possession_metrics as pm
from modeling import paths
from modeling.config import config_hash, load_config
from modeling.external.files import IDSSE_MATCHES, load_external
from modeling.pipeline import predict_labels
from modeling.scoring.metrics import score

SUMMARY_ROWS = ("mean", "se")
DEGRADE_CHOICES = ("default", "none", "harder", "easier", "fused")


def evaluate_config(cfg: dict, degrade: str = "default", matches=None) -> pd.DataFrame:
    """One row of metrics per match, then a ``mean`` row and an ``se`` row (std ddof=1 / sqrt n).

    With a single match the standard error is undefined and the ``se`` row is NaN.
    """
    matches = list(IDSSE_MATCHES if matches is None else matches)
    if not matches:
        raise ValueError("no matches to evaluate")
    clash = sorted(set(matches) & set(SUMMARY_ROWS))
    if clash:
        raise ValueError("match ids clash with summary rows: {}".format(clash))

    records = []
    for match in matches:
        master, gt = load_external(match, degrade)
        pred = predict_labels(master, cfg, holdout=match)
        records.append(score(gt, pm.label_codes(pred)))

    df = pd.DataFrame(records, index=pd.Index(matches, name="match"))
    n = len(df)
    mean_row = df.mean()
    se_row = df.std(ddof=1) / math.sqrt(n)  # NaN when n == 1
    summary = pd.DataFrame([mean_row, se_row], index=list(SUMMARY_ROWS))
    return pd.concat([df.astype(np.float64), summary])


def _match_rows(df: pd.DataFrame) -> list:
    return [m for m in df.index if m not in SUMMARY_ROWS]


def paired_delta(df_a: pd.DataFrame, df_b: pd.DataFrame,
                 metric: str = "spell_f1_mean") -> tuple[float, float]:
    """Mean and standard error of the per-match difference ``A - B``.

    Both tables must cover the same matches; the standard error is NaN with a single match.
    """
    matches_a, matches_b = _match_rows(df_a), _match_rows(df_b)
    if set(matches_a) != set(matches_b):
        raise ValueError("paired_delta needs the same matches in both tables: {} vs {}".format(
            sorted(matches_a), sorted(matches_b)))
    if not matches_a:
        raise ValueError("no matches to compare")
    deltas = (df_a.loc[matches_a, metric] - df_b.loc[matches_a, metric]).to_numpy(np.float64)
    n = deltas.size
    mean_delta = float(deltas.mean())
    se_delta = float(deltas.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    return mean_delta, se_delta


def _fmt(value) -> str:
    if isinstance(value, (float, np.floating)):
        return "nan" if np.isnan(value) else "{:.4f}".format(value)
    return str(value)


def to_markdown(df: pd.DataFrame) -> str:
    """GitHub markdown table with 4 decimals (no optional pandas dependency)."""
    header = [df.index.name or "match"] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join([":---"] + ["---:"] * len(df.columns)) + "|"]
    for label, row in df.iterrows():
        lines.append("| " + " | ".join([str(label)] + [_fmt(v) for v in row]) + " |")
    return "\n".join(lines)


def report_dir() -> Path:
    return paths.repo_root() / "reports" / "external"


def render_report(df: pd.DataFrame, cfg: dict, degrade: str) -> str:
    return "\n".join([
        "# External evaluation: {} (degrade: {})".format(cfg["name"], degrade),
        "",
        "- config: `{}`".format(cfg["name"]),
        "- config_hash: `{}`".format(config_hash(cfg)),
        "- degrade: `{}`".format(degrade),
        "- matches: {}".format(len(_match_rows(df))),
        "",
        to_markdown(df),
        "",
    ])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="External evaluation harness (IDSSE)")
    parser.add_argument("--config", required=True, help="path to a configs/*.yaml file")
    parser.add_argument("--degrade", default="default", choices=DEGRADE_CHOICES)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    print("config: {} ({})".format(cfg["name"], config_hash(cfg)))
    df = evaluate_config(cfg, degrade=args.degrade)
    report = render_report(df, cfg, args.degrade)
    print(report)

    out_dir = report_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "{}__{}.md".format(cfg["name"], args.degrade)
    with open(out_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(report)
    print("wrote {}".format(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
