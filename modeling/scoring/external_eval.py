"""External evaluation harness."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import possession_metrics as pm
from modeling.config import load_config
from modeling.scoring.metrics import score

try:
    from modeling.external.files import load_external, IDSSE_MATCHES
except ImportError:
    IDSSE_MATCHES = []
    def load_external(match, degrade):
        raise NotImplementedError("modeling.external.files is not available yet")

try:
    from modeling.pipeline import predict_labels
except ImportError:
    def predict_labels(master, cfg, holdout):
        raise NotImplementedError("modeling.pipeline is not available yet")

DEGRADE_CHOICES = ("default", "none", "harder", "easier", "hidelo", "hidehi", "fused")


def evaluate_config(cfg: dict, degrade: str = "default", matches: list[str] = None) -> pd.DataFrame:
    if matches is None:
        matches = IDSSE_MATCHES
        
    records = []
    for match in matches:
        master, gt = load_external(match, degrade)
        pred = predict_labels(master, cfg, holdout=match)
        pred_codes = pm.label_codes(pred)
        
        metrics = score(gt, pred_codes)
        metrics["match"] = match
        records.append(metrics)
        
    if not records:
        return pd.DataFrame()
        
    df = pd.DataFrame(records).set_index("match")
    
    # Compute mean and standard error (se)
    n = len(df)
    mean_row = df.mean().to_frame("mean").T
    se_row = (df.std(ddof=1) / np.sqrt(n)).to_frame("se").T if n > 1 else (df * 0.0).rename(index={df.index[0]: "se"})
    
    return pd.concat([df, mean_row, se_row])


def paired_delta(df_a: pd.DataFrame, df_b: pd.DataFrame, metric: str = "spell_f1_mean") -> tuple[float, float]:
    """Paired difference (A - B) over the individual matches."""
    matches = [m for m in df_a.index if m not in ("mean", "se")]
    deltas = df_a.loc[matches, metric] - df_b.loc[matches, metric]
    
    n = len(deltas)
    mean_delta = float(deltas.mean())
    se_delta = float(deltas.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    
    return mean_delta, se_delta


def main():
    parser = argparse.ArgumentParser(description="External evaluation harness")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--degrade", default="default", choices=DEGRADE_CHOICES)
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    cfg_name = cfg["name"]
    
    df = evaluate_config(cfg, degrade=args.degrade)
    
    if df.empty:
        print("No matches available to evaluate.")
        return
        
    markdown_table = df.to_markdown(floatfmt=".4f")
    print(markdown_table)
    
    out_dir = Path("reports/external")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cfg_name}__{args.degrade}.md"
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# External Evaluation: {cfg_name} (degrade: {args.degrade})\n\n")
        f.write(markdown_table)
        f.write("\n")

if __name__ == "__main__":
    main()