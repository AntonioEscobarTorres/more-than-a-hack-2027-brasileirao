"""Scoring metrics wrapper for the official evaluator."""
import numpy as np
from evaluation import possession_metrics as pm

def score(gt_codes: np.ndarray, pred_codes: np.ndarray) -> dict:
    """Wrapper around the official possession_metrics.evaluate."""
    res = pm.evaluate(gt_codes, pred_codes)
    
    latency = res["transitions"][25]["mean_abs_latency"]
    
    return {
        "spell_f1_mean": res["spell_f1_mean"],
        "macro_f1": res["macro_f1"],
        "alive_dead_f1": res["alive_dead_f1"],
        "trans_f1_25": res["transitions"][25]["f1"],
        "mean_abs_latency_25": latency if latency is not None else np.nan,
        "n_gt_spells": res["n_gt_spells"],
        "n_pred_spells": res["n_pred_spells"],
    }