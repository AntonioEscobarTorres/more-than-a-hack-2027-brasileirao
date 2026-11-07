"""Tests for scoring and external evaluation."""
import numpy as np
import pandas as pd
from evaluation import possession_metrics as pm

from modeling.scoring.metrics import score
from modeling.scoring.external_eval import evaluate_config, paired_delta


def test_perfect_prediction_gives_one():
    # 0=H, 1=A, 2=D, 3=X
    gt = np.array([0]*50 + [2]*15 + [3]*20 + [1]*50, dtype=np.int8)
    pred = gt.copy()
    res = score(gt, pred)
    assert res["spell_f1_mean"] == 1.0
    assert res["macro_f1"] == 1.0


def test_shifted_prediction_non_increasing_f1():
    gt = np.array([0]*200 + [1]*200, dtype=np.int8)
    f1s = []
    for shift in [0, 3, 8, 25]:
        pred = np.roll(gt, shift)
        if shift > 0:
            pred[:shift] = 0
        f1s.append(score(gt, pred)["spell_f1_mean"])
        
    for i in range(len(f1s) - 1):
        assert f1s[i] >= f1s[i + 1]


def test_score_coincides_with_pm_evaluate():
    gt = np.array([0]*100 + [1]*100, dtype=np.int8)
    pred = np.array([0]*80 + [1]*120, dtype=np.int8)
    
    wrapper_res = score(gt, pred)
    official_res = pm.evaluate(gt, pred)
    
    assert wrapper_res["spell_f1_mean"] == official_res["spell_f1_mean"]
    assert wrapper_res["macro_f1"] == official_res["macro_f1"]
    assert wrapper_res["alive_dead_f1"] == official_res["alive_dead_f1"]
    assert wrapper_res["trans_f1_25"] == official_res["transitions"][25]["f1"]
    assert wrapper_res["n_gt_spells"] == official_res["n_gt_spells"]
    assert wrapper_res["n_pred_spells"] == official_res["n_pred_spells"]


def test_evaluate_config(monkeypatch):
    def mock_load_external(match, degrade):
        master = pd.DataFrame({"dummy": np.zeros(100)})
        gt = np.array([0]*50 + [1]*50, dtype=np.int8)
        return master, gt
        
    def mock_predict_labels(master, cfg, holdout):
        return np.array(["H"]*50 + ["A"]*50, dtype="<U1")
        
    import modeling.scoring.external_eval as ext_eval
    monkeypatch.setattr(ext_eval, "load_external", mock_load_external)
    monkeypatch.setattr(ext_eval, "predict_labels", mock_predict_labels)
    
    cfg = {"name": "test"}
    matches = ["m1", "m2"]
    df = ext_eval.evaluate_config(cfg, matches=matches)
    
    assert len(df) == 4
    assert list(df.index) == ["m1", "m2", "mean", "se"]
    assert df.loc["mean", "spell_f1_mean"] == 1.0
    assert df.loc["se", "spell_f1_mean"] == 0.0


def test_paired_delta():
    df_a = pd.DataFrame({"spell_f1_mean": [0.8, 0.9, 0.85, 0.0]}, index=["m1", "m2", "mean", "se"])
    df_b = pd.DataFrame({"spell_f1_mean": [0.7, 0.8, 0.75, 0.0]}, index=["m1", "m2", "mean", "se"])
    
    mean_d, se_d = paired_delta(df_a, df_b, "spell_f1_mean")
    assert np.isclose(mean_d, 0.1)
    assert np.isclose(se_d, 0.0)