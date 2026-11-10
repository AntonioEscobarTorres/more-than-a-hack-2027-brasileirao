"""Tests for scoring and external evaluation (task F1-03).

Synthetic data only: masters are built in memory and every file lives in tmp_path.
"""
import math

import numpy as np
import pandas as pd
import pytest
from evaluation import possession_metrics as pm

import modeling.scoring.external_eval as ext_eval
from modeling.config import config_hash, load_config
from modeling.contracts import LABEL_CODES
from modeling.external.files import GAP_FRAMES, X_CODE, expected_half
from modeling.paths import repo_root
from modeling.scoring.metrics import score
from modeling.scoring.external_eval import evaluate_config, paired_delta
from preprocessing.pipeline_api import build_master_from_arrays

BASELINE_YAML = repo_root() / "configs" / "baseline.yaml"
N_HALF = 300


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
    latency = official_res["transitions"][25]["mean_abs_latency"]
    if latency is None:
        assert np.isnan(wrapper_res["mean_abs_latency_25"])
    else:
        assert wrapper_res["mean_abs_latency_25"] == latency
    assert wrapper_res["n_gt_spells"] == official_res["n_gt_spells"]
    assert wrapper_res["n_pred_spells"] == official_res["n_pred_spells"]


def test_evaluate_config_summary_rows(monkeypatch):
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


def test_paired_delta_single_match_se_is_nan():
    df_a = pd.DataFrame({"spell_f1_mean": [0.8]}, index=["m1"])
    df_b = pd.DataFrame({"spell_f1_mean": [0.6]}, index=["m1"])
    mean_d, se_d = paired_delta(df_a, df_b)
    assert np.isclose(mean_d, 0.2)
    assert math.isnan(se_d)


def test_paired_delta_rejects_different_matches():
    df_a = pd.DataFrame({"spell_f1_mean": [0.8, 0.9]}, index=["m1", "m2"])
    df_b = pd.DataFrame({"spell_f1_mean": [0.7, 0.8]}, index=["m1", "m3"])
    with pytest.raises(ValueError, match="same matches"):
        paired_delta(df_a, df_b)


def test_evaluate_config_rejects_empty_matches():
    with pytest.raises(ValueError, match="no matches"):
        evaluate_config({"name": "x"}, matches=[])


# -- C9 synthetic matches through the real pipeline ----------------------------------------------

def _c9_match(seed: int):
    """A C9-layout master (half 1, GAP_FRAMES of half 0, half 2) and its int8 C0 labels.

    Possession alternates every 100 frames; the owning team stands next to the ball, and the
    teams swap ends in the second half.
    """
    rng = np.random.default_rng(seed)
    half = expected_half(N_HALF, N_HALF)
    n = half.size
    first2 = N_HALF + GAP_FRAMES
    blocks = [(1, 0, N_HALF - 1), (2, first2, n - 1)]
    players = np.full((n, 6, 3), np.nan, dtype=np.float32)
    ball = np.full((n, 3), np.nan, dtype=np.float32)
    labels = np.full(n, X_CODE, dtype=np.int8)
    for h, f0, f1 in blocks:
        t = np.arange(f1 - f0 + 1)
        owner = (t // 100) % 2  # 0 = home, 1 = away
        bx = 0.5 + 0.0002 * np.sin(t / 30.0) + rng.normal(0, 1e-5, t.size)
        ball[f0:f1 + 1, 0] = bx
        ball[f0:f1 + 1, 1] = 0.5
        ball[f0:f1 + 1, 2] = 1.0
        near = np.where(owner == 0, 0.005, 0.20)
        players[f0:f1 + 1, :3, 0] = (bx - near)[:, None] - np.array([0, .05, .10])
        players[f0:f1 + 1, 3:, 0] = (bx + 0.205 - near)[:, None] + np.array([0, .05, .10])
        players[f0:f1 + 1, :, 1] = 0.5
        players[f0:f1 + 1, :, 2] = 3.0
        if h == 2:  # teams swap ends at halftime
            players[f0:f1 + 1, :, 0] = 1.0 - players[f0:f1 + 1, :, 0]
            ball[f0:f1 + 1, 0] = 1.0 - ball[f0:f1 + 1, 0]
        labels[f0:f1 + 1] = np.where(owner == 0, LABEL_CODES["H"], LABEL_CODES["A"])
    master = build_master_from_arrays(
        players, ball, np.array([0, 0, 0, 1, 1, 1], np.int8), blocks=blocks, pitch=(105.0, 68.0))
    assert (master["half"].to_numpy() == half).all()
    return master, labels


@pytest.fixture(scope="module")
def c9_matches():
    return {"S1": _c9_match(1), "S2": _c9_match(2)}


@pytest.fixture
def patched_external(monkeypatch, c9_matches):
    calls = []

    def fake_load_external(match, degrade):
        calls.append((match, degrade))
        master, labels = c9_matches[match]
        return master.copy(), labels.copy()

    monkeypatch.setattr(ext_eval, "load_external", fake_load_external)
    monkeypatch.setattr(ext_eval, "IDSSE_MATCHES", tuple(c9_matches))
    return calls


def test_evaluate_config_c9_baseline(patched_external):
    cfg = load_config(BASELINE_YAML)
    df = evaluate_config(cfg, degrade="none", matches=["S1", "S2"])

    assert list(df.index) == ["S1", "S2", "mean", "se"]
    assert patched_external == [("S1", "none"), ("S2", "none")]
    for match in ("S1", "S2"):
        assert 0.0 <= df.loc[match, "spell_f1_mean"] <= 1.0
        assert df.loc[match, "n_gt_spells"] > 0
    per_match = df.loc[["S1", "S2"], "spell_f1_mean"]
    assert np.isclose(df.loc["mean", "spell_f1_mean"], per_match.mean())
    assert np.isclose(df.loc["se", "spell_f1_mean"], per_match.std(ddof=1) / math.sqrt(2))


def test_evaluate_config_single_match_se_is_nan(patched_external):
    df = evaluate_config(load_config(BASELINE_YAML), degrade="none", matches=["S1"])
    assert list(df.index) == ["S1", "mean", "se"]
    assert df.loc["se"].isna().all()


def test_cli_writes_report_at_repo_reports_dir(patched_external, tmp_path, monkeypatch, capsys):
    out_dir = tmp_path / "reports" / "external"
    monkeypatch.setattr(ext_eval, "report_dir", lambda: out_dir)
    monkeypatch.chdir(tmp_path)  # the report location must not depend on the cwd

    assert ext_eval.main(["--config", str(BASELINE_YAML), "--degrade", "none"]) == 0

    report = (out_dir / "baseline__none.md").read_text(encoding="utf-8")
    cfg = load_config(BASELINE_YAML)
    assert config_hash(cfg) in report
    assert "| S1 |" in report and "| S2 |" in report and "| se |" in report
    assert config_hash(cfg) in capsys.readouterr().out


def test_report_dir_is_under_repo_root():
    assert ext_eval.report_dir() == repo_root() / "reports" / "external"
