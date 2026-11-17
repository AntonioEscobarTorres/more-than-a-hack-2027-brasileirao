"""Synthetic tests for the Section 6 selection protocol (task F2-06)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from modeling.scoring import selection as sel


def _table(spell_by_match: dict[str, float], macro: float = 0.5) -> pd.DataFrame:
    matches = list(spell_by_match)
    records = []
    for m in matches:
        records.append({
            "spell_f1_mean": spell_by_match[m],
            "macro_f1": macro,
            "alive_dead_f1": 0.8,
            "trans_f1_25": 0.3,
            "mean_abs_latency_25": 5.0,
            "n_gt_spells": 100.0,
            "n_pred_spells": 80.0,
        })
    df = pd.DataFrame(records, index=pd.Index(matches, name="match"))
    mean_row = df.mean()
    se_row = df.std(ddof=1) / math.sqrt(len(df))
    summary = pd.DataFrame([mean_row, se_row], index=["mean", "se"])
    return pd.concat([df.astype(np.float64), summary])


def test_gain_and_stress_rules():
    assert sel.gain_accepted(0.02, 0.01) is True
    assert sel.gain_accepted(0.01, 0.01) is False
    assert sel.stress_ok(-0.005, 0.01) is True
    assert sel.stress_ok(-0.02, 0.01) is False


def test_evaluate_step_accepts_clear_gain(monkeypatch):
    current = {"name": "a", "pipeline": "baseline"}
    candidate = {"name": "b", "pipeline": "baseline"}

    def fake_eval(cfg, degrade):
        base = {"M1": 0.40, "M2": 0.42, "M3": 0.41}
        if cfg["name"] == "b":
            if degrade == "default":
                return _table({k: v + 0.05 for k, v in base.items()})
            return _table({k: v + 0.02 for k, v in base.items()})
        return _table(base)

    decision = sel.evaluate_step(current, candidate, evaluate_fn=fake_eval)
    assert decision["accepted"] is True
    assert decision["primary"]["gain_ok"] is True
    assert all(row["ok"] for row in decision["stress"])


def test_evaluate_step_rejects_stress_regression():
    current = {"name": "a"}
    candidate = {"name": "b"}

    def fake_eval(cfg, degrade):
        base = {"M1": 0.50, "M2": 0.50, "M3": 0.50, "M4": 0.50}
        if cfg["name"] == "b" and degrade == "default":
            return _table({k: v + 0.05 for k, v in base.items()})
        if cfg["name"] == "b" and degrade == "harder":
            return _table({k: v - 0.10 for k, v in base.items()})
        return _table(base)

    decision = sel.evaluate_step(current, candidate, evaluate_fn=fake_eval)
    assert decision["primary"]["gain_ok"] is True
    assert decision["accepted"] is False
    harder = next(r for r in decision["stress"] if r["degrade"] == "harder")
    assert harder["ok"] is False


def test_fused_step_skips_stress():
    current = {"name": "a"}
    candidate = {"name": "a_fused"}

    def fake_eval(cfg, degrade):
        assert degrade == "default"
        base = {"M1": 0.40, "M2": 0.40, "M3": 0.40}
        if "fused" in cfg["name"]:
            return _table({k: v + 0.03 for k, v in base.items()})
        return _table(base)

    decision = sel.evaluate_step(
        current, candidate, evaluate_fn=fake_eval, require_stress=False)
    assert decision["accepted"] is True
    assert decision["stress"] == []


def test_run_ladder_with_fake_eval():
    scores = {
        ("baseline", "default"): 0.20,
        ("v1", "default"): 0.30,
        ("v1s", "default"): 0.32,
    }

    def fake_eval(cfg, degrade):
        name = cfg["name"]
        # Map mutated names back to a family score.
        key = "baseline"
        for cand in ("v2_gbdt_fused", "v2_gbdt", "v2", "v1s", "v1", "baseline"):
            if name.startswith(cand) or cand in name:
                key = cand
                break
        if name == "baseline":
            key = "baseline"
        elif name == "v1":
            key = "v1"
        elif name == "v1s":
            key = "v1s"
        elif name.startswith("v2_base"):
            key = "v2_base"
        elif name == "v2" or name.startswith("v2_"):
            key = "v2"
        base = {
            "baseline": 0.20, "v1": 0.30, "v1s": 0.33,
            "v2_base": 0.40, "v2": 0.45,
        }.get(key, 0.45)
        # Small degrade penalty so stress always passes when gain is real.
        penalty = {"default": 0.0, "harder": 0.01, "easier": 0.0,
                   "hidelo": 0.01, "hidehi": 0.0}.get(degrade, 0.0)
        val = base - penalty
        return _table({"M1": val, "M2": val + 0.01, "M3": val - 0.01})

    # Short ladder: baseline → v1 → v1s
    steps = [
        {"id": "baseline", "label": "baseline",
         "build": lambda _c: {"name": "baseline", "pipeline": "baseline",
                              "ball_source": "tactical",
                              "baseline": {"d_gap_frames": 100, "slow_speed_mps": 1.5,
                                           "min_spell_frames": 25}},
         "require_stress": True, "force_accept": True, "absolute": True},
        {"id": "v1", "label": "v1",
         "build": lambda _c: sel._load_named("v1"),
         "require_stress": True, "force_accept": False},
    ]
    # Avoid loading full hmm for baseline absolute seed only + skip second if heavy.
    # Use pure synthetic configs without validate for the unit test of run_ladder loop.
    def build_a(_):
        return {"name": "a"}

    def build_b(current):
        return {"name": "b"}

    short = [
        {"id": "a", "label": "a", "build": build_a, "require_stress": True,
         "force_accept": True, "absolute": True},
        {"id": "b", "label": "b", "build": build_b, "require_stress": True,
         "force_accept": False},
    ]

    def eval2(cfg, degrade):
        val = 0.2 if cfg["name"] == "a" else 0.35
        if degrade != "default":
            val -= 0.01
        return _table({"M1": val, "M2": val, "M3": val})

    result = sel.run_ladder(short, evaluate_fn=eval2)
    assert result["winner"]["name"] == "b"
    assert result["history"][0]["accepted"] is True
    assert result["history"][1]["accepted"] is True


def test_write_final_and_report(tmp_path):
    cfg = sel._load_named("baseline")
    path = sel.write_final(cfg, tmp_path / "final.yaml")
    loaded = __import__("modeling.config", fromlist=["load_config"]).load_config(path)
    assert loaded["name"] == "final"
    report = sel.render_report({
        "winner": loaded,
        "history": [{
            "id": "baseline", "label": "baseline", "accepted": True, "forced": True,
            "absolute": True, "candidate_name": "baseline",
            "primary": {
                "candidate_mean": 0.2, "candidate_se": 0.01, "candidate_ratio": 0.8,
                "candidate_table": _table({"M1": 0.2, "M2": 0.2}),
            },
            "stress": [],
        }],
    }, local_checks={"csv_valid": "PASS"})
    assert "Winner" in report and "PASS" in report
