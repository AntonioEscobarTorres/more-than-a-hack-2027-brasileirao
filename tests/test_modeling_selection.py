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


def test_ladder_has_the_gbdt_hazard_and_fused_steps_in_order():
    ids = [s["id"] for s in sel.default_ladder()]
    assert (ids.index("dead_pooled") < ids.index("team_gbdt") < ids.index("dead_gbdt")
            < ids.index("post_d_cross_75") < ids.index("durations") < ids.index("hazard")
            < ids.index("fused") == len(ids) - 1)
    steps = {s["id"]: s for s in sel.default_ladder()}
    for sid in ("dead_gbdt", "durations", "hazard"):
        assert steps[sid]["require_stress"] is True and steps[sid]["force_accept"] is False
    assert steps["durations"]["needs_eval"] and steps["hazard"]["needs_eval"]


def _no_resweep(monkeypatch, stages=None):
    monkeypatch.setattr(sel, "resweep_lambda", lambda cfg, evaluate_fn: cfg)
    monkeypatch.setattr(sel, "calibrated_stages",
                        lambda cfg: dict(stages or {"H": 1, "A": 1, "D": 2}))


def test_steps_change_only_their_own_section(monkeypatch):
    """v2 -> team GBDT is v2_gbdt; each later step changes only its own section (D-024..27)."""
    _no_resweep(monkeypatch)
    from modeling import paths
    from modeling.config import config_hash, load_config

    steps = {s["id"]: s for s in sel.default_ladder()}
    cur = steps["team_gbdt"]["build"](steps["v2"]["build"](None))
    ref = load_config(paths.repo_root() / "configs" / "v2_gbdt.yaml")
    assert config_hash({**cur, "name": "v2_gbdt"}) == config_hash(ref)

    def strip(cfg, *sections):
        return {k: v for k, v in cfg.items() if k not in ("name",) + sections}

    nxt = steps["dead_gbdt"]["build"](cur)
    assert nxt["dead"]["kind"] == "gbdt"
    assert nxt["dead"]["calibrated_file"] == "configs/calib/dead_gbdt.yaml"
    assert nxt["dead"]["params"] == cur["dead"]["params"]
    assert strip(nxt, "dead") == strip(cur, "dead")
    cur, nxt = nxt, steps["durations"]["build"](nxt, None)
    assert nxt["decoder"]["duration_stages"] == {"H": 1, "A": 1, "D": 2}
    assert {k: v for k, v in nxt["decoder"].items() if k != "duration_stages"} == \
        {k: v for k, v in cur["decoder"].items() if k != "duration_stages"}
    assert strip(nxt, "decoder") == strip(cur, "decoder")
    cur, nxt = nxt, steps["hazard"]["build"](nxt, None)
    hz = load_config(paths.repo_root() / "configs" / "v2_gbdt_hzn2.yaml")["hazard"]
    assert nxt["hazard"] == hz
    assert strip(nxt, "hazard") == strip(cur, "hazard")
    cur, nxt = nxt, steps["fused"]["build"](nxt)
    assert nxt["ball_source"] == "fused"
    assert strip(nxt, "ball_source") == strip(cur, "ball_source")


def test_ladder_configs_use_repo_relative_paths(monkeypatch):
    """A winner written to final.yaml must not carry a machine-specific absolute path."""
    _no_resweep(monkeypatch)
    steps = {s["id"]: s for s in sel.default_ladder()}
    cur = steps["v2"]["build"](None)
    for sid in ("dead_pooled", "dead_gbdt", "boundary_shift", "hazard"):
        build = steps[sid]["build"]
        cand = build(cur, None) if steps[sid].get("needs_eval") else build(cur)
        for section in ("dead", "hazard"):
            f = cand[section].get("calibrated_file")
            assert f is None or not f.startswith("/"), (sid, f)
        f = cand["postprocess"]["boundary_shift_file"]
        assert f is None or not f.startswith("/"), sid


def test_report_renders_the_seed_step():
    tables = {"M1": 0.40, "M2": 0.42, "M3": 0.41}

    def fake_eval(cfg, degrade):
        return _table(tables)

    steps = [{"id": "seed", "label": "seed", "build": lambda _c: {"name": "a"},
              "require_stress": True, "force_accept": True, "absolute": True}]
    result = sel.run_ladder(steps, evaluate_fn=fake_eval)
    text = sel.render_report(result)
    assert "seeded ladder with `a`" in text


def test_durations_identity_is_skipped(monkeypatch):
    """Stages equal to the current ones (e.g. v1s, D-027 item 6): no-op, nothing evaluated."""
    _no_resweep(monkeypatch, {"H": 1, "A": 1, "D": 1})
    calls = []

    def fake_eval(cfg, degrade):
        calls.append((cfg["name"], degrade))
        return _table({"M1": 0.40, "M2": 0.42, "M3": 0.41})

    v1s = sel._load_named("v1s")
    steps = [{"id": "seed", "label": "seed", "build": lambda _c: v1s, "absolute": True,
              "require_stress": True, "force_accept": True},
             {**{s["id"]: s for s in sel.default_ladder()}["durations"]}]
    result = sel.run_ladder(steps, evaluate_fn=fake_eval)
    assert result["history"][1]["skipped"] is True
    assert result["winner"] is v1s
    assert calls == [("v1s", "default")]


def test_resweep_lambda_picks_a_grid_point():
    """With a flat score every λ is on the plateau; the pick is a valid grid config."""
    def fake_eval(cfg, degrade):
        return _table({"M1": 0.40, "M2": 0.42, "M3": 0.41})

    base = sel._load_named("v2_gbdt")
    out = sel.resweep_lambda(base, fake_eval)
    from modeling.scoring.sweep import LAM_GRID
    assert out["decoder"]["lam_team"] in LAM_GRID and out["decoder"]["lam_dead"] in LAM_GRID
    assert {k: v for k, v in out.items() if k != "decoder"} == \
        {k: v for k, v in base.items() if k != "decoder"}


def test_step_must_also_beat_its_control():
    """A gain that the control (λ re-sweep alone) also gets is not credited (D-027 item 7)."""
    base = {"M1": 0.40, "M2": 0.42, "M3": 0.43}
    up = {"M1": 0.45, "M2": 0.46, "M3": 0.48}

    def fake_eval(cfg, degrade):
        return _table(up if cfg["name"] in ("cand", "ctrl") else base)

    def run(control_name):
        steps = [{"id": "seed", "label": "seed", "build": lambda _c: {"name": "cur"},
                  "absolute": True, "require_stress": True, "force_accept": True},
                 {"id": "step", "label": "step", "build": lambda c: {"name": "cand"},
                  "control_build": lambda c, ev: {"name": control_name},
                  "require_stress": True, "force_accept": False}]
        return sel.run_ladder(steps, evaluate_fn=fake_eval)

    same = run("ctrl")
    assert same["history"][1]["accepted"] is False
    assert same["history"][1]["control"]["accepted"] is False
    assert same["winner"]["name"] == "cur"
    weaker = run("cur")  # control scores like the current config: the step keeps its gain
    assert weaker["history"][1]["accepted"] is True and weaker["winner"]["name"] == "cand"
