"""Synthetic tests for the decoder sensitivity sweep helpers (task F2-05)."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import yaml

from modeling.config import load_config, validate_config
from modeling.scoring import sweep
from tests.modeling_fixtures import dead_params, team_params


def _hmm_cfg(**decoder_over) -> dict:
    cfg = {
        "name": "sweep_t",
        "pipeline": "hmm",
        "ball_source": "tactical",
        "team": {"kind": "rules_v1", "params": team_params(), "calibrated_file": None},
        "dead": {"kind": "none", "params": dead_params(), "calibrated_file": None},
        "decoder": {
            "lam_team": 0.2,
            "lam_dead": 0.2,
            "mean_duration_s": {"H": 15.0, "A": 15.0, "D": 20.0},
            "exit_split": {"H": {"A": 0.5, "D": 0.5}, "A": {"H": 0.5, "D": 0.5},
                           "D": {"H": 0.5, "A": 0.5}},
            "min_dwell_frames": {"H": 12, "A": 12, "D": 25},
        },
        "postprocess": {
            "midpoint_neutral_switch": False,
            "boundary_shift_file": None,
            "d_bridge_frames": 0,
            "d_cross_frames": 0,
            "min_spell_frames": 0,
        },
    }
    cfg["decoder"].update(decoder_over)
    return validate_config(cfg)


def test_select_plateau_prefers_centre_of_plateau():
    # means: 0.40, 0.48, 0.49, 0.48, 0.30 with SE of best = 0.02 → plateau around peak
    points = [
        {"id": "a", "mean": 0.40, "se": 0.02, "lam_team": 0.05, "lam_dead": 0.2},
        {"id": "b", "mean": 0.48, "se": 0.02, "lam_team": 0.1, "lam_dead": 0.2},
        {"id": "c", "mean": 0.49, "se": 0.02, "lam_team": 0.2, "lam_dead": 0.2},
        {"id": "d", "mean": 0.48, "se": 0.02, "lam_team": 0.4, "lam_dead": 0.2},
        {"id": "e", "mean": 0.30, "se": 0.02, "lam_team": 0.8, "lam_dead": 0.2},
    ]

    def neighbours(p, all_points):
        order = [q["lam_team"] for q in all_points]
        i = order.index(p["lam_team"])
        out = []
        if i > 0:
            out.append(all_points[i - 1])
        if i + 1 < len(all_points):
            out.append(all_points[i + 1])
        return out

    pick = sweep.select_plateau(points, neighbour_fn=neighbours)
    assert pick["id"] == "c"  # centre; neighbours b,d also in plateau


def test_select_plateau_falls_back_to_best_when_no_stable_centre():
    points = [
        {"id": "edge", "mean": 0.50, "se": 0.01, "lam_team": 0.05, "lam_dead": 0.2},
        {"id": "low", "mean": 0.40, "se": 0.01, "lam_team": 0.8, "lam_dead": 0.2},
    ]

    def neighbours(p, all_points):
        return [q for q in all_points if q is not p]

    pick = sweep.select_plateau(points, neighbour_fn=neighbours)
    # edge's neighbour is outside plateau → fall back to best
    assert pick["id"] == "edge"


def test_select_plateau_tiebreak_smaller_lambda():
    points = [
        {"id": "hi", "mean": 0.5, "se": 0.0, "lam_team": 0.4, "lam_dead": 0.4},
        {"id": "lo", "mean": 0.5, "se": 0.0, "lam_team": 0.2, "lam_dead": 0.2},
    ]
    pick = sweep.select_plateau(points, neighbour_fn=None)
    assert pick["id"] == "lo"


def test_select_robust_best_worst_case():
    candidates = [
        {"id": "A", "mean": 0.50},
        {"id": "B", "mean": 0.49},
    ]
    scores = {
        "default": {"A": {"mean": 0.50}, "B": {"mean": 0.49}},
        "hidelo": {"A": {"mean": 0.30}, "B": {"mean": 0.45}},
        "hidehi": {"A": {"mean": 0.48}, "B": {"mean": 0.44}},
    }
    # A worst=0.30, B worst=0.44 → B
    pick = sweep.select_robust(candidates, scores)
    assert pick["id"] == "B"


def test_with_decoder_and_postprocess_validate():
    cfg = _hmm_cfg()
    out = sweep.with_decoder(cfg, lam_team=0.4, min_dwell_frames={"H": 1, "A": 1, "D": 1})
    assert out["decoder"]["lam_team"] == 0.4
    assert out["decoder"]["min_dwell_frames"]["H"] == 1
    # mean_duration must still exceed min_dwell / fps — 15s * 25 > 1 ok
    post = sweep.with_postprocess(cfg, midpoint_neutral_switch=True, d_cross_frames=25)
    assert post["postprocess"]["midpoint_neutral_switch"] is True
    assert post["postprocess"]["d_cross_frames"] == 25


def test_overlay_decoder_from_idsse_file(tmp_path):
    base = _hmm_cfg()
    path = tmp_path / "sweep_t.yaml"
    path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    dec = tmp_path / "dec.yaml"
    dec.write_text(yaml.safe_dump({
        "decoder": {
            "lam_team": 0.4,
            "lam_dead": 0.2,
            "mean_duration_s": {"H": 7.0, "A": 6.0, "D": 25.0},
            "exit_split": base["decoder"]["exit_split"],
            "min_dwell_frames": {"H": 12, "A": 12, "D": 25},
        }
    }, sort_keys=False), encoding="utf-8")
    cfg = sweep.load_base_config(path, dec)
    assert cfg["decoder"]["lam_team"] == 0.4
    assert cfg["decoder"]["mean_duration_s"]["H"] == 7.0


def test_spell_ratio_and_summarize():
    df = pd.DataFrame({
        "spell_f1_mean": [0.4, 0.6, 0.5, 0.1],
        "n_gt_spells": [10, 20, 15, np.nan],
        "n_pred_spells": [12, 10, 15, np.nan],
    }, index=["M1", "M2", "mean", "se"])
    assert sweep.spell_ratio_from_frame(df) == pytest.approx((1.2 + 0.5) / 2)
    summary = sweep.summarize(df)
    assert summary["mean"] == 0.5
    assert summary["se"] == 0.1


def test_run_staged_sweep_with_fake_evaluator():
    cfg = _hmm_cfg()
    calls = []

    def fake_eval(c, degrade):
        calls.append((c["decoder"]["lam_team"], c["decoder"]["lam_dead"],
                      c["decoder"]["min_dwell_frames"]["H"],
                      c["decoder"]["min_dwell_frames"]["D"],
                      c["postprocess"]["d_bridge_frames"],
                      degrade))
        # Prefer lam 0.4/0.2, dwell 12/25, scale 1, post off
        lam_t = c["decoder"]["lam_team"]
        lam_d = c["decoder"]["lam_dead"]
        score = 0.3 + 0.1 * (lam_t == 0.4) + 0.1 * (lam_d == 0.2)
        ha = c["decoder"]["min_dwell_frames"]["H"]
        d = c["decoder"]["min_dwell_frames"]["D"]
        score += 0.05 * (ha == 12) + 0.05 * (d == 25)
        scale = c["decoder"]["mean_duration_s"]["H"] / 15.0
        score += 0.02 * (abs(scale - 1.0) < 1e-9)
        score -= 0.01 * c["postprocess"]["d_bridge_frames"] / 100.0
        score -= 0.01 * c["postprocess"]["min_spell_frames"] / 100.0
        if c["postprocess"]["midpoint_neutral_switch"]:
            score -= 0.02
        return {"mean": score, "se": 0.01, sweep.SPELL_RATIO: 1.0}

    # Monkeypatch stage evaluators that call evaluate_point internally by
    # patching evaluate_point used inside sweep_* via evaluate_fn in robustify only;
    # run_staged_sweep still calls sweep_lambdas → evaluate_point. Patch that.
    def fake_point(c, *, degrade="default", matches=None, metric=sweep.METRIC):
        return fake_eval(c, degrade)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(sweep, "evaluate_point", fake_point)
    try:
        history = sweep.run_staged_sweep(
            cfg, degrade="default", robust=True, include_d_cross=True,
            evaluate_fn=fake_eval)
    finally:
        monkey.undo()

    selected = history["selected_cfg"]
    assert selected["decoder"]["lam_team"] == 0.4
    assert selected["decoder"]["lam_dead"] == 0.2
    assert selected["decoder"]["min_dwell_frames"]["H"] == 12
    assert selected["decoder"]["min_dwell_frames"]["D"] == 25
    assert history["stages"][0]["selected"]["id"].startswith("lam|")
    # robust path evaluated hidelo/hidehi
    assert any(deg == "hidelo" for *_, deg in calls)


def test_write_v1s(tmp_path):
    cfg = _hmm_cfg(lam_team=0.4)
    path = sweep.write_v1s(cfg, tmp_path / "v1s.yaml")
    loaded = load_config(path)
    assert loaded["name"] == "v1s"
    assert loaded["decoder"]["lam_team"] == 0.4
