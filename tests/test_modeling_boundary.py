"""Synthetic tests for boundary-latency calibration (task F2-05)."""
from __future__ import annotations

import numpy as np
import pytest
import yaml

from modeling.calibration import boundary as bnd
from modeling.contracts import LABEL_CODES
from modeling.decode.postprocess import BOUNDARY_TYPES, postprocess


def _codes_from_labels(labels: str, half: np.ndarray) -> np.ndarray:
    arr = np.array([LABEL_CODES[c] for c in labels], dtype=np.int8)
    arr[half == 0] = LABEL_CODES["X"]
    return arr


def test_latencies_for_match_typed_pair():
    # One half of 20 frames: GT switches H→A at 10; pred switches at 12 → latency +2
    half = np.ones(20, np.int8)
    gt = _codes_from_labels("H" * 10 + "A" * 10, half)
    pred = _codes_from_labels("H" * 12 + "A" * 8, half)
    lat = bnd.latencies_for_match(gt, pred, tol=50)
    assert lat["H>A"] == [2.0]
    assert lat["A>H"] == []


def test_median_table_requires_min_pairs():
    lat = {k: [] for k in BOUNDARY_TYPES}
    lat["H>A"] = [1.0, 2.0, 3.0]  # < MIN_PAIRS
    lat["A>H"] = [5.0] * bnd.MIN_PAIRS
    table = bnd.median_table(lat)
    assert table["H>A"] == 0
    assert table["A>H"] == 5


def test_write_and_reload_boundary_yaml(tmp_path):
    table = {
        "all": {k: 0 for k in BOUNDARY_TYPES} | {"H>A": 3},
        "folds": {
            "S1": {k: 0 for k in BOUNDARY_TYPES} | {"H>A": 1},
        },
        "meta": {"degrade": "default", "tol": 50, "min_pairs": 20,
                 "types": list(BOUNDARY_TYPES), "n_pairs_all": {k: 0 for k in BOUNDARY_TYPES}},
    }
    path = bnd.write_boundary_yaml(table, tmp_path / "boundary.yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["all"]["H>A"] == 3
    assert raw["folds"]["S1"]["H>A"] == 1

    # postprocess shift loader understands the file (holdout + all)
    half = np.ones(10, np.int8)
    labels = np.array(list("HHHHHAAAAA"), dtype="<U1")
    master = __import__("pandas").DataFrame({
        "half": half, "is_set_piece": np.zeros(10, bool)})
    force = np.zeros(10, bool)
    cfg = {
        "midpoint_neutral_switch": False,
        "boundary_shift_file": str(path),
        "d_bridge_frames": 0,
        "d_cross_frames": 0,
        "min_spell_frames": 0,
    }
    out_all, stats = postprocess(
        labels, master, np.zeros(10), np.zeros(10), force, cfg, holdout=None)
    # median 3 → shift -3 → boundary at 2
    assert "".join(out_all) == "HHAAAAAAAA"
    assert stats["boundary_shift_file"] > 0

    out_fold, _ = postprocess(
        labels, master, np.zeros(10), np.zeros(10), force, cfg, holdout="S1")
    assert "".join(out_fold) == "HHHHAAAAAA"


def test_latencies_ignore_cross_half(tmp_path):
    half = np.array([1] * 5 + [0] * 3 + [2] * 5, np.int8)
    # Same label change pattern each half; matching stays inside runs.
    labels = "HHHHH" + "XXX" + "AAAAA"
    gt = _codes_from_labels(labels, half)
    pred = _codes_from_labels(labels, half)
    lat = bnd.latencies_for_match(gt, pred, tol=50)
    assert all(len(v) == 0 for v in lat.values())  # no transitions inside halves


# -- regressions (review of #123) ---------------------------------------------------------------

def test_median_rounds_half_away_from_zero():
    lat = {k: [] for k in BOUNDARY_TYPES}
    lat["H>A"] = [7.0, 8.0] * (bnd.MIN_PAIRS // 2)   # median 7.5
    lat["A>H"] = [-2.0, -3.0] * (bnd.MIN_PAIRS // 2)  # median -2.5
    lat["H>D"] = [2.4] * bnd.MIN_PAIRS
    table = bnd.median_table(lat)
    assert (table["H>A"], table["A>H"], table["H>D"]) == (8, -3, 2)


def test_calibrate_refuses_config_with_shift_on():
    cfg = {"postprocess": {"boundary_shift_file": "configs/calib/boundary.yaml"}}
    with pytest.raises(ValueError, match="boundary_shift_file"):
        bnd.calibrate_boundary(cfg, matches=["M1"])


def test_compare_boundary_shift_runs_off_and_on(tmp_path):
    import pandas as pd

    cfg = {"postprocess": {"boundary_shift_file": "old.yaml", "d_bridge_frames": 0}}
    seen = []

    def fake(c, degrade, matches):
        seen.append(c["postprocess"]["boundary_shift_file"])
        value = 0.5 if c["postprocess"]["boundary_shift_file"] else 0.4
        return pd.DataFrame({"spell_f1_mean": [value, value, 0.0]}, index=["M1", "mean", "se"])

    path = tmp_path / "b.yaml"
    result = bnd.compare_boundary_shift(cfg, path, evaluate_fn=fake)
    assert seen == [None, str(path)]
    assert cfg["postprocess"]["boundary_shift_file"] == "old.yaml"  # input untouched
    text = bnd.comparison_markdown(result)
    assert "off: spell_f1_mean 0.4000" in text and "on: spell_f1_mean 0.5000" in text
