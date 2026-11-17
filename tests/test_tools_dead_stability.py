"""Synthetic tests for tools/dead_stability.py (F2-06, stability check)."""
from __future__ import annotations

import re

import numpy as np
import pytest

from modeling import paths
from modeling.config import load_config
from preprocessing.make_mock_master import generate_mock_master
from tools import dead_stability as ds


def _cfg():
    return load_config(paths.repo_root() / "configs" / "v2_gbdt_fused.yaml")


def test_variants_change_only_the_dead_file():
    cfg = _cfg()
    out = ds.variant_configs(cfg)
    assert set(out) == {"default", "hidelo", "hidehi"}
    for key, v in out.items():
        assert v["dead"]["calibrated_file"] == ds.DEAD_VARIANTS[key]
        a = {k: x for k, x in v.items() if k != "dead"}
        b = {k: x for k, x in cfg.items() if k != "dead"}
        assert a == b
        assert {k: x for k, x in v["dead"].items() if k != "calibrated_file"} == \
            {k: x for k, x in cfg["dead"].items() if k != "calibrated_file"}
    assert cfg["dead"]["calibrated_file"] == "configs/calib/dead.yaml"  # input untouched
    for path in ds.DEAD_VARIANTS.values():
        assert (paths.repo_root() / path).is_file()


def test_variants_need_a_calibrated_dead():
    cfg = _cfg()
    cfg["dead"]["kind"] = "rules_v1"
    with pytest.raises(ValueError):
        ds.variant_configs(cfg)


def test_shares_ignore_frames_outside_the_halves():
    half = np.array([0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    a = np.array(list("HHHHHHAAAA"))
    b = a.copy()
    b[:2] = "D"          # outside the halves: ignored
    b[2] = "A"           # 1 of 8 inside
    shares = ds.changed_shares({"default": a, "hidelo": b, "hidehi": a}, half)
    assert shares[("default", "hidelo")] == pytest.approx(1 / 8)
    assert shares[("default", "hidehi")] == 0.0
    assert shares[("hidelo", "hidehi")] == pytest.approx(1 / 8)


def test_threshold_is_strict_and_default_output_has_no_numbers():
    ok, lines = ds.report({("default", "hidelo"): 0.0499, ("default", "hidehi"): 0.0})
    assert ok and lines[-1] == "overall: PASS"
    assert not any(re.search(r"\d\.\d", line) for line in lines)
    ok, lines = ds.report({("default", "hidelo"): 0.05})
    assert not ok and lines[0].endswith("FAIL") and lines[-1] == "overall: FAIL"
    _, verbose = ds.report({("default", "hidelo"): 0.01}, verbose=True)
    assert "0.0100" in verbose[0]


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        ds.changed_shares({"a": np.array(list("HH")), "b": np.array(list("H"))},
                          np.array([1, 1]))


def test_missing_master_exits_2(tmp_path, capsys):
    rc = ds.main(["--config", str(paths.repo_root() / "configs" / "v2_gbdt_fused.yaml"),
                  "--master", str(tmp_path / "missing.parquet")])
    assert rc == 2
    assert "master not found" in capsys.readouterr().err


def test_end_to_end_on_the_mock_master(tmp_path, capsys):
    mock_master = generate_mock_master(seed=42)
    path = tmp_path / "master_features.parquet"
    mock_master.to_parquet(path)
    rc = ds.main(["--config", str(paths.repo_root() / "configs" / "v2_gbdt_fused_hzn2.yaml"),
                  "--master", str(path)])
    out = capsys.readouterr().out.strip().splitlines()
    assert rc in (0, 1)
    assert len(out) == 4 and out[-1] in ("overall: PASS", "overall: FAIL")
    assert not any(re.search(r"\d\.\d", line) for line in out)
