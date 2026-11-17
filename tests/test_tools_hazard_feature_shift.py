"""tools/hazard_feature_shift.py (F2-07). Synthetic masters only: no match data, no IDSSE."""
from __future__ import annotations

import re

import numpy as np
import pytest

from modeling import paths
from tests.modeling_fixtures import make_master
from tools import hazard_feature_shift as hfs

HAZARD_FILE = paths.repo_root() / "configs" / "calib" / "hazard.yaml"


def master(scale=1.0, n=1500, seed=0, nan_acc=0.0):
    """Synthetic master with every column the hazard and team_logit_rules_v1 read."""
    rng = np.random.default_rng(seed)
    d_home = rng.uniform(0.5, 12.0, n) * scale
    d_away = rng.uniform(0.5, 12.0, n) * scale
    acc = rng.gamma(2.0, 8.0, n)
    acc[rng.random(n) < nan_acc] = np.nan
    half = np.r_[np.ones(n // 2), np.full(n - n // 2, 2)]
    return make_master(
        n, half=half, d_near_home=d_home, d_near_away=d_away, ball_acc=acc,
        ball_dir_change=rng.uniform(0.0, np.pi, n), gap_length=0.0,
        since_last_obs=0.0, until_next_obs=0.0)


def run_cli(monkeypatch, capsys, ref, ext, argv=()):
    monkeypatch.setattr(hfs, "_n_frames", lambda: len(ref))
    monkeypatch.setattr(hfs.io, "load_master", lambda *_a, **_k: ref)
    monkeypatch.setattr(hfs, "load_external", lambda _m, _d: (ext, None))
    code = hfs.main(["--match", "J03WMX", "J03WN1", "--hazard-file", str(HAZARD_FILE), *argv])
    return code, capsys.readouterr().out.splitlines()


def test_same_distribution_passes_and_default_output_has_no_numbers(monkeypatch, capsys):
    code, lines = run_cli(monkeypatch, capsys, master(seed=1), master(seed=2))
    assert code == 0 and lines[-1] == "PASS"
    assert all(re.match(r"^(PASS|FAIL)\b", line) for line in lines)
    assert not any(re.search(r"\d", line) for line in lines)
    code, lines = run_cli(monkeypatch, capsys, master(seed=1), master(seed=2), ["--verbose"])
    assert code == 0 and any(re.search(r"\d\.\d{3}", line) for line in lines)


def test_distances_out_of_range_fail_a_with_feature_names(monkeypatch, capsys):
    code, lines = run_cli(monkeypatch, capsys, master(scale=4.0, seed=1), master(seed=2))
    assert code == 1
    assert lines[0].startswith("FAIL  (a)")
    names = lines[1].split(":", 1)[1]
    assert "abs_d_diff" in names and "d_min" in names
    assert not re.search(r"\d", names)
    assert lines[-1].startswith("FAIL")


def test_nan_share_shift_fails_a(monkeypatch, capsys):
    code, lines = run_cli(monkeypatch, capsys, master(seed=1, nan_acc=0.5), master(seed=2))
    assert code == 1 and "ball_acc" in lines[1]


def test_mean_probability_check_is_relative_and_eliminatory():
    ref = {"features": np.zeros((4, 1)), "names": ["x"], "mean_prob": 0.10, "strong": 0.03}
    ok = dict(ref, mean_prob=0.10 * (1 + hfs.MEAN_PROB_TOL_REL) - 1e-9)
    bad = dict(ref, mean_prob=0.10 * (1 + hfs.MEAN_PROB_TOL_REL) + 1e-6)
    by_key = {r.key: r for r in hfs.run_checks(ok, [ref])}
    assert by_key["b"].passed and by_key["b"].eliminatory
    by_key = {r.key: r for r in hfs.run_checks(bad, [ref])}
    assert not by_key["b"].passed
    informative = {r.key: r for r in hfs.run_checks(dict(ref, strong=0.3), [ref])}["c"]
    assert not informative.passed and not informative.eliminatory


def test_undefined_statistics_fail():
    ref = {"features": np.zeros((4, 1)), "names": ["x"], "mean_prob": None, "strong": None}
    by_key = {r.key: r for r in hfs.run_checks(ref, [ref])}
    assert not by_key["b"].passed and by_key["b"].detail == "undefined"


def test_unloadable_inputs_exit_2(monkeypatch, capsys, tmp_path):
    code = hfs.main(["--hazard-file", str(tmp_path / "missing.yaml")])
    out = capsys.readouterr().out
    assert code == 2 and out.startswith("FAIL  could not load inputs")
    assert str(tmp_path) not in out


def test_statistics_use_only_frames_inside_the_halves():
    m = master(seed=3)
    m.loc[:99, "half"] = 0
    calib = hfs.load_hazard_file(HAZARD_FILE)
    stats = hfs.master_statistics(m, calib, holdout=None)
    inside = int((m["half"] != 0).sum())
    assert stats["features"].shape == (len(range(0, inside, hfs.SUBSAMPLE)), len(calib["features"]))
    assert 0.0 < stats["mean_prob"] < 1.0 and 0.0 <= stats["strong"] <= 1.0


@pytest.mark.parametrize("holdout", [None, "J03WMX"])
def test_holdout_selects_the_fold_model(holdout):
    calib = hfs.load_hazard_file(HAZARD_FILE)
    stats = hfs.master_statistics(master(seed=4), calib, holdout=holdout)
    assert np.isfinite(stats["mean_prob"])
