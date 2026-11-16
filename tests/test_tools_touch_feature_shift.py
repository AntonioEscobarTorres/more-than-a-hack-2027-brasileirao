"""tools/touch_feature_shift.py (F1-05 review). Synthetic masters only: no match data, no IDSSE."""
from __future__ import annotations

import re

import numpy as np
import pytest

from modeling.calibration.team import calibrate_team, write_team_yaml
from tests.modeling_fixtures import make_master, team_params
from tests.test_modeling_calibration_team import _rich_synthetic_master
from tools import touch_feature_shift as tfs

P = team_params()


@pytest.fixture(scope="module")
def calib_file(tmp_path_factory):
    matches = {"m1": _rich_synthetic_master(300, seed=10),
               "m2": _rich_synthetic_master(300, seed=20)}
    cfg = {"name": "v1", "team": {"kind": "rules_v1", "params": P, "calibrated_file": None}}
    table = calibrate_team(cfg, variant="touch", matches=list(matches),
                           load_fn=lambda m, d: matches[m])
    return str(write_team_yaml(table, tmp_path_factory.mktemp("calib") / "team.yaml"))


def master(acc, n=400):
    """Home close to the ball all the time; ``acc`` sets the ball acceleration everywhere."""
    half = np.ones(n, dtype=np.int8)
    half[n // 2:] = 2
    return make_master(n, half=half, d_near_home=np.full(n, 1.0), d_near_away=np.full(n, 6.0),
                       ball_acc=np.full(n, float(acc)), ball_dir_change=np.zeros(n))


def run_cli(monkeypatch, capsys, calib_file, ref, ext, argv=()):
    monkeypatch.setattr(tfs, "_n_frames", lambda: len(ref))
    monkeypatch.setattr(tfs, "load_config", lambda _path: {"team": {"params": P}})
    monkeypatch.setattr(tfs.io, "load_master", lambda *_a, **_k: ref)
    monkeypatch.setattr(tfs, "load_external", lambda _m, _d: (ext, None))
    code = tfs.main(["--match", "M1", "M2", "--calibration", calib_file, *argv])
    return code, capsys.readouterr().out.splitlines()


def test_default_output_is_only_pass_fail_and_verbose_adds_numbers(monkeypatch, capsys, calib_file):
    code, lines = run_cli(monkeypatch, capsys, calib_file, master(20.0), master(20.0))
    assert code == 0 and lines[-1] == "PASS"
    assert all(re.match(r"^(PASS|FAIL)\b", line) for line in lines)
    assert not any(re.search(r"\d", line) for line in lines)
    code, lines = run_cli(monkeypatch, capsys, calib_file, master(20.0), master(20.0),
                          ["--verbose"])
    assert code == 0 and any(re.search(r"\d\.\d{4}", line) for line in lines)


def test_a_smooth_ball_on_the_provided_master_fails_a(monkeypatch, capsys, calib_file):
    code, lines = run_cli(monkeypatch, capsys, calib_file, master(0.0), master(20.0))
    assert code == 1
    assert lines[0].startswith("FAIL  (a)")


def test_statistics_follow_the_touch_conditions(calib_file):
    noisy = tfs.master_statistics(master(20.0), P, calib_file)
    smooth = tfs.master_statistics(master(0.0), P, calib_file)
    assert noisy["abrupt_share"] == 1.0 and smooth["abrupt_share"] == 0.0
    assert noisy["touch_share"] > 0.9 and smooth["touch_share"] == 0.0


def test_informative_checks_never_change_the_result():
    ref = {"abrupt_share": 0.99, "touch_share": 0.1, "strong_share": 0.1}
    ext = [{"abrupt_share": 0.98, "touch_share": 0.5, "strong_share": 0.5}]
    results = tfs.run_checks(ref, ext)
    assert [r.passed for r in results] == [True, False, False]
    assert tfs.report(results)[-1] == "PASS"


def _calib(tmp_path_factory, variant, **params):
    matches = {"m1": _rich_synthetic_master(300, seed=10),
               "m2": _rich_synthetic_master(300, seed=20)}
    cfg = {"name": "v1", "team": {"kind": "rules_v1", "params": {**P, **params},
                                  "calibrated_file": None}}
    table = calibrate_team(cfg, variant=variant, matches=list(matches),
                           load_fn=lambda m, d: matches[m])
    return str(write_team_yaml(table, tmp_path_factory.mktemp("calib") / "team.yaml"))


def test_persistence_only_touch_ignores_a_and_makes_b_eliminatory(
        monkeypatch, capsys, tmp_path_factory):
    calib = _calib(tmp_path_factory, "touch", touch_require_abrupt=False)
    assert tfs.calibration_mode(calib) == (True, False)
    # A smooth ball no longer matters: the touch is persistence only.
    code, lines = run_cli(monkeypatch, capsys, calib, master(0.0), master(20.0))
    assert code == 0 and lines[0] == "PASS  (a) abrupt-change share (not used)"
    assert tfs.master_statistics(master(0.0), P, calib)["touch_share"] > 0.9
    # A touch share far from the reference fails (b), and (b) is eliminatory.
    ref = {"abrupt_share": 0.0, "touch_share": 0.1, "strong_share": 0.1}
    ext = [{"abrupt_share": 0.99, "touch_share": 0.5, "strong_share": 0.1}]
    results = tfs.run_checks(ref, ext, uses_touch=True, require_abrupt=False)
    assert [(r.key, r.passed, r.eliminatory) for r in results][:2] == [
        ("a", True, False), ("b", False, True)]
    assert tfs.report(results)[-1].startswith("FAIL")


def test_calibration_without_touch_passes_a_and_b(monkeypatch, capsys, tmp_path_factory):
    calib = _calib(tmp_path_factory, "base")
    assert tfs.calibration_mode(calib)[0] is False
    code, lines = run_cli(monkeypatch, capsys, calib, master(0.0), master(20.0))
    assert code == 0
    assert lines[:2] == ["PASS  (a) abrupt-change share (not used)",
                         "PASS  (b) touch share (not used)"]
