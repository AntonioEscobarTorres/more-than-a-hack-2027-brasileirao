"""tools/dead_feature_shift.py (F3-04 review). Synthetic masters only: no match data, no IDSSE."""
from __future__ import annotations

import re

import numpy as np
import pytest

from modeling.calibration import dead as cd
from tests.modeling_fixtures import dead_params, deadball_master
from tools import dead_feature_shift as dfs

P = dead_params()


def toy_doc():
    """A valid calibration whose model ``all`` gives ``d_fit = 2 - speed`` (prior 0.5)."""
    features = list(cd.FULL_FEATURES)
    model = {"mean": {f: 0.0 for f in features}, "std": {f: 1.0 for f in features},
             "weights": {f: 0.0 for f in features}, "intercept": 2.0, "pi_d": 0.5, "n_train": 1}
    model["weights"]["mean_speed"] = -1.0
    doc = {"format_version": cd.FORMAT_VERSION, "features": features,
           "gap_log_clip_frames": cd.GAP_LOG_CLIP_FRAMES, "feature_params": cd.feature_params(P),
           "all": model, "folds": {}}
    return cd.validate_calibration(doc)


def master(speed, n=600, half=1):
    x = np.linspace(10.0, 90.0, n)
    return deadball_master(x, 34.0, vx=5.0, half=half, mean_speed_home=speed,
                           mean_speed_away=speed)


def stats(m):
    return dfs.master_statistics(m, P, toy_doc())


def run_cli(monkeypatch, capsys, ref, ext, argv=()):
    monkeypatch.setattr(dfs, "_n_frames", lambda: len(ref))
    monkeypatch.setattr(dfs, "load_config", lambda _path: {"dead": {"params": P}})
    monkeypatch.setattr(dfs, "load_calibration", lambda _path: toy_doc())
    monkeypatch.setattr(dfs.io, "load_master", lambda *_a, **_k: ref)
    monkeypatch.setattr(dfs, "load_external", lambda _m, _d: (ext, None))
    code = dfs.main(["--match", "M1", "M2", *argv])
    return code, capsys.readouterr().out.splitlines()


def test_default_output_is_only_pass_fail_and_verbose_adds_numbers(monkeypatch, capsys):
    code, lines = run_cli(monkeypatch, capsys, master(2.0), master(2.05))
    assert code == 0 and lines[-1] == "PASS"
    assert all(re.match(r"^(PASS|FAIL)\b", line) for line in lines)
    assert not any(re.search(r"\d", line) for line in lines)
    code, lines = run_cli(monkeypatch, capsys, master(2.0), master(2.05), ["--verbose"])
    assert code == 0 and any(re.search(r"\d\.\d{3}", line) for line in lines)


def test_a_speed_shift_beyond_the_tolerance_fails(monkeypatch, capsys):
    shift = dfs.SPEED_QUANTILE_TOL_MPS + 0.1
    code, lines = run_cli(monkeypatch, capsys, master(2.0 + shift), master(2.0))
    assert code == 1
    assert any(line.startswith("FAIL  (c)") for line in lines)
    assert lines[-1].startswith("FAIL")


def test_missing_speed_on_the_provided_master_fails_a(monkeypatch, capsys):
    speed = np.full(600, 2.0)
    speed[100:300] = np.nan                            # longer than smooth_frames
    code, lines = run_cli(monkeypatch, capsys, master(speed), master(2.0))
    assert code == 1
    assert any(line.startswith("FAIL  (a)") for line in lines)
    assert any(line.startswith("PASS  (b)") for line in lines)


def test_the_informative_check_never_changes_the_result():
    ref, ext = stats(master(2.0)), stats(master(2.0))
    ref = {**ref, "d_positive_share": ext["d_positive_share"] * 3 + 0.1}
    results = dfs.run_checks(ref, [ext])
    d = [r for r in results if r.key == "d"][0]
    assert not d.passed and not d.eliminatory
    assert dfs.report(results)[-1] == "PASS"


def test_statistics_use_half_frames_only():
    half = np.r_[np.zeros(200, dtype=np.int8), np.ones(400, dtype=np.int8)]
    speed = np.r_[np.full(200, 9.0), np.full(400, 1.0)]
    s = stats(master(speed, half=half))
    assert s["speed_finite"] and s["still_finite"]
    assert s["speed_quantiles"] == pytest.approx([1.0, 1.0, 1.0])
    assert s["d_positive_share"] == pytest.approx(1.0)  # d_fit = 2 - 1 > 0 on every half frame
    assert stats(master(3.0))["d_positive_share"] == 0.0


def test_reference_is_the_mean_over_external_masters():
    ext = [stats(master(1.0)), stats(master(3.0))]
    ok = dfs.run_checks(stats(master(2.0)), ext)
    assert [r.passed for r in ok if r.key == "c"] == [True]
