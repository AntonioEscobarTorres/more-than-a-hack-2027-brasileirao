"""tools/stack_feature_shift.py (F2-09). Synthetic masters only: no match data, no IDSSE."""
from __future__ import annotations

import copy
import re

import numpy as np
import yaml

from tests.test_modeling_stack import stacked  # noqa: F401  (module fixture)
from tools import stack_feature_shift as sfs


def _cfg_file(stacked, tmp_path):  # noqa: F811
    cfg = copy.deepcopy(stacked["full"])
    cfg["name"] = "stackcfg"
    path = tmp_path / "stackcfg.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def _run(monkeypatch, capsys, stacked, tmp_path, ref, argv=()):  # noqa: F811
    monkeypatch.setattr(sfs, "_n_frames", lambda: len(ref))
    monkeypatch.setattr(sfs.io, "load_master", lambda *_a, **_k: ref)
    monkeypatch.setattr(sfs, "load_external", lambda m, _d: stacked["matches"][m])
    code = sfs.main(["--match", "m1", "m2", "m3", "--config", str(_cfg_file(stacked, tmp_path)),
                     *argv])
    return code, capsys.readouterr().out.splitlines()


def test_default_output_is_pass_fail_only(monkeypatch, capsys, stacked, tmp_path):  # noqa: F811
    ref = stacked["matches"]["m1"][0]
    code, lines = _run(monkeypatch, capsys, stacked, tmp_path, ref)
    assert code in (0, 1)
    assert lines[-1].startswith(("PASS", "FAIL"))
    keys = [re.match(r"^(PASS|FAIL)  \((\w)\)", ln) for ln in lines[:-1]]
    assert [k.group(2) for k in keys if k] == ["a", "b", "c", "d", "e"]
    for ln in lines:
        assert not re.search(r"\d\.\d", ln), ln          # no numbers without --verbose


def test_out_of_range_master_fails_and_names_features(monkeypatch, capsys, stacked,  # noqa: F811
                                                      tmp_path):
    ref = stacked["matches"]["m1"][0].copy()
    for c in ("d_near_home", "d_near_away"):
        ref[c] = ref[c] * 25.0
    ref["d_diff"] = ref["d_near_home"] - ref["d_near_away"]
    ref["ball_acc"] = ref["ball_acc"] * 50.0
    code, lines = _run(monkeypatch, capsys, stacked, tmp_path, ref)
    assert code == 1
    assert any(ln.startswith("FAIL  (a)") for ln in lines)
    assert any("failing features:" in ln for ln in lines)
    assert lines[-1].startswith("FAIL")


def test_unloadable_inputs_exit_2(monkeypatch, capsys, tmp_path):
    bad = tmp_path / "final_d028.yaml"
    bad.write_text("name: final_d028\n", encoding="utf-8")
    code = sfs.main(["--config", str(bad)])
    assert code == 2
    assert capsys.readouterr().out.startswith("FAIL  could not load inputs")
    assert np.isfinite(sfs.Q_HI)
