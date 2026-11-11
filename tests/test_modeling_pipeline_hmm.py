"""C8 predict_labels with ``pipeline: hmm`` and configs/v1.yaml (task F5-04).

Synthetic data only: the master is preprocessing.make_mock_master and every file lives in
tmp_path. No test reads MTAH_DATA_DIR or the real master.
"""
from __future__ import annotations

import copy
import hashlib
import logging
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import games
from evaluation import possession_metrics as pm
from modeling import io, make_submission, paths, pipeline, variants
from modeling.config import load_config
from modeling.contracts import validate_labels
from modeling.deadball.dead import dead_logit_rules_v1
from modeling.deadball.force import build_force_d
from modeling.decode.emission import logits_to_logscores
from modeling.decode.params import DecoderParams
from modeling.decode.viterbi import decode_match
from modeling.evidence.team import team_logit_rules_v1
from modeling.pipeline import final_imposition, predict_labels
from tests.modeling_fixtures import dead_params, team_params
from preprocessing.make_mock_master import generate_mock_master

N_GAME = games.GAMES[games.GAME]["n_video_frames"]
REPO = paths.repo_root()
V1_YAML = REPO / "configs" / "v1.yaml"
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


@pytest.fixture(scope="module")
def mock_master() -> pd.DataFrame:
    return generate_mock_master(seed=42)


@pytest.fixture(scope="module")
def v1_cfg() -> dict:
    return load_config(V1_YAML)


@pytest.fixture(scope="module")
def v1_labels(mock_master, v1_cfg) -> np.ndarray:
    return predict_labels(mock_master, v1_cfg)


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _with_post(cfg: dict, **post) -> dict:
    out = copy.deepcopy(cfg)
    out["postprocess"].update(post)
    return out


# -- configs/v1.yaml -----------------------------------------------------------------------------

def test_v1_config_is_appendix_a(v1_cfg):
    assert v1_cfg["name"] == "v1" and v1_cfg["pipeline"] == "hmm"
    assert v1_cfg["ball_source"] == "tactical"
    assert v1_cfg["team"] == {"kind": "rules_v1", "params": team_params(),
                              "calibrated_file": None}
    assert v1_cfg["dead"] == {"kind": "rules_v1", "params": dead_params(),
                              "calibrated_file": None}
    assert v1_cfg["decoder"] == {
        "lam_team": 0.2, "lam_dead": 0.2,
        "mean_duration_s": {"H": 15.0, "A": 15.0, "D": 20.0},
        "exit_split": {"H": {"A": 0.5, "D": 0.5}, "A": {"H": 0.5, "D": 0.5},
                       "D": {"H": 0.5, "A": 0.5}},
        "min_dwell_frames": {"H": 12, "A": 12, "D": 25},
    }
    assert v1_cfg["postprocess"] == {"midpoint_neutral_switch": False,
                                     "boundary_shift_file": None, "d_bridge_frames": 0,
                                     "min_spell_frames": 0}


# -- end to end on the mock ----------------------------------------------------------------------

def test_hmm_labels_are_valid(mock_master, v1_labels):
    half = mock_master["half"].to_numpy()
    assert v1_labels.dtype == np.dtype("<U1") and v1_labels.shape == (len(mock_master),)
    validate_labels(v1_labels, half)
    assert set(np.unique(v1_labels[half != 0])) <= {"H", "A", "D"}
    # the decoder is not degenerate on the mock: both teams and D all appear
    assert {"H", "A", "D"} <= set(np.unique(v1_labels).tolist())


def test_hmm_no_x_inside_halves_and_x_outside(mock_master, v1_labels):
    half = mock_master["half"].to_numpy()
    assert not (v1_labels[half != 0] == "X").any()
    assert (v1_labels[half == 0] == "X").all()


def test_hmm_pre_kickoff_is_d(mock_master, v1_labels):
    pre = mock_master["pre_kickoff"].to_numpy(dtype=bool)
    assert pre.any()                                   # the mock has a pre-kickoff segment
    assert (v1_labels[pre] == "D").all()


def _manual_labels(master, cfg):
    """The chain of the F5-04 specification (steps 1-10), written out by hand."""
    half = master["half"].to_numpy()
    t = team_logit_rules_v1(master, cfg["team"]["params"])
    d = dead_logit_rules_v1(master, cfg["dead"]["params"])
    force_d = build_force_d(master)
    params = DecoderParams.from_config(cfg["decoder"])
    scores = logits_to_logscores(t, d, params.lam_team, params.lam_dead, force_d)
    return decode_match(scores, half, force_d, params)


def test_hmm_matches_manual_composition(mock_master, v1_cfg, v1_labels):
    np.testing.assert_array_equal(v1_labels, _manual_labels(mock_master, v1_cfg))


def test_hmm_matches_manual_composition_asymmetric_decoder(mock_master, v1_cfg):
    """lam_team != lam_dead and unequal durations, so a swapped argument changes the labels."""
    cfg = copy.deepcopy(v1_cfg)
    cfg["decoder"]["lam_team"], cfg["decoder"]["lam_dead"] = 0.35, 0.1
    cfg["decoder"]["mean_duration_s"] = {"H": 8.0, "A": 12.0, "D": 25.0}
    cfg["decoder"]["exit_split"]["H"] = {"A": 0.7, "D": 0.3}
    labels = predict_labels(mock_master, cfg)
    np.testing.assert_array_equal(labels, _manual_labels(mock_master, cfg))
    swapped = copy.deepcopy(cfg)
    swapped["decoder"]["lam_team"], swapped["decoder"]["lam_dead"] = 0.1, 0.35
    assert (predict_labels(mock_master, swapped) != labels).any()


def test_hmm_is_deterministic_in_process(mock_master, v1_cfg, v1_labels):
    np.testing.assert_array_equal(predict_labels(mock_master, v1_cfg), v1_labels)


def test_hmm_does_not_modify_inputs(mock_master, v1_cfg):
    master = mock_master.copy()
    cfg = copy.deepcopy(v1_cfg)
    predict_labels(master, cfg)
    pd.testing.assert_frame_equal(master, mock_master)
    assert cfg == v1_cfg


def test_hmm_rules_v1_ignores_holdout(mock_master, v1_cfg, v1_labels):
    np.testing.assert_array_equal(predict_labels(mock_master, v1_cfg, holdout="J03WMX"),
                                  v1_labels)


def test_hmm_on_external_layout_master(mock_master, v1_cfg):
    """IDSSE masters: other length, no pre-kickoff, no tags. Same function, valid labels."""
    k1 = int(np.flatnonzero(mock_master["half"].to_numpy() == 1)[0])
    k2 = int(np.flatnonzero(mock_master["half"].to_numpy() == 2)[0])
    master = pd.concat([mock_master.iloc[k1 - 50:k1 + 3000], mock_master.iloc[k2:k2 + 3000]],
                       ignore_index=True)
    master["video_frame"] = np.arange(len(master), dtype=master["video_frame"].dtype)
    master["pre_kickoff"] = False
    labels = predict_labels(master, v1_cfg, holdout="S1")
    half = master["half"].to_numpy()
    validate_labels(labels, half)
    assert (labels[half == 0] == "X").all() and not (labels[half != 0] == "X").any()


def test_hmm_csv_sha256_is_reproducible(mock_master, tmp_path):
    """Two runs of the single command give byte-identical CSVs (fresh interpreters)."""
    root = tmp_path / "data"
    path = root / "processed" / io.MASTER_FILENAME
    path.parent.mkdir(parents=True)
    mock_master.to_parquet(path, index=False)
    env = {k: v for k, v in os.environ.items() if k not in THREAD_VARS}
    env["MTAH_DATA_DIR"] = str(root)
    hashes = []
    for i in range(2):
        out = tmp_path / "v1_run{}.csv".format(i)
        proc = subprocess.run(
            [sys.executable, "-m", "modeling.make_submission", "--config", str(V1_YAML),
             "--skip-preprocessing", "--out", str(out)],
            cwd=REPO, env=env, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        assert pm.validate_submission(pm.read_label_csv(out), N_GAME) is None
        hashes.append(_sha256(out))
        assert hashes[-1] in proc.stdout
        assert "config: v1 (sha256 " in proc.stdout
        assert "postprocess (v1, holdout=None): " in proc.stderr     # post_stats logged
    assert hashes[0] == hashes[1]


def test_hmm_cli_in_process(mock_master, v1_labels, tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("MTAH_DATA_DIR", str(root))
    monkeypatch.setattr(games, "DATA_DIR", str(root))
    path = root / "processed" / io.MASTER_FILENAME
    path.parent.mkdir(parents=True)
    mock_master.to_parquet(path, index=False)
    out = tmp_path / "v1.csv"
    assert make_submission.main(["--config", str(V1_YAML), "--skip-preprocessing",
                                 "--out", str(out)]) == 0
    table = pm.read_label_csv(out)
    np.testing.assert_array_equal(table["label"].to_numpy().astype("<U1"), v1_labels)


# -- post-processing hook (F2-04) ----------------------------------------------------------------

def test_post_stats_are_logged(mock_master, v1_cfg, caplog):
    with caplog.at_level(logging.INFO, logger="modeling.pipeline"):
        predict_labels(mock_master, v1_cfg, holdout="S9")
    msgs = [r.getMessage() for r in caplog.records if r.name == "modeling.pipeline"]
    assert len(msgs) == 1
    assert msgs[0].startswith("postprocess (v1, holdout=S9): ")
    for key in pipeline.POSTPROCESS_RULES + ("final_imposition",):
        assert "{}=".format(key) in msgs[0]


@pytest.mark.parametrize("post", [{"midpoint_neutral_switch": True},
                                  {"boundary_shift_file": "configs/calib/boundary.yaml"},
                                  {"d_bridge_frames": 5}, {"min_spell_frames": 5}])
def test_enabled_rule_without_f2_04_is_an_error(mock_master, v1_cfg, post, monkeypatch):
    monkeypatch.setattr(pipeline, "_postprocess_fn", lambda: pipeline._identity_postprocess)
    with pytest.raises(NotImplementedError, match="F2-04"):
        predict_labels(mock_master, _with_post(v1_cfg, **post))


def test_postprocess_module_is_used_when_present(mock_master, v1_cfg, v1_labels, monkeypatch):
    calls = []

    def fake_postprocess(labels, master, team_logit, dead_logit, force_d, cfg_post, *,
                         holdout=None):
        calls.append((team_logit.dtype, dead_logit.dtype, force_d.dtype, cfg_post, holdout))
        out = labels.copy()
        return out, {"midpoint_neutral_switch": 0}

    module = types.ModuleType(pipeline.POSTPROCESS_MODULE)
    module.postprocess = fake_postprocess
    monkeypatch.setitem(sys.modules, pipeline.POSTPROCESS_MODULE, module)
    labels = predict_labels(mock_master, v1_cfg, holdout="S1")
    np.testing.assert_array_equal(labels, v1_labels)
    assert calls == [(np.float32, np.float32, np.bool_, v1_cfg["postprocess"], "S1")]


def test_postprocess_output_is_validated(mock_master, v1_cfg, monkeypatch):
    def bad_postprocess(labels, master, *args, **kwargs):
        out = labels.copy()
        out[master["half"].to_numpy() == 0] = "H"
        return out, {}

    module = types.ModuleType(pipeline.POSTPROCESS_MODULE)
    module.postprocess = bad_postprocess
    monkeypatch.setitem(sys.modules, pipeline.POSTPROCESS_MODULE, module)
    with pytest.raises(ValueError, match="half == 0"):
        predict_labels(mock_master, v1_cfg)


def test_broken_postprocess_import_is_not_hidden(monkeypatch):
    def fail(name):
        raise ModuleNotFoundError("No module named 'missing_dep'", name="missing_dep")

    monkeypatch.setattr(pipeline.importlib, "import_module", fail)
    with pytest.raises(ModuleNotFoundError, match="missing_dep"):
        pipeline._postprocess_fn()


def test_missing_postprocess_module_falls_back(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError("No module named {!r}".format(name), name=name)

    monkeypatch.setattr(pipeline.importlib, "import_module", missing)
    assert pipeline._postprocess_fn() is pipeline._identity_postprocess


def test_final_imposition():
    half = np.array([0, 1, 1, 1, 0, 2, 2, 2], np.int8)
    force = np.array([0, 0, 0, 0, 0, 1, 1, 0], bool)
    labels = np.array(list("HHAHXHAA"), dtype="<U1")
    out, n = final_imposition(labels, half, force)
    np.testing.assert_array_equal(out, np.array(list("XHAHXDDA"), dtype="<U1"))
    assert n == 3
    assert labels.tolist() == list("HHAHXHAA")                 # input not modified
    again, n2 = final_imposition(out, half, force)               # idempotent
    np.testing.assert_array_equal(again, out)
    assert n2 == 0
    with pytest.raises(ValueError, match="X inside a half"):
        final_imposition(np.array(list("XHXHXHAA"), dtype="<U1"), half, force)


# -- evidence wiring -----------------------------------------------------------------------------

def test_variants_receive_params_and_holdout(mock_master, v1_cfg, monkeypatch):
    seen = {}

    def team(master, params, *, holdout=None):
        seen["team"] = (params, holdout)
        return np.zeros(len(master), np.float32)

    def dead(master, params, *, holdout=None):
        seen["dead"] = (params, holdout)
        return np.zeros(len(master), np.float32)

    variants.load_kinds()
    monkeypatch.setitem(variants.TEAM_KINDS, "rules_v1", team)
    monkeypatch.setitem(variants.DEAD_KINDS, "rules_v1", dead)
    predict_labels(mock_master, v1_cfg, holdout="S2")
    assert seen == {"team": (v1_cfg["team"]["params"], "S2"),
                    "dead": (v1_cfg["dead"]["params"], "S2")}


def test_kind_none_gives_valid_labels(mock_master, v1_cfg):
    cfg = copy.deepcopy(v1_cfg)
    cfg["team"]["kind"] = cfg["dead"]["kind"] = "none"
    labels = predict_labels(mock_master, cfg)
    validate_labels(labels, mock_master["half"].to_numpy())
    assert (labels[mock_master["pre_kickoff"].to_numpy(dtype=bool)] == "D").all()


@pytest.mark.parametrize("bad", ["float64", "nan", "too_big", "nonzero_gap"])
def test_invalid_logits_are_rejected(mock_master, v1_cfg, bad, monkeypatch):
    def team(master, params, *, holdout=None):
        n = len(master)
        out = np.zeros(n, np.float32)
        if bad == "float64":
            return out.astype(np.float64)
        if bad == "nan":
            out[n // 2] = np.nan
        elif bad == "too_big":
            out[n // 2] = 9.0
        else:
            out[master["half"].to_numpy() == 0] = 1.0
        return out

    variants.load_kinds()
    monkeypatch.setitem(variants.TEAM_KINDS, "rules_v1", team)
    with pytest.raises(ValueError, match="team_logit"):
        predict_labels(mock_master, v1_cfg)


def test_unregistered_kind_is_an_error(mock_master, v1_cfg, monkeypatch):
    variants.load_kinds()
    monkeypatch.delitem(variants.DEAD_KINDS, "rules_v1")
    monkeypatch.setattr(variants, "load_kinds", lambda: None)
    with pytest.raises(NotImplementedError, match="dead.kind 'rules_v1'"):
        predict_labels(mock_master, v1_cfg)
