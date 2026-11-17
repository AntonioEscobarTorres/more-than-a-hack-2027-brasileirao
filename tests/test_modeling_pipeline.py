"""C8 predict_labels, the make_submission CLI and tools/check_public_window.py (task F5-02).

Synthetic data only: the master is preprocessing.make_mock_master, the ground truth is built in
the test, and every file lives in tmp_path. No test reads MTAH_DATA_DIR or the real master.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import games
from evaluation import possession_metrics as pm
from modeling import io, make_submission, paths
from modeling.baseline import baseline_labels
from modeling.config import load_config
from modeling.pipeline import predict_labels
from preprocessing.make_mock_master import generate_mock_master
from preprocessing.params import KICKOFF_FALLBACK
from tools import check_public_window as cpw

N_GAME = games.GAMES[games.GAME]["n_video_frames"]
REPO = paths.repo_root()
BASELINE_YAML = REPO / "configs" / "baseline.yaml"
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


# -- fixtures -----------------------------------------------------------------------------------

@pytest.fixture(scope="session")
def mock_master() -> pd.DataFrame:
    return generate_mock_master(seed=42)


@pytest.fixture(scope="session")
def baseline_cfg() -> dict:
    return load_config(BASELINE_YAML)


@pytest.fixture
def data_root(tmp_path, monkeypatch) -> Path:
    """Redirect the data directory to tmp_path (both the env var and the loaded constant)."""
    root = tmp_path / "data"
    monkeypatch.setenv("MTAH_DATA_DIR", str(root))
    monkeypatch.setattr(games, "DATA_DIR", str(root))
    return root


def _save_master(master: pd.DataFrame, root: Path) -> Path:
    path = root / "processed" / io.MASTER_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    master.to_parquet(path, index=False)
    return path


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_cfg(tmp_path: Path, cfg: dict) -> Path:
    path = tmp_path / "{}.yaml".format(cfg["name"])
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


# -- C8: predict_labels --------------------------------------------------------------------------

def test_predict_labels_baseline(mock_master, baseline_cfg):
    labels = predict_labels(mock_master, baseline_cfg)
    np.testing.assert_array_equal(labels, baseline_labels(mock_master, baseline_cfg["baseline"]))
    assert labels.dtype == np.dtype("<U1") and labels.shape == (len(mock_master),)


def test_predict_labels_ignores_holdout_for_baseline(mock_master, baseline_cfg):
    np.testing.assert_array_equal(predict_labels(mock_master, baseline_cfg, holdout="J03WMX"),
                                  predict_labels(mock_master, baseline_cfg))


def test_predict_labels_unknown_pipeline(mock_master):
    with pytest.raises(ValueError, match="unknown pipeline"):
        predict_labels(mock_master, {"pipeline": "magic"})


def test_predict_labels_always_validates(mock_master, baseline_cfg, monkeypatch):
    from modeling import pipeline

    def broken(master, params):
        return np.full(len(master), "H", dtype="<U1")      # H on the halftime gap

    monkeypatch.setattr(pipeline, "baseline_labels", broken)
    with pytest.raises(ValueError, match="half == 0"):
        pipeline.predict_labels(mock_master, baseline_cfg)


# -- make_submission -----------------------------------------------------------------------------

def test_thread_limits_are_set_before_any_other_import():
    source = Path(make_submission.__file__).read_text(encoding="utf-8")
    body = ast.parse(source).body
    stmts = [s for s in body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    assert isinstance(stmts[0], ast.ImportFrom) and stmts[0].module == "__future__"
    assert isinstance(stmts[1], ast.Import) and [a.name for a in stmts[1].names] == ["os"]
    loop = stmts[2]
    assert isinstance(loop, ast.For)
    assert sorted(e.value for e in loop.iter.elts) == sorted(THREAD_VARS)
    assert "setdefault" in ast.unparse(loop)


def test_thread_limits_in_a_fresh_interpreter():
    env = {k: v for k, v in os.environ.items() if k not in THREAD_VARS}
    code = ("import sys; assert 'numpy' not in sys.modules; import os; "
            "import modeling.make_submission; "
            "print(','.join(os.environ[v] for v in {!r}))".format(THREAD_VARS))
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, check=True,
                         capture_output=True, text=True).stdout.strip()
    assert out == "1,1,1"
    env["OMP_NUM_THREADS"] = "4"
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, check=True,
                         capture_output=True, text=True).stdout.strip()
    assert out == "4,1,1"                              # setdefault keeps an explicit choice


def test_cli_skip_preprocessing_writes_valid_csv(mock_master, baseline_cfg, data_root, tmp_path,
                                                 capsys, monkeypatch):
    _save_master(mock_master, data_root)
    import preprocessing.run_all as run_all
    monkeypatch.setattr(run_all, "run_pipeline",
                        lambda **kw: pytest.fail("preprocessing must be skipped"))
    out = tmp_path / "sub.csv"
    rc = make_submission.main(["--config", str(BASELINE_YAML), "--skip-preprocessing",
                               "--out", str(out)])
    assert rc == 0
    table = pm.read_label_csv(out)
    assert pm.validate_submission(table, N_GAME) is None
    np.testing.assert_array_equal(table["label"].to_numpy().astype("<U1"),
                                  predict_labels(mock_master, baseline_cfg))

    printed = capsys.readouterr().out
    from modeling.config import config_hash
    assert "config: baseline (sha256 {})".format(config_hash(baseline_cfg)) in printed
    assert _sha256(out) in printed
    assert "graded frames" in printed and "possessions: half 1 = " in printed


def test_cli_default_output_path(mock_master, data_root, tmp_path, monkeypatch):
    _save_master(mock_master, data_root)
    monkeypatch.setattr(paths, "outputs_dir", lambda: tmp_path / "outputs")
    assert make_submission.main(["--config", str(BASELINE_YAML), "--skip-preprocessing"]) == 0
    assert (tmp_path / "outputs" / "baseline.csv").is_file()


def test_cli_data_dir_overrides_env(mock_master, data_root, tmp_path):
    other = tmp_path / "other"
    _save_master(mock_master, other)
    out = tmp_path / "sub.csv"
    rc = make_submission.main(["--config", str(BASELINE_YAML), "--skip-preprocessing",
                               "--data-dir", str(other), "--out", str(out)])
    assert rc == 0 and out.is_file()


def test_cli_runs_preprocessing_and_aborts_on_failure(mock_master, data_root, tmp_path,
                                                     monkeypatch):
    import preprocessing.run_all as run_all
    calls = []

    def fake_pipeline(rc):
        def run(data_dir=None, **kw):
            calls.append(data_dir)
            if rc == 0:
                _save_master(mock_master, data_root)
            return rc
        return run

    out = tmp_path / "sub.csv"
    monkeypatch.setattr(run_all, "run_pipeline", fake_pipeline(3))
    assert make_submission.main(["--config", str(BASELINE_YAML), "--out", str(out)]) == 3
    assert not out.exists()

    monkeypatch.setattr(run_all, "run_pipeline", fake_pipeline(0))
    assert make_submission.main(["--config", str(BASELINE_YAML), "--out", str(out),
                                 "--data-dir", str(data_root)]) == 0
    assert calls == [None, str(data_root)]
    assert pm.validate_submission(pm.read_label_csv(out), N_GAME) is None


def test_cli_missing_master(data_root, tmp_path, capsys):
    out = tmp_path / "sub.csv"
    rc = make_submission.main(["--config", str(BASELINE_YAML), "--skip-preprocessing",
                               "--out", str(out)])
    assert rc == 1 and not out.exists()
    assert "master not found" in capsys.readouterr().err


def test_cli_rejects_bad_config(baseline_cfg, data_root, tmp_path, capsys):
    bad = copy.deepcopy(baseline_cfg)
    bad["extra"] = 1
    assert make_submission.main(["--config", str(_write_cfg(tmp_path, bad)),
                                 "--skip-preprocessing"]) == 2
    assert make_submission.main(["--config", str(tmp_path / "nope.yaml"),
                                 "--skip-preprocessing"]) == 2
    assert "error:" in capsys.readouterr().err


def test_cli_fused_ball_source_reads_processed_fused(baseline_cfg, data_root, tmp_path, capsys):
    """F4-06: ball_source fused is accepted and loads the master from <data>/processed_fused only."""
    fused = copy.deepcopy(baseline_cfg)
    fused["ball_source"] = "fused"
    out = tmp_path / "sub.csv"
    assert make_submission.main(["--config", str(_write_cfg(tmp_path, fused)),
                                 "--skip-preprocessing", "--out", str(out)]) == 1
    assert not out.exists()
    err = capsys.readouterr().err
    assert "master not found" in err and "processed_fused" in err


def test_cli_as_module_is_byte_reproducible(mock_master, tmp_path):
    root = tmp_path / "data"
    _save_master(mock_master, root)
    env = {k: v for k, v in os.environ.items() if k not in THREAD_VARS}
    env["MTAH_DATA_DIR"] = str(root)
    hashes = []
    for i in range(2):
        out = tmp_path / "run{}.csv".format(i)
        proc = subprocess.run(
            [sys.executable, "-m", "modeling.make_submission", "--config", str(BASELINE_YAML),
             "--skip-preprocessing", "--out", str(out)],
            cwd=REPO, env=env, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        assert pm.validate_submission(pm.read_label_csv(out), N_GAME) is None
        hashes.append(_sha256(out))
        assert hashes[-1] in proc.stdout
    assert hashes[0] == hashes[1]


def test_label_summary_counts_possessions_per_half():
    half = np.array([0, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2], np.int8)
    labels = np.array(list("XHHDAAXAHHD"), dtype="<U1")
    summary = make_submission.label_summary(labels, half)
    assert summary["n_graded"] == 9
    assert summary["possessions"] == {1: 2, 2: 2}
    assert summary["shares"] == pytest.approx({"H": 4 / 9, "A": 3 / 9, "D": 2 / 9})
    lines = make_submission.format_summary(summary)
    assert lines[1] == "possessions: half 1 = 2, half 2 = 2"


# -- tools/check_public_window.py ----------------------------------------------------------------

K1 = KICKOFF_FALLBACK[1]
WINDOW = (K1, K1 + 3000)                          # synthetic public window [start, stop)


@pytest.fixture(scope="session")
def window_case(mock_master):
    """(master, gt_codes, pred_codes) built so that every check passes."""
    master = mock_master.copy()
    n = len(master)
    gt = np.full(n, pm.X, dtype=np.int8)
    a, b = WINDOW
    gt[a:a + 1000] = pm.H
    gt[a + 1000:a + 2000] = pm.A
    gt[a + 2000:b] = pm.D

    d_diff = master["d_diff"].to_numpy().copy()
    d_diff[gt == pm.H] = -5.0
    d_diff[gt == pm.A] = 5.0
    master["d_diff"] = d_diff.astype(np.float32)
    home = master["n_tags_favor_home"].to_numpy().copy()
    away = master["n_tags_favor_away"].to_numpy().copy()
    home[gt == pm.H], away[gt == pm.H] = 2, 0
    home[gt == pm.A], away[gt == pm.A] = 0, 2
    master["n_tags_favor_home"] = home.astype(np.int16)
    master["n_tags_favor_away"] = away.astype(np.int16)
    for half, kick in KICKOFF_FALLBACK.items():
        cx_h = master["centroid_home_x"].to_numpy().copy()
        cx_a = master["centroid_away_x"].to_numpy().copy()
        cx_h[kick:kick + cpw.KICKOFF_WINDOW_FRAMES] = 30.0
        cx_a[kick:kick + cpw.KICKOFF_WINDOW_FRAMES] = 70.0
        master["centroid_home_x"] = cx_h.astype(np.float32)
        master["centroid_away_x"] = cx_a.astype(np.float32)

    half = master["half"].to_numpy()
    pred = np.where(half == 0, pm.X, pm.H).astype(np.int8)
    pred[gt != pm.X] = gt[gt != pm.X]
    return master, gt, pred


def _results(master, gt, pred, error=None):
    return {r.key: r.passed for r in cpw.run_checks(pred, error, gt, master)}


def test_check_all_pass(window_case):
    master, gt, pred = window_case
    assert all(_results(master, gt, pred).values())


def test_check_invalid_submission_fails_dependent_checks(window_case):
    master, gt, _ = window_case
    res = _results(master, gt, None, "row count 3 != {}".format(N_GAME))
    assert not res["i"] and not res["ii"] and not res["vi"] and not res["vii"]
    assert res["iii"] and res["iv"] and res["v"]


def test_check_x_rules(window_case):
    master, gt, pred = window_case
    graded = np.flatnonzero(master["half"].to_numpy() != 0)
    gap = np.flatnonzero(master["half"].to_numpy() == 0)
    x_inside = pred.copy()
    x_inside[graded[-1]] = pm.X
    assert not _results(master, gt, x_inside)["ii"]
    label_outside = pred.copy()
    label_outside[gap[0]] = pm.H
    assert not _results(master, gt, label_outside)["ii"]


def test_check_swapped_teams(window_case):
    master, gt, pred = window_case
    swapped = pred.copy()
    swapped[pred == pm.H], swapped[pred == pm.A] = pm.A, pm.H
    res = _results(master, gt, swapped)
    assert not res["vi"] and res["i"] and res["ii"] and res["vii"]


def test_check_no_dead_prediction(window_case):
    master, gt, pred = window_case
    no_d = np.where(pred == pm.D, pm.H, pred).astype(np.int8)
    assert not _results(master, gt, no_d)["vii"]


def test_check_master_sanity_failures(window_case):
    master, gt, pred = window_case
    m = master.copy()
    m["d_diff"] = (-m["d_diff"]).astype(np.float32)
    assert not _results(m, gt, pred)["iii"]

    m = master.copy()
    m["n_tags_favor_home"], m["n_tags_favor_away"] = (master["n_tags_favor_away"],
                                                      master["n_tags_favor_home"])
    assert not _results(m, gt, pred)["iv"]

    m = master.copy()
    m["centroid_home_x"], m["centroid_away_x"] = (master["centroid_away_x"],
                                                  master["centroid_home_x"])
    assert not _results(m, gt, pred)["v"]

    # one kickoff without centroids is enough to fail
    m = master.copy()
    cx = m["centroid_home_x"].to_numpy().copy()
    k2 = KICKOFF_FALLBACK[2]
    cx[k2:k2 + cpw.KICKOFF_WINDOW_FRAMES] = np.nan
    m["centroid_home_x"] = cx
    assert not _results(m, gt, pred)["v"]


def test_check_gt_without_dead_frames_fails_vii(window_case):
    master, gt, pred = window_case
    gt2 = np.where(gt == pm.D, pm.H, gt).astype(np.int8)
    assert not _results(master, gt2, pred)["vii"]


def _write_codes(codes, path):
    labels = np.array(pm.LABELS, dtype="<U1")[codes]
    io.write_submission(labels, path, n_expected=len(codes))
    return path


def test_check_cli_pass_and_fail(window_case, tmp_path, capsys):
    master, gt, pred = window_case
    master_path = tmp_path / "master.parquet"
    master.to_parquet(master_path, index=False)
    gt_path = _write_codes(gt, tmp_path / "gt.csv")
    pred_path = _write_codes(pred, tmp_path / "pred.csv")
    base = ["--gt", str(gt_path), "--master", str(master_path)]

    assert cpw.main(["--pred", str(pred_path)] + base) == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 8 and out[-1] == "PASS"
    assert all(line.startswith("PASS  (") for line in out[:-1])
    assert not any(ch.isdigit() for line in out for ch in line)   # no numbers without --verbose

    assert cpw.main(["--pred", str(pred_path), "--verbose"] + base) == 0
    verbose = capsys.readouterr().out
    assert "team accuracy" in verbose

    swapped = pred.copy()
    swapped[pred == pm.H], swapped[pred == pm.A] = pm.A, pm.H
    bad_path = _write_codes(swapped, tmp_path / "bad.csv")
    assert cpw.main(["--pred", str(bad_path)] + base) == 1
    out = capsys.readouterr().out.splitlines()
    assert "FAIL  (vi) " in "\n".join(out) and out[-1] == "FAIL (1 of 7 checks)"

    short = tmp_path / "short.csv"
    short.write_text("frame,label\n0,H\n", encoding="utf-8")
    assert cpw.main(["--pred", str(short)] + base) == 1
    assert "FAIL  (i) " in capsys.readouterr().out

    missing = tmp_path / "missing.csv"
    assert cpw.main(["--pred", str(missing)] + base) == 1


def test_check_cli_missing_master(tmp_path, data_root, capsys):
    pred = tmp_path / "pred.csv"
    assert cpw.main(["--pred", str(pred)]) == 1
    assert capsys.readouterr().out.startswith("FAIL  could not load inputs")


def test_check_script_runs_from_any_directory(window_case, tmp_path):
    master, gt, pred = window_case
    master_path = tmp_path / "master.parquet"
    master.to_parquet(master_path, index=False)
    gt_path = _write_codes(gt, tmp_path / "gt.csv")
    pred_path = _write_codes(pred, tmp_path / "pred.csv")
    proc = subprocess.run(
        [sys.executable, str(REPO / "tools" / "check_public_window.py"), "--pred", str(pred_path),
         "--gt", str(gt_path), "--master", str(master_path)],
        cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines()[-1] == "PASS"


def test_default_gt_is_the_public_file():
    assert cpw.DEFAULT_GT == REPO / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv"
    assert cpw.DEFAULT_GT.is_file()
