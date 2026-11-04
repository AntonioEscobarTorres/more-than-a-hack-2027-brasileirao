"""Synthetic tests for preprocessing.run_all (task E1).

Unit tests use only synthetic / stubbed steps — match assets never enter tests/.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from preprocessing import run_all
from preprocessing.build_master_table import file_sha256

# Fixed payload used by the stubbed D step so two full runs share a byte-identical master.
_SYNTH_MASTER = b"e1-synthetic-master-features-v1\n"


def test_run_pipeline_step_order_and_v1(monkeypatch, tmp_path):
    """Pipeline must call C → C5 → A(V1) → B → D; A must pass --stage V1."""
    calls = []

    def fake_export_tagging(output_dir, blocks, exact_kickoff_frames=None, data_dir=None):
        calls.append(("C", str(output_dir), data_dir))

    def fake_export_homography(output_dir, blocks, data_dir=None, n_frames=None):
        calls.append(("C5", str(output_dir), data_dir))

    def fake_tracking_main(argv=None):
        argv = list(argv or [])
        calls.append(("A", tuple(argv)))
        assert "--stage" in argv
        assert argv[argv.index("--stage") + 1] == "V1"
        return 0

    def fake_proximities_main(argv=None):
        calls.append(("B", tuple(argv or [])))
        return 0

    def fake_master_main(argv=None):
        calls.append(("D", tuple(argv or [])))
        return 0

    monkeypatch.setattr(
        "preprocessing.export_tagging.export_tagging_data", fake_export_tagging)
    monkeypatch.setattr(
        "preprocessing.prep_homography.export_homography_data", fake_export_homography)
    monkeypatch.setattr("preprocessing.prep_tracking.main", fake_tracking_main)
    monkeypatch.setattr("preprocessing.prep_proximities.main", fake_proximities_main)
    monkeypatch.setattr("preprocessing.build_master_table.main", fake_master_main)

    processed = tmp_path / "processed"
    rc = run_all.run_pipeline(
        data_dir=str(tmp_path / "data"),
        processed_dir=str(processed),
    )
    assert rc == 0
    assert [c[0] for c in calls] == ["C", "C5", "A", "B", "D"]
    assert calls[2][0] == "A"
    assert "--stage" in calls[2][1]
    assert calls[2][1][list(calls[2][1]).index("--stage") + 1] == "V1"


def test_run_pipeline_skip_homography(monkeypatch, tmp_path):
    calls = []

    monkeypatch.setattr(
        "preprocessing.export_tagging.export_tagging_data",
        lambda *a, **k: calls.append("C"))
    monkeypatch.setattr(
        "preprocessing.prep_homography.export_homography_data",
        lambda *a, **k: calls.append("C5"))
    monkeypatch.setattr(
        "preprocessing.prep_tracking.main",
        lambda argv=None: calls.append("A") or 0)
    monkeypatch.setattr(
        "preprocessing.prep_proximities.main",
        lambda argv=None: calls.append("B") or 0)
    monkeypatch.setattr(
        "preprocessing.build_master_table.main",
        lambda argv=None: calls.append("D") or 0)

    rc = run_all.run_pipeline(
        data_dir=str(tmp_path / "data"),
        processed_dir=str(tmp_path / "processed"),
        skip_homography=True,
    )
    assert rc == 0
    assert calls == ["C", "A", "B", "D"]


def test_run_pipeline_homography_failure_is_optional(monkeypatch, tmp_path):
    calls = []

    monkeypatch.setattr(
        "preprocessing.export_tagging.export_tagging_data",
        lambda *a, **k: calls.append("C"))

    def boom(*a, **k):
        calls.append("C5")
        raise FileNotFoundError("no homography.csv")

    monkeypatch.setattr(
        "preprocessing.prep_homography.export_homography_data", boom)
    monkeypatch.setattr(
        "preprocessing.prep_tracking.main",
        lambda argv=None: calls.append("A") or 0)
    monkeypatch.setattr(
        "preprocessing.prep_proximities.main",
        lambda argv=None: calls.append("B") or 0)
    monkeypatch.setattr(
        "preprocessing.build_master_table.main",
        lambda argv=None: calls.append("D") or 0)

    rc = run_all.run_pipeline(
        data_dir=str(tmp_path / "data"),
        processed_dir=str(tmp_path / "processed"),
    )
    assert rc == 0
    assert calls == ["C", "C5", "A", "B", "D"]


def _stub_pipeline(monkeypatch, d_payload=lambda folder: _SYNTH_MASTER, homography=True):
    """Stub every step with file-writing fakes; returns the list of folders D was run in."""
    d_runs = []

    def stub_c(output_dir, blocks, exact_kickoff_frames=None, data_dir=None):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "tagging.parquet").write_bytes(b"tag")
        (Path(output_dir) / "kickoffs.json").write_text("{}", encoding="utf-8")

    def stub_c5(output_dir, blocks, data_dir=None, n_frames=None):
        if not homography:
            raise FileNotFoundError("no homography.csv")
        (Path(output_dir) / "homography.parquet").write_bytes(b"hom")

    def stub_a(argv=None):
        argv = list(argv or [])
        out = Path(argv[argv.index("--output-dir") + 1])
        assert (out / "kickoffs.json").exists()          # C wrote it in the same folder A reads
        (out / "tracking_clean.parquet").write_bytes(b"trk")
        (out / "slot_teams.parquet").write_bytes(b"slt")
        return 0

    def stub_b(argv=None):
        argv = list(argv or [])
        folder = Path(argv[argv.index("--processed-dir") + 1])
        (folder / "proximities.parquet").write_bytes(b"prox")
        return 0

    def stub_d(argv=None):
        argv = list(argv or [])
        folder = Path(argv[argv.index("--processed-dir") + 1])
        d_runs.append(folder)
        (folder / "master_features.parquet").write_bytes(d_payload(folder))
        return 0

    monkeypatch.setattr(
        "preprocessing.export_tagging.export_tagging_data", stub_c)
    monkeypatch.setattr(
        "preprocessing.prep_homography.export_homography_data", stub_c5)
    monkeypatch.setattr("preprocessing.prep_tracking.main", stub_a)
    monkeypatch.setattr("preprocessing.prep_proximities.main", stub_b)
    monkeypatch.setattr("preprocessing.build_master_table.main", stub_d)
    return d_runs


def test_check_determinism_runs_the_whole_pipeline_twice(monkeypatch, tmp_path):
    """--check-determinism re-runs C → D from scratch in another folder and compares SHA-256."""
    d_runs = _stub_pipeline(monkeypatch)
    out = tmp_path / "processed"
    assert run_all.run_pipeline(
        data_dir=str(tmp_path / "data"), processed_dir=str(out), check_determinism=True) == 0
    assert len(d_runs) == 2 and d_runs[0] == out and d_runs[1] != out
    assert file_sha256(out / "master_features.parquet") == hashlib.sha256(_SYNTH_MASTER).hexdigest()


def test_check_determinism_fails_when_the_two_masters_differ(monkeypatch, tmp_path):
    counter = iter(range(10))
    _stub_pipeline(monkeypatch, d_payload=lambda folder: b"run-%d" % next(counter))
    assert run_all.run_pipeline(
        data_dir=str(tmp_path / "data"), processed_dir=str(tmp_path / "p"),
        check_determinism=True) == 1


def test_stale_homography_is_removed_when_c5_is_skipped(monkeypatch, tmp_path):
    _stub_pipeline(monkeypatch)
    out = tmp_path / "processed"
    out.mkdir()
    (out / "homography.parquet").write_bytes(b"old run")
    assert run_all.run_pipeline(
        data_dir=str(tmp_path / "data"), processed_dir=str(out), skip_homography=True) == 0
    assert not (out / "homography.parquet").exists()


def test_stale_homography_is_removed_when_c5_fails(monkeypatch, tmp_path):
    _stub_pipeline(monkeypatch, homography=False)
    out = tmp_path / "processed"
    out.mkdir()
    (out / "homography.parquet").write_bytes(b"old run")
    assert run_all.run_pipeline(
        data_dir=str(tmp_path / "data"), processed_dir=str(out)) == 0
    assert not (out / "homography.parquet").exists()


def test_main_cli_wires_flags(monkeypatch, tmp_path):
    seen = {}

    def fake_run(**kwargs):
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(run_all, "run_pipeline", fake_run)
    rc = run_all.main([
        "--data-dir", str(tmp_path / "data"),
        "--processed-dir", str(tmp_path / "out"),
        "--skip-homography",
        "--check-determinism",
    ])
    assert rc == 0
    assert seen["data_dir"] == str(tmp_path / "data")
    assert seen["processed_dir"] == str(tmp_path / "out")
    assert seen["skip_homography"] is True
    assert seen["check_determinism"] is True


def test_file_sha256_helper_stable(tmp_path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"abc123")
    assert file_sha256(path) == hashlib.sha256(b"abc123").hexdigest()
