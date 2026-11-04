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


def test_two_full_runs_same_master_sha256(monkeypatch, tmp_path):
    """Two complete pipeline runs must produce an identical master SHA-256."""

    def stub_c(output_dir, blocks, exact_kickoff_frames=None, data_dir=None):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "tagging.parquet").write_bytes(b"tag")
        (Path(output_dir) / "kickoffs.json").write_text("{}", encoding="utf-8")

    def stub_c5(output_dir, blocks, data_dir=None, n_frames=None):
        (Path(output_dir) / "homography.parquet").write_bytes(b"hom")

    def stub_a(argv=None):
        argv = list(argv or [])
        out = Path(argv[argv.index("--output-dir") + 1])
        out.mkdir(parents=True, exist_ok=True)
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
        master = folder / "master_features.parquet"
        master.write_bytes(_SYNTH_MASTER)
        if "--check-determinism" in argv:
            second = folder / "master_features_second.parquet"
            second.write_bytes(_SYNTH_MASTER)
            assert file_sha256(master) == file_sha256(second)
        return 0

    monkeypatch.setattr(
        "preprocessing.export_tagging.export_tagging_data", stub_c)
    monkeypatch.setattr(
        "preprocessing.prep_homography.export_homography_data", stub_c5)
    monkeypatch.setattr("preprocessing.prep_tracking.main", stub_a)
    monkeypatch.setattr("preprocessing.prep_proximities.main", stub_b)
    monkeypatch.setattr("preprocessing.build_master_table.main", stub_d)

    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"
    assert run_all.run_pipeline(
        data_dir=str(tmp_path / "data"), processed_dir=str(run1)) == 0
    assert run_all.run_pipeline(
        data_dir=str(tmp_path / "data"), processed_dir=str(run2),
        check_determinism=True) == 0

    h1 = file_sha256(run1 / "master_features.parquet")
    h2 = file_sha256(run2 / "master_features.parquet")
    assert h1 == h2
    assert len(h1) == 64


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
