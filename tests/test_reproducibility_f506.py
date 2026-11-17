"""[F5-06] Tests for reproducibility documentation and final config determinism."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from modeling import make_submission, paths
from modeling.config import config_hash, load_config


EXPECTED_CONFIG_SHA256 = "a84e812099d2df782dd2b66576ff47cd4892c01ab7d227e0c93f0d394066f1f8"
EXPECTED_CSV_SHA256 = "25064d90ff2808ffb325f43d51253a5dd58be765d70c2ddee6d608f5fb9f0503"


def test_reproducibility_docs_exist_and_cover_spec():
    root = paths.repo_root()
    repro_doc = root / "docs" / "reproducibility.md"
    readme = root / "README.md"
    task_doc = root / "docs" / "tasks" / "F5-06.md"

    assert repro_doc.is_file(), "docs/reproducibility.md must exist"
    repro_text = repro_doc.read_text(encoding="utf-8")
    for token in (
        "Python 3.12",
        "pip install -r requirements-modeling.txt",
        "./data",
        "MTAH_DATA_DIR",
        "python -m modeling.make_submission --config configs/final.yaml",
        EXPECTED_CSV_SHA256,
        "CPU Model",
    ):
        assert token in repro_text, f"missing {token!r} in docs/reproducibility.md"

    readme_text = readme.read_text(encoding="utf-8")
    assert "## How to run" in readme_text
    assert "requirements-modeling.txt" in readme_text
    assert "python -m modeling.make_submission --config configs/final.yaml" in readme_text
    assert EXPECTED_CSV_SHA256 in readme_text
    assert "docs/reproducibility.md" in readme_text

    task_text = task_doc.read_text(encoding="utf-8")
    assert "AVX-512" in task_text


def test_final_config_and_single_thread_determinism_guards(tmp_path: Path):
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        assert os.environ.get(var) == "1"

    cfg = load_config(paths.repo_root() / "configs" / "final.yaml")
    assert cfg["name"] == "final"
    assert config_hash(cfg) == EXPECTED_CONFIG_SHA256

    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"
    payload = "frame,label\n0,X\n1,H\n2,A\n3,D\n"
    csv_a.write_text(payload, encoding="utf-8")
    csv_b.write_text(payload, encoding="utf-8")
    assert make_submission.file_sha256(csv_a) == make_submission.file_sha256(csv_b)

    labels = np.array(["X", "H", "H", "D", "A", "A", "X"], dtype="U1")
    half = np.array([0, 1, 1, 1, 2, 2, 0], dtype=np.int64)
    summary = make_submission.label_summary(labels, half)
    assert summary["n_graded"] == 5
    assert summary["possessions"] == {1: 1, 2: 1}
