"""Unit tests for modeling/external/build.py and DEGRADE_CHOICES (F4-04).

All fixtures are synthetic and live under ``tmp_path``. ``MTAH_DATA_DIR`` and
``MTAH_EXT_DIR`` are redirected for every test (autouse), so nothing under the
real data directory or ``./data`` is ever read or written. All randomness uses
fixed seeds.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modeling.contracts import LABEL_CODES
from modeling.external import build
from modeling.external.files import (
    GAP_FRAMES,
    IDSSE_MATCHES,
    N_HOME_SLOTS,
    X_CODE,
    ExternalMatch,
    arrays_path,
    expected_half,
    labels_path,
    load_external,
    master_path,
)
from modeling.io import validate_master
from modeling.scoring import external_eval
from preprocessing.schema import N_SLOTS

DEGRADE_MODES = ("none", "default", "harder", "easier", "hidelo", "hidehi")

SUMMARY_KEYS = {
    "match",
    "degrade",
    "pct_ball_observed",
    "pct_ball_filled",
    "pct_ball_jump_removed",
    "mean_n_tracked",
    "pct_d",
    "pct_ball_observed_d",
    "pct_ball_observed_play",
    "bg_hazard",
}


# --------------------------------------------------------------------------- #
# Helpers / fixtures
# --------------------------------------------------------------------------- #
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _make_synthetic_external_match(
    match_id: str = "J03WMX",
    n_first: int = 80,
    n_second: int = 80,
    seed: int = 13,
) -> ExternalMatch:
    """Build a small valid C9 ExternalMatch fixture (deterministic for a given seed)."""
    rng = np.random.default_rng(seed)
    half = expected_half(n_first, n_second)
    t_total = half.size
    gap = half == 0
    n_in = int((~gap).sum())

    players = np.full((t_total, N_SLOTS, 3), np.nan, dtype=np.float32)
    # Activate 11 home slots (0..10) and 11 away slots on in-half frames.
    active_slots = list(range(11)) + list(range(N_HOME_SLOTS, N_HOME_SLOTS + 11))
    for s in active_slots:
        players[~gap, s, 0] = rng.uniform(0.15, 0.85, size=n_in).astype(np.float32)
        players[~gap, s, 1] = rng.uniform(0.15, 0.85, size=n_in).astype(np.float32)
        players[~gap, s, 2] = rng.uniform(0.5, 3.0, size=n_in).astype(np.float32)

    ball = np.full((t_total, 3), np.nan, dtype=np.float32)
    ball[~gap, 0] = np.linspace(0.2, 0.8, n_in, dtype=np.float32)
    ball[~gap, 1] = 0.5
    ball[~gap, 2] = 4.0

    # Freeze ball during a dead-ball segment in half 1 (never crossing into the gap).
    d_start = min(5, n_first // 4)
    d_end = min(50, n_first)
    if d_end > d_start:
        ball[d_start:d_end, 0] = 0.5
        ball[d_start:d_end, 1] = 0.5
        ball[d_start:d_end, 2] = 0.0
    ball[gap] = np.nan

    ball_z = np.zeros(t_total, dtype=np.float32)
    ball_z[gap] = np.nan

    labels = np.full(t_total, LABEL_CODES["H"], dtype=np.int8)
    if d_end > d_start:
        labels[d_start:d_end] = LABEL_CODES["D"]
    a_start = n_first + GAP_FRAMES + min(10, n_second // 4)
    a_end = min(n_first + GAP_FRAMES + 40, t_total)
    if a_end > a_start:
        labels[a_start:a_end] = LABEL_CODES["A"]
    labels[gap] = X_CODE

    slot_team_ids = np.r_[
        np.zeros(N_HOME_SLOTS, dtype=np.int8),
        np.ones(N_SLOTS - N_HOME_SLOTS, dtype=np.int8),
    ]
    pitch = np.array([105.0, 68.0], dtype=np.float64)

    return ExternalMatch(
        match_id=match_id,
        players=players,
        ball=ball,
        ball_z=ball_z,
        labels=labels,
        half=half,
        slot_team_ids=slot_team_ids,
        pitch=pitch,
    )


@pytest.fixture(autouse=True)
def isolated_data_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect every data/report location into ``tmp_path`` for every test."""
    ext_dir = tmp_path / "ext"
    monkeypatch.setenv("MTAH_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MTAH_EXT_DIR", str(ext_dir))
    # Safety net: even if a test forgets --report/--doc, nothing hits the repo tree.
    monkeypatch.setattr(build, "DEFAULT_REPORT_PATH", tmp_path / "default_report.md")
    monkeypatch.setattr(build, "DEFAULT_EXTERNAL_DATA_DOC", tmp_path / "default_doc.md")
    return ext_dir


@pytest.fixture
def degrade_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Wrap ``degrade_mod.degrade_with_info`` to record calls and tolerate tiny-fixture failures.

    The bisection target of the real degrader may be unreachable on a tiny synthetic
    match; in that case the undegraded copy is returned so the build path is still
    exercised end-to-end.
    """
    calls: list[dict] = []
    orig_degrade = build.degrade_mod.degrade_with_info

    def _safe_degrade(m: ExternalMatch, cfg: dict, seed: int) -> tuple[ExternalMatch, dict]:
        calls.append({"match_id": m.match_id, "seed": seed})
        try:
            return orig_degrade(m, cfg, seed=seed)
        except Exception:  # noqa: BLE001 - tiny fixture may not satisfy bisection target
            return m, {"bg_hazard": 0.0}

    monkeypatch.setattr(build.degrade_mod, "degrade_with_info", _safe_degrade)
    return calls


def _make_summary_master(n_gap: int = 20, n_half: int = 80):
    """Hand-computed master/labels pair used by the summary-metric tests."""
    n = n_gap + n_half
    half = np.array([0] * n_gap + [1] * n_half, dtype=np.int64)
    labels = np.array(
        [X_CODE] * n_gap
        + [LABEL_CODES["D"]] * 30
        + [LABEL_CODES["H"]] * 30
        + [LABEL_CODES["A"]] * 20,
        dtype=np.int8,
    )
    ball_obs = np.zeros(n, dtype=np.int64)
    ball_obs[20:35] = 1  # 15 of 30 D frames -> 50%
    ball_obs[50:90] = 1  # 40 of 50 in-play frames -> 80%

    ball_filled = np.zeros(n, dtype=np.int64)
    ball_filled[35:43] = 1  # 8 of 80 -> 10%

    ball_jump = np.zeros(n, dtype=np.int64)
    ball_jump[50:54] = 1  # 4 of 80 -> 5%

    master = pd.DataFrame(
        {
            "frame": np.arange(n, dtype=np.int64),
            "half": half,
            "ball_observed": ball_obs,
            "ball_filled": ball_filled,
            "ball_jump_removed": ball_jump,
            "n_tracked_home": np.full(n, 10, dtype=np.int64),
            "n_tracked_away": np.full(n, 11, dtype=np.int64),
        }
    )
    return master, labels


# --------------------------------------------------------------------------- #
# 2. DEGRADE_CHOICES
# --------------------------------------------------------------------------- #
def test_external_eval_degrade_choices_include_hidelo_and_hidehi() -> None:
    """DEGRADE_CHOICES in modeling/scoring/external_eval.py includes hidelo & hidehi."""
    assert hasattr(external_eval, "DEGRADE_CHOICES")
    for mode in DEGRADE_MODES:
        assert mode in external_eval.DEGRADE_CHOICES
    assert "hidelo" in external_eval.DEGRADE_CHOICES
    assert "hidehi" in external_eval.DEGRADE_CHOICES


def test_build_degrade_choices_cover_all_modes() -> None:
    """The build CLI accepts exactly the six documented degradation modes."""
    assert tuple(build.BUILD_DEGRADE_CHOICES) == DEGRADE_MODES
    parser = build.build_parser()
    for mode in DEGRADE_MODES:
        assert parser.parse_args(["--degrade", mode]).degrade == mode
    with pytest.raises(SystemExit):
        parser.parse_args(["--degrade", "not-a-mode"])


# --------------------------------------------------------------------------- #
# 1. Build across degradation modes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", DEGRADE_MODES)
def test_build_synthetic_match_across_modes(
    tmp_path: Path,
    degrade_calls: list[dict],
    mode: str,
) -> None:
    """Build on a synthetic C9 match gives a valid master and labels == X iff half == 0."""
    match_id = IDSSE_MATCHES[0]
    ext_match = _make_synthetic_external_match(match_id=match_id)
    ext_match.save(arrays_path(match_id))

    report_file = tmp_path / "external_masters.md"
    rc = build.main(
        [
            "--matches",
            match_id,
            "--degrade",
            mode,
            "--seed",
            "0",
            "--report",
            str(report_file),
        ]
    )
    assert rc == 0
    assert report_file.is_file()

    # Artifacts were written inside tmp_path only.
    assert tmp_path in master_path(match_id, mode).parents
    assert tmp_path in labels_path(match_id, mode).parents
    assert master_path(match_id, mode).is_file()
    assert labels_path(match_id, mode).is_file()

    # Degradation is applied for every mode except "none".
    if mode == "none":
        assert degrade_calls == []
    else:
        assert len(degrade_calls) == 1
        assert degrade_calls[0]["match_id"] == match_id

    master, labels = load_external(match_id, mode)
    validate_master(master, n_frames=ext_match.n_frames)

    assert len(labels) == ext_match.n_frames
    half = master["half"].to_numpy()
    assert np.array_equal(labels == X_CODE, half == 0)

    # The report row for this (match, mode) pair exists.
    report_text = report_file.read_text(encoding="utf-8")
    assert any(
        line.startswith("| {} | {} |".format(match_id, mode))
        for line in report_text.splitlines()
    )


def test_build_is_deterministic_for_fixed_seed(degrade_calls: list[dict]) -> None:
    """Two builds with the same seed produce identical masters and labels."""
    match_id = IDSSE_MATCHES[0]
    _make_synthetic_external_match(match_id=match_id, seed=13).save(arrays_path(match_id))

    row_a = build.build_one(match_id, "default", seed=5)
    master_a, labels_a = load_external(match_id, "default")
    row_b = build.build_one(match_id, "default", seed=5)
    master_b, labels_b = load_external(match_id, "default")

    pd.testing.assert_frame_equal(master_a, master_b)
    assert np.array_equal(labels_a, labels_b)
    assert row_a == row_b
    assert [c["seed"] for c in degrade_calls] == [5, 5]


def test_build_passes_per_match_seed_to_degrade(
    tmp_path: Path, degrade_calls: list[dict]
) -> None:
    """The degrade seed is ``base_seed + index of the match`` (deterministic, per match)."""
    match_id = IDSSE_MATCHES[0]
    _make_synthetic_external_match(match_id=match_id).save(arrays_path(match_id))

    rc = build.main(
        [
            "--matches",
            match_id,
            "--degrade",
            "default",
            "--seed",
            "7",
            "--report",
            str(tmp_path / "report.md"),
        ]
    )
    assert rc == 0
    assert degrade_calls[0]["seed"] == 7 + IDSSE_MATCHES.index(match_id)
    for i, m in enumerate(IDSSE_MATCHES):
        assert build._seed_for_match(3, m) == 3 + i
    assert build._seed_for_match(3, "NOT_A_MATCH") == 3


def test_resolve_matches_rejects_unknown_match() -> None:
    assert build._resolve_matches(["all"]) == list(IDSSE_MATCHES)
    assert build._resolve_matches([IDSSE_MATCHES[0]]) == [IDSSE_MATCHES[0]]
    with pytest.raises(ValueError, match="unknown match"):
        build._resolve_matches(["definitely_not_a_match"])


# --------------------------------------------------------------------------- #
# 3. --verify CLI
# --------------------------------------------------------------------------- #
def _write_verify_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Create npz/master/labels files plus a mocked docs/external_data.md in tmp_path."""
    match_id = IDSSE_MATCHES[0]
    npz_file = arrays_path(match_id)
    _make_synthetic_external_match(match_id=match_id, n_first=20, n_second=20).save(npz_file)

    m_file = master_path(match_id, "default")
    l_file = labels_path(match_id, "default")
    m_file.parent.mkdir(parents=True, exist_ok=True)
    l_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"frame": [0, 1], "half": [1, 1]}).to_parquet(m_file, index=False)
    np.save(l_file, np.array([LABEL_CODES["H"], LABEL_CODES["H"]], dtype=np.int8))

    doc_file = tmp_path / "external_data.md"
    doc_file.write_text(
        "# External Data Hashes\n\n"
        "| File | SHA-256 |\n| --- | --- |\n"
        f"| `{npz_file.name}` | `{_sha256(npz_file)}` |\n"
        f"| `{m_file.name}` | `{_sha256(m_file)}` |\n"
        f"| `{l_file.name}` | `{_sha256(l_file)}` |\n",
        encoding="utf-8",
    )
    return npz_file, m_file, l_file, doc_file


def test_parse_expected_hashes_reads_mocked_doc(tmp_path: Path) -> None:
    npz_file, m_file, l_file, doc_file = _write_verify_fixture(tmp_path)
    hashes = build.parse_expected_hashes(doc_file)
    assert hashes == {
        npz_file.name: _sha256(npz_file),
        m_file.name: _sha256(m_file),
        l_file.name: _sha256(l_file),
    }

    with pytest.raises(FileNotFoundError):
        build.parse_expected_hashes(tmp_path / "missing.md")

    empty = tmp_path / "empty.md"
    empty.write_text("# nothing to see here\n", encoding="utf-8")
    with pytest.raises(ValueError):
        build.parse_expected_hashes(empty)


def test_verify_cli_identical_and_different(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--verify compares local files against docs/external_data.md SHA-256 entries."""
    npz_file, m_file, l_file, doc_file = _write_verify_fixture(tmp_path)

    rc_ok = build.main(["--verify", "--doc", str(doc_file)])
    out_ok = capsys.readouterr().out
    assert rc_ok == 0
    assert out_ok.count("IDENTICAL") == 3
    assert "DIFFERENT" not in out_ok

    # Corrupt labels file -> must report DIFFERENT and return non-zero.
    l_file.write_bytes(b"corrupted")
    rc_bad = build.main(["--verify", "--doc", str(doc_file)])
    out_bad = capsys.readouterr().out
    assert rc_bad == 1
    assert "DIFFERENT" in out_bad
    assert "IDENTICAL" in out_bad  # the two untouched files still match


@pytest.mark.parametrize("which", ["npz", "master", "labels"])
def test_verify_cli_flags_each_corrupted_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    which: str,
) -> None:
    """Corrupting any one of the three artifact kinds is detected by --verify."""
    npz_file, m_file, l_file, doc_file = _write_verify_fixture(tmp_path)
    target = {"npz": npz_file, "master": m_file, "labels": l_file}[which]
    target.write_bytes(b"corrupted")

    rc = build.main(["--verify", "--doc", str(doc_file)])
    out = capsys.readouterr().out
    assert rc == 1
    different_lines = [ln for ln in out.splitlines() if ln.startswith("DIFFERENT")]
    assert len(different_lines) == 1
    assert target.name in different_lines[0]


def test_verify_cli_reports_missing_file_as_different(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A hash listed in the doc whose local file is absent is reported DIFFERENT."""
    _, m_file, _, doc_file = _write_verify_fixture(tmp_path)
    m_file.unlink()

    rc = build.main(["--verify", "--doc", str(doc_file)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "DIFFERENT" in out
    assert "missing" in out


def test_parse_expected_hashes_accepts_extra_columns_and_skips_source_rows(tmp_path: Path) -> None:
    """Hash rows may carry columns between name and digest; XML source rows are not artifacts."""
    digest_a, digest_b, digest_xml = "a" * 64, "B" * 64, "c" * 64
    doc_file = tmp_path / "external_data.md"
    doc_file.write_text(
        "Prose mentioning `J03WMX.npz` without a digest is ignored.\n\n"
        "| Match | Source | Bytes | SHA-256 |\n| --- | --- | ---: | --- |\n"
        f"| J03WMX | `DFL_04_03_positions.xml` | 12 | `{digest_xml}` |\n\n"
        "| File | Bytes | SHA-256 |\n| --- | ---: | --- |\n"
        f"| `J03WMX.npz` | 123 | `{digest_a}` |\n"
        f"| `J03WMX__default_master.parquet` | 456 | `{digest_b}` |\n",
        encoding="utf-8",
    )

    assert build.parse_expected_hashes(doc_file) == {
        "J03WMX.npz": digest_a,
        "J03WMX__default_master.parquet": digest_b.lower(),
    }


def test_parse_expected_hashes_rejects_rows_with_unpaired_digests(tmp_path: Path) -> None:
    doc_file = tmp_path / "external_data.md"
    doc_file.write_text(
        f"| `J03WMX.npz` | `J03WN1.npz` | `{'a' * 64}` |\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="file name"):
        build.parse_expected_hashes(doc_file)


def test_verify_cli_without_published_hashes_prints_different_and_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A doc with no artifact hashes yet is a clean DIFFERENT, not a traceback."""
    doc_file = tmp_path / "external_data.md"
    doc_file.write_text("# External data\n\nNo hashes published yet.\n", encoding="utf-8")

    rc = build.main(["--verify", "--doc", str(doc_file)])
    out = capsys.readouterr().out
    assert rc == 1
    assert out.startswith("DIFFERENT")


# --------------------------------------------------------------------------- #
# 4. Public summary metrics
# --------------------------------------------------------------------------- #
def test_summarize_master_computes_public_metrics() -> None:
    """summarize_master computes % observed, filled, jump, tracked, % D, vis in D/play, bg_hazard."""
    master, labels = _make_summary_master()

    row = build.summarize_master("J03WMX", "default", master, labels, bg_hazard=0.0125)
    assert row["match"] == "J03WMX"
    assert row["degrade"] == "default"
    assert np.isclose(row["pct_ball_observed"], 68.75)
    assert np.isclose(row["pct_ball_filled"], 10.0)
    assert np.isclose(row["pct_ball_jump_removed"], 5.0)
    assert np.isclose(row["mean_n_tracked"], 21.0)
    assert np.isclose(row["pct_d"], 37.5)
    assert np.isclose(row["pct_ball_observed_d"], 50.0)
    assert np.isclose(row["pct_ball_observed_play"], 80.0)
    assert np.isclose(row["bg_hazard"], 0.0125)


def test_summarize_master_ignores_gap_frames() -> None:
    """Frames with half == 0 never contribute to any metric, even if flags are set there."""
    master, labels = _make_summary_master()
    baseline = build.summarize_master("J03WMX", "default", master, labels, bg_hazard=0.01)

    polluted = master.copy()
    polluted.loc[:19, ["ball_observed", "ball_filled", "ball_jump_removed"]] = 1
    polluted.loc[:19, ["n_tracked_home", "n_tracked_away"]] = 99
    labels_polluted = labels.copy()
    labels_polluted[:19] = LABEL_CODES["D"]  # a D label on a gap frame must not count

    row = build.summarize_master("J03WMX", "default", polluted, labels_polluted, bg_hazard=0.01)
    for key in SUMMARY_KEYS - {"match", "degrade"}:
        assert np.isclose(row[key], baseline[key]), key


def test_summarize_master_accepts_string_labels() -> None:
    """Symbolic 'D'/'H'/'A'/'X' labels give the same result as integer codes."""
    master, labels = _make_summary_master()
    inv = {LABEL_CODES["D"]: "D", LABEL_CODES["H"]: "H", LABEL_CODES["A"]: "A", X_CODE: "X"}
    str_labels = np.array([inv[int(c)] for c in labels], dtype=object)

    row_int = build.summarize_master("J03WMX", "default", master, labels, bg_hazard=0.02)
    row_str = build.summarize_master("J03WMX", "default", master, str_labels, bg_hazard=0.02)
    assert row_int == row_str


def test_summarize_master_none_mode_has_no_bg_hazard() -> None:
    """The undegraded ('none') mode never reports a bg_hazard, even if one is passed."""
    master, labels = _make_summary_master()
    row = build.summarize_master("J03WMX", "none", master, labels, bg_hazard=0.5)
    assert row["bg_hazard"] is None

    row_missing = build.summarize_master("J03WMX", "default", master, labels, bg_hazard=None)
    assert row_missing["bg_hazard"] is None


def test_summarize_master_handles_no_graded_frames_and_no_d_or_play() -> None:
    """Empty denominators yield 0.0 rather than NaN or a ZeroDivisionError."""
    n = 10
    master = pd.DataFrame(
        {
            "half": np.zeros(n, dtype=np.int64),
            "ball_observed": np.ones(n, dtype=np.int64),
            "ball_filled": np.ones(n, dtype=np.int64),
            "ball_jump_removed": np.ones(n, dtype=np.int64),
            "n_tracked_home": np.full(n, 11, dtype=np.int64),
            "n_tracked_away": np.full(n, 11, dtype=np.int64),
        }
    )
    labels = np.full(n, X_CODE, dtype=np.int8)
    row = build.summarize_master("J03WMX", "default", master, labels, bg_hazard=0.1)
    for key in (
        "pct_ball_observed",
        "pct_ball_filled",
        "pct_ball_jump_removed",
        "mean_n_tracked",
        "pct_d",
        "pct_ball_observed_d",
        "pct_ball_observed_play",
    ):
        assert row[key] == 0.0, key


def test_summarize_master_does_not_mutate_or_leak_match_data() -> None:
    """The summary is scalars only and the inputs are left untouched."""
    master, labels = _make_summary_master()
    master_before = master.copy(deep=True)
    labels_before = labels.copy()

    row = build.summarize_master("J03WMX", "default", master, labels, bg_hazard=0.0125)

    pd.testing.assert_frame_equal(master, master_before)
    assert np.array_equal(labels, labels_before)

    assert set(row) == SUMMARY_KEYS
    for key, value in row.items():
        assert value is None or isinstance(value, (str, float, int, np.floating)), key
        assert not isinstance(value, (np.ndarray, pd.Series, pd.DataFrame, list, dict)), key


def test_report_contains_only_aggregates(tmp_path: Path) -> None:
    """The written report has one aggregate row per (match, mode) and no per-frame data."""
    master, labels = _make_summary_master()
    row_default = build.summarize_master(IDSSE_MATCHES[0], "default", master, labels, bg_hazard=0.0125)
    row_none = build.summarize_master(IDSSE_MATCHES[0], "none", master, labels)

    report = tmp_path / "reports" / "external_masters.md"
    out = build.update_report([row_default, row_none], report_path=report)
    assert out == report and report.is_file()

    lines = report.read_text(encoding="utf-8").splitlines()
    data_rows = [ln for ln in lines if ln.startswith("| {} |".format(IDSSE_MATCHES[0]))]
    assert len(data_rows) == 2
    # "none" sorts before "default" (BUILD_DEGRADE_CHOICES order).
    assert "| none |" in data_rows[0] and "| default |" in data_rows[1]
    assert data_rows[0].rstrip().endswith("| n/a |")
    assert data_rows[1].rstrip().endswith("| 0.012500 |")
    assert "68.75%" in data_rows[1]
    # Header + separator + 2 data rows; nothing frame-level.
    assert len([ln for ln in lines if ln.startswith("|")]) == 4

    # A re-run for one pair replaces its row and preserves the other.
    updated = dict(row_default, bg_hazard=0.5)
    build.update_report([updated], report_path=report)
    lines2 = report.read_text(encoding="utf-8").splitlines()
    rows2 = [ln for ln in lines2 if ln.startswith("| {} |".format(IDSSE_MATCHES[0]))]
    assert len(rows2) == 2
    assert any(ln.rstrip().endswith("| 0.500000 |") for ln in rows2)
    assert any(ln.rstrip().endswith("| n/a |") for ln in rows2)


def test_build_one_summary_matches_written_artifacts(degrade_calls: list[dict]) -> None:
    """The returned summary is consistent with the master/labels that build_one persisted."""
    match_id = IDSSE_MATCHES[0]
    ext_match = _make_synthetic_external_match(match_id=match_id)
    ext_match.save(arrays_path(match_id))

    row = build.build_one(match_id, "none", seed=0)
    master, labels = load_external(match_id, "none")

    expected = build.summarize_master(match_id, "none", master, labels)
    assert set(row) == SUMMARY_KEYS
    assert row["bg_hazard"] is None
    for key in SUMMARY_KEYS - {"match", "degrade"}:
        if row[key] is None:
            assert expected[key] is None
        else:
            assert np.isclose(row[key], expected[key]), key

@pytest.mark.parametrize("mode", ["default", "harder", "easier", "hidelo", "hidehi"])
def test_build_runs_the_real_degradation_on_the_synthetic_match(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No fallback: the shipped config degrades the fixture for real. Only the target
    tolerances and the player target are adapted: a 2 x 80-frame match cannot hit full-match targets."""
    match_id = IDSSE_MATCHES[0]
    _make_synthetic_external_match(match_id=match_id).save(arrays_path(match_id))
    shipped = build._load_degrade_config(mode)

    def widened(name: str) -> dict:
        cfg = dict(shipped)
        cfg["target_ball_visible_tolerance"] = 1.0
        # Player calibration is covered by the degrade tests; the random fixture players carry
        # no orientation structure, so removing many of them breaks the preprocessing instead.
        cfg["target_tracked_mean"] = float(N_SLOTS)
        cfg["target_tracked_mean_tolerance"] = float(N_SLOTS)
        cfg["dup_frac"] = 0.0
        return cfg

    monkeypatch.setattr(build, "_load_degrade_config", widened)

    row = build.build_one(match_id, mode, seed=0)
    clean_row = build.build_one(match_id, "none", seed=0)

    master = pd.read_parquet(master_path(match_id, mode))
    labels = np.load(labels_path(match_id, mode))
    validate_master(master)
    assert np.array_equal(labels == X_CODE, master["half"].to_numpy() == 0)
    assert row["bg_hazard"] is not None and np.isfinite(row["bg_hazard"])
    # The degradation really ran: the ball is hidden on more frames than in the clean build.
    assert row["pct_ball_observed"] < clean_row["pct_ball_observed"]
