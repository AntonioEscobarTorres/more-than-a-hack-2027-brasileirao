"""tools/deadball_diagnostics.py (task F3-06).

Synthetic data only: hand-built masters, the synthetic mock master and labels written to
tmp_path. Nothing reads the match data or a ground truth.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from modeling import io
from modeling.config import load_config
from modeling.contracts import LABEL_CODES
from modeling.scoring.label_stats import half_stats
from preprocessing.make_mock_master import generate_mock_master
from tests.modeling_fixtures import dead_params, deadball_master, with_gap
from tools import deadball_diagnostics as dd

REPO = Path(__file__).resolve().parent.parent
H, A, D, X = (LABEL_CODES[k] for k in "HADX")
FPS = 25.0
MINUTE = int(60 * FPS)                      # 1500 frames
P = dead_params()
# Bands that the standard half below sits well inside, before and after the slack.
BANDS = {"d_fraction": (0.3, 0.6), "possessions_per_min": (3.5, 5.7),
         "mean_possession_duration_s": (4.3, 9.7)}
MIDFIELD = (50.0, 34.0, 0.0, 0.0)


def master_for(half, **cols) -> pd.DataFrame:
    """Ball in midfield, players moving, no tags, no pre-kickoff: no dead-ball evidence."""
    half = np.asarray(half)
    pre = cols.pop("pre_kickoff", False)
    master = deadball_master(np.full(len(half), 50.0), 34.0, half=half, **cols)
    master["pre_kickoff"] = np.broadcast_to(np.asarray(pre, dtype=bool), (len(half),)).copy()
    return master


def pattern(*runs) -> np.ndarray:
    """Labels from ``(code, n_frames)`` runs."""
    return np.concatenate([np.full(n, code, dtype=np.int64) for code, n in runs])


def standard_half() -> np.ndarray:
    """One minute: H, D, A, D, H, D, A, D, H, D with 150 frames each.

    5 possessions per minute, mean possession 6 s, D fraction 0.5.
    """
    return pattern(*[(c, 150) for c in (H, D, A, D, H, D, A, D, H, D)])


def two_halves(first=None, second=None, gap=50):
    first = standard_half() if first is None else first
    second = standard_half() if second is None else second
    codes = np.r_[first, np.full(gap, X), second]
    half = np.r_[np.full(len(first), 1), np.full(gap, 0), np.full(len(second), 2)]
    return codes, half


def by_key(results):
    return {r.key: r for r in results}


def check(codes, master, bands=BANDS):
    return by_key(dd.run_checks(codes, master, bands, P, fps=FPS))


# -- bands --------------------------------------------------------------------------------------
def test_real_plausibility_file_loads():
    bands = dd.load_bands()
    assert set(bands) == {name for name, *_ in dd.BAND_CHECKS}
    assert all(0 <= low <= high for low, high in bands.values())


def test_widened_band_is_forty_percent_on_each_side():
    assert dd.BAND_SLACK == 0.4
    low, high = dd.widened((5.0, 10.0))
    assert low == pytest.approx(3.0) and high == pytest.approx(14.0)


@pytest.mark.parametrize("content, match", [
    ("[1, 2]", "mapping"),
    ("d_fraction: {min: 0.1, max: 0.5}", "possessions_per_min"),
    ("d_fraction: {min: 0.6, max: 0.5}\npossessions_per_min: {min: 1, max: 2}\n"
     "mean_possession_duration_s: {min: 1, max: 2}", "min <= max"),
    ("d_fraction: {min: .nan, max: 0.5}\npossessions_per_min: {min: 1, max: 2}\n"
     "mean_possession_duration_s: {min: 1, max: 2}", "min <= max"),
    ("d_fraction: {min: x, max: 0.5}\npossessions_per_min: {min: 1, max: 2}\n"
     "mean_possession_duration_s: {min: 1, max: 2}", "numbers"),
    ("d_fraction: {min: 0.1, max: 0.5, mean: 0.3}\npossessions_per_min: {min: 1, max: 2}\n"
     "mean_possession_duration_s: {min: 1, max: 2}", "exactly the keys"),
])
def test_bad_plausibility_file_is_rejected(tmp_path, content, match):
    path = tmp_path / "plausibility.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        dd.load_bands(path)


# -- (i)-(iii): eliminatory band checks ---------------------------------------------------------
def test_statistics_are_the_ones_of_the_idsse_bands():
    codes, half = two_halves()
    s = half_stats(codes, half, 1, fps=FPS)
    assert s.possessions_per_min == pytest.approx(5.0)
    assert s.mean_possession_duration_s == pytest.approx(6.0)
    assert s.d_fraction == pytest.approx(0.5)


def test_plausible_prediction_passes_every_check():
    codes, half = two_halves()
    results = check(codes, master_for(half))
    assert all(results[k].passed for k in ("i", "ii", "iii"))
    assert all(results[k].eliminatory for k in ("i", "ii", "iii"))
    assert not results["iv"].eliminatory and not results["v"].eliminatory


@pytest.mark.parametrize("name, key, inside, outside", [
    # widened low = low * 0.6: 0.5 is inside (0.8, 1) but outside (0.85, 1).
    ("d_fraction", "i", (0.8, 1.0), (0.85, 1.0)),
    # widened high = high * 1.4: 5 is inside (1, 3.6) but outside (1, 3.5).
    ("possessions_per_min", "ii", (1.0, 3.6), (1.0, 3.5)),
    ("mean_possession_duration_s", "iii", (1.0, 4.3), (1.0, 4.2)),
])
def test_each_band_uses_the_slack(name, key, inside, outside):
    codes, half = two_halves()
    master = master_for(half)
    assert check(codes, master, {**BANDS, name: inside})[key].passed
    failed = check(codes, master, {**BANDS, name: outside})
    assert not failed[key].passed
    assert all(failed[k].passed for k in ("i", "ii", "iii") if k != key)


def test_one_implausible_half_is_enough_to_fail():
    all_dead = pattern((D, MINUTE))                      # no possession at all in half 2
    codes, half = two_halves(second=all_dead)
    results = check(codes, master_for(half))
    assert not results["i"].passed                       # D fraction 1.0
    assert not results["ii"].passed                      # 0 possessions per minute
    assert not results["iii"].passed                     # mean duration NaN
    assert "half 1: 0.5000 (in)" in results["i"].detail
    assert "half 2: 1.0000 (OUT)" in results["i"].detail


def test_flickering_prediction_fails_possession_checks():
    # 1 s possessions: 30 per minute, mean 1 s; D fraction stays 0.5.
    flicker = pattern(*[(c, 25) for c in (H, D, A, D) * 15])
    codes, half = two_halves(first=flicker, second=flicker)
    results = check(codes, master_for(half))
    assert results["i"].passed
    assert not results["ii"].passed and not results["iii"].passed


def test_d_splits_a_possession_and_x_is_not_counted():
    # H D H = two possessions; frames outside the halves (X) never count.
    first = pattern((H, 600), (D, 300), (H, 600))
    codes, half = two_halves(first=first, second=first, gap=500)
    s = half_stats(codes, half, 1, fps=FPS)
    assert s.possessions_per_min == pytest.approx(2.0) and s.d_fraction == pytest.approx(0.2)


# -- (iv): out of the pitch predicted alive -----------------------------------------------------
def test_out_of_pitch_predicted_alive_is_reported():
    codes, half = two_halves()
    out = np.zeros(len(codes))
    out[:300] = 1.0                                      # frames 0-149 H, 150-299 D
    results = check(codes, master_for(half, out_of_pitch=out))
    assert not results["iv"].passed and "150 of 300" in results["iv"].detail
    out_dead = np.zeros(len(codes))
    out_dead[150:300] = 1.0                              # only during the D run
    assert check(codes, master_for(half, out_of_pitch=out_dead))["iv"].passed


def test_out_of_pitch_threshold_is_strict():
    codes, half = two_halves()
    out = np.zeros(len(codes))
    out[100:300] = 1.0                                   # 50 alive (H) of 200: exactly 0.25
    assert not check(codes, master_for(half, out_of_pitch=out))["iv"].passed
    out[100:110] = 0.0                                   # 40 of 190 = 0.21
    assert not check(codes, master_for(half, out_of_pitch=out))["iv"].passed
    out[100:125] = 0.0                                   # 25 of 175 = 0.143
    assert check(codes, master_for(half, out_of_pitch=out))["iv"].passed
    exact = np.zeros(len(codes))
    exact[120:270] = 1.0                                 # 30 alive (H) of 150: exactly 0.2
    assert not check(codes, master_for(half, out_of_pitch=exact))["iv"].passed


def test_out_of_pitch_outside_the_halves_and_nan_are_ignored():
    codes, half = two_halves()
    out = np.where(half == 0, 1.0, np.nan)
    results = check(codes, master_for(half, out_of_pitch=out))
    assert results["iv"].passed and "0 of 0" in results["iv"].detail


# -- (v): D segments without evidence -----------------------------------------------------------
def test_d_segments_without_any_evidence_are_reported():
    codes, half = two_halves()
    results = check(codes, master_for(half))
    assert not results["v"].passed and "10 of 10" in results["v"].detail


def evidence_count(codes, master):
    detail = check(codes, master)["v"].detail
    return int(detail.split(" of ")[0])                  # D segments without evidence


@pytest.mark.parametrize("cols", [
    {"out_of_pitch": 1.0},
    {"n_still_home": 10.0, "n_still_away": 10.0},        # still fraction 1.0 > 0.3
    {"mean_speed_home": 0.5, "mean_speed_away": 0.5},    # slow players < 1.5 m/s
    {"is_set_piece": True},
    {"pre_kickoff": True},
])
def test_each_kind_of_evidence_counts(cols):
    codes, half = two_halves()
    assert evidence_count(codes, master_for(half)) == 10
    assert evidence_count(codes, master_for(half, **cols)) == 0


def test_exit_gap_is_evidence_and_a_high_ball_is_not():
    exit_gap = with_gap((50.0, 1.0, 0.0, -4.0), MIDFIELD, gap=20, lead=40, trail=40)
    high_ball = with_gap((40.0, 34.0, 20.0, 0.0), MIDFIELD, gap=20, lead=40, trail=40)
    codes = pattern((H, 40), (D, 20), (A, 40))
    for master, expected in ((exit_gap, 0), (high_ball, 1)):
        master["pre_kickoff"] = False
        bands = {k: (0.0, 1e9) for k in BANDS}
        detail = by_key(dd.run_checks(codes, master, bands, P, fps=FPS))["v"].detail
        assert detail.startswith("{} of 1 ".format(expected))


def test_one_frame_of_evidence_is_enough_for_a_segment():
    codes, half = two_halves()
    out = np.zeros(len(codes))
    out[[150, 450, 750]] = 1.0                           # one frame in three D segments
    assert evidence_count(codes, master_for(half, out_of_pitch=out)) == 7


def test_d_segment_threshold_is_strict():
    codes, half = two_halves()                           # 10 D segments, 5 per half
    starts = np.flatnonzero((codes == D) & (np.r_[-1, codes[:-1]] != D))

    def master_with_evidence(n_segments_with_evidence):
        out = np.zeros(len(codes))
        out[starts[:n_segments_with_evidence]] = 1.0
        return master_for(half, out_of_pitch=out)

    assert not check(codes, master_with_evidence(7))["v"].passed   # 3 of 10 = 0.3
    assert check(codes, master_with_evidence(8))["v"].passed       # 2 of 10 = 0.2


def test_no_d_segment_passes():
    codes, half = two_halves(first=pattern((H, MINUTE)), second=pattern((A, MINUTE)))
    results = check(codes, master_for(half))
    assert results["v"].passed and "0 of 0" in results["v"].detail


def test_missing_set_piece_value_is_no_evidence():
    codes, half = two_halves()
    master = master_for(half, is_set_piece=True)
    master["is_set_piece"] = pd.array([pd.NA] * len(master), dtype="boolean")
    assert evidence_count(codes, master) == 10


# -- overall result and report ------------------------------------------------------------------
def test_informative_checks_never_change_the_overall_result():
    codes, half = two_halves()
    results = dd.run_checks(codes, master_for(half, out_of_pitch=1.0), BANDS, P, fps=FPS)
    flags = by_key(results)
    assert not flags["iv"].passed                        # every out frame alive on H/A runs
    assert flags["v"].passed
    assert dd.overall_passed(results)
    no_evidence = dd.run_checks(codes, master_for(half), BANDS, P, fps=FPS)
    assert not by_key(no_evidence)["v"].passed and dd.overall_passed(no_evidence)


def test_report_prints_numbers_only_when_verbose():
    codes, half = two_halves()
    results = dd.run_checks(codes, master_for(half), {**BANDS, "d_fraction": (0.85, 1.0)}, P,
                            fps=FPS)
    quiet = dd.report(results)
    assert quiet == [
        "FAIL  (i) D fraction of each half inside the IDSSE band",
        "PASS  (ii) possessions per minute of each half inside the IDSSE band",
        "PASS  (iii) mean possession duration of each half inside the IDSSE band",
        "PASS  (iv) ball out of the pitch is predicted as D [informative]",
        "FAIL  (v) predicted D segments have dead-ball evidence [informative]",
        "FAIL (1 of 3 eliminatory checks)",
    ]
    assert not any(ch.isdigit() for line in quiet for ch in line.split(")", 1)[1])
    loud = dd.report(results, verbose=True)
    assert len(loud) > len(quiet) and any("half 1: 0.5000" in line for line in loud)


# -- inputs -------------------------------------------------------------------------------------
@pytest.mark.parametrize("column", dd.MASTER_COLUMNS)
def test_missing_master_column_is_an_error(column):
    codes, half = two_halves()
    with pytest.raises(ValueError, match=column):
        dd.run_checks(codes, master_for(half).drop(columns=[column]), BANDS, P, fps=FPS)


def test_length_mismatch_and_no_half_are_errors():
    codes, half = two_halves()
    with pytest.raises(ValueError, match="frames"):
        dd.run_checks(codes[:-1], master_for(half), BANDS, P, fps=FPS)
    with pytest.raises(ValueError, match="no frames inside a half"):
        dd.run_checks(np.full(10, X), master_for(np.zeros(10, dtype=int)), BANDS, P, fps=FPS)


def test_inputs_are_not_modified():
    codes, half = two_halves()
    master = master_for(half)
    before_master, before_codes = master.copy(deep=True), codes.copy()
    dd.run_checks(codes, master, BANDS, P, fps=FPS)
    pd.testing.assert_frame_equal(master, before_master)
    assert np.array_equal(codes, before_codes)


# -- command line, on the synthetic mock master ----------------------------------------------------
def mock_labels(master: pd.DataFrame) -> np.ndarray:
    """Plausible labels: pre-kickoff D, then the standard one-minute pattern in each half."""
    half = master["half"].to_numpy()
    labels = np.full(len(master), "X", dtype="<U1")
    unit = np.array(list("HDADHDADHD")).repeat(150)
    for h in (1, 2):
        idx = np.flatnonzero(half == h)
        labels[idx] = np.resize(unit, idx.size)
    labels[master["pre_kickoff"].to_numpy()] = "D"
    return labels


@pytest.fixture(scope="module")
def mock_files(tmp_path_factory):
    root = tmp_path_factory.mktemp("f306")
    master = generate_mock_master(seed=42)
    master_path = root / "master.parquet"
    master.to_parquet(master_path)
    good = io.write_submission(mock_labels(master), root / "good.csv")
    flicker_labels = mock_labels(master)
    graded = master["half"].to_numpy() != 0
    flicker_labels[graded] = np.resize(np.array(list("HA")).repeat(5), int(graded.sum()))
    flicker = io.write_submission(flicker_labels, root / "flicker.csv")
    return {"master": master_path, "good": good, "flicker": flicker, "root": root}


def test_mock_master_with_plausible_labels_passes(mock_files, capsys):
    args = ["--pred", str(mock_files["good"]), "--master", str(mock_files["master"])]
    assert dd.main(args) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[:3] == ["PASS  (i) D fraction of each half inside the IDSSE band",
                       "PASS  (ii) possessions per minute of each half inside the IDSSE band",
                       "PASS  (iii) mean possession duration of each half inside the IDSSE band"]
    assert out[-1] == "PASS" and len(out) == 6


def test_flickering_labels_fail_and_verbose_adds_numbers(mock_files, capsys):
    args = ["--pred", str(mock_files["flicker"]), "--master", str(mock_files["master"])]
    assert dd.main(args) == 1
    quiet = capsys.readouterr().out.splitlines()
    assert quiet[-1].startswith("FAIL (")
    assert dd.main(args + ["--verbose"]) == 1
    assert "half 1:" in capsys.readouterr().out


def test_unusable_inputs_give_exit_code_2(mock_files, capsys, tmp_path):
    base = ["--master", str(mock_files["master"])]
    assert dd.main(["--pred", str(tmp_path / "nope.csv")] + base) == 2
    assert capsys.readouterr().out.startswith("ERROR  could not load inputs")

    short = tmp_path / "short.csv"
    short.write_text("frame,label\n0,X\n", encoding="utf-8")
    assert dd.main(["--pred", str(short)] + base) == 2
    assert "invalid submission" in capsys.readouterr().out

    good = ["--pred", str(mock_files["good"])]
    assert dd.main(good + ["--master", str(tmp_path / "nope.parquet")]) == 2
    capsys.readouterr()
    bad_bands = tmp_path / "bands.yaml"
    bad_bands.write_text(yaml.safe_dump({"d_fraction": {"min": 0.1, "max": 0.5}}), encoding="utf-8")
    assert dd.main(good + base + ["--plausibility", str(bad_bands)]) == 2
    capsys.readouterr()
    assert dd.main(good + base + ["--config", str(REPO / "configs" / "baseline.yaml")]) == 2
    assert "no dead section" in capsys.readouterr().out


def test_default_config_has_the_dead_params_of_appendix_a():
    cfg = load_config(dd.DEFAULT_CONFIG)
    assert cfg["dead"]["params"] == {**P}


def test_script_runs_from_the_command_line(mock_files):
    done = subprocess.run(
        [sys.executable, str(REPO / "tools" / "deadball_diagnostics.py"),
         "--pred", str(mock_files["good"]), "--master", str(mock_files["master"])],
        capture_output=True, text=True, cwd=str(mock_files["root"]))
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines()[-1] == "PASS"
