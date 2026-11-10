"""Synthetic tests for IDSSE label statistics (task F2-03)."""
from __future__ import annotations

import numpy as np
import pytest
import yaml

from modeling.contracts import LABEL_CODES
from modeling.scoring.label_stats import (
    A4_LAM_DEAD,
    A4_LAM_TEAM,
    A4_MIN_DWELL,
    FPS,
    aggregate_halves,
    decoder_idsse_config,
    half_stats,
    segments_from_labels,
    write_all,
)


def _codes(text: str) -> np.ndarray:
    table = {"H": LABEL_CODES["H"], "A": LABEL_CODES["A"], "D": LABEL_CODES["D"],
             "X": LABEL_CODES["X"]}
    return np.asarray([table[c] for c in text.replace(" ", "")], dtype=np.int8)


def test_segments_ignore_x_and_never_count_x_as_segment():
    # half 1 over the whole array; X breaks runs and is not a segment.
    labels = _codes("HHHXHH DDD AA")
    half = np.ones(len(labels), dtype=np.int8)
    segs = segments_from_labels(labels, half, 1)
    assert [(s.state, s.length) for s in segs] == [
        ("H", 3), ("H", 2), ("D", 3), ("A", 2),
    ]
    assert all(s.state != "X" for s in segs)


def test_x_outside_half_is_irrelevant():
    labels = _codes("HHHXXXAAA")
    half = np.array([1, 1, 1, 0, 0, 0, 2, 2, 2], dtype=np.int8)
    assert [(s.state, s.length) for s in segments_from_labels(labels, half, 1)] == [("H", 3)]
    assert [(s.state, s.length) for s in segments_from_labels(labels, half, 2)] == [("A", 3)]
    assert segments_from_labels(labels, half, 0) == []


def test_known_sequence_durations_and_exit_split():
    # Half 1: HHH AA DDDD H  → lengths 3,2,4,1 at 25 fps.
    # Transitions: H→A, A→D, D→H.
    labels = _codes("HHHAADDDDH")
    half = np.ones(len(labels), dtype=np.int8)
    stats = half_stats(labels, half, 1, match_id="SYN", fps=FPS)
    assert [ (s.state, s.length) for s in stats.segments ] == [
        ("H", 3), ("A", 2), ("D", 4), ("H", 1),
    ]
    np.testing.assert_allclose(stats.mean_duration_s["H"], (3 + 1) / 2 / FPS)
    np.testing.assert_allclose(stats.mean_duration_s["A"], 2 / FPS)
    np.testing.assert_allclose(stats.mean_duration_s["D"], 4 / FPS)
    np.testing.assert_allclose(stats.mean_duration_s["HA"], (3 + 2 + 1) / 3 / FPS)

    # One half only → aggregate equals this half's transitions.
    agg = aggregate_halves([stats], fps=FPS)
    assert agg.exit_split["H"] == {"A": 1.0, "D": 0.0}
    assert agg.exit_split["A"] == {"H": 0.0, "D": 1.0}
    assert agg.exit_split["D"] == {"H": 1.0, "A": 0.0}
    np.testing.assert_allclose(agg.mean_duration_s["H"], (3 + 1) / 2 / FPS)
    np.testing.assert_allclose(agg.mean_duration_s["D"], 4 / FPS)

    # Possessions = H and A segments: 3 of them over 10/25/60 minutes.
    minutes = 10 / (FPS * 60.0)
    np.testing.assert_allclose(stats.possessions_per_min, 3 / minutes)
    np.testing.assert_allclose(stats.mean_possession_duration_s, (3 + 2 + 1) / 3 / FPS)
    np.testing.assert_allclose(stats.d_fraction, 4 / 10)


def test_exit_split_row_normalisation_with_two_exits():
    # H→A, H→A, H→D  ⇒ A: 2/3, D: 1/3.
    labels = _codes("HA HA HD")
    # Build explicitly without spaces: HAHAHD
    labels = _codes("HAHAHD")
    half = np.ones(len(labels), dtype=np.int8)
    # Segments: H,A,H,A,H,D — transitions H→A, A→H, H→A, A→H, H→D
    agg = aggregate_halves([half_stats(labels, half, 1, fps=FPS)], fps=FPS)
    np.testing.assert_allclose(agg.exit_split["H"]["A"], 2.0 / 3.0)
    np.testing.assert_allclose(agg.exit_split["H"]["D"], 1.0 / 3.0)
    np.testing.assert_allclose(agg.exit_split["A"]["H"], 1.0)
    np.testing.assert_allclose(agg.exit_split["A"]["D"], 0.0)


def test_plausibility_min_max_over_halves():
    # Two halves with different possession rates.
    h1 = half_stats(_codes("HHHHAAAA"), np.ones(8, dtype=np.int8), 1, fps=FPS)
    labels2 = _codes("HDAHDAHDAHDA")
    h2 = half_stats(labels2, np.full(len(labels2), 2, dtype=np.int8), 2, fps=FPS)
    agg = aggregate_halves([h1, h2], fps=FPS)
    ppm = [h1.possessions_per_min, h2.possessions_per_min]
    assert agg.plausibility["possessions_per_min"]["min"] == min(ppm)
    assert agg.plausibility["possessions_per_min"]["max"] == max(ppm)
    dfrac = [h1.d_fraction, h2.d_fraction]
    assert agg.plausibility["d_fraction"]["min"] == min(dfrac)
    assert agg.plausibility["d_fraction"]["max"] == max(dfrac)


def test_decoder_idsse_keeps_appendix_a4_knobs(tmp_path):
    labels = _codes("HHHAADDD")
    half = np.ones(len(labels), dtype=np.int8)
    agg = aggregate_halves([half_stats(labels, half, 1, match_id="S", fps=FPS)], fps=FPS)
    dec = decoder_idsse_config(agg)
    assert dec["lam_team"] == A4_LAM_TEAM
    assert dec["lam_dead"] == A4_LAM_DEAD
    assert dec["min_dwell_frames"] == A4_MIN_DWELL
    written = write_all(agg, root=tmp_path)
    loaded = yaml.safe_load(written["decoder_idsse"].read_text(encoding="utf-8"))
    assert loaded["decoder"]["lam_team"] == A4_LAM_TEAM
    assert "mean_duration_s" in loaded["decoder"]
    plaus = yaml.safe_load(written["plausibility"].read_text(encoding="utf-8"))
    assert set(plaus) == {
        "possessions_per_min", "mean_possession_duration_s", "d_fraction",
    }
    assert written["report"].is_file()


def test_empty_half_returns_empty_segments():
    labels = _codes("HHH")
    half = np.ones(3, dtype=np.int8)
    assert segments_from_labels(labels, half, 2) == []


def test_labels_and_half_from_xml_map_inserts_gap():
    from modeling.external.files import GAP_FRAMES
    from modeling.scoring.label_stats import labels_and_half_from_xml_map

    xml = {}
    for fid in range(1, 6):
        xml[(1, fid)] = LABEL_CODES["H"]
    for fid in range(10, 14):
        xml[(2, fid)] = LABEL_CODES["A"]
    labels, half = labels_and_half_from_xml_map(xml)
    assert labels.shape == half.shape
    assert int((half == 1).sum()) == 5
    assert int((half == 2).sum()) == 4
    assert int((half == 0).sum()) == GAP_FRAMES
    assert (labels[half == 0] == LABEL_CODES["X"]).all()
    assert (labels[half == 1] == LABEL_CODES["H"]).all()
    assert (labels[half == 2] == LABEL_CODES["A"]).all()
