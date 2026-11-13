"""Synthetic tests for per-half possession post-processing (task F2-04)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from modeling.contracts import LABEL_DTYPE, validate_labels
from modeling.decode.postprocess import BOUNDARY_TYPES, postprocess
from modeling.pipeline import POSTPROCESS_RULES, final_imposition


def _cfg(**overrides) -> dict:
    base = {
        "midpoint_neutral_switch": False,
        "boundary_shift_file": None,
        "d_bridge_frames": 0,
        "d_cross_frames": 0,
        "min_spell_frames": 0,
    }
    base.update(overrides)
    return base


def _master(half, *, is_set_piece=None) -> pd.DataFrame:
    half = np.asarray(half, dtype=np.int8)
    data = {"half": half}
    if is_set_piece is None:
        data["is_set_piece"] = np.zeros(half.shape[0], dtype=bool)
    else:
        data["is_set_piece"] = np.asarray(is_set_piece, dtype=bool)
    return pd.DataFrame(data)


def _labels(s: str) -> np.ndarray:
    return np.array(list(s), dtype=LABEL_DTYPE)


def _zeros(n: int, dtype=np.float64):
    return np.zeros(n, dtype=dtype)


def _run(labels, half, cfg, *, team_logit=None, dead_logit=None, force_d=None,
         is_set_piece=None, holdout=None):
    labels = np.asarray(labels, dtype=LABEL_DTYPE)
    n = labels.shape[0]
    half = np.asarray(half, dtype=np.int8)
    if team_logit is None:
        team_logit = _zeros(n)
    if dead_logit is None:
        dead_logit = _zeros(n)
    if force_d is None:
        force_d = np.zeros(n, dtype=bool)
    master = _master(half, is_set_piece=is_set_piece)
    return postprocess(labels, master, team_logit, dead_logit, force_d, cfg, holdout=holdout)


# -- all rules off: identity + final imposition -------------------------------------------------

def test_all_rules_off_is_identity_plus_final():
    half = np.array([0, 1, 1, 1, 0, 2, 2, 2], np.int8)
    force = np.array([0, 0, 1, 0, 0, 0, 0, 0], bool)
    labels = _labels("XHHDXHAA")
    out, stats = _run(labels, half, _cfg(), force_d=force)
    expected, n_final = final_imposition(labels, half, force)
    np.testing.assert_array_equal(out, expected)
    assert stats["final_imposition"] == n_final
    for key in POSTPROCESS_RULES:
        assert stats[key] == 0
    validate_labels(out, half)


# -- d_bridge: H D H / A D A --------------------------------------------------------------------

def test_d_bridge_collapses_short_d_between_same_team():
    half = np.ones(7, np.int8)
    labels = _labels("HHDHHHH")  # wait - need H D H pattern
    labels = _labels("HHDDHHH")
    # Short D (2) between H: with max_len=3 → collapse; with max_len=2 → keep
    out, stats = _run(labels, half, _cfg(d_bridge_frames=3))
    assert "".join(out) == "HHHHHHH"
    assert stats["d_bridge_frames"] == 2

    out2, stats2 = _run(labels, half, _cfg(d_bridge_frames=2))
    assert "".join(out2) == "HHDDHHH"
    assert stats2["d_bridge_frames"] == 0


def test_d_bridge_a_d_a_and_skips_force_d():
    half = np.ones(7, np.int8)
    labels = _labels("AADDAAA")
    out, _ = _run(labels, half, _cfg(d_bridge_frames=5))
    assert "".join(out) == "AAAAAAA"

    force = np.array([0, 0, 1, 0, 0, 0, 0], bool)
    out_f, stats = _run(labels, half, _cfg(d_bridge_frames=5), force_d=force)
    # force_d frame becomes D in final imposition; bridge skipped because force touches D seg
    assert out_f[2] == "D"
    assert stats["d_bridge_frames"] == 0


def test_d_bridge_does_not_join_different_teams():
    half = np.ones(7, np.int8)
    labels = _labels("HHDDAAA")
    out, stats = _run(labels, half, _cfg(d_bridge_frames=5))
    assert "".join(out) == "HHDDAAA"
    assert stats["d_bridge_frames"] == 0


# -- min_spell: H A H ---------------------------------------------------------------------------

def test_min_spell_absorbs_short_ha_flanked_by_same_label():
    half = np.ones(9, np.int8)
    labels = _labels("HHHAHHHAA")  # short A (1) between H → absorb; trailing AA kept
    out, stats = _run(labels, half, _cfg(min_spell_frames=2))
    assert "".join(out) == "HHHHHHHAA"
    assert stats["min_spell_frames"] == 1


def test_min_spell_left_to_right_cascades():
    # H A H A H with min_spell=2: first short A absorbed → HHH A H → then second A absorbed
    half = np.ones(5, np.int8)
    labels = _labels("HAHAH")
    out, stats = _run(labels, half, _cfg(min_spell_frames=2))
    assert "".join(out) == "HHHHH"
    assert stats["min_spell_frames"] == 2


# -- midpoint_neutral_switch --------------------------------------------------------------------

def test_midpoint_moves_boundary_inside_neutral_run():
    # HHHHAAAA, boundary at 4; zeros on [2, 5] → mid = 2 + 4//2 = 4 (unchanged)
    # zeros on [1, 4] → mid = 1 + 4//2 = 3 → HHH AAAAA
    half = np.ones(8, np.int8)
    labels = _labels("HHHHAAAA")
    logit = np.array([1, 0, 0, 0, 0, 1, 1, 1], np.float64)
    out, stats = _run(labels, half, _cfg(midpoint_neutral_switch=True), team_logit=logit)
    assert "".join(out) == "HHHAAAAA"
    assert stats["midpoint_neutral_switch"] > 0


def test_midpoint_no_change_when_evidence_at_boundary():
    half = np.ones(8, np.int8)
    labels = _labels("HHHHAAAA")
    logit = np.array([1, 1, 1, 1, 1, 1, 1, 1], np.float64)
    out, stats = _run(labels, half, _cfg(midpoint_neutral_switch=True), team_logit=logit)
    assert "".join(out) == "HHHHAAAA"
    assert stats["midpoint_neutral_switch"] == 0


def test_midpoint_on_edge_of_zero_run():
    # Boundary at 4; zero run ends at 3 (left edge) → mid of [1,3] = 1+3//2 = 2
    half = np.ones(8, np.int8)
    labels = _labels("HHHHAAAA")
    logit = np.array([1, 0, 0, 0, 1, 1, 1, 1], np.float64)
    out, stats = _run(labels, half, _cfg(midpoint_neutral_switch=True), team_logit=logit)
    assert "".join(out) == "HHAAAAAA"
    assert stats["midpoint_neutral_switch"] > 0


# -- boundary_shift_file ------------------------------------------------------------------------

def test_boundary_shift_by_type_and_fold(tmp_path):
    half = np.ones(10, np.int8)
    labels = _labels("HHHHHAAAAA")  # H>A at 5
    path = tmp_path / "boundary.yaml"
    path.write_text(yaml.safe_dump({
        "all": {t: 0 for t in BOUNDARY_TYPES} | {"H>A": 2},
        "folds": {
            "S1": {t: 0 for t in BOUNDARY_TYPES} | {"H>A": 1},
        },
    }), encoding="utf-8")
    # median 2 → shift -2 → boundary at 3 → HHH AAAAAAA
    out, stats = _run(labels, half, _cfg(boundary_shift_file=str(path)))
    assert "".join(out) == "HHHAAAAAAA"
    assert stats["boundary_shift_file"] == 2

    out_f, _ = _run(labels, half, _cfg(boundary_shift_file=str(path)), holdout="S1")
    assert "".join(out_f) == "HHHHAAAAAA"


def test_boundary_shift_clamped_to_neighbours(tmp_path):
    half = np.ones(6, np.int8)
    labels = _labels("HHAADD")  # H>A at 2; A length 2
    path = tmp_path / "boundary.yaml"
    path.write_text(yaml.safe_dump({
        "all": {t: 0 for t in BOUNDARY_TYPES} | {"H>A": 10},  # would push far left
    }), encoding="utf-8")
    out, _ = _run(labels, half, _cfg(boundary_shift_file=str(path)))
    # left H keeps ≥1 frame → boundary clamped to 1 → H AA DD
    assert "".join(out) == "HAAADD"


def test_boundary_shift_skips_force_d_neighbours(tmp_path):
    half = np.ones(8, np.int8)
    labels = _labels("HHHHAAAA")
    force = np.zeros(8, bool)
    force[4] = True  # right side of H>A boundary
    path = tmp_path / "boundary.yaml"
    path.write_text(yaml.safe_dump({
        "all": {t: 0 for t in BOUNDARY_TYPES} | {"H>A": 2},
    }), encoding="utf-8")
    out, stats = _run(labels, half, _cfg(boundary_shift_file=str(path)), force_d=force)
    # shift skipped; final imposition paints force frame as D
    assert stats["boundary_shift_file"] == 0
    assert out[4] == "D"


def test_boundary_shift_then_final_imposition_valid(tmp_path):
    half = np.array([0, 1, 1, 1, 1, 0], np.int8)
    labels = _labels("XHHHAX")
    path = tmp_path / "boundary.yaml"
    path.write_text(yaml.safe_dump({
        "all": {t: 0 for t in BOUNDARY_TYPES} | {"H>A": 1},
    }), encoding="utf-8")
    out, stats = _run(labels, half, _cfg(boundary_shift_file=str(path)))
    validate_labels(out, half)
    assert out[0] == "X" and out[-1] == "X"
    assert stats["final_imposition"] >= 0


# -- d_cross_frames -----------------------------------------------------------------------------

def test_d_cross_splits_short_d_between_different_teams():
    half = np.ones(9, np.int8)
    # H H D D D A A A A  length 3 D; odd → mid to prev: H H H H D→ wait
    # length 3: mid = a + 2 → [a:a+2]=H, [a+2:b]=A → HH HH A AAAA = HHHHAAAAA? indices:
    # 0:H 1:H 2:D 3:D 4:D 5:A 6:A 7:A 8:A
    # mid = 2 + 2 = 4 → [2:4]=H, [4:5]=A → H H H H A A A A A
    labels = _labels("HHDDDAAAA")
    out, stats = _run(labels, half, _cfg(d_cross_frames=4))
    assert "".join(out) == "HHHHAAAAA"
    assert stats["d_cross_frames"] == 3


def test_d_cross_even_length():
    half = np.ones(8, np.int8)
    labels = _labels("HHDDAAAA")  # D len 2; mid = 2+1 = 3 → HH H A AAAA
    out, _ = _run(labels, half, _cfg(d_cross_frames=3))
    assert "".join(out) == "HHHAAAAA"


def test_d_cross_skips_force_d_set_piece_and_same_team():
    half = np.ones(9, np.int8)
    labels = _labels("HHDDDAAAA")

    force = np.zeros(9, bool)
    force[3] = True
    out_f, s_f = _run(labels, half, _cfg(d_cross_frames=4), force_d=force)
    assert s_f["d_cross_frames"] == 0
    assert out_f[3] == "D"

    sp = np.zeros(9, bool)
    sp[2] = True
    out_sp, s_sp = _run(labels, half, _cfg(d_cross_frames=4), is_set_piece=sp)
    assert s_sp["d_cross_frames"] == 0
    assert "".join(out_sp) == "HHDDDAAAA"

    same = _labels("HHDDDHHHH")
    out_same, s_same = _run(same, half, _cfg(d_cross_frames=4))
    assert s_same["d_cross_frames"] == 0
    # d_bridge is off, so D remains
    assert "".join(out_same) == "HHDDDHHHH"


# -- idempotency of rules 1, 3, 4 ----------------------------------------------------------------

@pytest.mark.parametrize("cfg_kw,builder", [
    ({"midpoint_neutral_switch": True}, "mid"),
    ({"d_bridge_frames": 5}, "bridge"),
    ({"min_spell_frames": 2}, "spell"),
])
def test_rules_1_3_4_are_idempotent(cfg_kw, builder):
    half = np.ones(12, np.int8)
    if builder == "mid":
        labels = _labels("HHHHHHAAAAAA")
        logit = np.array([1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], np.float64)
        kwargs = {"team_logit": logit}
    elif builder == "bridge":
        labels = _labels("HHHDDDHHHHHH")
        kwargs = {}
    else:
        labels = _labels("HHHAHHHAAAAA")
        kwargs = {}
    cfg = _cfg(**cfg_kw)
    out1, _ = _run(labels, half, cfg, **kwargs)
    out2, stats2 = _run(out1, half, cfg, **kwargs)
    np.testing.assert_array_equal(out1, out2)
    rule = next(iter(cfg_kw))
    assert stats2[rule] == 0


def test_per_half_isolation():
    # D-bridge in half 1 must not look across the gap into half 2
    half = np.array([1, 1, 1, 0, 0, 2, 2, 2], np.int8)
    labels = _labels("HDHXXHDH")
    out, stats = _run(labels, half, _cfg(d_bridge_frames=2))
    assert "".join(out) == "HHHXXHHH"
    assert stats["d_bridge_frames"] == 2
    assert (out[half == 0] == "X").all()


def test_stats_keys_match_pipeline_rules():
    half = np.ones(4, np.int8)
    _, stats = _run(_labels("HHHH"), half, _cfg())
    assert set(stats) == set(POSTPROCESS_RULES) | {"final_imposition"}


# -- regressions: consecutive boundaries and many short spells (review of #122) ---------------

def _runs(out) -> list[tuple[str, int]]:
    s = "".join(out)
    runs, i = [], 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        runs.append((s[i], j - i))
        i = j
    return runs


def test_midpoint_moves_consecutive_boundaries():
    # H|A|H, both boundaries inside their own neutral run → both go to the midpoints 23 and 43
    labels = _labels("H" * 20 + "A" * 20 + "H" * 20)
    logit = np.ones(60)
    logit[18:28] = 0.0
    logit[38:48] = 0.0
    cfg = _cfg(midpoint_neutral_switch=True)
    out, _ = _run(labels, np.ones(60, np.int8), cfg, team_logit=logit)
    assert _runs(out) == [("H", 23), ("A", 20), ("H", 17)]
    out2, stats2 = _run(out, np.ones(60, np.int8), cfg, team_logit=logit)
    np.testing.assert_array_equal(out, out2)
    assert stats2["midpoint_neutral_switch"] == 0


def test_boundary_shift_moves_consecutive_boundaries(tmp_path):
    # H|A|D with median +3 on every type → both boundaries move 3 frames earlier
    path = tmp_path / "boundary.yaml"
    path.write_text(yaml.safe_dump({"all": {t: 3 for t in BOUNDARY_TYPES}}), encoding="utf-8")
    labels = _labels("H" * 20 + "A" * 20 + "D" * 20)
    out, stats = _run(labels, np.ones(60, np.int8), _cfg(boundary_shift_file=str(path)))
    assert _runs(out) == [("H", 17), ("A", 20), ("D", 23)]
    assert stats["boundary_shift_file"] == 6


def test_boundary_shift_neighbours_never_cross(tmp_path):
    # Opposite shifts squeeze the 2-frame A segment: it keeps exactly one frame
    path = tmp_path / "boundary.yaml"
    path.write_text(yaml.safe_dump({"all": {t: 0 for t in BOUNDARY_TYPES}
                                    | {"H>A": -5, "A>D": 5}}), encoding="utf-8")
    labels = _labels("HHHHHAADDDDD")
    out, _ = _run(labels, np.ones(12, np.int8), _cfg(boundary_shift_file=str(path)))
    assert [lab for lab, _n in _runs(out)] == ["H", "A", "D"]
    assert min(n for _lab, n in _runs(out)) >= 1


def test_boundary_shift_does_not_paint_over_force_d(tmp_path):
    # H>A shifted 4 frames later would cover a force_d frame at 6: it stops just before it
    path = tmp_path / "boundary.yaml"
    path.write_text(yaml.safe_dump({"all": {t: 0 for t in BOUNDARY_TYPES} | {"H>A": -4}}),
                    encoding="utf-8")
    force = np.zeros(12, bool)
    force[6] = True
    labels = _labels("HHHHAAAAAAAA")
    out, _ = _run(labels, np.ones(12, np.int8), _cfg(boundary_shift_file=str(path)),
                  force_d=force)
    assert "".join(out) == "HHHHHHDAAAAA"


def test_min_spell_absorbs_more_than_twenty_spells():
    # 30 short A blips inside H: one pass must absorb all of them (no per-pass cap)
    labels = _labels("".join("H" * 20 + "A" * 3 for _ in range(30)) + "H" * 20)
    half = np.ones(labels.shape[0], np.int8)
    cfg = _cfg(min_spell_frames=13)
    out, stats = _run(labels, half, cfg)
    assert set("".join(out)) == {"H"}
    assert stats["min_spell_frames"] == 90
    out2, stats2 = _run(out, half, cfg)
    np.testing.assert_array_equal(out, out2)
    assert stats2["min_spell_frames"] == 0
