"""Synthetic tests for the semi-Markov (negative binomial) durations of the decoder (F2-08, D-027).

Nothing here reads the match data or the external IDSSE files: scores and segment lengths are
random or built by hand, the master is ``preprocessing.make_mock_master`` and configs are
validated in memory.
"""
from __future__ import annotations

import copy
import hashlib
import math

import numpy as np
import pytest
import yaml

from modeling import paths
from modeling.calibration import durations
from modeling.config import ConfigError, load_config, validate_config
from modeling.contracts import STATES
from modeling.decode import viterbi
from modeling.decode.params import DecoderParams, _p_stay_from_mean
from modeling.decode.viterbi import decode_match
from modeling.pipeline import predict_labels
from preprocessing.make_mock_master import generate_mock_master
from tests.test_modeling_contracts import HMM_CFG

REPO = paths.repo_root()
# sha256 of predict_labels(generate_mock_master(seed=42), configs/<name>.yaml).tobytes(), computed
# on main at aa55168, before F2-08 (numpy 2.0.2, the pinned version). The v1 value equals the one
# recorded by F2-07 on 185819a.
MOCK_LABELS_SHA256 = {
    "v1": "249da6b1f87e08f403b857f5823d58211c1dded7dc4e13f735c7868be9441522",
    "v1s": "d7c5a5ada37c6515ed5814c1a4709d7afeb59ad5097f77186066db0844015510",
    "v2_gbdt": "3814be2763c94fb9907310bb2d9bbe3be66a9f5015b518ec59a7fa02ed818fc9",
}
CODE = {name: i for i, name in enumerate(STATES)}
SPLIT = {"H": {"A": 0.6, "D": 0.4}, "A": {"H": 0.7, "D": 0.3}, "D": {"H": 0.5, "A": 0.5}}


def _decoder_cfg(min_dwell=(1, 1, 1), mean_extra=10.0, stages=(1, 1, 1), exit_split=None,
                 fps=25.0) -> dict:
    """Decoder section; mean duration = min_dwell + mean_extra frames."""
    return {
        "lam_team": 0.2, "lam_dead": 0.2,
        "mean_duration_s": {name: (int(min_dwell[i]) + float(mean_extra)) / fps
                            for i, name in enumerate(STATES)},
        "exit_split": copy.deepcopy(SPLIT if exit_split is None else exit_split),
        "min_dwell_frames": {name: int(min_dwell[i]) for i, name in enumerate(STATES)},
        "duration_stages": {name: int(stages[i]) for i, name in enumerate(STATES)},
    }


def _params(min_dwell=(1, 1, 1), mean_extra=10.0, stages=(1, 1, 1), exit_split=None,
            fps=25.0) -> DecoderParams:
    return DecoderParams.from_config(
        _decoder_cfg(min_dwell, mean_extra, stages, exit_split, fps), fps=fps)


def _codes(labels) -> np.ndarray:
    table = dict(CODE, X=3)
    return np.asarray([table[x] for x in labels], dtype=np.int8)


def _sha(labels: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(labels).tobytes()).hexdigest()


def _one_block(n):
    return np.ones(n, dtype=np.int8), np.zeros(n, dtype=bool)


# -- independent references ---------------------------------------------------------------------

def _f202_dense_predecessors(params: DecoderParams):
    """The F2-02 topology (before F2-08), rebuilt here as the byte-for-byte reference."""
    m = [int(v) for v in params.min_dwell]
    offsets = [0, m[0], m[0] + m[1]]
    k_total = sum(m)
    preds = [[] for _ in range(k_total)]
    terminal = [offsets[s] + m[s] - 1 for s in range(3)]
    state_of = np.repeat(np.arange(3, dtype=np.int16), m)
    for s in range(3):
        for k in range(1, m[s]):
            preds[offsets[s] + k].append((offsets[s] + k - 1, 0.0))
    for s in range(3):
        preds[terminal[s]].append((terminal[s], float(params.log_p_stay[s])))
        for sp in range(3):
            if sp != s and np.isfinite(params.log_exit[s, sp]):
                preds[offsets[sp]].append((terminal[s], float(params.log_exit[s, sp])))
    for row in preds:
        row.sort(key=lambda pair: pair[0])
    width = max(len(row) for row in preds)
    pred_src = np.full((k_total, width), k_total, dtype=np.int64)
    pred_log = np.full((k_total, width), -np.inf)
    for j, row in enumerate(preds):
        for col, (i, la) in enumerate(row):
            pred_src[j, col], pred_log[j, col] = i, la
    return state_of, np.asarray(terminal, dtype=np.int16), pred_src, pred_log


def _spec_matrix(params: DecoderParams):
    """Dense ``(K, K)`` transition matrix built from the F2-08 specification (probabilities).

    Sub-states of ``s``: ``m_s − 1`` chain states then ``r_s`` stages; chain → next with 1, stage
    loops with ``p_s`` and advances with ``1 − p_s``; the last stage exits with
    ``(1 − p_s) · exit_split[s, ·]`` into the first sub-state of the other label.
    """
    m = [int(v) for v in params.min_dwell]
    r = [int(v) for v in params.duration_stages]
    subs = [(s, k) for s in range(3) for k in range(1, m[s] + r[s])]
    index = {sub: i for i, sub in enumerate(subs)}
    k_total = len(subs)
    p = 1.0 - np.asarray([r[s] / (params.mean_duration_frames[s] - m[s] + 1.0)
                          for s in range(3)])
    mat = np.zeros((k_total, k_total))
    for s in range(3):
        last = m[s] + r[s] - 1
        for k in range(1, m[s]):
            mat[index[(s, k)], index[(s, k + 1)]] = 1.0
        for k in range(m[s], last + 1):
            mat[index[(s, k)], index[(s, k)]] = p[s]
            if k < last:
                mat[index[(s, k)], index[(s, k + 1)]] = 1.0 - p[s]
        for sp in range(3):
            if sp != s:
                mat[index[(s, last)], index[(sp, 1)]] = (1.0 - p[s]) * params.exit_split[s, sp]
    state_of = np.asarray([s for s, _ in subs])
    terminal = [index[(s, m[s] + r[s] - 1)] for s in range(3)]
    return mat, state_of, terminal


def _matrix_from_tables(params: DecoderParams) -> np.ndarray:
    """Dense ``(K, K)`` probability matrix read back from the decoder's predecessor tables."""
    state_of, _term, pred_src, pred_log = viterbi._dense_predecessors(params)
    k_total = state_of.shape[0]
    mat = np.zeros((k_total, k_total))
    for j in range(k_total):
        for i, la in zip(pred_src[j], pred_log[j]):
            if i < k_total and np.isfinite(la):
                mat[i, j] += math.exp(la)
    return mat


def _reference_block(scores: np.ndarray, params: DecoderParams) -> np.ndarray:
    """Dense Viterbi on the specification matrix; init at the last stage, smallest-index ties."""
    mat, state_of, terminal = _spec_matrix(params)
    with np.errstate(divide="ignore"):
        logm = np.log(mat)
    t_len, k_total = scores.shape[0], mat.shape[0]
    delta = np.full(k_total, -np.inf)
    for s in range(3):
        delta[terminal[s]] = np.log(1.0 / 3.0) + scores[0, s]
    ptr = np.zeros((t_len, k_total), dtype=np.int64)
    for t in range(1, t_len):
        cand = delta[:, None] + logm
        ptr[t] = np.argmax(cand, axis=0)
        delta = cand[ptr[t], np.arange(k_total)] + scores[t, state_of]
    path = np.empty(t_len, dtype=np.int64)
    path[-1] = int(np.argmax(delta))
    for t in range(t_len - 1, 0, -1):
        path[t - 1] = ptr[t, path[t]]
    return state_of[path].astype(np.int8)


# -- byte-for-byte regression with one stage ----------------------------------------------------

@pytest.mark.parametrize("name", sorted(MOCK_LABELS_SHA256))
def test_one_stage_mock_labels_identical_to_main(name):
    cfg = load_config(REPO / "configs" / "{}.yaml".format(name))
    assert cfg["decoder"]["duration_stages"] == {"H": 1, "A": 1, "D": 1}
    labels = predict_labels(generate_mock_master(seed=42), cfg)
    assert _sha(labels) == MOCK_LABELS_SHA256[name]


@pytest.mark.parametrize("name", ["v1", "v1s", "v2", "v2_base", "v2_gbdt"])
def test_one_stage_params_identical_to_f201(name):
    cfg = load_config(REPO / "configs" / "{}.yaml".format(name))
    params = DecoderParams.from_config(cfg["decoder"])
    legacy = dict(cfg["decoder"])
    del legacy["duration_stages"]                     # the F2-01 mapping
    params_legacy = DecoderParams.from_config(legacy)
    for field in ("p_stay", "log_p_stay", "log_exit", "min_dwell", "mean_duration_frames"):
        assert getattr(params, field).tobytes() == getattr(params_legacy, field).tobytes()
    for s in range(3):
        mean = params.mean_duration_frames[s]
        m = int(params.min_dwell[s])
        expected = 1.0 - 1.0 / (mean - m + 1.0)        # F2-01 closed form, bit for bit
        assert params.p_stay[s] == expected
        for sp in range(3):
            if sp != s:
                assert params.log_exit[s, sp] == np.log(1.0 - expected) + np.log(
                    params.exit_split[s, sp])
    np.testing.assert_array_equal(params.duration_stages, [1, 1, 1])


@pytest.mark.parametrize("min_dwell", [(1, 1, 1), (12, 12, 25), (25, 25, 50), (3, 1, 2)])
def test_one_stage_topology_identical_to_f202(min_dwell):
    params = _params(min_dwell=min_dwell, mean_extra=20.0)
    new = viterbi._dense_predecessors(params)
    old = _f202_dense_predecessors(params)
    for a, b in zip(new, old):
        assert a.shape == b.shape
        np.testing.assert_array_equal(a, b)
        assert a.tobytes() == b.astype(a.dtype).tobytes()


# -- transition table ---------------------------------------------------------------------------

@pytest.mark.parametrize("min_dwell,stages", [
    ((1, 1, 1), (1, 1, 1)), ((1, 1, 1), (3, 2, 4)), ((12, 12, 25), (1, 1, 2)),
    ((3, 1, 2), (8, 6, 1)), ((5, 5, 5), (2, 3, 4))])
def test_transition_rows_sum_to_one(min_dwell, stages):
    params = _params(min_dwell=min_dwell, mean_extra=30.0, stages=stages)
    mat = _matrix_from_tables(params)
    np.testing.assert_allclose(mat.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
    spec, _state_of, _term = _spec_matrix(params)
    np.testing.assert_allclose(mat, spec, rtol=0.0, atol=1e-15)
    assert mat.shape[0] == int(np.sum(np.asarray(min_dwell) + np.asarray(stages) - 1))


def test_at_most_three_predecessors_and_terminal_is_last_stage():
    params = _params(min_dwell=(1, 4, 2), mean_extra=30.0, stages=(3, 2, 5))
    state_of, terminal_idx, pred_src, _log = viterbi._dense_predecessors(params)
    assert pred_src.shape[1] <= 3
    for s in range(3):
        idx = np.flatnonzero(state_of == s)
        assert int(terminal_idx[s]) == int(idx[-1])


@pytest.mark.parametrize("min_dwell,stages", [
    ((1, 1, 1), (2, 3, 1)), ((3, 2, 4), (2, 2, 2)), ((1, 4, 2), (4, 1, 3)), ((5, 5, 1), (6, 1, 8))])
@pytest.mark.parametrize("seed", [0, 1])
def test_decoder_matches_dense_spec_reference(min_dwell, stages, seed):
    rng = np.random.default_rng(100 * seed + sum(min_dwell) + 10 * sum(stages))
    n = 200
    scores = rng.normal(scale=1.5, size=(n, 3))
    params = _params(min_dwell=min_dwell, mean_extra=12.0, stages=stages)
    labels = decode_match(scores, *_one_block(n), params)
    np.testing.assert_array_equal(_codes(labels), _reference_block(scores, params))


def test_decoder_matches_reference_with_integer_ties():
    """Integer scores create exact ties: the smallest predecessor index must win, as in F2-02."""
    rng = np.random.default_rng(3)
    n = 300
    scores = rng.integers(-2, 3, size=(n, 3)).astype(np.float64)
    params = _params(min_dwell=(2, 3, 1), mean_extra=8.0, stages=(3, 2, 2))
    labels = decode_match(scores, *_one_block(n), params)
    np.testing.assert_array_equal(_codes(labels), _reference_block(scores, params))


# -- durations ----------------------------------------------------------------------------------

def _simulate_lengths(params: DecoderParams, state: int, n: int, seed: int) -> np.ndarray:
    """Segment lengths of ``state`` by walking the decoder's own transition table from its entry."""
    mat = _matrix_from_tables(params)
    state_of, _term, _src, _log = viterbi._dense_predecessors(params)
    entry = int(np.flatnonzero(state_of == state)[0])
    rng = np.random.default_rng(seed)
    cum = np.cumsum(mat, axis=1)
    pos = np.full(n, entry, dtype=np.int64)
    length = np.ones(n, dtype=np.int64)
    alive = np.ones(n, dtype=bool)
    while alive.any():
        idx = np.flatnonzero(alive)
        u = rng.random(idx.size)
        nxt = (u[:, None] > cum[pos[idx]]).sum(axis=1)
        nxt = np.minimum(nxt, mat.shape[0] - 1)
        stay = state_of[nxt] == state
        pos[idx] = nxt
        length[idx[stay]] += 1
        alive[idx[~stay]] = False
    return length


@pytest.mark.parametrize("stages", [(1, 1, 1), (2, 3, 4), (8, 6, 2)])
def test_simulated_mean_duration_matches_config(stages):
    params = _params(min_dwell=(5, 3, 8), mean_extra=40.0, stages=stages)
    for s in range(3):
        lengths = _simulate_lengths(params, s, n=20000, seed=11 + s)
        mean = float(params.mean_duration_frames[s])
        assert abs(lengths.mean() - mean) <= 0.02 * mean
        assert lengths.min() >= int(params.min_dwell[s]) + int(params.duration_stages[s]) - 1


def test_simulated_lengths_follow_calibration_pmf():
    """The decoder chain and ``durations.length_logpmf`` describe the same distribution."""
    params = _params(min_dwell=(4, 4, 4), mean_extra=20.0, stages=(3, 1, 1))
    lengths = _simulate_lengths(params, 0, n=40000, seed=5)
    support = np.arange(6, 40)
    pmf = np.exp(durations.length_logpmf(support, params.mean_duration_frames[0], 4, 3))
    freq = np.asarray([(lengths == v).mean() for v in support])
    np.testing.assert_allclose(freq, pmf, atol=0.006)


def test_short_segments_less_likely_with_stages():
    """Lengths of 1–2 frames beyond min_dwell lose probability when r > 1 (mean unchanged)."""
    m, extra = 12, 150.0
    mean = m + extra
    for r in (2, 3, 4):
        for beyond in (1, 2):
            length = np.asarray([m + beyond])
            lp1 = durations.length_logpmf(length, mean, m, 1)[0]
            lpr = durations.length_logpmf(length, mean, m, r)[0]
            assert lpr < lp1


@pytest.mark.parametrize("blip,delta", [(6, 1.7), (7, 1.45)])
def test_weak_evidence_short_spell_survives_only_with_one_stage(blip, delta):
    """A short H spell (1–2 frames beyond min_dwell 5) inside A with weak evidence: kept with one
    stage, dropped with three stages."""
    n = 200
    scores = np.zeros((n, 3))
    scores[:, 2] = -5.0
    scores[:, 0] = -1.0                                 # A favoured everywhere ...
    scores[100:100 + blip, 0] = delta                   # ... except a weak H blip
    one = _params(min_dwell=(5, 5, 5), mean_extra=40.0, stages=(1, 1, 1),
                  exit_split={"H": {"A": .5, "D": .5}, "A": {"H": .5, "D": .5},
                              "D": {"H": .5, "A": .5}})
    three = _params(min_dwell=(5, 5, 5), mean_extra=40.0, stages=(3, 3, 3),
                    exit_split={"H": {"A": .5, "D": .5}, "A": {"H": .5, "D": .5},
                                "D": {"H": .5, "A": .5}})
    lab1 = decode_match(scores, *_one_block(n), one)
    lab3 = decode_match(scores, *_one_block(n), three)
    assert (lab1 == "H").sum() == blip
    assert (lab3 == "H").sum() == 0


# -- errors and limits --------------------------------------------------------------------------

def test_error_when_mean_does_not_exceed_stages():
    # mean − m + 1 = 4 frames: r = 3 is fine, r = 4 is not.
    cfg = _decoder_cfg(min_dwell=(10, 10, 10), mean_extra=3.0, stages=(3, 3, 3))
    DecoderParams.from_config(cfg)
    cfg["duration_stages"]["A"] = 4
    with pytest.raises(ValueError, match=r"must exceed duration_stages\[A\]"):
        DecoderParams.from_config(cfg)
    with pytest.raises(ValueError, match="must exceed duration_stages"):
        _p_stay_from_mean(13.0, 10, 4)
    with pytest.raises(ValueError, match="must exceed duration_stages"):
        durations.advance_prob(13.0, 10, 4)


def test_config_error_when_mean_does_not_exceed_stages():
    raw = copy.deepcopy(HMM_CFG)
    # H: 0.6 s = 15 frames, min_dwell 12 → mean − m + 1 = 4.
    raw["decoder"]["mean_duration_s"]["H"] = 0.6
    raw["decoder"]["duration_stages"]["H"] = 3
    validate_config(raw)
    raw["decoder"]["duration_stages"]["H"] = 4
    with pytest.raises(ConfigError, match="must exceed decoder.duration_stages.H"):
        validate_config(raw)


@pytest.mark.parametrize("bad", [0, -1, 1.0, True, None])
def test_config_rejects_invalid_stage_counts(bad):
    raw = copy.deepcopy(HMM_CFG)
    raw["decoder"]["duration_stages"]["D"] = bad
    with pytest.raises(ConfigError):
        validate_config(raw)


def test_config_requires_duration_stages(tmp_path):
    raw = copy.deepcopy(HMM_CFG)
    del raw["decoder"]["duration_stages"]
    path = tmp_path / "{}.yaml".format(raw["name"])
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="missing keys.*duration_stages"):
        load_config(path)
    raw = copy.deepcopy(HMM_CFG)
    raw["decoder"]["duration_stages"]["X"] = 1
    with pytest.raises(ConfigError, match="unknown keys"):
        validate_config(raw)


def test_every_repo_hmm_config_declares_duration_stages():
    found = 0
    for path in sorted((REPO / "configs").glob("*.yaml")):
        if path.name.startswith(("degrade_", "broadcast", "decoder_", "plausibility", "hazard_")):
            continue
        cfg = load_config(path)
        if cfg["pipeline"] != "hmm":
            continue
        found += 1
        assert set(cfg["decoder"]["duration_stages"]) == {"H", "A", "D"}
    assert found >= 5
    overlay = yaml.safe_load((REPO / "configs" / "decoder_idsse.yaml").read_text(encoding="utf-8"))
    assert overlay["decoder"]["duration_stages"] == {"H": 1, "A": 1, "D": 1}


def test_k_stays_within_int16():
    with pytest.raises(ValueError, match="int16"):
        viterbi._expand_topology(np.asarray([10000, 10000, 10000]), np.asarray([1000, 1000, 1001]))
    k_total, *_ = viterbi._expand_topology(np.asarray([10000, 10000, 10000]),
                                           np.asarray([1000, 1000, 768]))
    assert k_total == 32765
    params = _params(min_dwell=(25, 25, 50), mean_extra=200.0,
                     stages=tuple([max(durations.R_GRID)] * 3))
    state_of, *_ = viterbi._dense_predecessors(params)
    assert state_of.dtype == np.int16 and state_of.shape[0] == 100 + 3 * 7


def test_params_reject_invalid_stage_array_and_are_read_only():
    params = _params(stages=(2, 1, 3), mean_extra=20.0)
    assert not params.duration_stages.flags.writeable
    with pytest.raises(ValueError, match="duration_stages"):
        DecoderParams(lam_team=0.2, lam_dead=0.2, mean_duration_frames=[20.0] * 3,
                      exit_split=np.full((3, 3), 0.5) - np.eye(3) * 0.5, min_dwell=[1, 1, 1],
                      neg=-20.0, p_stay=[0.9] * 3, log_p_stay=np.log([0.9] * 3),
                      log_exit=np.zeros((3, 3)), duration_stages=[1, 0, 1])


def test_determinism_and_half_blocks_with_stages():
    rng = np.random.default_rng(9)
    n = 400
    scores = rng.normal(scale=1.0, size=(n, 3))
    half = np.ones(n, dtype=np.int8)
    half[150:170] = 0
    half[170:] = 2
    force = np.zeros(n, dtype=bool)
    params = _params(min_dwell=(3, 3, 4), mean_extra=15.0, stages=(2, 3, 2))
    a = decode_match(scores, half, force, params)
    b = decode_match(scores, half, force, params)
    assert a.tobytes() == b.tobytes()
    assert (a[150:170] == "X").all()
    # Each half is decoded alone: decoding the second block by itself gives the same labels.
    alone = decode_match(scores[170:], np.full(n - 170, 2, np.int8), force[170:], params)
    np.testing.assert_array_equal(a[170:], alone)


# -- calibration (modeling.calibration.durations) -----------------------------------------------

@pytest.mark.parametrize("m,r", [(1, 1), (12, 1), (12, 2), (25, 3), (5, 8)])
def test_length_pmf_normalises_and_keeps_the_mean(m, r):
    mean = m + 60.0
    support = np.arange(0, 4000)
    pmf = np.exp(durations.length_logpmf(support, mean, m, r))
    assert pmf[: m - 1 + r].sum() == 0.0
    assert abs(pmf.sum() - 1.0) < 1e-9
    assert abs((support * pmf).sum() - mean) < 1e-6


def _draw_lengths(rng, n, mean, m, r):
    q = durations.advance_prob(mean, m, r)
    return (m - 1) + rng.geometric(q, size=(n, r)).sum(axis=1)


@pytest.mark.parametrize("true_r", [1, 2, 3, 6])
def test_choose_stages_recovers_r(true_r):
    rng = np.random.default_rng(true_r)
    m, mean = 12, 150.0
    lengths = _draw_lengths(rng, 3000, mean, m, true_r)
    chosen, table = durations.choose_stages(lengths, mean, m)
    assert chosen == true_r
    assert set(table) == set(durations.R_GRID)


def test_overdispersed_lengths_choose_one_stage():
    """A mixture of short and long segments is more spread than a geometric: r = 1 wins."""
    rng = np.random.default_rng(4)
    m = 12
    lengths = np.concatenate([_draw_lengths(rng, 2000, 40.0, m, 1),
                              _draw_lengths(rng, 1000, 400.0, m, 1)])
    mean = float(lengths.mean())
    chosen, _ = durations.choose_stages(lengths, mean, m)
    assert chosen == 1
    floor = durations.support_floor(m)
    assert durations.dispersion_ratio(lengths, mean, m, floor) > 1.0


def test_choose_stages_skips_infeasible_r_and_breaks_ties_to_smallest():
    # mean − m + 1 = 5: r in {6, 8} infeasible.
    lengths = np.asarray([20, 21, 22])
    chosen, table = durations.choose_stages(lengths, 16.0, 12)
    assert table[6] == -math.inf and table[8] == -math.inf
    assert chosen in (1, 2, 3, 4)
    # No segment reaches the common floor: every feasible r scores 0, the smallest wins.
    chosen, table = durations.choose_stages(np.asarray([12, 13]), 100.0, 12)
    assert chosen == 1 and table[1] == 0.0


def test_truncated_likelihood_uses_the_same_segments_for_every_r():
    rng = np.random.default_rng(2)
    m, mean = 5, 60.0
    lengths = _draw_lengths(rng, 500, mean, m, 1)
    floor = durations.support_floor(m)
    short = np.asarray([m, m + 1])                      # impossible for large r
    for r in durations.R_GRID:
        a = durations.truncated_loglik(lengths, mean, m, r, floor)
        b = durations.truncated_loglik(np.concatenate([lengths, short]), mean, m, r, floor)
        assert a == b and math.isfinite(a)


def _synthetic_per_match():
    rng = np.random.default_rng(8)
    per_match = {}
    for i, name in enumerate(("M1", "M2", "M3", "M4")):
        per_match[name] = {
            "H": _draw_lengths(rng, 400, 190.0, 12, 1),
            "A": _draw_lengths(rng, 400, 170.0, 12, 1),
            # M4's D segments are near-deterministic (r = 8); the others are r = 2.
            "D": _draw_lengths(rng, 400 if i < 3 else 4000, 630.0, 25, 2 if i < 3 else 8),
        }
    return per_match


def test_calibrate_durations_all_and_leave_one_out_folds():
    per_match = _synthetic_per_match()
    cfg_dec = {"mean_duration_s": {"H": 190.0 / 25, "A": 170.0 / 25, "D": 630.0 / 25},
               "min_dwell_frames": {"H": 12, "A": 12, "D": 25}}
    result = durations.calibrate_durations(per_match, cfg_dec, fps=25.0)
    assert result["all"]["H"] == 1 and result["all"]["A"] == 1
    assert set(result["folds"]) == set(per_match)
    # Leaving M4 out removes the r = 8 segments: that fold is the only one at r = 2.
    assert result["folds"]["M4"]["D"] == 2
    assert all(result["folds"][m]["D"] > 2 for m in ("M1", "M2", "M3"))
    # Each fold equals a fit on the other matches only.
    rest = durations.pool(per_match, ["M1", "M2", "M3"])
    mean, dwell = durations.decoder_durations(cfg_dec, 25.0)
    chosen, _ = durations.fit_stages(rest, mean, dwell)
    assert chosen == result["folds"]["M4"]


def test_write_calib_has_only_all_and_folds(tmp_path):
    per_match = _synthetic_per_match()
    cfg_dec = {"mean_duration_s": {"H": 190.0 / 25, "A": 170.0 / 25, "D": 630.0 / 25},
               "min_dwell_frames": {"H": 12, "A": 12, "D": 25}}
    result = durations.calibrate_durations(per_match, cfg_dec, fps=25.0)
    path = durations.write_calib(result, tmp_path / "durations.yaml", config_name="v1s")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert set(loaded) == {"all", "folds"}
    assert loaded["all"] == result["all"]
    assert set(loaded["folds"]) == set(per_match)
    lines = durations.report_lines(result, per_match, config_name="v1s", fps=25.0)
    hist = durations.histogram_lines(durations.pool(per_match, list(per_match)), 25.0)
    assert any("Leave-one-match-out" in line for line in lines)
    assert any(line.lstrip().startswith("≥15 |") for line in hist)
    assert any(line.lstrip().startswith("≥60 |") for line in hist)


def test_repo_calibration_files_are_consistent_with_their_configs():
    """configs/calib/durations*.yaml exist, are LOMO-stable, and every config built from one
    uses its ``all`` stages (the files themselves come from IDSSE, never read in CI)."""
    calib = yaml.safe_load((REPO / "configs" / "calib" / "durations.yaml").read_text("utf-8"))
    assert calib["all"] == {"H": 1, "A": 1, "D": 1}
    assert load_config(REPO / "configs" / "v1s.yaml")["decoder"]["duration_stages"] == calib["all"]


# -- selection step (modeling.scoring.durations_sweep) -------------------------------------------

def _fake_frame_fn(gain_by_mode):
    """Per-match metric tables whose spellF1 depends on the D stages, λ and the degrade mode."""
    import pandas as pd

    from modeling.scoring.external_eval import SUMMARY_ROWS

    matches = ["M1", "M2", "M3", "M4"]
    noise = np.asarray([0.0, 0.002, -0.001, 0.001])

    def frame(cfg, degrade):
        dec = cfg["decoder"]
        staged = dec["duration_stages"]["D"] > 1
        f1 = 0.40 + noise + (gain_by_mode[degrade] if staged else 0.0) \
            - 0.01 * abs(dec["lam_team"] - 0.2) - 0.01 * abs(dec["lam_dead"] - 0.2)
        df = pd.DataFrame({"spell_f1_mean": f1, "n_gt_spells": 100.0,
                           "n_pred_spells": 70.0 + 5.0 * staged}, index=matches)
        summary = pd.DataFrame([df.mean(), df.std(ddof=1) / 2.0], index=list(SUMMARY_ROWS))
        return pd.concat([df, summary])
    return frame


def test_durations_step_is_identity_when_stages_do_not_change():
    from modeling.scoring import durations_sweep as ds

    calls = []
    evaluate = ds.Evaluator(matches=["M1"], frame_fn=lambda c, d: calls.append(d))
    base = load_config(REPO / "configs" / "v1s.yaml")
    history = ds.run_durations_step(base, {"H": 1, "A": 1, "D": 1}, evaluate)
    assert history["identity"] and not calls
    text = ds.render_section(history, config_name="v1s", calib="configs/calib/durations.yaml")
    assert "identity" in text


@pytest.mark.parametrize("gains,accepted", [
    ({"default": 0.01, "harder": 0.0, "easier": 0.0, "hidelo": 0.0, "hidehi": 0.0}, True),
    ({"default": 0.01, "harder": -0.01, "easier": 0.0, "hidelo": 0.0, "hidehi": 0.0}, False),
    ({"default": -0.003, "harder": 0.0, "easier": 0.0, "hidelo": 0.0, "hidehi": 0.0}, False)])
def test_durations_step_applies_the_f206_rule(gains, accepted):
    from modeling.scoring import durations_sweep as ds

    evaluate = ds.Evaluator(matches=["M1", "M2", "M3", "M4"], frame_fn=_fake_frame_fn(gains))
    base = validate_config(copy.deepcopy(HMM_CFG))
    history = ds.run_durations_step(base, {"H": 1, "A": 1, "D": 2}, evaluate)
    assert not history["identity"]
    pick = history["lam_pick"]["cfg"]["decoder"]
    assert (pick["lam_team"], pick["lam_dead"]) == (0.2, 0.2)
    assert pick["duration_stages"] == {"H": 1, "A": 1, "D": 2}
    assert history["accept"]["stages_lam"]["accepted"] is accepted
    # the control re-sweeps λ with one stage; against it only the stage effect is left
    ctrl = history["control_pick"]["cfg"]["decoder"]
    assert ctrl["duration_stages"] == {"H": 1, "A": 1, "D": 1}
    assert history["accept"]["stages_lam vs base_lam"]["accepted"] is accepted
    summary = ds.summary_frame(evaluate, history["configs"])
    assert set(summary["mode"]) == set(ds.MODES)
    text = ds.render_section(history, config_name="x", calib="c.yaml", summary=summary,
                             extra=ds.extra_d_rows(evaluate, base, history["stages"], [3]))
    assert ("**accepted**" in text) is accepted
    assert "base_lam" in text and "control" in text


def test_evaluator_disk_cache_round_trips(tmp_path):
    from modeling.scoring import durations_sweep as ds

    calls = []
    fake = _fake_frame_fn({m: 0.0 for m in ("default",)})

    def frame_fn(cfg, degrade):
        calls.append(degrade)
        return fake(cfg, degrade)

    cfg = validate_config(copy.deepcopy(HMM_CFG))
    first = ds.Evaluator(matches=["M1", "M2", "M3", "M4"], frame_fn=frame_fn, cache_dir=tmp_path)
    a = first.frame(cfg, "default")
    first.frame(cfg, "default")
    second = ds.Evaluator(matches=["M1", "M2", "M3", "M4"], frame_fn=frame_fn, cache_dir=tmp_path)
    b = second.frame(cfg, "default")
    assert calls == ["default"]
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=0, atol=1e-15)
    assert list(a.index) == list(b.index)


def test_read_stages_rejects_other_layouts(tmp_path):
    from modeling.scoring import durations_sweep as ds

    path = tmp_path / "d.yaml"
    path.write_text(yaml.safe_dump({"all": {"H": 1, "A": 1, "D": 2}, "folds": {}}), "utf-8")
    assert ds.read_stages(path) == {"H": 1, "A": 1, "D": 2}
    path.write_text(yaml.safe_dump({"all": {"H": 1, "A": 1, "D": 2}}), "utf-8")
    with pytest.raises(ValueError, match="'all' and 'folds'"):
        ds.read_stages(path)


def test_v2_gbdt_nb_is_v2_gbdt_with_calibrated_stages_and_reswept_lambda():
    nb = load_config(REPO / "configs" / "v2_gbdt_nb.yaml")
    base = load_config(REPO / "configs" / "v2_gbdt.yaml")
    calib = yaml.safe_load(
        (REPO / "configs" / "calib" / "durations_v2_gbdt.yaml").read_text("utf-8"))
    assert calib["all"] == {"H": 1, "A": 1, "D": 2}
    assert all(row == calib["all"] for row in calib["folds"].values())
    assert nb["decoder"]["duration_stages"] == calib["all"]
    assert (nb["decoder"]["lam_team"], nb["decoder"]["lam_dead"]) == (0.4, 0.4)
    expected = copy.deepcopy(base)
    expected["name"] = "v2_gbdt_nb"
    expected["decoder"].update(lam_team=0.4, lam_dead=0.4, duration_stages=calib["all"])
    assert nb == expected


# -- interaction with the hazard-guided exits of F2-07 (D-025) ----------------------------------

def _reference_block_modulated(scores: np.ndarray, params: DecoderParams, x) -> np.ndarray:
    """Dense Viterbi with a per-frame matrix: the F2-08 chain, and the F2-07 modulation applied to
    the row of the last stage of H and of A (``q_o e^x / Z``, ``p / Z``, ``q_d / Z``)."""
    base, state_of, terminal = _spec_matrix(params)
    h, a = CODE["H"], CODE["A"]

    def matrix(xt):
        mat = base.copy()
        for s, other in ((h, a), (a, h)):
            term = terminal[s]
            row = mat[term].copy()
            first_other = int(np.flatnonzero(state_of == other)[0])
            row[first_other] *= np.exp(xt)
            mat[term] = row / row.sum()
        with np.errstate(divide="ignore"):
            return np.log(mat)

    t_len, k_total = scores.shape[0], base.shape[0]
    delta = np.full(k_total, -np.inf)
    for s in range(3):
        delta[terminal[s]] = np.log(1.0 / 3.0) + scores[0, s]
    ptr = np.zeros((t_len, k_total), dtype=np.int64)
    for t in range(1, t_len):
        cand = delta[:, None] + matrix(x[t])
        ptr[t] = np.argmax(cand, axis=0)
        delta = cand[ptr[t], np.arange(k_total)] + scores[t, state_of]
    path = np.empty(t_len, dtype=np.int64)
    path[-1] = int(np.argmax(delta))
    for t in range(t_len - 1, 0, -1):
        path[t - 1] = ptr[t, path[t]]
    return state_of[path].astype(np.int8)


@pytest.mark.parametrize("min_dwell,stages", [
    ((1, 1, 1), (2, 3, 1)), ((3, 2, 4), (2, 2, 2)), ((5, 5, 1), (1, 1, 3)), ((2, 4, 2), (4, 1, 2))])
@pytest.mark.parametrize("seed", [0, 1])
def test_hazard_modulation_acts_on_the_last_stage(min_dwell, stages, seed):
    rng = np.random.default_rng(1000 + 10 * seed + sum(stages))
    n = 160
    scores = rng.normal(scale=1.5, size=(n, 3))
    x = rng.normal(scale=3.0, size=n)
    params = _params(min_dwell=min_dwell, mean_extra=8.0, stages=stages)
    labels = decode_match(scores, *_one_block(n), params, exit_logit=x)
    np.testing.assert_array_equal(_codes(labels), _reference_block_modulated(scores, params, x))
    # zero exit logit = the constant decoder with stages (no tie changes on these scores)
    zero = decode_match(scores, *_one_block(n), params, exit_logit=np.zeros(n))
    np.testing.assert_array_equal(zero, decode_match(scores, *_one_block(n), params))
