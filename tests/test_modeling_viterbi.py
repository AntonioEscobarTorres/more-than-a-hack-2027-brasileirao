"""Synthetic tests for the per-half min-dwell Viterbi decoder (task F2-02)."""
from __future__ import annotations

import itertools
import time

import numpy as np
import pytest

from modeling.contracts import STATES
from modeling.decode.params import DecoderParams
from modeling.decode.viterbi import decode_match


def _params(min_dwell=(1, 1, 1), mean_extra=10.0, exit_split=None, fps=25.0) -> DecoderParams:
    """DecoderParams with controllable min_dwell; mean duration = dwell + mean_extra."""
    if exit_split is None:
        exit_split = {
            "H": {"A": 0.5, "D": 0.5},
            "A": {"H": 0.5, "D": 0.5},
            "D": {"H": 0.5, "A": 0.5},
        }
    cfg = {
        "lam_team": 0.2,
        "lam_dead": 0.2,
        "mean_duration_s": {
            name: (int(min_dwell[i]) + float(mean_extra)) / fps
            for i, name in enumerate(STATES)
        },
        "exit_split": exit_split,
        "min_dwell_frames": {name: int(min_dwell[i]) for i, name in enumerate(STATES)},
    }
    return DecoderParams.from_config(cfg, fps=fps)


def _path_logprob(labels_codes: np.ndarray, scores: np.ndarray, params: DecoderParams) -> float:
    """Log-probability of a label-code path under the expanded HMM, or -inf if illegal."""
    n = labels_codes.shape[0]
    if n == 0:
        return 0.0
    m = params.min_dwell
    log_p_stay = params.log_p_stay
    log_exit = params.log_exit

    s0 = int(labels_codes[0])
    dwell = int(m[s0])  # init at terminal
    total = np.log(1.0 / 3.0) + float(scores[0, s0])

    for t in range(1, n):
        s = int(labels_codes[t])
        prev = int(labels_codes[t - 1])
        if s == prev:
            if dwell < int(m[s]):
                total += 0.0 + float(scores[t, s])
                dwell += 1
            else:
                total += float(log_p_stay[s]) + float(scores[t, s])
                dwell = int(m[s])
        else:
            if dwell < int(m[prev]):
                return -np.inf
            la = float(log_exit[prev, s])
            if not np.isfinite(la):
                return -np.inf
            total += la + float(scores[t, s])
            dwell = 1
    return float(total)


def _exhaustive_best(scores: np.ndarray, params: DecoderParams):
    """Return (best_logprob, best_codes) over all label sequences of length T."""
    t_len = scores.shape[0]
    best_lp = -np.inf
    best_codes = None
    for seq in itertools.product(range(3), repeat=t_len):
        codes = np.asarray(seq, dtype=np.int8)
        lp = _path_logprob(codes, scores, params)
        if lp > best_lp:
            best_lp = lp
            best_codes = codes
    return best_lp, best_codes


def _segments(codes: np.ndarray):
    """List of (state, start, stop) over a 1-D code array."""
    out = []
    n = codes.shape[0]
    if n == 0:
        return out
    start = 0
    for i in range(1, n):
        if codes[i] != codes[start]:
            out.append((int(codes[start]), start, i))
            start = i
    out.append((int(codes[start]), start, n))
    return out


def _labels_to_codes(labels: np.ndarray) -> np.ndarray:
    table = {name: i for i, name in enumerate(STATES)}
    table["X"] = 3
    return np.asarray([table[x] for x in labels], dtype=np.int8)


# -- (a) equality with exhaustive enumeration ---------------------------------------------------


@pytest.mark.parametrize("min_dwell", [
    (1, 1, 1),
    (2, 1, 1),
    (1, 2, 3),
    (2, 2, 2),
    (3, 2, 1),
])
@pytest.mark.parametrize("n", [4, 6, 8])
def test_matches_exhaustive_enumeration(min_dwell, n):
    rng = np.random.default_rng(1000 + 10 * n + sum(min_dwell))
    scores = rng.normal(size=(n, 3)).astype(np.float64)
    params = _params(min_dwell=min_dwell, mean_extra=5.0)
    half = np.ones(n, dtype=np.int8)
    force = np.zeros(n, dtype=bool)

    labels = decode_match(scores, half, force, params)
    codes = _labels_to_codes(labels)
    assert (codes < 3).all()

    vit_lp = _path_logprob(codes, scores, params)
    best_lp, best_codes = _exhaustive_best(scores, params)
    assert np.isfinite(vit_lp)
    assert np.isfinite(best_lp)
    np.testing.assert_allclose(vit_lp, best_lp, rtol=0.0, atol=1e-9)
    # Viterbi path must be one of the optimal sequences.
    assert vit_lp == _path_logprob(codes, scores, params)
    assert best_lp == _path_logprob(best_codes, scores, params)


# -- (b) min dwell on interior segments ---------------------------------------------------------


def test_min_dwell_respected_except_first_and_last():
    rng = np.random.default_rng(7)
    n = 40
    # Strong alternating blocks longer than min_dwell so the decoder switches.
    scores = np.zeros((n, 3), dtype=np.float64)
    for t in range(n):
        s = (t // 8) % 3
        scores[t, s] = 5.0
    params = _params(min_dwell=(3, 3, 3), mean_extra=8.0)
    half = np.ones(n, dtype=np.int8)
    labels = decode_match(scores, half, np.zeros(n, dtype=bool), params)
    codes = _labels_to_codes(labels)
    segs = _segments(codes)
    assert len(segs) >= 2
    for idx, (s, a, b) in enumerate(segs):
        length = b - a
        if idx == 0 or idx == len(segs) - 1:
            continue
        assert length >= int(params.min_dwell[s]), (idx, s, length, segs)


# -- (c) zero-score gap keeps the same team -----------------------------------------------------


def test_zero_gap_between_same_team_keeps_label():
    n = 30
    scores = np.zeros((n, 3), dtype=np.float64)
    scores[:10, 0] = 3.0   # H
    # frames 10..19: all zeros (no evidence)
    scores[20:, 0] = 3.0   # H again
    params = _params(min_dwell=(2, 2, 2), mean_extra=10.0)
    half = np.ones(n, dtype=np.int8)
    labels = decode_match(scores, half, np.zeros(n, dtype=bool), params)
    assert set(labels.tolist()) == {"H"}


# -- (d) sharp evidence change places the boundary near t0 --------------------------------------


def test_sharp_change_boundary_near_switch():
    n = 40
    t0 = 20
    scores = np.zeros((n, 3), dtype=np.float64)
    scores[:t0, 0] = 8.0
    scores[t0:, 1] = 8.0
    params = _params(min_dwell=(1, 1, 1), mean_extra=20.0)
    half = np.ones(n, dtype=np.int8)
    labels = decode_match(scores, half, np.zeros(n, dtype=bool), params)
    codes = _labels_to_codes(labels)
    # First frame of A after the run of H.
    switch = int(np.flatnonzero(codes == 1)[0])
    assert abs(switch - t0) <= 3, switch


# -- (e) determinism ----------------------------------------------------------------------------


def test_determinism():
    rng = np.random.default_rng(123)
    n = 50
    scores = rng.normal(size=(n, 3))
    half = np.ones(n, dtype=np.int8)
    half[40:] = 2
    force = np.zeros(n, dtype=bool)
    force[45:48] = True
    params = _params(min_dwell=(2, 3, 2), mean_extra=6.0)
    a = decode_match(scores, half, force, params)
    b = decode_match(scores, half, force, params)
    assert a.dtype == np.dtype("<U1")
    np.testing.assert_array_equal(a, b)


# -- (f) force_d --------------------------------------------------------------------------------


def test_force_d_respected():
    n = 25
    scores = np.zeros((n, 3), dtype=np.float64)
    scores[:, 0] = 10.0  # strong H everywhere
    force = np.zeros(n, dtype=bool)
    force[5:10] = True
    params = _params(min_dwell=(1, 1, 1), mean_extra=5.0)
    half = np.ones(n, dtype=np.int8)
    labels = decode_match(scores, half, force, params)
    assert (labels[force] == "D").all()
    assert (labels[~force] == "H").all()


# -- (g) X exactly where half == 0 --------------------------------------------------------------


def test_x_exactly_where_half_zero():
    n = 30
    half = np.zeros(n, dtype=np.int8)
    half[5:15] = 1
    half[20:28] = 2
    scores = np.zeros((n, 3), dtype=np.float64)
    scores[:, 0] = 1.0
    params = _params(min_dwell=(1, 1, 1))
    labels = decode_match(scores, half, np.zeros(n, dtype=bool), params)
    assert (labels[half == 0] == "X").all()
    assert (labels[half != 0] != "X").all()


# -- acceptance: runtime on a long synthetic half -----------------------------------------------


def test_runtime_under_ten_seconds_for_80k_frames():
    n = 80_000
    rng = np.random.default_rng(0)
    scores = rng.normal(scale=0.5, size=(n, 3)).astype(np.float64)
    # K ≈ 50: e.g. 16+16+18.
    params = _params(min_dwell=(16, 16, 18), mean_extra=30.0)
    half = np.ones(n, dtype=np.int8)
    force = np.zeros(n, dtype=bool)
    t0 = time.perf_counter()
    labels = decode_match(scores, half, force, params)
    elapsed = time.perf_counter() - t0
    assert labels.shape == (n,)
    assert elapsed < 10.0, "decode took {:.2f}s (limit 10s)".format(elapsed)
