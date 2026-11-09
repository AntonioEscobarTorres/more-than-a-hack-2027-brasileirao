"""Per-half Viterbi decoder with minimum dwell via an expanded-state chain (C5 / F2-02).

``decode_match(scores, half, force_d, params)`` decodes each contiguous ``half ∈ {1, 2}``
block independently, maps ``half == 0`` to ``X``, and finally forces ``force_d`` frames
to ``D``.
"""
from __future__ import annotations

import numpy as np

from modeling.contracts import LABEL_DTYPE, STATES
from modeling.decode.params import DecoderParams

_NEG_INF = -np.inf
_LOG_ONE_THIRD = np.log(1.0 / 3.0)


def _expand_topology(min_dwell: np.ndarray):
    """Build the expanded sub-state index and sparse predecessor lists.

    Sub-states are ordered H, A, D; within each label ``s``, indices run
    ``(s, 1) … (s, m_s)``. Returns ``(K, state_of, dwell_of, is_terminal, preds)``
    where ``preds[j]`` is a list of ``(i, log_a_ij)`` predecessors of sub-state ``j``.
    """
    min_dwell = np.asarray(min_dwell, dtype=np.int64).reshape(3)
    if (min_dwell < 1).any():
        raise ValueError("min_dwell must be >= 1 for every state, got {}".format(min_dwell))
    offsets = np.zeros(3, dtype=np.int64)
    offsets[1] = min_dwell[0]
    offsets[2] = min_dwell[0] + min_dwell[1]
    k_total = int(min_dwell.sum())
    if k_total > 32767:
        raise ValueError("expanded state count K={} exceeds int16 range".format(k_total))

    state_of = np.empty(k_total, dtype=np.int16)
    dwell_of = np.empty(k_total, dtype=np.int16)
    is_terminal = np.zeros(k_total, dtype=bool)
    terminal_idx = np.empty(3, dtype=np.int16)
    for s in range(3):
        m = int(min_dwell[s])
        for k in range(1, m + 1):
            j = int(offsets[s]) + (k - 1)
            state_of[j] = s
            dwell_of[j] = k
            if k == m:
                is_terminal[j] = True
                terminal_idx[s] = j

    # preds[j] = list of (i, log_transition); built so i increases (argmax tie-break).
    preds = [[] for _ in range(k_total)]
    for s in range(3):
        m = int(min_dwell[s])
        base = int(offsets[s])
        # Chain: (s,k) → (s,k+1) with log-prob 0.
        for k in range(1, m):
            preds[base + k].append((base + k - 1, 0.0))
        # Terminal self-loop with log p_stay[s] — filled by caller via log_p_stay.
        # Exit into (s', 1) — filled by caller via log_exit.
        # Placeholder: topology only knows structure; log values attached below.
        _ = base  # silence linters when m == 1 and the chain loop is empty

    return k_total, state_of, dwell_of, is_terminal, terminal_idx, offsets, preds


def _attach_transition_logs(
    preds,
    min_dwell: np.ndarray,
    offsets: np.ndarray,
    terminal_idx: np.ndarray,
    log_p_stay: np.ndarray,
    log_exit: np.ndarray,
) -> None:
    """Fill self-loop and exit edges on the predecessor lists (mutates ``preds``)."""
    for s in range(3):
        term = int(terminal_idx[s])
        preds[term].append((term, float(log_p_stay[s])))
        for sp in range(3):
            if sp == s:
                continue
            dest = int(offsets[sp])  # (sp, 1)
            la = float(log_exit[s, sp])
            if np.isfinite(la):
                preds[dest].append((term, la))
    # Sort each predecessor list by source index so a left-to-right max picks the
    # smallest index on ties (same rule as np.argmax on a dense row).
    for j in range(len(preds)):
        preds[j].sort(key=lambda pair: pair[0])


def _viterbi_block(scores: np.ndarray, params: DecoderParams) -> np.ndarray:
    """Decode one contiguous half block. ``scores`` is float64 ``(T, 3)``."""
    t_len = int(scores.shape[0])
    if t_len == 0:
        return np.empty(0, dtype=np.int8)

    min_dwell = params.min_dwell
    k_total, state_of, _dwell_of, is_terminal, terminal_idx, offsets, preds = (
        _expand_topology(min_dwell)
    )
    _attach_transition_logs(
        preds, min_dwell, offsets, terminal_idx, params.log_p_stay, params.log_exit
    )

    # Pack sparse preds into arrays for the inner loop.
    pred_i = []
    pred_log = []
    pred_ptr = np.zeros(k_total + 1, dtype=np.int32)
    cursor = 0
    for j in range(k_total):
        pred_ptr[j] = cursor
        for i, la in preds[j]:
            pred_i.append(i)
            pred_log.append(la)
            cursor += 1
    pred_ptr[k_total] = cursor
    pred_i = np.asarray(pred_i, dtype=np.int16)
    pred_log = np.asarray(pred_log, dtype=np.float64)

    delta = np.full(k_total, _NEG_INF, dtype=np.float64)
    # Init: only terminal sub-states (s, m_s) get log(1/3) + scores[0, s].
    for s in range(3):
        j = int(terminal_idx[s])
        delta[j] = _LOG_ONE_THIRD + scores[0, s]

    ptr = np.empty((t_len, k_total), dtype=np.int16)
    ptr[0, :] = -1

    emit = np.empty(k_total, dtype=np.float64)

    for t in range(1, t_len):
        for s in range(3):
            emit[offsets[s]:offsets[s] + min_dwell[s]] = scores[t, s]
        prev = delta
        nxt = np.full(k_total, _NEG_INF, dtype=np.float64)
        for j in range(k_total):
            a = int(pred_ptr[j])
            b = int(pred_ptr[j + 1])
            if a == b:
                continue
            # Predecessors are sorted by source index; strict '>' keeps the
            # smallest index on ties (same rule as np.argmax on a dense row).
            best_v = _NEG_INF
            best_i = -1
            for p in range(a, b):
                i = int(pred_i[p])
                v = prev[i] + pred_log[p]
                if v > best_v:
                    best_v = v
                    best_i = i
            if best_i < 0 or not np.isfinite(best_v):
                continue
            nxt[j] = best_v + emit[j]
            ptr[t, j] = best_i
        delta = nxt

    end = int(np.argmax(delta))
    path_sub = np.empty(t_len, dtype=np.int16)
    path_sub[t_len - 1] = end
    for t in range(t_len - 1, 0, -1):
        path_sub[t - 1] = ptr[t, path_sub[t]]

    return state_of[path_sub].astype(np.int8, copy=False)


def _contiguous_blocks(half: np.ndarray):
    """Yield ``(h, start, stop)`` for every contiguous run with ``half == h`` and ``h in {1,2}``."""
    n = half.shape[0]
    i = 0
    while i < n:
        h = int(half[i])
        j = i + 1
        while j < n and int(half[j]) == h:
            j += 1
        if h in (1, 2):
            yield h, i, j
        i = j


def decode_match(scores, half, force_d, params: DecoderParams) -> np.ndarray:
    """Decode a full match into labels ``H``/``A``/``D``/``X`` (dtype ``<U1``).

    Parameters
    ----------
    scores:
        float64 array ``(n, 3)`` of emission log-scores in order (H, A, D).
    half:
        integer array ``(n,)`` with values in ``{0, 1, 2}``.
    force_d:
        bool array ``(n,)``; those frames are set to ``D`` after decoding.
    params:
        ``DecoderParams`` with ``min_dwell``, ``p_stay`` / ``log_p_stay`` and ``log_exit``.
    """
    scores = np.asarray(scores, dtype=np.float64)
    half = np.asarray(half)
    force = np.asarray(force_d, dtype=bool)
    if scores.ndim != 2 or scores.shape[1] != 3:
        raise ValueError("scores must have shape (n, 3), got {}".format(scores.shape))
    n = scores.shape[0]
    if half.shape != (n,):
        raise ValueError("half must have shape ({},), got {}".format(n, half.shape))
    if force.shape != (n,):
        raise ValueError("force_d must have shape ({},), got {}".format(n, force.shape))
    if not isinstance(params, DecoderParams):
        raise TypeError("params must be DecoderParams, got {}".format(type(params).__name__))

    # Default outside play; blocks overwrite with H/A/D.
    state_codes = np.full(n, 3, dtype=np.int8)  # X = 3
    for _h, start, stop in _contiguous_blocks(half):
        block_states = _viterbi_block(scores[start:stop], params)
        state_codes[start:stop] = block_states

    if force.any():
        state_codes[force] = 2  # D

    labels = np.empty(n, dtype=LABEL_DTYPE)
    # X for half==0 (and anything still 3); STATES for 0,1,2.
    labels[:] = "X"
    for code, name in enumerate(STATES):
        labels[state_codes == code] = name
    # Guarantees: never X inside a half block unless somehow undecoded (should not happen).
    inside = (half == 1) | (half == 2)
    if (labels[inside] == "X").any():
        raise RuntimeError("decoder produced X inside a half block")
    return labels
