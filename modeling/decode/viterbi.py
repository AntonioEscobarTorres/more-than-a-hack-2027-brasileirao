"""Per-half Viterbi decoder with minimum dwell via an expanded-state chain (C5 / F2-02).

``decode_match(scores, half, force_d, params)`` decodes each contiguous ``half ∈ {1, 2}``
block independently, maps ``half == 0`` to ``X``, and finally forces ``force_d`` frames
to ``D``.

Durations (F2-08, decision D-025): after the ``min_dwell`` chain, a label holds through
``duration_stages[s]`` stages with self-loop ``p_stay[s]``, so the segment length is a shifted
negative binomial with the configured mean. One stage is the F2-01 geometric duration; the
sub-states, their order and the transition values are then exactly the original ones.
"""
from __future__ import annotations

import numpy as np

from modeling.contracts import LABEL_DTYPE, STATES
from modeling.decode.params import DecoderParams

_NEG_INF = -np.inf
_LOG_ONE_THIRD = np.log(1.0 / 3.0)


def _expand_topology(min_dwell: np.ndarray, duration_stages=None):
    """Build the expanded sub-state index and sparse predecessor lists.

    Sub-states are ordered H, A, D; within each label ``s``, indices run
    ``(s, 1) … (s, m_s − 1)`` (the ``min_dwell`` chain, no self-loop) followed by the
    ``r_s = duration_stages[s]`` stages ``(s, m_s) … (s, m_s + r_s − 1)``. The last stage is
    the *terminal* sub-state: it alone exits to other labels. With ``r_s == 1`` the layout is the
    F2-02 one. Returns ``(K, state_of, dwell_of, is_terminal, terminal_idx, offsets, preds)``
    where ``preds[j]`` is a list of ``(i, log_a_ij)`` predecessors of sub-state ``j``.
    """
    min_dwell = np.asarray(min_dwell, dtype=np.int64).reshape(3)
    if (min_dwell < 1).any():
        raise ValueError("min_dwell must be >= 1 for every state, got {}".format(min_dwell))
    stages = np.ones(3, dtype=np.int64) if duration_stages is None \
        else np.asarray(duration_stages, dtype=np.int64).reshape(3)
    if (stages < 1).any():
        raise ValueError("duration_stages must be >= 1 for every state, got {}".format(stages))
    length = min_dwell + stages - 1  # sub-states per label
    offsets = np.zeros(3, dtype=np.int64)
    offsets[1] = length[0]
    offsets[2] = length[0] + length[1]
    k_total = int(length.sum())
    if k_total > 32767:
        raise ValueError("expanded state count K={} exceeds int16 range".format(k_total))

    state_of = np.empty(k_total, dtype=np.int16)
    dwell_of = np.empty(k_total, dtype=np.int16)
    is_terminal = np.zeros(k_total, dtype=bool)
    terminal_idx = np.empty(3, dtype=np.int16)
    for s in range(3):
        n_sub = int(length[s])
        for k in range(1, n_sub + 1):
            j = int(offsets[s]) + (k - 1)
            state_of[j] = s
            dwell_of[j] = k
            if k == n_sub:
                is_terminal[j] = True
                terminal_idx[s] = j

    # preds[j] = list of (i, log_transition); built so i increases (argmax tie-break).
    preds = [[] for _ in range(k_total)]
    for s in range(3):
        m = int(min_dwell[s])
        base = int(offsets[s])
        # Chain: (s,k) → (s,k+1) with log-prob 0, up to the first stage (s, m).
        for k in range(1, m):
            preds[base + k].append((base + k - 1, 0.0))
        # Stage self-loops and stage advances (s, k) → (s, k+1), k >= m: filled by the
        # caller from log_p_stay / log(1 − p_stay).
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
    duration_stages=None,
    log_advance=None,
) -> None:
    """Fill stage, self-loop and exit edges on the predecessor lists (mutates ``preds``).

    Each stage ``(s, k)``, ``k >= m_s``, loops with ``log_p_stay[s]``; every stage but the last
    advances to the next with ``log_advance[s] = log(1 − p_stay[s])``; the last (terminal) stage
    exits with ``log_exit[s, ·]``. With one stage per state only the terminal edges exist.
    """
    stages = np.ones(3, dtype=np.int64) if duration_stages is None \
        else np.asarray(duration_stages, dtype=np.int64).reshape(3)
    if (stages > 1).any() and log_advance is None:
        raise ValueError("log_advance is required when a state has more than one stage")
    for s in range(3):
        term = int(terminal_idx[s])
        first_stage = term - int(stages[s]) + 1
        for j in range(first_stage, term):
            preds[j].append((j, float(log_p_stay[s])))
            preds[j + 1].append((j, float(log_advance[s])))
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


def _dense_predecessors(params: DecoderParams):
    """Predecessor table of the expanded chain as dense ``(K, P)`` arrays.

    Returns ``(state_of, terminal_idx, pred_src, pred_log)``. Row ``j`` lists the predecessors
    of sub-state ``j`` sorted by source index; unused slots point at the sentinel index ``K``
    (whose score is always -inf) with log-prob -inf. Every sub-state has at most 3 predecessors
    (the first stage: chain end or two exits, plus its self-loop).
    """
    min_dwell = params.min_dwell
    stages = params.duration_stages
    k_total, state_of, _dwell_of, _is_terminal, terminal_idx, offsets, preds = (
        _expand_topology(min_dwell, stages)
    )
    log_advance = None
    if (stages > 1).any():
        with np.errstate(divide="ignore"):
            log_advance = np.log(1.0 - params.p_stay)
    _attach_transition_logs(
        preds, min_dwell, offsets, terminal_idx, params.log_p_stay, params.log_exit,
        stages, log_advance,
    )
    width = max(1, max(len(row) for row in preds))
    pred_src = np.full((k_total, width), k_total, dtype=np.int64)
    pred_log = np.full((k_total, width), _NEG_INF, dtype=np.float64)
    for j, row in enumerate(preds):
        for col, (i, la) in enumerate(row):  # already sorted by source index
            pred_src[j, col] = i
            pred_log[j, col] = la
    return state_of, terminal_idx, pred_src, pred_log


def _viterbi_block(scores: np.ndarray, params: DecoderParams) -> np.ndarray:
    """Decode one contiguous half block. ``scores`` is float64 ``(T, 3)``.

    The recursion is vectorised over the sub-states: at each frame every sub-state takes the
    best of its (at most 3) predecessors with one gather and one ``argmax``. Predecessor
    columns are sorted by source index, so ``np.argmax`` keeps the smallest index on ties.
    """
    t_len = int(scores.shape[0])
    if t_len == 0:
        return np.empty(0, dtype=np.int8)

    state_of, terminal_idx, pred_src, pred_log = _dense_predecessors(params)
    k_total = int(state_of.shape[0])
    rows = np.arange(k_total)
    emit = scores[:, state_of.astype(np.int64)]  # (T, K): sub-state (s, k) emits scores[t, s]

    # delta has one extra sentinel slot (index K) that stays -inf for padded predecessors.
    delta = np.full(k_total + 1, _NEG_INF, dtype=np.float64)
    # Init: only terminal sub-states (the last stage of each s) get log(1/3) + scores[0, s]: a
    # half starts inside a segment that has already passed its min_dwell (D-025).
    for s in range(3):
        j = int(terminal_idx[s])
        delta[j] = _LOG_ONE_THIRD + scores[0, s]

    ptr = np.full((t_len, k_total), -1, dtype=np.int16)
    for t in range(1, t_len):
        candidates = delta[pred_src] + pred_log  # (K, P)
        best = np.argmax(candidates, axis=1)
        best_v = candidates[rows, best]
        ptr[t] = pred_src[rows, best]
        delta[:k_total] = best_v + emit[t]

    end = int(np.argmax(delta[:k_total]))
    path_sub = np.empty(t_len, dtype=np.int64)
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
    if not np.isfinite(scores).all():
        bad = np.flatnonzero(~np.isfinite(scores).all(axis=1))
        raise ValueError("scores must be finite (C3 never yields NaN/inf); first bad frame {}, "
                         "{} bad frames".format(int(bad[0]), int(bad.size)))
    if not np.isin(half, (0, 1, 2)).all():
        raise ValueError("half must only contain 0, 1 or 2")

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
