"""Synthetic tests for the hazard-guided H<->A transitions of the decoder (task F2-07, D-025).

Nothing here reads the match data or the external IDSSE files: scores are random or built by
hand, the master is ``preprocessing.make_mock_master`` and configs are validated in memory.
"""
from __future__ import annotations

import copy
import hashlib

import numpy as np
import pandas as pd
import pytest

from modeling import paths, pipeline, variants
from modeling.config import ConfigError, HMM_SECTIONS, load_config, validate_config
from modeling.contracts import STATES
from modeling.deadball.force import build_force_d
from modeling.decode import viterbi
from modeling.decode.emission import logits_to_logscores
from modeling.decode.params import A, D, H, DecoderParams, team_exit_logs
from modeling.decode.viterbi import decode_match
from modeling.pipeline import hazard_exit_logit, predict_labels
from modeling.scoring import hazard_sweep, sweep
from preprocessing.make_mock_master import generate_mock_master
from tests.modeling_fixtures import dead_params, team_params

REPO = paths.repo_root()
# sha256 of predict_labels(generate_mock_master(seed=42), configs/v1.yaml).tobytes(), computed on
# main at 185819a, before F2-07 (numpy 2.0.2, the pinned version).
V1_MOCK_LABELS_SHA256 = "249da6b1f87e08f403b857f5823d58211c1dded7dc4e13f735c7868be9441522"
CODE = {name: i for i, name in enumerate(STATES)}


def _params(min_dwell=(1, 1, 1), mean_extra=10.0, exit_split=None, fps=25.0) -> DecoderParams:
    """DecoderParams with controllable min_dwell; mean duration = dwell + mean_extra frames."""
    if exit_split is None:
        exit_split = {"H": {"A": 0.6, "D": 0.4}, "A": {"H": 0.7, "D": 0.3},
                      "D": {"H": 0.5, "A": 0.5}}
    cfg = {
        "lam_team": 0.2, "lam_dead": 0.2,
        "mean_duration_s": {name: (int(min_dwell[i]) + float(mean_extra)) / fps
                            for i, name in enumerate(STATES)},
        "exit_split": exit_split,
        "min_dwell_frames": {name: int(min_dwell[i]) for i, name in enumerate(STATES)},
    }
    return DecoderParams.from_config(cfg, fps=fps)


def _codes(labels) -> np.ndarray:
    table = dict(CODE, X=3)
    return np.asarray([table[x] for x in labels], dtype=np.int8)


def _sha(labels: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(labels).tobytes()).hexdigest()


# -- independent reference decoder --------------------------------------------------------------

def _reference_block(scores: np.ndarray, params: DecoderParams, x) -> np.ndarray:
    """Dense Viterbi over the expanded chain with a full (K, K) matrix per frame, built from the
    probabilities of the specification (``np.exp``/``np.log``, no ``logaddexp``)."""
    m = [int(v) for v in params.min_dwell]
    subs = [(s, k) for s in range(3) for k in range(1, m[s] + 1)]
    index = {sub: i for i, sub in enumerate(subs)}
    k_total = len(subs)
    p = np.exp(params.log_p_stay)
    q = np.exp(params.log_exit)                       # (3, 3), q[s, s'] = (1 - p_s) split[s, s']

    def matrix(xt):
        mat = np.zeros((k_total, k_total))
        for s in range(3):
            for k in range(1, m[s]):
                mat[index[(s, k)], index[(s, k + 1)]] = 1.0
            term = index[(s, m[s])]
            row = {s: p[s], **{o: q[s, o] for o in range(3) if o != s}}
            if xt is not None and s in (H, A):
                other = A if s == H else H
                row[other] = row[other] * np.exp(xt)
                z = sum(row.values())
                row = {k2: v / z for k2, v in row.items()}
            for dest, prob in row.items():
                j = term if dest == s else index[(dest, 1)]
                mat[term, j] += prob
        with np.errstate(divide="ignore"):
            return np.log(mat)

    const = matrix(None)
    t_len = scores.shape[0]
    delta = np.full(k_total, -np.inf)
    for s in range(3):
        delta[index[(s, m[s])]] = np.log(1.0 / 3.0) + scores[0, s]
    ptr = np.zeros((t_len, k_total), dtype=np.int64)
    state_of = np.array([s for s, _ in subs])
    for t in range(1, t_len):
        mat = const if x is None else matrix(x[t])
        cand = delta[:, None] + mat                    # (from, to)
        ptr[t] = np.argmax(cand, axis=0)
        delta = cand[ptr[t], np.arange(k_total)] + scores[t, state_of]
    path = np.empty(t_len, dtype=np.int64)
    path[-1] = int(np.argmax(delta))
    for t in range(t_len - 1, 0, -1):
        path[t - 1] = ptr[t, path[t]]
    return state_of[path].astype(np.int8)


@pytest.mark.parametrize("min_dwell", [(1, 1, 1), (3, 2, 4), (1, 4, 2), (5, 5, 1)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_modulated_decoder_matches_dense_reference(min_dwell, seed):
    rng = np.random.default_rng(100 * seed + sum(min_dwell))
    n = 160
    scores = rng.normal(scale=1.5, size=(n, 3))
    x = rng.normal(scale=3.0, size=n)
    params = _params(min_dwell=min_dwell, mean_extra=6.0)
    labels = decode_match(scores, np.ones(n, np.int8), np.zeros(n, bool), params, exit_logit=x)
    np.testing.assert_array_equal(_codes(labels), _reference_block(scores, params, x))
    # and the reference without modulation is the existing decoder
    labels0 = decode_match(scores, np.ones(n, np.int8), np.zeros(n, bool), params)
    np.testing.assert_array_equal(_codes(labels0), _reference_block(scores, params, None))


def test_modulated_decoder_with_a_zero_team_exit_matches_reference():
    """exit_split H->A = 0: the H->A edge stays absent, only H's other edges are renormalised."""
    split = {"H": {"A": 0.0, "D": 1.0}, "A": {"H": 0.7, "D": 0.3}, "D": {"H": 0.5, "A": 0.5}}
    params = _params(min_dwell=(2, 3, 2), mean_extra=5.0, exit_split=split)
    rng = np.random.default_rng(7)
    n = 120
    scores = rng.normal(scale=1.5, size=(n, 3))
    x = rng.normal(scale=3.0, size=n)
    labels = decode_match(scores, np.ones(n, np.int8), np.zeros(n, bool), params, exit_logit=x)
    codes = _codes(labels)
    np.testing.assert_array_equal(codes, _reference_block(scores, params, x))
    assert not ((codes[:-1] == H) & (codes[1:] == A)).any()


# -- transition rows ----------------------------------------------------------------------------

@pytest.mark.parametrize("split", [
    None,
    {"H": {"A": 0.0, "D": 1.0}, "A": {"H": 1.0, "D": 0.0}, "D": {"H": 0.5, "A": 0.5}},
])
def test_transition_rows_sum_to_one_on_every_frame(split):
    params = _params(min_dwell=(12, 12, 25), mean_extra=150.0, exit_split=split)
    x = np.concatenate([np.linspace(-60.0, 60.0, 241), [0.0, -1e-9, 1e-9, 700.0, -700.0]])
    logs = team_exit_logs(params, x)
    for s in (H, A):
        total = sum(np.exp(part) for part in logs[s])
        np.testing.assert_allclose(total, 1.0, rtol=0.0, atol=1e-12)
        assert all(part.dtype == np.float64 and part.shape == x.shape for part in logs[s])


def test_transition_rows_follow_the_formula():
    params = _params(min_dwell=(3, 3, 3), mean_extra=20.0)
    x = np.array([-2.0, 0.0, 1.5])
    logs = team_exit_logs(params, x)
    for s, other in ((H, A), (A, H)):
        p = np.exp(params.log_p_stay[s])
        qo, qd = np.exp(params.log_exit[s, other]), np.exp(params.log_exit[s, D])
        z = p + qd + qo * np.exp(x)
        np.testing.assert_allclose(np.exp(logs[s][0]), p / z, rtol=1e-13)
        np.testing.assert_allclose(np.exp(logs[s][1]), qo * np.exp(x) / z, rtol=1e-13)
        np.testing.assert_allclose(np.exp(logs[s][2]), qd / z, rtol=1e-13)
    # x = 0 gives back the constant row (up to the rounding of Z = 1)
    np.testing.assert_allclose(logs[H][0][1], params.log_p_stay[H], atol=1e-14)
    np.testing.assert_allclose(logs[H][1][1], params.log_exit[H, A], atol=1e-14)


def test_transition_rows_reject_non_finite_or_2d():
    params = _params()
    with pytest.raises(ValueError, match="finite"):
        team_exit_logs(params, np.array([0.0, np.nan]))
    with pytest.raises(ValueError, match="one-dimensional"):
        team_exit_logs(params, np.zeros((2, 2)))


def test_only_the_six_team_exit_slots_are_modulated():
    params = _params(min_dwell=(2, 3, 4))
    state_of, terminal_idx, pred_src, _ = viterbi._dense_predecessors(params)
    rows, cols, keys = viterbi._team_exit_slots(state_of, terminal_idx, pred_src)
    assert len(rows) == 6 and keys == [(H, 0), (H, 1), (H, 2), (A, 0), (A, 1), (A, 2)]
    for (s, j), r, c in zip(keys, rows, cols):
        assert pred_src[r, c] == terminal_idx[s]      # every slot leaves a terminal of H or A
    assert terminal_idx[D] not in set(pred_src[rows, cols].tolist())


# -- regression and equivalence -----------------------------------------------------------------

@pytest.fixture(scope="module")
def mock_master() -> pd.DataFrame:
    return generate_mock_master(seed=42)


@pytest.fixture(scope="module")
def v1_cfg() -> dict:
    return load_config(REPO / "configs" / "v1.yaml")


def _scores(master, cfg):
    variants.load_kinds()
    team = variants.TEAM_KINDS[cfg["team"]["kind"]](master, cfg["team"]["params"])
    dead = variants.DEAD_KINDS[cfg["dead"]["kind"]](master, cfg["dead"]["params"])
    force_d = build_force_d(master)
    params = DecoderParams.from_config(cfg["decoder"])
    scores = logits_to_logscores(team, dead, params.lam_team, params.lam_dead, force_d,
                                 neg=params.neg)
    return scores, master["half"].to_numpy(), force_d, params


def test_v1_labels_are_byte_identical_to_main(mock_master, v1_cfg):
    """kind none (the configs as committed) and exit_logit None give the labels of main."""
    assert v1_cfg["hazard"] == {"kind": "none", "calibrated_file": None, "beta": 0.0}
    labels = predict_labels(mock_master, v1_cfg)
    assert _sha(labels) == V1_MOCK_LABELS_SHA256
    scores, half, force_d, params = _scores(mock_master, v1_cfg)
    direct = decode_match(scores, half, force_d, params, exit_logit=None)
    assert _sha(pipeline.final_imposition(direct, half, force_d)[0]) == V1_MOCK_LABELS_SHA256


def test_calibrated_kind_with_beta_zero_is_the_constant_path(mock_master, v1_cfg, monkeypatch):
    """beta == 0 passes None: the hazard is not even computed."""
    cfg = copy.deepcopy(v1_cfg)
    cfg["hazard"] = {"kind": "calibrated", "calibrated_file": "configs/calib/hazard.yaml",
                     "beta": 0.0}
    variants.load_kinds()

    def boom(*_a, **_k):
        raise AssertionError("the hazard must not be computed when it is off")

    monkeypatch.setitem(variants.HAZARD_KINDS, "calibrated", boom)
    assert hazard_exit_logit(mock_master, cfg) is None
    assert _sha(predict_labels(mock_master, validate_config(cfg))) == V1_MOCK_LABELS_SHA256


def test_zero_exit_logit_equals_none(mock_master, v1_cfg):
    """All-zero exit_logit gives the same labels as None. Z_t = p + q_d + q_o rounds to 1 within
    1e-16, so a difference could only come from an exact tie; none occurs on these inputs."""
    scores, half, force_d, params = _scores(mock_master, v1_cfg)
    base = decode_match(scores, half, force_d, params)
    zero = decode_match(scores, half, force_d, params, exit_logit=np.zeros(len(half)))
    np.testing.assert_array_equal(zero, base)
    rng = np.random.default_rng(3)
    for dwell in ((1, 1, 1), (12, 12, 25)):
        p = _params(min_dwell=dwell, mean_extra=40.0)
        sc = rng.normal(size=(3000, 3))
        hb = np.ones(3000, np.int8)
        fd = np.zeros(3000, bool)
        np.testing.assert_array_equal(decode_match(sc, hb, fd, p, exit_logit=np.zeros(3000)),
                                      decode_match(sc, hb, fd, p))


# -- behaviour on hand-built cases --------------------------------------------------------------

def _late_switch_case():
    """H clearly until frame 100, then 60 frames that lean slightly to H, then A clearly.

    Without modulation the switch comes at frame 160 (late); the true change is at 110.
    """
    n = 260
    scores = np.zeros((n, 3))
    scores[:, D] = -8.0
    scores[:100, H], scores[:100, A] = 0.0, -4.0
    scores[100:160, H], scores[100:160, A] = 0.0, -0.05
    scores[160:, H], scores[160:, A] = -4.0, 0.0
    return scores, _params(min_dwell=(12, 12, 25), mean_extra=150.0)


def test_exit_logit_spike_pulls_the_switch_to_its_frame():
    scores, params = _late_switch_case()
    n = scores.shape[0]
    half, fd = np.ones(n, np.int8), np.zeros(n, bool)
    base = _codes(decode_match(scores, half, fd, params))
    assert base[159] == H and base[160] == A and set(base.tolist()) == {H, A}
    x = np.zeros(n)
    x[110] = 10.0
    spiked = _codes(decode_match(scores, half, fd, params, exit_logit=x))
    # convention: the transition into frame t uses exit_logit[t], so frame 110 is the first A
    assert spiked[109] == H and spiked[110] == A and (spiked[110:] == A).all()
    assert (spiked[:110] == H).all()


def test_spike_one_frame_early_or_late_moves_the_switch_with_it():
    scores, params = _late_switch_case()
    n = scores.shape[0]
    half, fd = np.ones(n, np.int8), np.zeros(n, bool)
    for t0 in (105, 111, 140):
        x = np.zeros(n)
        x[t0] = 10.0
        codes = _codes(decode_match(scores, half, fd, params, exit_logit=x))
        assert int(np.flatnonzero(codes == A)[0]) == t0


def test_very_negative_exit_logit_blocks_direct_switch_but_not_through_d():
    n = 300
    scores = np.zeros((n, 3))
    scores[:150, H], scores[:150, A], scores[:150, D] = 0.0, -3.0, -1.0
    scores[150:, H], scores[150:, A], scores[150:, D] = -3.0, 0.0, -1.0
    params = _params(min_dwell=(12, 12, 3), mean_extra=150.0)
    half, fd = np.ones(n, np.int8), np.zeros(n, bool)
    free = _codes(decode_match(scores, half, fd, params))
    assert ((free[:-1] == H) & (free[1:] == A)).sum() == 1 and D not in free
    blocked = _codes(decode_match(scores, half, fd, params, exit_logit=np.full(n, -60.0)))
    direct = ((blocked[:-1] == H) & (blocked[1:] == A)) | ((blocked[:-1] == A) & (blocked[1:] == H))
    assert not direct.any()
    assert blocked[0] == H and blocked[-1] == A and D in blocked   # H -> D -> A
    first_a = int(np.flatnonzero(blocked == A)[0])
    assert (blocked[first_a - 3:first_a] == D).all()


def test_d_chain_and_d_exits_are_not_modulated():
    """A D segment between two teams is decoded the same with any exit_logit on D frames."""
    n = 200
    scores = np.zeros((n, 3))
    scores[:80, H], scores[:80, A], scores[:80, D] = 0.0, -5.0, -5.0
    scores[80:120, H], scores[80:120, A], scores[80:120, D] = -5.0, -5.0, 0.0
    scores[120:, H], scores[120:, A], scores[120:, D] = -5.0, 0.0, -5.0
    params = _params(min_dwell=(5, 5, 5), mean_extra=100.0)
    half, fd = np.ones(n, np.int8), np.zeros(n, bool)
    base = decode_match(scores, half, fd, params)
    x = np.zeros(n)
    x[82:118] = np.random.default_rng(0).normal(scale=20.0, size=36)  # inside the D run
    np.testing.assert_array_equal(decode_match(scores, half, fd, params, exit_logit=x), base)


# -- halves, determinism, validation ------------------------------------------------------------

def test_nothing_crosses_half_zero_and_first_frame_of_a_block_is_unused():
    rng = np.random.default_rng(11)
    n1, gap, n2 = 150, 20, 170
    n = n1 + gap + n2
    scores = rng.normal(scale=1.5, size=(n, 3))
    half = np.r_[np.ones(n1), np.zeros(gap), np.full(n2, 2)].astype(np.int8)
    fd = np.zeros(n, bool)
    params = _params(min_dwell=(3, 3, 3), mean_extra=10.0)
    x = rng.normal(scale=3.0, size=n)
    labels = decode_match(scores, half, fd, params, exit_logit=x)
    assert (labels[n1:n1 + gap] == "X").all()
    # each block equals the block decoded alone
    for lo, hi in ((0, n1), (n1 + gap, n)):
        alone = decode_match(scores[lo:hi], np.ones(hi - lo, np.int8), np.zeros(hi - lo, bool),
                             params, exit_logit=x[lo:hi])
        np.testing.assert_array_equal(labels[lo:hi], alone)
    # values at half == 0 (even NaN) and at the first frame of each block do not matter
    y = x.copy()
    y[n1:n1 + gap] = np.nan
    y[0] += 50.0
    y[n1 + gap] -= 50.0
    np.testing.assert_array_equal(decode_match(scores, half, fd, params, exit_logit=y), labels)


def test_modulated_decoder_is_deterministic():
    rng = np.random.default_rng(5)
    n = 2000
    scores = rng.normal(size=(n, 3))
    half = np.r_[np.ones(900), np.zeros(100), np.full(1000, 2)].astype(np.int8)
    fd = rng.random(n) < 0.05
    x = rng.normal(scale=2.0, size=n)
    params = _params(min_dwell=(12, 12, 25), mean_extra=100.0)
    a = decode_match(scores, half, fd, params, exit_logit=x)
    b = decode_match(scores, half, fd, params, exit_logit=x.copy())
    np.testing.assert_array_equal(a, b)
    assert (a[fd] == "D").all()


def test_decode_match_validates_exit_logit():
    params = _params()
    scores, half, fd = np.zeros((10, 3)), np.ones(10, np.int8), np.zeros(10, bool)
    with pytest.raises(ValueError, match="shape"):
        decode_match(scores, half, fd, params, exit_logit=np.zeros(9))
    bad = np.zeros(10)
    bad[4] = np.inf
    with pytest.raises(ValueError, match="finite"):
        decode_match(scores, half, fd, params, exit_logit=bad)


# -- config -------------------------------------------------------------------------------------

def _cfg(**hazard) -> dict:
    cfg = load_config(REPO / "configs" / "v1.yaml")
    cfg["hazard"].update(hazard)
    return cfg


def test_hazard_is_a_required_hmm_section():
    assert "hazard" in HMM_SECTIONS
    cfg = _cfg()
    del cfg["hazard"]
    with pytest.raises(ConfigError, match="missing keys"):
        validate_config(cfg)
    cfg = _cfg()
    del cfg["hazard"]["beta"]
    with pytest.raises(ConfigError, match="missing keys"):
        validate_config(cfg)
    with pytest.raises(ConfigError, match="unknown keys"):
        validate_config(_cfg(params={}))


@pytest.mark.parametrize("hazard, message", [
    ({"kind": "calibrated"}, "hazard.calibrated_file is required"),
    ({"kind": "bogus", "calibrated_file": "x.yaml"}, "must be one of"),
    ({"beta": -0.1}, "must be >= 0"),
    ({"beta": True}, "expected a number"),
    ({"beta": float("nan")}, "finite"),
])
def test_hazard_section_is_validated(hazard, message):
    with pytest.raises(ConfigError, match=message):
        validate_config(_cfg(**hazard))


def test_beta_is_normalised_to_float():
    cfg = validate_config(_cfg(kind="calibrated", calibrated_file="configs/calib/hazard.yaml",
                               beta=1))
    assert cfg["hazard"]["beta"] == 1.0 and isinstance(cfg["hazard"]["beta"], float)


def test_every_committed_hmm_config_has_the_hazard_section():
    seen = 0
    for path in sorted((REPO / "configs").glob("*.yaml")):
        if path.name.startswith(("degrade_", "broadcast", "hazard_fit", "decoder_", "plausib")):
            continue
        cfg = load_config(path)
        if cfg["pipeline"] != "hmm":
            continue
        seen += 1
        hz = cfg["hazard"]
        assert hz["kind"] in ("none", "calibrated", "gbdt")
        if hz["kind"] != "none":
            assert (REPO / hz["calibrated_file"]).is_file()
        for name in ("stack", "dead_stack"):
            if cfg.get(name) is not None and cfg[name]["kind"] != "none":
                assert (REPO / cfg[name]["calibrated_file"]).is_file()
    assert seen >= 5


# -- pipeline -----------------------------------------------------------------------------------

def _stub_hazard(record):
    def fn(master, params, *, holdout=None):
        record.append((dict(params), holdout))
        half = master["half"].to_numpy()
        h = np.sin(np.arange(len(master)) / 7.0).astype(np.float32) * 6.0
        return np.where(half != 0, h, 0.0).astype(np.float32)
    return fn


def test_pipeline_passes_beta_times_hazard(mock_master, v1_cfg, monkeypatch):
    variants.load_kinds()
    record = []
    monkeypatch.setitem(variants.HAZARD_KINDS, "calibrated", _stub_hazard(record))
    cfg = validate_config(_cfg(kind="calibrated", calibrated_file="configs/calib/hazard.yaml",
                               beta=0.5))
    labels = predict_labels(mock_master, cfg, holdout="J03WMX")
    params, holdout = record[-1]
    assert holdout == "J03WMX"
    assert params == {"calibrated_file": str(REPO / "configs" / "calib" / "hazard.yaml")}
    h = _stub_hazard([])(mock_master, {})
    scores, half, force_d, dparams = _scores(mock_master, cfg)
    manual = decode_match(scores, half, force_d, dparams, exit_logit=0.5 * h.astype(np.float64))
    np.testing.assert_array_equal(labels, pipeline.final_imposition(manual, half, force_d)[0])
    assert _sha(labels) != V1_MOCK_LABELS_SHA256          # the hazard does reach the decoder


@pytest.mark.parametrize("corrupt, message", [
    (lambda h, half: h[:-1], "shape"),
    (lambda h, half: np.where(np.arange(len(h)) == 5, np.nan, h), "finite"),
    (lambda h, half: np.where(half == 0, 1.0, h), "half == 0"),
])
def test_pipeline_rejects_a_bad_hazard(mock_master, monkeypatch, corrupt, message):
    variants.load_kinds()
    half = mock_master["half"].to_numpy()

    def bad(master, params, *, holdout=None):
        return corrupt(_stub_hazard([])(master, params), half)

    monkeypatch.setitem(variants.HAZARD_KINDS, "calibrated", bad)
    cfg = validate_config(_cfg(kind="calibrated", calibrated_file="configs/calib/hazard.yaml",
                               beta=1.0))
    with pytest.raises(ValueError, match=message):
        hazard_exit_logit(mock_master, cfg)


def test_pipeline_with_the_committed_hazard_file_runs_on_the_mock(mock_master):
    cfg = validate_config(_cfg(kind="calibrated", calibrated_file="configs/calib/hazard.yaml",
                               beta=1.0))
    a = predict_labels(mock_master, cfg)
    b = predict_labels(mock_master, cfg)
    np.testing.assert_array_equal(a, b)
    half = mock_master["half"].to_numpy()
    assert (a[half == 0] == "X").all() and not (a[half != 0] == "X").any()
    x = hazard_exit_logit(mock_master, cfg)
    assert x.dtype == np.float64 and np.isfinite(x).all() and (x[half == 0] == 0).all()


# -- sweep --------------------------------------------------------------------------------------

def _fake_eval(table):
    """evaluate_fn whose spellF1 depends on (beta, lam_team) only; SE 0.002."""
    calls = []

    def fn(cfg, degrade):
        calls.append((cfg["hazard"]["beta"], cfg["decoder"]["lam_team"], degrade))
        mean = table(cfg["hazard"]["beta"], cfg["decoder"]["lam_team"], degrade)
        return {"mean": mean, "se": 0.002, sweep.SPELL_RATIO: 0.7}

    fn.calls = calls
    return fn


def test_sweep_beta_needs_the_hazard_on(v1_cfg):
    with pytest.raises(ValueError, match="hazard.kind"):
        sweep.sweep_beta(v1_cfg, evaluate_fn=_fake_eval(lambda *a: 0.3))


def test_hazard_sweep_picks_the_plateau_then_resweeps_lambda(v1_cfg):
    def table(beta, lam, degrade):
        bump = {"default": 0.0, "hidelo": -0.02, "hidehi": 0.01}[degrade]
        return 0.30 + 0.01 * min(beta, 0.5) - 0.02 * abs(lam - 0.4) + bump

    fake = _fake_eval(table)
    hist = hazard_sweep.run_hazard_sweep(v1_cfg, evaluate_fn=fake, lam_grid=(0.2, 0.4, 0.8))
    beta_stage, lam_stage = hist["stages"]
    assert [p["beta"] for p in beta_stage["points"]] == list(sweep.BETA_GRID)
    assert beta_stage["selected"]["beta"] in (0.5, 1.0)          # plateau, robust over modes
    assert hist["selected_cfg"]["hazard"]["kind"] == "calibrated"
    assert hist["selected_cfg"]["hazard"]["beta"] == beta_stage["selected"]["beta"]
    assert hist["selected_cfg"]["decoder"]["lam_team"] == 0.4
    assert len(lam_stage["points"]) == 9                         # 3 x 3, lam_dead grid = lam grid
    assert {d for _, _, d in fake.calls} == set(sweep.ROBUST_DEGRADES)


def test_hazard_sweep_with_beta_zero_skips_lambda(v1_cfg):
    fake = _fake_eval(lambda beta, lam, deg: 0.4 - 0.05 * beta)
    hist = hazard_sweep.run_hazard_sweep(v1_cfg, evaluate_fn=fake, lam_grid=(0.2, 0.4),
                                         lam_dead_grid=(0.2,))
    assert [s["name"] for s in hist["stages"]] == ["beta"]
    assert hist["selected_cfg"] is hist["base"]
    assert hist["base"]["hazard"]["kind"] == "none"


def test_hazard_sweep_rejects_bad_inputs(v1_cfg):
    fake = _fake_eval(lambda *a: 0.3)
    with pytest.raises(ValueError, match="contain 0"):
        hazard_sweep.run_hazard_sweep(v1_cfg, evaluate_fn=fake, beta_grid=(0.5, 1.0))
    on = validate_config(_cfg(kind="calibrated", calibrated_file="configs/calib/hazard.yaml",
                              beta=0.5))
    with pytest.raises(ValueError, match="hazard off"):
        hazard_sweep.run_hazard_sweep(on, evaluate_fn=fake)


def test_sweep_lambdas_accepts_a_separate_dead_grid(v1_cfg):
    cfg = validate_config(_cfg(kind="calibrated", calibrated_file="configs/calib/hazard.yaml",
                               beta=0.5))
    fake = _fake_eval(lambda *a: 0.3)
    points = sweep.sweep_lambdas(cfg, grid=(0.2, 0.3), dead_grid=(0.2,), evaluate_fn=fake)
    assert [(p["lam_team"], p["lam_dead"]) for p in points] == [(0.2, 0.2), (0.3, 0.2)]
    assert all(p["cfg"]["hazard"]["beta"] == 0.5 for p in points)


def test_latency_summary_and_written_config_round_trip(tmp_path, v1_cfg):
    out = hazard_sweep.latency_summary(np.array([-3.0, 1.0, 5.0, -10.0]))
    assert out == {"median_latency": -1.0, "median_abs_latency": 4.0, "n_pairs": 4}
    assert np.isnan(hazard_sweep.latency_summary(np.array([]))["median_latency"])
    cfg = validate_config(_cfg(kind="calibrated", calibrated_file="configs/calib/hazard.yaml",
                               beta=0.25))
    path = hazard_sweep.write_config(cfg, tmp_path / "v1_hz.yaml", "# header\n")
    back = load_config(path)
    assert back["name"] == "v1_hz" and back["hazard"] == cfg["hazard"]
    assert path.read_text(encoding="utf-8").startswith("# header\n")


class _FakeEvaluator:
    """``details`` with a synthetic per-match table whose spellF1 grows with beta."""

    matches = ["m1", "m2", "m3"]

    def details(self, cfg, degrade):
        beta = cfg["hazard"]["beta"] if cfg["hazard"]["kind"] != "none" else 0.0
        rows = [{"spell_f1_mean": 0.40 + 0.01 * beta + 0.001 * i, "n_gt_spells": 100.0,
                 "n_pred_spells": 70.0 + i} for i in range(3)]
        df = pd.DataFrame(rows, index=pd.Index(self.matches, name="match"))
        summary = pd.DataFrame([df.mean(), df.std(ddof=1) / np.sqrt(3)], index=["mean", "se"])
        return {"frame": pd.concat([df, summary]), "lat": np.array([-2.0, 0.0, 3.0])}

    def __call__(self, cfg, degrade):
        return sweep.summarize(self.details(cfg, degrade)["frame"])


def test_report_renders_stages_summary_and_extra_section(v1_cfg):
    ev = _FakeEvaluator()
    hist = hazard_sweep.run_hazard_sweep(v1_cfg, evaluate_fn=ev, lam_grid=(0.2, 0.4),
                                         lam_dead_grid=(0.2,))
    configs = {"base (hazard off)": hist["base"], "beta, base λ": hist["beta_cfg"],
               "beta + λ re-swept": hist["selected_cfg"]}
    summary = hazard_sweep.summary_rows(ev, configs, ["default", "hidelo"])
    assert list(summary["config"][:3]) == list(configs)
    base_rows = summary[summary["config"] == "base (hazard off)"]
    assert (base_rows["delta vs base"] == 0.0).all()
    best = summary[summary["config"] == "beta, base λ"].iloc[0]
    assert best["delta vs base"] == pytest.approx(0.01 * hist["beta_cfg"]["hazard"]["beta"])
    assert best["matches better"] == "3/3" and best["median_latency"] == 0.0
    extra_cfgs = hazard_sweep.extra_beta_configs(hist, [2.0, 4.0])
    assert list(extra_cfgs) == ["base (hazard off)", "beta = 2", "beta = 4"]
    assert extra_cfgs["beta = 4"]["decoder"] == hist["base"]["decoder"]
    with pytest.raises(ValueError, match="> 0"):
        hazard_sweep.extra_beta_configs(hist, [0.0])
    extra = hazard_sweep.summary_rows(ev, extra_cfgs, ["default"])
    text = hazard_sweep.render_report(hist, summary, None, extra=extra, args={
        "config": "configs/v1.yaml", "hazard_file": hazard_sweep.HAZARD_FILE,
        "degrade": "default", "robust": True, "beta_grid": list(sweep.BETA_GRID),
        "lam_grid": [0.2, 0.4], "lam_dead_grid": [0.2], "out_config": None})
    for heading in ("## Stage: beta", "## Stage: lam", "## Summary per degrade mode",
                    "## Beyond the pre-registered grid"):
        assert heading in text
    assert "beta = 4" in text and "## Decoder time" not in text


def test_nodir_hazard_file_matches_its_fit_config():
    """configs/calib/hazard_nodir.yaml = the F1-08 fit without ball_dir_change (D-025)."""
    from modeling.calibration.hazard import load_fit_config
    from modeling.evidence.hazard import design_columns, load_hazard_file

    fit = load_fit_config(REPO / "configs" / "hazard_fit_nodir.yaml")
    base = load_fit_config(REPO / "configs" / "hazard_fit.yaml")
    assert fit["features"] == [f for f in base["features"] if f != "ball_dir_change"]
    assert {k: v for k, v in fit.items() if k != "features"} == \
        {k: v for k, v in base.items() if k != "features"}
    calib = load_hazard_file(REPO / "configs" / "calib" / "hazard_nodir.yaml")
    assert calib["features"] == fit["features"]
    assert "ball_dir_change" not in design_columns(calib["features"])
    assert sorted(calib["folds"]) == sorted(load_hazard_file(
        REPO / "configs" / "calib" / "hazard.yaml")["folds"])
