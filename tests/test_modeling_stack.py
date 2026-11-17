"""F2-09 (D-029): GBDT hazard, stacked team_logit, their calibration and the pipeline wiring.

Synthetic data only (never the match data).
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from modeling import variants
from modeling.calibration import stack_gbdt as cal
from modeling.calibration.team import calibrate_team, write_team_yaml
from modeling.calibration.team_gbdt import MODEL_PARAMS, calibrate_team_gbdt, write_gbdt_yaml
from modeling.config import ConfigError, config_hash, load_config, validate_config
from modeling.contracts import TEAM_LOGIT_MAX
from modeling.evidence import hazard_gbdt as hg
from modeling.evidence import stack as st
from modeling.pipeline import hazard_exit_logit, hazard_values, predict_labels
from tests.modeling_fixtures import swap_teams, team_params
from tests.test_modeling_hazard import hazard_master
from tests.test_modeling_team_gbdt import _full_master

REPO_CFG = "configs/final_d028.yaml"
SMALL = {**MODEL_PARAMS, "max_iter": 15, "min_samples_leaf": 5}
CONTROL = {**team_params(), "bridge_median_frames": 7, "r_touch_m": 1.5, "touch_min_frames": 5,
           "touch_win_frames": 9, "touch_dir_change_rad": 0.5, "touch_acc_mps2": 8.0,
           "touch_require_abrupt": False}
SETTINGS = {"acc_clip_mps2": 100.0, "dist_clip_m": 30.0, "gap_clip_frames": 1500.0,
            "h_clip": 8.0, "team_params": team_params()}


def _two_halves(n):
    half = np.ones(n, dtype=np.int8)
    half[n // 2 - 5:n // 2 + 5] = 0
    half[n // 2 + 5:] = 2
    return half


# --- GBDT hazard features ----------------------------------------------------------------------

def test_hazard_gbdt_features_are_even_under_the_team_swap():
    m = hazard_master(400, seed=3, half=_two_halves(400))
    f = hg.build_hazard_gbdt_features(m, SETTINGS, CONTROL)
    fs = hg.build_hazard_gbdt_features(swap_teams(m), SETTINGS, CONTROL)
    assert list(f.columns) == hg.hazard_gbdt_feature_names()
    np.testing.assert_allclose(fs.to_numpy(), f.to_numpy(), rtol=1e-12, atol=1e-12)


def test_hazard_gbdt_features_are_nan_off_the_halves_and_stay_in_their_half():
    n = 400
    half = _two_halves(n)
    m = hazard_master(n, seed=4, half=half)
    f = hg.build_hazard_gbdt_features(m, SETTINGS, CONTROL).to_numpy()
    assert np.isnan(f[half == 0]).all()
    # Changing the second half leaves every first-half feature unchanged.
    m2 = m.copy()
    second = np.flatnonzero(half == 2)
    m2.loc[second, "d_near_home"] = m2.loc[second, "d_near_home"] + 3.0
    m2.loc[second, "d_diff"] = m2.loc[second, "d_near_home"] - m2.loc[second, "d_near_away"]
    f2 = hg.build_hazard_gbdt_features(m2, SETTINGS, CONTROL).to_numpy()
    first = half == 1
    np.testing.assert_array_equal(np.isnan(f2[first]), np.isnan(f[first]))
    np.testing.assert_allclose(f2[first], f[first], equal_nan=True)


def test_ema_pair_is_causal_and_anticausal_per_block():
    half = np.array([1, 1, 1, 0, 2, 2, 2], dtype=np.int8)
    c = np.array([1.0, 0, 0, 0, 0, 0, 1.0])
    b, f = hg.ema_pair(c, half, 2.0)
    np.testing.assert_allclose(b[:3], [0.5, 0.25, 0.125])
    assert f[0] == pytest.approx(0.5) and f[1] == 0 and f[2] == 0
    np.testing.assert_allclose(f[4:], [0.125, 0.25, 0.5])
    assert b[4] == 0 and b[5] == 0


# --- stacked team_logit ------------------------------------------------------------------------

def test_stack_features_flip_with_t_and_keep_h():
    rng = np.random.default_rng(0)
    n = 300
    half = _two_halves(n)
    t, h = rng.normal(size=n), rng.normal(size=n)
    X = st.build_stack_features(t, h, half)
    Xs = st.build_stack_features(-t, h, half)
    assert X.shape == (n, len(st.stack_feature_names()))
    np.testing.assert_array_equal(Xs, X * st.stack_sign())
    assert np.isnan(X[half == 0]).all()


def test_stacked_logit_is_exactly_antisymmetric_and_clipped():
    rng = np.random.default_rng(1)
    n = 600
    half = _two_halves(n)
    t = np.where((np.arange(n) // 50) % 2 == 0, 2.0, -2.0) + rng.normal(size=n)
    h = rng.normal(size=n)
    X = st.build_stack_features(t, h, half)
    y = (t > 0).astype(np.int8)
    model = cal.fit_antisymmetric(X[half != 0], y[half != 0], None, st.stack_sign(), SMALL)
    t2 = st.stack_from_model(model, t, h, half)
    t2s = st.stack_from_model(model, -t, h, half)
    assert t2.dtype == np.float32
    np.testing.assert_array_equal(t2s, -t2)
    assert (t2[half == 0] == 0).all()
    assert np.abs(t2).max() <= TEAM_LOGIT_MAX
    assert np.mean(np.sign(t2[half != 0]) == np.sign(t[half != 0])) > 0.8


# --- config -------------------------------------------------------------------------------------

def _raw_cfg():
    import yaml
    with open(REPO_CFG, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw["name"] = "x"
    return raw


def test_stack_section_is_optional_and_absent_from_the_hash():
    raw = _raw_cfg()
    cfg = validate_config(raw)
    assert "stack" not in cfg
    with_none = validate_config({**copy.deepcopy(raw),
                                 "stack": {"kind": "none", "calibrated_file": None}})
    assert with_none["stack"]["kind"] == "none"
    assert config_hash(cfg) != config_hash(with_none)


def test_stack_needs_a_file_and_a_hazard():
    raw = _raw_cfg()
    bad = {**copy.deepcopy(raw), "stack": {"kind": "gbdt", "calibrated_file": None}}
    with pytest.raises(ConfigError, match="stack.calibrated_file"):
        validate_config(bad)
    bad = {**copy.deepcopy(raw), "stack": {"kind": "gbdt", "calibrated_file": "x.yaml"}}
    bad["hazard"] = {"kind": "none", "calibrated_file": None, "beta": 0.0}
    with pytest.raises(ConfigError, match="needs a hazard"):
        validate_config(bad)
    with pytest.raises(ConfigError, match="unknown keys"):
        validate_config({**copy.deepcopy(raw), "stack": {"kind": "gbdt", "calibrated_file": "a",
                                                          "beta": 1.0}})


def test_repository_configs_still_load():
    for name in ("final", "final_d028", "v2_gbdt_fused_hzn2"):
        cfg = load_config("configs/{}.yaml".format(name))
        assert cfg["name"] == name


# --- calibration and pipeline end to end --------------------------------------------------------

def _match(seed):
    master, gt = _full_master(400, seed=seed)
    rng = np.random.default_rng(seed + 7)
    has_ball = np.isfinite(master["ball_x"].to_numpy())
    master["ball_acc"] = np.where(has_ball, rng.uniform(0, 150, len(master)), np.nan)
    master["ball_dir_change"] = np.where(has_ball, rng.uniform(0, np.pi, len(master)), np.nan)
    master["gap_length"] = np.where(has_ball, 0.0, 5.0)
    half = master["half"].to_numpy()
    master.loc[half == 0, ["ball_acc", "ball_dir_change", "gap_length"]] = np.nan
    return master, gt


@pytest.fixture(scope="module")
def stacked(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("stack")
    matches = {"m1": _match(10), "m2": _match(20), "m3": _match(30)}
    load = lambda match, degrade: matches[match]  # noqa: E731
    base = {"name": "t", "team": {"kind": "rules_v1", "params": team_params(),
                                  "calibrated_file": None}}
    logistic = calibrate_team(base, variant="touch", matches=list(matches), load_fn=load,
                              touch_params={"r_touch_m": 1.5, "touch_min_frames": 5})
    logistic_path = write_team_yaml(logistic, tmp / "team.yaml")
    table = calibrate_team_gbdt(base, logistic_file=logistic_path, matches=list(matches), step=1,
                                window_sizes=[3, 6], model_params=SMALL,
                                all_model_path=tmp / "team_all.joblib", fold_dir=tmp / "team_folds",
                                load_fn=load, log=lambda *_: None)
    team_path = write_gbdt_yaml(table, tmp / "team_gbdt.yaml", gap_model="logistic")
    cfg = load_config(REPO_CFG)
    cfg["team"]["calibrated_file"] = str(team_path)
    cfg["team"]["params"] = team_params()
    cfg["dead"]["kind"], cfg["dead"]["calibrated_file"] = "none", None

    def run(sub):
        hz_table, st_table = cal.calibrate(
            cfg, matches=list(matches), jobs=1, load_fn=load, gbdt_params=SMALL,
            team_model_params=SMALL, hazard_dir=str(tmp / sub / "hz"),
            stack_dir=str(tmp / sub / "st"), pair_dir=str(tmp / sub / "pairs"),
            log=lambda *_: None)
        hz_path = cal.write_yaml(hz_table, tmp / sub / "hazard_gbdt.yaml", "")
        st_path = cal.write_yaml(st_table, tmp / sub / "stack.yaml", "")
        return hz_path, st_path

    hz_path, st_path = run("a")
    full = copy.deepcopy(cfg)
    full["name"] = "stacked"
    full["hazard"] = {"kind": "gbdt", "calibrated_file": str(hz_path), "beta": 1.5}
    full["stack"] = {"kind": "gbdt", "calibrated_file": str(st_path)}
    return {"tmp": tmp, "matches": matches, "cfg": cfg, "full": full, "run": run,
            "hz_path": hz_path, "st_path": st_path}


def test_calibration_writes_all_fold_and_pair_models(stacked):
    tmp = stacked["tmp"] / "a"
    for sub in ("hz", "st"):
        assert (tmp / sub / "all.joblib").is_file()
        for m in stacked["matches"]:
            assert (tmp / sub / "{}.joblib".format(m)).is_file()
    pairs = sorted(p.name for p in (tmp / "pairs").iterdir())
    assert pairs == sorted("{}_{}_{}.joblib".format(k, a, b) for k in ("hazard", "team")
                           for a, b in (("m1", "m2"), ("m1", "m3"), ("m2", "m3")))
    hz_cal = hg.load_hazard_gbdt_calibration(stacked["hz_path"])
    assert set(hz_cal["meta"]["fold_metrics"]) == set(stacked["matches"])


def test_hazard_gbdt_kind_contract_and_symmetry(stacked):
    variants.load_kinds()
    assert variants.HAZARD_KINDS["gbdt"] is hg.hazard_gbdt
    master, _ = stacked["matches"]["m1"]
    params = {"calibrated_file": str(stacked["hz_path"])}
    h = hg.hazard_gbdt(master, params)
    half = master["half"].to_numpy()
    assert h.dtype == np.float32 and h.shape == (len(master),)
    assert np.isfinite(h).all() and (h[half == 0] == 0).all()
    np.testing.assert_allclose(hg.hazard_gbdt(swap_teams(master), params), h, atol=1e-6)
    h_fold = hg.hazard_gbdt(master, params, holdout="m1")
    assert h_fold.shape == h.shape
    with pytest.raises(FileNotFoundError):
        hg.hazard_gbdt(master, params, holdout="nope")


def test_pipeline_with_stack_is_valid_antisymmetric_and_beta_scaled(stacked):
    master, _ = stacked["matches"]["m2"]
    cfg = stacked["full"]
    labels = predict_labels(master, cfg, holdout="m2")
    half = master["half"].to_numpy()
    assert set(np.unique(labels[half != 0])) <= {"H", "A", "D"}
    assert (labels[half == 0] == "X").all()
    labels_s = predict_labels(swap_teams(master), cfg, holdout="m2")
    flip = {"H": "A", "A": "H", "D": "D", "X": "X"}
    agree = np.mean(np.array([flip[x] for x in labels]) == labels_s)
    assert agree > 0.99
    h = hazard_values(master, cfg, holdout="m2")
    np.testing.assert_allclose(hazard_exit_logit(master, cfg, holdout="m2"), 1.5 * h)


def test_stack_none_equals_no_section(stacked):
    master, _ = stacked["matches"]["m3"]
    cfg = copy.deepcopy(stacked["full"])
    cfg["stack"] = {"kind": "none", "calibrated_file": None}
    no_section = copy.deepcopy(cfg)
    del no_section["stack"]
    np.testing.assert_array_equal(predict_labels(master, cfg, holdout="m3"),
                                  predict_labels(master, no_section, holdout="m3"))


def test_calibration_is_deterministic(stacked):
    hz_b, st_b = stacked["run"]("b")
    master, _ = stacked["matches"]["m1"]
    cfg_b = copy.deepcopy(stacked["full"])
    cfg_b["hazard"]["calibrated_file"] = str(hz_b)
    cfg_b["stack"]["calibrated_file"] = str(st_b)
    np.testing.assert_array_equal(predict_labels(master, stacked["full"]),
                                  predict_labels(master, cfg_b))
    np.testing.assert_array_equal(hg.hazard_gbdt(master, {"calibrated_file": str(hz_b)}),
                                  hg.hazard_gbdt(master, {"calibrated_file":
                                                          str(stacked["hz_path"])}))


# --- dead_logit second stage ---------------------------------------------------------------------

def _dead_toy(n=600, seed=2):
    rng = np.random.default_rng(seed)
    half = _two_halves(n)
    dead = (np.arange(n) // 60) % 3 == 0
    d_fit = np.where(dead, 2.0, -2.0) + rng.normal(size=n)
    h = rng.normal(size=n)
    return half, dead, d_fit, h


def _dead_model(half, dead, d_fit, h):
    X = st.build_dead_stack_features(d_fit, h, half)
    inside = half != 0
    return cal._fit_classifier(X[inside], dead[inside].astype(np.int8), SMALL), \
        float(dead[inside].mean())


def test_dead_stack_features_and_output_contract():
    half, dead, d_fit, h = _dead_toy()
    X = st.build_dead_stack_features(d_fit, h, half)
    assert X.shape == (len(half), len(st.dead_stack_feature_names()))
    assert np.isnan(X[half == 0]).all()
    model, pi = _dead_model(half, dead, d_fit, h)
    d2 = st.dead_stack_from_model(model, pi, d_fit, h, half)
    assert d2.dtype == np.float32 and (d2[half == 0] == 0).all()
    assert np.abs(d2).max() <= st.DEAD_LOGIT_MAX
    inside = half != 0
    assert np.mean((d2[inside] > 0) == dead[inside]) > 0.8
    prior = np.where(half != 0, 0.75, 0.0)
    d2p = st.dead_stack_from_model(model, pi, d_fit, h, half, prior)
    free = inside & (np.abs(d2.astype(np.float64) + 0.75) < st.DEAD_LOGIT_MAX - 1e-3)
    np.testing.assert_allclose(d2p[free], d2[free] + 0.75, atol=1e-5)
    assert (d2p[half == 0] == 0).all()
    with pytest.raises(ValueError):
        st.dead_stack_from_model(model, 1.0, d_fit, h, half)


def _write_dead_stack(tmp_path, model, pi):
    import joblib
    import sklearn
    import yaml
    bundle = {"sklearn_version": sklearn.__version__, "features": st.dead_stack_feature_names(),
              "model": model, "pi_d": pi}
    (tmp_path / "ds").mkdir(exist_ok=True)
    joblib.dump(bundle, tmp_path / "ds" / "all.joblib")
    joblib.dump(bundle, tmp_path / "ds" / "M1.joblib")
    doc = {"variant": "gbdt", "target": "dead", "sklearn_version": sklearn.__version__,
           "model_file": str(tmp_path / "ds" / "all.joblib"), "fold_dir": str(tmp_path / "ds"),
           "features": st.dead_stack_feature_names()}
    path = tmp_path / "dead_stack.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return path


def test_dead_stack_logit_reads_its_file_and_folds(tmp_path):
    half, dead, d_fit, h = _dead_toy()
    model, pi = _dead_model(half, dead, d_fit, h)
    path = _write_dead_stack(tmp_path, model, pi)
    section = {"kind": "gbdt", "calibrated_file": str(path)}
    expected = st.dead_stack_from_model(model, pi, d_fit, h, half)
    np.testing.assert_array_equal(st.dead_stack_logit(d_fit, h, half, section), expected)
    np.testing.assert_array_equal(st.dead_stack_logit(d_fit, h, half, section, holdout="M1"),
                                  expected)
    with pytest.raises(FileNotFoundError):
        st.dead_stack_logit(d_fit, h, half, section, holdout="M2")
    with pytest.raises(ValueError, match="stack file"):
        st.stack_team_logit(d_fit, h, half, section)


def test_dead_stack_config_rules():
    raw = _raw_cfg()
    ok = {**copy.deepcopy(raw), "dead_stack": {"kind": "gbdt", "calibrated_file": "d.yaml"}}
    assert validate_config(ok)["dead_stack"]["kind"] == "gbdt"
    bad = copy.deepcopy(ok)
    bad["dead"]["kind"], bad["dead"]["calibrated_file"] = "rules_v1", None
    with pytest.raises(ConfigError, match="dead.kind 'gbdt'"):
        validate_config(bad)
    bad = copy.deepcopy(ok)
    bad["hazard"] = {"kind": "none", "calibrated_file": None, "beta": 0.0}
    with pytest.raises(ConfigError, match="needs a hazard"):
        validate_config(bad)
    bad = copy.deepcopy(ok)
    bad["dead_stack"]["calibrated_file"] = None
    with pytest.raises(ConfigError, match="dead_stack.calibrated_file"):
        validate_config(bad)
    bad = {**copy.deepcopy(raw), "stack": {"kind": "gbdt", "calibrated_file": "s.yaml"}}
    bad["team"]["params"]["w_sp_team"] = 1.0
    with pytest.raises(ConfigError, match="tag prior off"):
        validate_config(bad)


def test_evidence_logits_applies_the_dead_stack_with_the_tag_prior(stacked, tmp_path,
                                                                    monkeypatch):
    from modeling import pipeline
    from modeling.deadball import dead as dead_mod
    from modeling.deadball import dead_gbdt as dg

    master, _ = stacked["matches"]["m1"]
    half = master["half"].to_numpy()
    rng = np.random.default_rng(9)
    fake_fit = np.where(half != 0, rng.normal(size=len(master)), 0.0)
    prior = np.where(half != 0, 0.5, 0.0)
    variants.load_kinds()
    monkeypatch.setitem(variants.DEAD_KINDS, "gbdt",
                        lambda m, p, *, holdout=None: np.clip(fake_fit + prior, -8, 8)
                        .astype(np.float32))
    monkeypatch.setattr(dg, "gbdt_d_fit", lambda m, p, holdout=None: fake_fit)
    monkeypatch.setattr(dead_mod, "dead_tag_prior", lambda m, p: prior)
    h = hazard_values(master, stacked["full"])
    d_toy = (np.arange(len(master)) // 60) % 3 == 0
    model, pi = _dead_model(half, d_toy, fake_fit, h)
    path = _write_dead_stack(tmp_path, model, pi)
    cfg = copy.deepcopy(stacked["full"])
    cfg["dead"]["kind"], cfg["dead"]["calibrated_file"] = "gbdt", "unused.yaml"
    cfg["dead_stack"] = {"kind": "gbdt", "calibrated_file": str(path)}
    team, dead, h2 = pipeline.evidence_logits(master, cfg)
    np.testing.assert_array_equal(h2, h)
    np.testing.assert_array_equal(
        dead, st.dead_stack_from_model(model, pi, fake_fit, h, half, prior))


def test_reusing_the_hazard_models_gives_the_same_stack(stacked):
    tmp = stacked["tmp"]
    _, st_table = cal.calibrate(
        stacked["cfg"], matches=list(stacked["matches"]), jobs=1,
        load_fn=lambda m, d: stacked["matches"][m], gbdt_params=SMALL, team_model_params=SMALL,
        hazard_dir=str(tmp / "a" / "hz"), stack_dir=str(tmp / "r" / "st"),
        pair_dir=str(tmp / "r" / "pairs"), hazard_pair_dir=str(tmp / "a" / "pairs"),
        reuse_hazard=True, log=lambda *_: None)
    st_path = cal.write_yaml(st_table, tmp / "r" / "stack.yaml", "")
    cfg = copy.deepcopy(stacked["full"])
    cfg["stack"]["calibrated_file"] = str(st_path)
    master, _ = stacked["matches"]["m2"]
    for holdout in (None, "m2"):
        np.testing.assert_array_equal(predict_labels(master, cfg, holdout=holdout),
                                      predict_labels(master, stacked["full"], holdout=holdout))
    assert not (tmp / "r" / "pairs" / "hazard_m1_m2.joblib").exists()


# --- control EMA features of the first stage (tried in D-029, kept in the code) -----------------

def test_ema_features_are_odd_and_only_present_when_requested():
    from modeling.evidence.team_gbdt import (build_gbdt_features, gbdt_feature_layout,
                                             swap_matrix, swap_permutation)
    from tests.test_modeling_team_gbdt import F_PARAMS
    m, _ = _full_master(300, seed=4)
    base = gbdt_feature_layout(F_PARAMS)
    fp = {**F_PARAMS, "ema_taus": [10, 25], "touch_ema_taus": [25]}
    layout = gbdt_feature_layout(fp)
    extra = [c for c in layout["features"] if c not in base["features"]]
    assert extra == ["ctl_bwd_10", "ctl_fwd_10", "ctl_bwd_25", "ctl_fwd_25", "tch_bwd_25",
                     "tch_fwd_25"]
    assert set(extra) <= set(layout["odd"])
    p = team_params()
    feats, valid = build_gbdt_features(m, p, fp)
    feats_s, valid_s = build_gbdt_features(swap_teams(m), p, fp)
    perm, sign = swap_permutation(layout)
    np.testing.assert_array_equal(valid, valid_s)
    np.testing.assert_allclose(feats_s.to_numpy(), swap_matrix(feats.to_numpy(), perm, sign),
                               rtol=1e-9, atol=1e-9)
    with pytest.raises(ValueError):
        gbdt_feature_layout({**F_PARAMS, "ema_taus": [10, 10]})
