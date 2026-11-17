"""Tests for the F3-08 GBDT dead_logit, its features and its calibration.

Synthetic data only: nothing from the match or from IDSSE.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import yaml

from evaluation import possession_metrics as pm
from modeling import config as config_mod
from modeling import contracts, paths, variants
from modeling.calibration import dead as cd
from modeling.calibration import dead_gbdt as cg
from modeling.contracts import LABEL_CODES
from modeling.deadball import dead_gbdt as dg
from modeling.deadball.dead import dead_logit_gbdt, dead_tag_prior
from modeling.deadball.features import GAP_EXIT, GAP_LONG
from modeling.evidence.windows import STAT_NAMES, rolling_window_stats
from modeling.pipeline import predict_labels
from preprocessing.make_mock_master import generate_mock_master
from tests.modeling_fixtures import dead_params, deadball_master
from tests.test_modeling_calibration_dead import toy_match

P = dead_params()
D = LABEL_CODES["D"]
MATCHES = ("M1", "M2", "M3")
WINDOWS = (3, 6)
SMALL_MODEL = {**cg.MODEL_PARAMS, "max_iter": 15, "min_samples_leaf": 5}


def _logit(p):
    return float(np.log(p) - np.log1p(-p))


# -- shared windows ------------------------------------------------------------------------------
def test_window_max_statistics_cover_the_specified_frames():
    x = np.array([0, 5, 1, 2, 9, 3, 4, 0, 1, 2], dtype=float)
    r = rolling_window_stats(x, np.ones(10, dtype=np.int8), 2, ("past_max", "fut_max", "cent_max"))
    assert r["past_max"][5] == 9.0                       # [3, 5]
    assert r["fut_max"][5] == 4.0                        # [5, 7]
    assert r["cent_max"][5] == 9.0                       # [3, 7]
    assert r["cent_max"][8] == 4.0                       # [6, 9]: the end of the half
    assert set(STAT_NAMES) >= {"past_max", "fut_max", "cent_max", "past_mean", "cent_min"}
    with pytest.raises(ValueError, match="unknown window statistics"):
        rolling_window_stats(x, np.ones(10, dtype=np.int8), 2, ("cent_median",))


def test_window_max_skips_nan_and_never_crosses_a_half():
    half = np.array([1, 1, 1, 0, 2, 2, 2], dtype=np.int8)
    x = np.array([np.nan, np.nan, 1.0, 50.0, np.nan, 0.0, np.nan])
    r = rolling_window_stats(x, half, 3, ("cent_max",))
    np.testing.assert_array_equal(r["cent_max"][:3], [1.0, 1.0, 1.0])
    assert np.isnan(r["cent_max"][3])                    # half == 0
    np.testing.assert_array_equal(r["cent_max"][4:], [0.0, 0.0, 0.0])


# -- features ------------------------------------------------------------------------------------
def test_feature_names_follow_the_spec():
    names = dg.feature_names((25, 50, 75))
    assert names[:len(dg.FRAME_FEATURES)] == list(dg.FRAME_FEATURES)
    assert len(names) == len(dg.FRAME_FEATURES) + 9 * 3 == len(set(names))
    for w in (25, 50, 75):
        for name in ("still_past_mean_{}", "still_fut_mean_{}", "still_fut_minus_past_{}",
                     "speed_past_mean_{}", "speed_fut_mean_{}", "speed_fut_minus_past_{}",
                     "d_touchline_cent_min_{}", "d_goalline_cent_min_{}",
                     "out_of_pitch_cent_max_{}"):
            assert name.format(w) in names
    for bad in dg.FORBIDDEN_FEATURES:
        assert not any(bad in n for n in names)
    assert not any("ball_obs" in n or "observed" in n for n in names)
    for bad_sizes in ([], [3, 3], [0], [2.5]):
        with pytest.raises(ValueError):
            dg.feature_names(bad_sizes)


def _two_half_master(seed=0, n=120):
    rng = np.random.default_rng(seed)
    x = 30.0 + np.cumsum(rng.normal(0, 0.5, n))
    y = 20.0 + np.cumsum(rng.normal(0, 0.5, n))
    x[40:60] = np.nan                                     # a long gap in the first half
    half = np.r_[np.ones(n // 2 - 5), np.zeros(10), np.full(n - n // 2 - 5, 2)].astype(np.int8)
    speed = rng.uniform(0.5, 4.0, n)
    return deadball_master(x, y, vx=1.0, vy=0.5, half=half, mean_speed_home=speed,
                           mean_speed_away=speed[::-1].copy())


def test_features_match_their_definition():
    m = _two_half_master()
    feats = dg.build_features(m, P, WINDOWS)
    assert list(feats.columns) == dg.feature_names(WINDOWS)
    assert (feats.dtypes == np.float64).all() and feats.index.equals(m.index)
    db = cd.raw_features(m, P)
    for col in ("db_gap_kind", "db_still_frac", "db_mean_speed", "db_fast_in_pitch"):
        np.testing.assert_array_equal(feats[col].to_numpy()[m["half"] != 0],
                                      db[col].to_numpy()[m["half"] != 0])
    for w in WINDOWS:
        np.testing.assert_array_equal(
            feats["speed_fut_minus_past_{}".format(w)].to_numpy(),
            feats["speed_fut_mean_{}".format(w)].to_numpy()
            - feats["speed_past_mean_{}".format(w)].to_numpy())
    # Centred min of the touchline distance at a frame inside the gap: the ball seen around it.
    t, w = 45, 6
    seen = m["d_touchline"].to_numpy()[t - w:t + w + 1]
    assert feats.loc[t, "d_touchline_cent_min_6"] == np.nanmin(seen)
    assert np.isnan(feats.loc[50, "d_touchline_cent_min_3"])          # whole window in the gap
    assert np.isnan(feats.loc[50, "d_touchline"]) and np.isnan(feats.loc[50, "out_of_pitch"])


def test_features_are_nan_outside_the_halves_and_never_cross_a_half():
    m = _two_half_master()
    half = m["half"].to_numpy()
    feats = dg.build_features(m, P, WINDOWS)
    assert feats[half == 0].isna().all().all()
    changed = m.copy()
    second = half == 2
    changed.loc[second, ["mean_speed_home", "mean_speed_away"]] = 9.0
    changed.loc[second, "d_touchline"] = 0.1
    changed.loc[second, "out_of_pitch"] = 1.0
    other = dg.build_features(changed, P, WINDOWS)
    pd.testing.assert_frame_equal(feats[half == 1], other[half == 1])
    assert not feats[second].equals(other[second])


# -- model: prior correction, determinism, persistence --------------------------------------------
def _known_rate_data(rng, n, pi, p_x_dead=0.6, p_x_alive=0.1):
    """One binary informative column x with P(x | D) and P(x | alive) known, plus noise."""
    y = rng.random(n) < pi
    x = np.where(y, rng.random(n) < p_x_dead, rng.random(n) < p_x_alive).astype(float)
    return np.column_stack([x, rng.normal(size=n)]), y


def test_prior_correction_with_a_known_pi_d():
    rng = np.random.default_rng(0)
    params = {**cg.MODEL_PARAMS, "max_iter": 100}
    # Uninformative features: d_fit is about 0 whatever the share of D.
    X = rng.normal(size=(20000, 3))
    y = rng.random(20000) < 0.2
    model, pi = cg.fit_model(X, y, params)
    assert pi == y.mean()
    d = dg.d_fit_values(model, X, pi)
    np.testing.assert_array_equal(d, model.decision_function(X) - _logit(pi))
    assert abs(float(np.mean(d))) < 0.05 and float(np.mean(np.abs(d))) < 0.25
    # Informative feature: d_fit is the likelihood ratio, whatever the share of D in the training.
    expected = {1.0: np.log(0.6 / 0.1), 0.0: np.log(0.4 / 0.9)}
    for pi_true in (0.2, 0.5):
        X, y = _known_rate_data(rng, 40000, pi_true)
        model, pi = cg.fit_model(X, y, params)
        for x_value, lr in expected.items():
            rows = X[:, 0] == x_value
            assert abs(float(np.mean(dg.d_fit_values(model, X[rows], pi))) - lr) < 0.1


def test_prior_logit_rejects_degenerate_pi():
    for bad in (0.0, 1.0, -0.1, float("nan")):
        with pytest.raises(ValueError):
            dg.prior_logit(bad)
    with pytest.raises(ValueError, match="both classes"):
        cg.fit_model(np.zeros((5, 2)), np.zeros(5, dtype=bool))


def test_training_is_deterministic():
    rng = np.random.default_rng(1)
    X, y = _known_rate_data(rng, 5000, 0.3)
    X[rng.random(X.shape) < 0.1] = np.nan                  # NaN routed natively, no imputation
    a, pa = cg.fit_model(X, y, SMALL_MODEL)
    b, pb = cg.fit_model(X, y, SMALL_MODEL)
    assert pa == pb
    np.testing.assert_array_equal(a.decision_function(X), b.decision_function(X))


def test_save_and_load_preserve_predictions_bit_for_bit(tmp_path):
    import sklearn

    rng = np.random.default_rng(2)
    X, y = _known_rate_data(rng, 3000, 0.3)
    model, pi = cg.fit_model(X, y, SMALL_MODEL)
    path = dg.save_bundle(tmp_path / "b.joblib", model, ["a", "b"], pi)
    loaded = dg.load_bundle(path, sklearn_version=sklearn.__version__, features=["a", "b"])
    assert loaded["pi_d"] == pi
    np.testing.assert_array_equal(loaded["model"].decision_function(X), model.decision_function(X))
    np.testing.assert_array_equal(dg.d_fit_values(loaded["model"], X, loaded["pi_d"]),
                                  dg.d_fit_values(model, X, pi))
    with pytest.raises(RuntimeError, match="scikit-learn"):
        dg.load_bundle(path, sklearn_version="0.0.0", features=["a", "b"])
    with pytest.raises(ValueError, match="feature list"):
        dg.load_bundle(path, sklearn_version=sklearn.__version__, features=["a"])
    with pytest.raises(FileNotFoundError):
        dg.load_bundle(tmp_path / "nope.joblib", sklearn_version=sklearn.__version__,
                       features=["a", "b"])


# -- scenarios -----------------------------------------------------------------------------------
def test_scenarios_are_the_f3_04_stories_with_coherent_windows():
    rows = cg.scenario_rows(WINDOWS)
    assert list(rows.columns) == dg.feature_names(WINDOWS)
    assert list(rows.index) == ["R1", "R2", "R2b", "R3"]
    r1, r2, r2b, r3 = (rows.loc[s] for s in ("R1", "R2", "R2b", "R3"))
    assert r1["db_gap_kind"] == GAP_LONG and r2["db_gap_kind"] == r2b["db_gap_kind"] == GAP_EXIT
    assert r1["gap_length"] == 50 and r1["db_still_frac"] == 0.2 and r1["db_mean_speed"] == 2.5
    assert r3["db_still_frac"] == 0.45 and r3["db_mean_speed"] == 1.8
    for r in (r1, r2, r2b, r3):                          # the ball is not seen inside the gap
        assert np.isnan(r[["out_of_pitch", "d_touchline", "d_goalline"]].astype(float)).all()
        assert r["db_fast_in_pitch"] == 0
    for w in WINDOWS:
        assert r1["speed_past_mean_{}".format(w)] == r1["speed_fut_mean_{}".format(w)] == 2.5
        assert r1["still_fut_minus_past_{}".format(w)] == 0.0
        assert r1["d_touchline_cent_min_{}".format(w)] > P["exit_line_m"]       # midfield
        assert r1["out_of_pitch_cent_max_{}".format(w)] == 0
        assert r2["d_touchline_cent_min_{}".format(w)] < 0                      # seen out
        assert r2["out_of_pitch_cent_max_{}".format(w)] == 1
        assert 0 <= r2b["d_touchline_cent_min_{}".format(w)] < P["exit_line_m"]  # just inside
        assert r2b["out_of_pitch_cent_max_{}".format(w)] == 0

    def changed(a, b):
        return {c for c in rows.columns if not np.array_equal(a[[c]], b[[c]], equal_nan=True)}
    touch = {"d_touchline_cent_min_{}".format(w) for w in WINDOWS}
    out = {"out_of_pitch_cent_max_{}".format(w) for w in WINDOWS}
    assert changed(r1, r2) == {"db_gap_kind"} | touch | out
    assert changed(r1, r2b) == {"db_gap_kind"} | touch
    assert all(c.startswith(("db_still", "db_mean", "still_", "speed_")) for c in changed(r1, r3))
    assert set(cg.REPORTED_ONLY) == {"R2b", "R3"}


def _cell_matrix(rows):
    """Feature matrix from ``{feature: value}`` rows (missing features NaN)."""
    names = dg.feature_names(WINDOWS)
    return np.array([[r.get(n, np.nan) for n in names] for r in rows], dtype=float)


def _r1_like(gap_length, **kw):
    w = cg.cell_window(WINDOWS)
    row = {"db_gap_kind": GAP_LONG, "gap_length": gap_length, "db_mean_speed": 2.5,
           "speed_fut_minus_past_{}".format(w): 0.0, "d_touchline_cent_min_{}".format(w): 10.0,
           "out_of_pitch_cent_max_{}".format(w): 0.0}
    row.update(kw)
    return row


def test_scenario_cells_select_the_right_frames_and_resample_whole_gaps():
    w = cg.cell_window(WINDOWS)
    assert w == WINDOWS[1] and cg.cell_window((25, 50, 75)) == 50
    rows = ([_r1_like(30.0)] * 4 + [_r1_like(40.0)] * 6               # two gaps in the R1 cell
            + [_r1_like(30.0, db_mean_speed=4.0)]                     # too fast
            + [_r1_like(500.0)]                                       # gap too long
            + [_r1_like(30.0, **{"speed_fut_minus_past_{}".format(w): -1.5})]   # slowing down
            + [_r1_like(30.0, **{"d_touchline_cent_min_{}".format(w): 1.0})]    # near the line
            + [_r1_like(0.0, db_gap_kind=0)])                         # no gap
    X = _cell_matrix(rows)
    mask = cg.scenario_cell_mask(X, "R1", WINDOWS)
    np.testing.assert_array_equal(mask, [True] * 10 + [False] * 5)
    runs = cg.gap_runs(X, WINDOWS)
    assert list(runs[:10]) == [0] * 4 + [1] * 6 and runs[-1] == -1
    y = np.array([True] * 4 + [False] * 6 + [False] * 5)
    data = {"M": {"X": X, "y": y}}
    d_fit = {"GBDT": {"M": np.full(len(y), -1.0)}}
    cells = cg.scenario_cells(data, d_fit, 0.5, WINDOWS)
    c = cells["R1"]
    assert c["n_frames"] == 10 and c["n_gaps"] == 2 and c["p_dead"] == 0.4
    assert c["d"] == pytest.approx(_logit(0.4)) and c["low"] <= c["d"] <= c["high"]
    assert c["models"]["GBDT"] == -1.0
    assert cells["R2"]["n_frames"] == 0 and np.isnan(cells["R2"]["d"])


# -- calibration end to end ----------------------------------------------------------------------
def _toy_matches():
    return {m: toy_match(10 + i, n_cycles=10) for i, m in enumerate(MATCHES)}


def _reference(tmp, matches):
    data = {m: [cd.match_design(*matches[m], P)] for m in matches}
    result = cd.calibrate_with_fallback(data, step=2)
    doc = cd.calibration_document(result, P, train_modes=("default",), step=2,
                                  description="toy reference")
    return cd.write_calibration(doc, tmp / "ref.yaml")


def _run(tmp, matches, ref, **kw):
    return cg.run(P, list(matches), out_dir=tmp / "calib", model_dir=tmp / "models",
                  report_path=tmp / "report.md", window_sizes=WINDOWS, fold_root=tmp / "ext",
                  reference_files={"dead_gbdt": ref, "dead_gbdt_pooled": ref},
                  load_fn=lambda m, mode: matches[m], model_params=SMALL_MODEL,
                  log=lambda *_: None, **kw)


@pytest.fixture(scope="module")
def calibrated(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("dead_gbdt")
    matches = _toy_matches()
    ref = _reference(tmp, matches)
    out = _run(tmp, matches, ref)
    return {"tmp": tmp, "matches": matches, "ref": ref, "out": out,
            "params": {**P, "calibrated_file": str(out["files"]["dead_gbdt"])}}


def test_run_writes_files_models_and_report(calibrated):
    tmp, out = calibrated["tmp"], calibrated["out"]
    assert set(out["files"]) == {"dead_gbdt", "dead_gbdt_pooled"}
    for name in out["files"]:
        doc = dg.load_calibration(out["files"][name])
        assert (tmp / "models" / name / "all.joblib").is_file()
        for m in MATCHES:
            assert (tmp / "ext" / "models" / name / "{}.joblib".format(m)).is_file()
        assert set(doc["pi_d"]["folds"]) == set(MATCHES)
        assert doc["sklearn_version"] == __import__("sklearn").__version__
        assert doc["features"] == dg.feature_names(WINDOWS)
        assert doc["model_params"] == SMALL_MODEL
        assert set(doc["meta"]["fold_metrics"]) == set(cg.EVAL_MODES)
        assert set(doc["robustness"]["values"]) == {"all"} | {"fold " + m for m in MATCHES}
        assert doc["robustness"]["reference_values"]["all"].keys() == {"R1", "R2", "R3"}
        assert set(doc["robustness"]["values"]["all"]) == set(cg.SCENARIOS)
        assert set(doc["meta"]["scenario_cells"]) == set(cg.SCENARIOS)
        assert set(doc["meta"]["scenario_cells"]["R1"]["models"]) == {"GBDT", "F3-04"}
        assert "none" not in doc["train_modes"]
    pooled = dg.load_calibration(out["files"]["dead_gbdt_pooled"])
    assert pooled["train_modes"] == list(cg.POOLED_MODES)
    assert pooled["subsample"] == cg.POOLED_SUBSAMPLE
    single = dg.load_calibration(out["files"]["dead_gbdt"])
    assert single["train_modes"] == ["default"] and single["subsample"] == cg.SUBSAMPLE
    report = out["report"].read_text(encoding="utf-8")
    for text in ("## `dead_gbdt`", "## `dead_gbdt_pooled`", "#### `hidelo`", "#### `hidehi`",
                 "F3-04 log-loss", "Robustness scenarios", "Features alone", "`d_goalline`",
                 "The scenarios against the data", "| R2b |"):
        assert text in report


def test_pi_d_and_metrics_are_those_of_each_fold(calibrated):
    doc = dg.load_calibration(calibrated["out"]["files"]["dead_gbdt"])
    matches = calibrated["matches"]
    for holdout in MATCHES:
        ys = [cd.match_design(*matches[m], P)[1][::cg.SUBSAMPLE] for m in MATCHES if m != holdout]
        assert doc["pi_d"]["folds"][holdout] == float(np.concatenate(ys).mean())
        assert doc["meta"]["n_train"]["folds"][holdout] == sum(len(y) for y in ys)
    rob = doc["robustness"]
    assert rob["r1_pass"] == all(v["R1"] < 0 for v in rob["values"].values())
    assert rob["r2_pass"] == all(v["R2"] > 0 for v in rob["values"].values())


def test_fold_models_never_see_their_match(calibrated, tmp_path):
    """Changing the labels of M1 changes every model but the fold of M1."""
    matches = copy.deepcopy(calibrated["matches"])
    master, labels = matches["M1"]
    flipped = labels.copy()
    flipped[(labels == D)] = LABEL_CODES["H"]
    flipped[::7] = D
    matches["M1"] = (master, flipped)
    out = _run(tmp_path, matches, calibrated["ref"], candidates=("dead_gbdt",))
    a = dg.load_calibration(calibrated["out"]["files"]["dead_gbdt"])
    b = dg.load_calibration(out["files"]["dead_gbdt"])
    assert a["pi_d"]["folds"]["M1"] == b["pi_d"]["folds"]["M1"]
    assert a["robustness"]["values"]["fold M1"] == b["robustness"]["values"]["fold M1"]
    assert a["pi_d"]["folds"]["M2"] != b["pi_d"]["folds"]["M2"]
    assert a["pi_d"]["all"] != b["pi_d"]["all"]


def test_calibration_is_deterministic(calibrated, tmp_path):
    out = _run(tmp_path, calibrated["matches"], calibrated["ref"])
    for name in out["files"]:
        a = dg.load_calibration(calibrated["out"]["files"][name])
        b = dg.load_calibration(out["files"][name])
        a, b = copy.deepcopy(a), copy.deepcopy(b)
        for doc in (a, b):
            doc.pop("model_file"), doc.pop("fold_dir")
        assert a == b
    m, _ = calibrated["matches"]["M2"]
    d1 = dead_logit_gbdt(m, calibrated["params"])
    d2 = dead_logit_gbdt(m, {**P, "calibrated_file": str(out["files"]["dead_gbdt"])})
    assert d1.tobytes() == d2.tobytes()


# -- the registered kind -------------------------------------------------------------------------
def test_gbdt_dead_kind_is_registered():
    variants.load_kinds()
    assert variants.DEAD_KINDS["gbdt"] is dead_logit_gbdt
    assert "gbdt" in config_mod.DEAD_KIND_NAMES


def test_holdout_uses_its_fold_model(calibrated):
    import sklearn

    m, _ = calibrated["matches"]["M2"]
    params = calibrated["params"]
    d_all = dead_logit_gbdt(m, params)
    d_m2 = dead_logit_gbdt(m, params, holdout="M2")
    d_m3 = dead_logit_gbdt(m, params, holdout="M3")
    assert not np.array_equal(d_all, d_m2) and not np.array_equal(d_m2, d_m3)
    doc = dg.load_calibration(params["calibrated_file"])
    bundle = dg.load_bundle(calibrated["tmp"] / "ext" / "models" / "dead_gbdt" / "M2.joblib",
                            sklearn_version=sklearn.__version__, features=doc["features"])
    assert bundle["pi_d"] == doc["pi_d"]["folds"]["M2"]
    X = dg.build_features(m, P, doc["window_sizes"]).to_numpy()
    direct = dg.d_fit_values(bundle["model"], X, bundle["pi_d"]) + dead_tag_prior(m, P)
    expected = np.clip(direct, -contracts.DEAD_LOGIT_MAX, contracts.DEAD_LOGIT_MAX)
    np.testing.assert_array_equal(d_m2, expected.astype(np.float32))
    with pytest.raises(ValueError, match="no fold"):
        dead_logit_gbdt(m, params, holdout="unknown")


def test_logit_points_to_dead_balls_on_the_toy_matches(calibrated):
    m, labels = toy_match(99, n_cycles=6)
    d = dead_logit_gbdt(m, calibrated["params"])
    assert d[labels == D].mean() > 0 > d[labels != D].mean()


def test_c2_contract_and_zero_outside_the_halves_on_the_mock(calibrated):
    mock = generate_mock_master(seed=42)
    half = mock["half"].to_numpy()
    d = dead_logit_gbdt(mock, calibrated["params"])
    contracts.validate_logits(np.zeros(len(mock), np.float32), d, len(mock), half=half)
    assert (d[half == 0] == 0).all() and not np.signbit(d[half == 0]).any()
    assert dead_logit_gbdt(mock, calibrated["params"]).tobytes() == d.tobytes()
    assert dg.gbdt_d_fit(mock, calibrated["params"]).dtype == np.float64


def test_tag_prior_is_added_and_the_sum_is_clipped(calibrated):
    m, _ = toy_match(7, n_cycles=3)
    params = calibrated["params"]
    base = dead_logit_gbdt(m, params)
    tagged = m.copy()
    tagged["is_set_piece"] = True
    with_tag = dead_logit_gbdt(tagged, params)
    expected = np.clip(base.astype(np.float64) + P["weights"]["tag_set_piece"], -12, 12)
    np.testing.assert_allclose(with_tag, expected, atol=1e-5)
    big = {**params, "weights": {**P["weights"], "tag_set_piece": 40.0}}
    assert (dead_logit_gbdt(tagged, big) == np.float32(contracts.DEAD_LOGIT_MAX)).all()


def test_errors_on_bad_inputs(calibrated, tmp_path):
    m, _ = toy_match(3, n_cycles=2)
    params = calibrated["params"]
    with pytest.raises(ValueError, match="calibrated_file"):
        dead_logit_gbdt(m, P)
    with pytest.raises(FileNotFoundError):
        dead_logit_gbdt(m, {**P, "calibrated_file": str(tmp_path / "nope.yaml")})
    with pytest.raises(ValueError, match="smooth_frames"):
        dead_logit_gbdt(m, {**params, "smooth_frames": P["smooth_frames"] + 2})
    with pytest.raises(ValueError, match="is_set_piece"):
        dead_logit_gbdt(m.drop(columns=["is_set_piece"]), params)
    # The weights are not feature parameters: changing them is allowed.
    dead_logit_gbdt(m, {**params, "weights": {**P["weights"], "tag_sp_prox": 0.5}})


def test_calibration_file_is_validated_and_tied_to_its_models(calibrated, tmp_path):
    raw = yaml.safe_load(calibrated["out"]["files"]["dead_gbdt"].read_text(encoding="utf-8"))
    for key, value, match in (("features", ["db_ball_rest"], "feature list"),
                              ("variant", "logistic", "not a GBDT"),
                              ("pi_d", {"all": 1.0, "folds": {}}, "pi_D")):
        bad = {**raw, key: value}
        path = tmp_path / "bad_{}.yaml".format(key)
        path.write_text(yaml.safe_dump(bad), encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            dg.load_calibration(path)
    stale = copy.deepcopy(raw)
    stale["pi_d"]["all"] = 0.5 if raw["pi_d"]["all"] != 0.5 else 0.4
    path = tmp_path / "stale.yaml"
    path.write_text(yaml.safe_dump(stale), encoding="utf-8")
    m, _ = toy_match(3, n_cycles=2)
    with pytest.raises(ValueError, match="regenerate"):
        dead_logit_gbdt(m, {**P, "calibrated_file": str(path)})
    missing = {k: v for k, v in raw.items() if k != "fold_dir"}
    path = tmp_path / "missing.yaml"
    path.write_text(yaml.safe_dump(missing), encoding="utf-8")
    with pytest.raises(ValueError, match="fold_dir"):
        dg.load_calibration(path)


# -- pipeline and decoder ------------------------------------------------------------------------
def _v1_with_gbdt(path, tmp_path):
    raw = yaml.safe_load((paths.repo_root() / "configs" / "v1.yaml").read_text(encoding="utf-8"))
    raw["name"] = "v1_dead_gbdt"
    raw["dead"]["kind"] = "gbdt"
    raw["dead"]["calibrated_file"] = str(path)
    target = tmp_path / "v1_dead_gbdt.yaml"
    target.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return config_mod.load_config(target)


def test_predict_labels_with_a_gbdt_config(calibrated, tmp_path):
    cfg = _v1_with_gbdt(calibrated["out"]["files"]["dead_gbdt"], tmp_path)
    mock = generate_mock_master(seed=42)
    labels = predict_labels(mock, cfg)
    contracts.validate_labels(labels, mock["half"].to_numpy())


def test_decoder_table_compares_each_base_config_with_the_gbdt_files(calibrated, tmp_path):
    files = calibrated["out"]["files"]
    base = config_mod.load_config(paths.repo_root() / "configs" / "v1.yaml")
    cfgs, own = cg.decoder_configs({"v1": base}, files)
    assert own == {"v1 (dead dead_gbdt)": "v1 (dead {})".format(base["dead"]["kind"]),
                   "v1 (dead dead_gbdt_pooled)": "v1 (dead {})".format(base["dead"]["kind"])}
    assert list(cfgs) == ["v1 (dead {})".format(base["dead"]["kind"]),
                          "v1 (dead dead_gbdt)", "v1 (dead dead_gbdt_pooled)"]
    assert cfgs["v1 (dead dead_gbdt)"]["dead"]["kind"] == "gbdt"
    assert base["dead"]["kind"] != "gbdt"                 # the base config is not modified
    mock = generate_mock_master(seed=42)
    gt = pm.label_codes(predict_labels(mock, base))          # the base config scores 1 on it
    lam = float(base["decoder"]["lam_dead"])
    df = cg.evaluate_decoder(cfgs, [lam, lam + 0.2], degrade="default", matches=["M1"],
                             load_fn=lambda m, mode: (mock, gt))
    assert len(df) == 6 and df["spell_f1_mean"].between(0, 1).all()
    s = cg.summarize_decoder(df)
    assert s.loc[("v1 (dead {})".format(base["dead"]["kind"]), lam), "mean"] == 1.0
    paired = cg.paired_decoder(df, own)
    assert len(paired) == 4 and (paired["n"] == 1).all()
    row = paired[(paired["config"] == "v1 (dead dead_gbdt)") & (paired["lam_dead"] == lam)].iloc[0]
    gb = df[(df["config"] == "v1 (dead dead_gbdt)") & (df["lam_dead"] == lam)]["spell_f1_mean"]
    assert row["diff"] == pytest.approx(float(gb.iloc[0]) - 1.0)
    assert row["wins"] == 0                                # the base scores 1, nothing beats it
    report = cg.report_markdown({"dead_gbdt": dg.load_calibration(files["dead_gbdt"])},
                                decoder={"default": df}, decoder_note="note", decoder_own=own)
    assert "lam_dead" in report and "v1 (dead dead_gbdt)" in report
    assert "Matches higher" in report


# -- tools/dead_feature_shift.py with a GBDT file ------------------------------------------------
def test_dead_feature_shift_tool_reads_gbdt_files(calibrated):
    from tools import dead_feature_shift as dfs

    path = calibrated["out"]["files"]["dead_gbdt"]
    doc = dfs.load_calibration(path)
    assert doc["variant"] == dg.VARIANT
    assert "folds" in dfs.load_calibration(calibrated["ref"])           # F3-04 file
    m, _ = toy_match(5, n_cycles=4)
    stats = dfs.master_statistics(m, P, doc)
    d = dg.gbdt_d_fit(m, calibrated["params"])[m["half"].to_numpy() != 0]
    assert stats["d_positive_share"] == float(np.mean(d > 0))
    assert stats["speed_finite"] and stats["still_finite"]
