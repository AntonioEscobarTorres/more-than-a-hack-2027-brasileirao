"""F3-04 calibration of dead_logit. Synthetic data only: nothing from the match or from IDSSE."""
import numpy as np
import pandas as pd
import pytest
import yaml

from modeling import config as config_mod
from modeling import contracts, paths, variants
from modeling.calibration import dead as cd
from modeling.contracts import LABEL_CODES
from modeling.deadball.dead import dead_logit_calibrated, dead_tag_prior
from modeling.deadball.features import GAP_AIR, GAP_EXIT, GAP_LONG
from modeling.pipeline import predict_labels
from preprocessing.make_mock_master import generate_mock_master
from tests.modeling_fixtures import dead_params, deadball_master

P = dead_params()
D, H, A, X = (LABEL_CODES[k] for k in "DHAX")
MATCHES = ("M1", "M2", "M3", "M4")


# -- synthetic data ------------------------------------------------------------------------------
def random_raw(n, rng):
    """Raw feature rows with every kind of value, NaN included."""
    has_ball = rng.random(n) < 0.6
    out = np.where(has_ball, (rng.random(n) < 0.1).astype(float), np.nan)
    kind = np.where(has_ball, 0, rng.choice([GAP_EXIT, GAP_AIR, GAP_LONG], n))
    gap = np.where(has_ball, 0.0, rng.integers(13, 3000, n).astype(float))
    still = np.where(rng.random(n) < 0.05, np.nan, rng.random(n))
    speed = np.where(rng.random(n) < 0.05, np.nan, rng.uniform(0, 6, n))
    fast = np.where(has_ball, (rng.random(n) < 0.3).astype(float), 0.0)
    return pd.DataFrame({"has_ball": has_ball.astype(float), "out_of_pitch": out,
                         "db_gap_kind": kind.astype(float), "gap_length": gap,
                         "db_still_frac": still, "db_mean_speed": speed,
                         "db_fast_in_pitch": fast})


TRUE_WEIGHTS = {"out_of_pitch": 3.0, "gap_exit": 2.0, "gap_air": -1.5, "fast_in_pitch": -2.5,
                "mean_speed": -0.8, "still_frac": 1.5}


def synthetic_data(seed=0, n=4000, weights=TRUE_WEIGHTS, intercept=0.5, matches=MATCHES):
    """``{match: [(X_full, y)]}`` drawn from a logistic model with known weights."""
    rng = np.random.default_rng(seed)
    data = {}
    for m in matches:
        X = cd.design_matrix(random_raw(n, rng), cd.ALL_FEATURES)
        logit = intercept + sum(w * X[:, cd.ALL_FEATURES.index(f)] for f, w in weights.items())
        y = rng.random(n) < 1.0 / (1.0 + np.exp(-logit))
        data[m] = [(X, y)]
    return data


def toy_match(seed, n_cycles=30):
    """A synthetic external match: live play (fast ball in midfield) and dead balls (still ball
    out of the pitch, slow players), with some noise so the classes overlap."""
    rng = np.random.default_rng(seed)
    xs, ys, labels, speeds = [], [], [], []
    for _ in range(n_cycles):
        n_live, n_dead = int(rng.integers(150, 300)), int(rng.integers(60, 200))
        x = 30.0 + np.cumsum(rng.normal(0.0, 0.4, n_live))
        xs.append(np.clip(x, 5.0, 100.0))
        ys.append(np.full(n_live, 34.0) + rng.normal(0.0, 3.0, n_live))
        labels.append(np.where(rng.random(n_live) < 0.5, H, A))
        speeds.append(np.full(n_live, rng.uniform(2.0, 4.0)))
        out = rng.random() < 0.7
        xs.append(np.full(n_dead, 50.0))
        ys.append(np.full(n_dead, -3.0 if out else 30.0))
        labels.append(np.full(n_dead, D))
        speeds.append(np.full(n_dead, rng.uniform(0.5, 1.5)))
    x, y = np.concatenate(xs), np.concatenate(ys)
    lab = np.concatenate(labels).astype(np.int8)
    speed = np.concatenate(speeds)
    vx = np.gradient(x) * 25.0
    hide = rng.random(len(x)) < 0.2
    x[hide] = np.nan
    master = deadball_master(x, y, vx=vx, vy=0.0, mean_speed_home=speed, mean_speed_away=speed)
    return master, lab


def doc_from(result, params=P, step=cd.SUBSAMPLE, modes=("default",)):
    return cd.calibration_document(result, params, train_modes=modes, step=step,
                                   description="test")


@pytest.fixture(scope="module")
def toy_calibration(tmp_path_factory):
    data = {m: [cd.match_design(*toy_match(i), P)] for i, m in enumerate(MATCHES)}
    result = cd.calibrate_with_fallback(data, step=2)
    path = cd.write_calibration(doc_from(result), tmp_path_factory.mktemp("calib") / "dead.yaml")
    return result, path


def params_with(path):
    return {**P, "calibrated_file": str(path)}


# -- features ------------------------------------------------------------------------------------
def test_design_matrix_has_no_nan_and_imputes_with_indicators():
    raw = pd.DataFrame({"has_ball": [1.0, 0.0], "out_of_pitch": [1.0, np.nan],
                        "db_gap_kind": [0.0, GAP_EXIT], "gap_length": [0.0, 4000.0],
                        "db_still_frac": [0.4, np.nan], "db_mean_speed": [np.nan, 1.0],
                        "db_fast_in_pitch": [1.0, 0.0]})
    X = pd.DataFrame(cd.design_matrix(raw, cd.FULL_FEATURES), columns=cd.FULL_FEATURES)
    assert np.isfinite(X.to_numpy()).all()
    assert X["out_of_pitch"].tolist() == [1.0, 0.0]
    assert X["gap_exit"].tolist() == [0.0, 1.0] and X["gap_long"].tolist() == [0.0, 0.0]
    assert X["still_frac"].tolist() == [0.4, 0.0] and X["still_frac_missing"].tolist() == [0, 1]
    assert X["mean_speed"].tolist() == [0.0, 1.0] and X["mean_speed_missing"].tolist() == [1, 0]
    assert X["log_gap"].tolist() == [0.0, pytest.approx(np.log1p(cd.GAP_LOG_CLIP_FRAMES))]
    raw.loc[0, "gap_length"] = np.nan
    X = pd.DataFrame(cd.design_matrix(raw, cd.FULL_FEATURES), columns=cd.FULL_FEATURES)
    assert X["log_gap"][0] == 0.0 and X["log_gap_missing"].tolist() == [1.0, 0.0]


def test_ball_rest_and_ball_speed_are_never_features():
    for features in cd.FEATURE_SETS.values():
        assert not any("rest" in f or f == "ball_speed" for f in features)
    assert "ball_speed" not in cd.RAW_COLUMNS
    assert set(cd.FULL_FEATURES) - set(cd.REDUCED_FEATURES) == {"has_ball", "log_gap",
                                                               "log_gap_missing"}


def test_interactions_are_products_of_the_imputed_factors():
    raw = pd.DataFrame({"has_ball": [0.0, 0.0, 1.0], "out_of_pitch": [np.nan, np.nan, 0.0],
                        "db_gap_kind": [GAP_EXIT, GAP_LONG, 0.0],
                        "gap_length": [50.0, 400.0, 0.0],
                        "db_still_frac": [0.2, 0.2, 0.2], "db_mean_speed": [2.5, np.nan, 1.0],
                        "db_fast_in_pitch": [0.0, 0.0, 1.0]})
    X = pd.DataFrame(cd.design_matrix(raw, cd.ALL_FEATURES), columns=cd.ALL_FEATURES)
    assert np.isfinite(X.to_numpy()).all()
    lg = np.log1p([50.0, 400.0, 0.0])
    assert X["exit_x_log_gap"].tolist() == pytest.approx([lg[0], 0.0, 0.0])
    assert X["long_x_log_gap"].tolist() == pytest.approx([0.0, lg[1], 0.0])
    assert X["exit_x_speed"].tolist() == [2.5, 0.0, 0.0]
    assert X["long_x_speed"].tolist() == [0.0, 0.0, 0.0]            # missing speed -> 0
    assert X["mean_speed_missing"].tolist() == [0.0, 1.0, 0.0]      # ... flagged by its indicator
    assert X["speed_x_log_gap"].tolist() == pytest.approx([2.5 * lg[0], 0.0, 0.0])
    for name, (a, b) in cd.INTERACTIONS.items():
        np.testing.assert_array_equal(X[name], X[a] * X[b])


def test_feature_sets_and_interactions_are_consistent():
    assert cd.ALL_FEATURES == cd.FULL_FEATURES + cd.INTERACTION_FEATURES
    assert cd.FEATURE_SETS["interactions"] == cd.ALL_FEATURES
    assert cd.CANDIDATE_SETS[0] == "full" and cd.FALLBACK_SET == "reduced"
    for a, b in cd.INTERACTIONS.values():
        assert a in cd.FULL_FEATURES and b in cd.FULL_FEATURES
    # The reduced set (item 4) has no gap length, so no interaction either.
    assert not set(cd.REDUCED_FEATURES) & set(cd.INTERACTION_FEATURES)
    # The full set keeps the order of the base columns, so a full-feature design is a prefix.
    assert cd._columns(cd.FULL_FEATURES) == list(range(len(cd.FULL_FEATURES)))


def test_raw_features_follow_the_master():
    master, _ = toy_match(0, n_cycles=3)
    raw = cd.raw_features(master, P)
    assert list(raw.columns) == list(cd.RAW_COLUMNS)
    assert raw["has_ball"].tolist() == np.isfinite(master["ball_x"]).astype(float).tolist()
    with pytest.raises(ValueError, match="gap_length"):
        cd.raw_features(master.drop(columns=["gap_length"]), P)


def test_match_design_uses_only_labelled_frames_inside_the_halves():
    master, labels = toy_match(1, n_cycles=2)
    labels = labels.copy()
    master.loc[:9, "half"] = 0
    labels[:10] = X
    labels[20:25] = X
    X_full, y = cd.match_design(master, labels, P)
    assert X_full.shape == (len(master) - 15, len(cd.ALL_FEATURES))
    assert y.dtype == bool and y.sum() == (labels == D).sum()


# -- fitting ----------------------------------------------------------------------------------
def test_signs_are_recovered_on_synthetic_data():
    result = cd.calibrate(synthetic_data(), cd.FULL_FEATURES, step=1)
    for model in [result["all"], *result["folds"].values()]:
        for f, w in TRUE_WEIGHTS.items():
            assert np.sign(model["weights"][f]) == np.sign(w), f
    assert set(result["folds"]) == set(MATCHES)
    assert result["C"] in cd.C_GRID and set(result["cv_log_loss"]) == set(cd.C_GRID)


def test_prior_correction_neutral_features_give_d_fit_near_zero():
    # Labels independent of the features, P(D) = 0.3: the model logit is logit(0.3) and
    # d_fit ~ 0 everywhere.
    rng = np.random.default_rng(3)
    data = {}
    for m in MATCHES:
        Xm = cd.design_matrix(random_raw(5000, rng), cd.FULL_FEATURES)
        data[m] = [(Xm, rng.random(5000) < 0.3)]
    result = cd.calibrate(data, cd.FULL_FEATURES, step=1)
    model = result["all"]
    assert model["pi_d"] == pytest.approx(0.3, abs=0.01)
    X_all = np.vstack([d[0][0] for d in data.values()])
    values = cd.d_fit(X_all, model, cd.FULL_FEATURES)
    assert abs(values.mean()) < 0.02 and np.abs(values).max() < 0.25
    logit = cd.model_logit(X_all, model, cd.FULL_FEATURES)
    assert abs(logit.mean() - np.log(0.3 / 0.7)) < 0.02


def test_prior_correction_shifts_by_logit_pi():
    model = {"mean": {"fast_in_pitch": 0.0}, "std": {"fast_in_pitch": 1.0},
             "weights": {"fast_in_pitch": -1.0}, "intercept": 0.7, "pi_d": 0.4, "n_train": 10}
    X1 = np.array([[1.0], [0.0]])
    np.testing.assert_allclose(cd.d_fit(X1, model, ["fast_in_pitch"]),
                               [0.7 - 1.0 - np.log(0.4 / 0.6), 0.7 - np.log(0.4 / 0.6)])


def test_fold_models_never_see_their_match():
    data = synthetic_data(seed=5)
    data["M2"] = [(data["M2"][0][0], ~data["M2"][0][1])]          # M2 has inverted labels
    result = cd.calibrate(data, cd.FULL_FEATURES, step=1)
    others = [m for m in MATCHES if m != "M2"]
    # Every fold that includes M2 is pulled towards the inverted signs; the M2 fold is not.
    w_m2 = result["folds"]["M2"]["weights"]["out_of_pitch"]
    assert all(w_m2 > result["folds"][m]["weights"]["out_of_pitch"] for m in others)
    assert result["folds"]["M2"]["n_train"] == 3 * 4000


def test_c_is_chosen_by_held_out_log_loss_and_ties_go_to_the_smaller_c(monkeypatch):
    data = synthetic_data(seed=7, n=800)
    calls = []
    real = cd.log_loss

    def fake(y, logit):
        calls.append(1)
        return 1.0                                                  # all C tie
    monkeypatch.setattr(cd, "log_loss", fake)
    result = cd.calibrate(data, cd.FULL_FEATURES, step=1)
    assert result["C"] == min(cd.C_GRID) and len(calls) == len(cd.C_GRID) * len(MATCHES)
    monkeypatch.setattr(cd, "log_loss", real)


def test_subsampling_keeps_one_frame_in_step():
    data = synthetic_data(seed=8, n=1000)
    X_tr, y_tr = cd.training_rows(data, ["M1", "M2"], 5, list(range(len(cd.ALL_FEATURES))))
    assert X_tr.shape[0] == 2 * 200
    assert np.array_equal(X_tr[:200], data["M1"][0][0][::5])


def test_pooled_folds_drop_the_match_from_every_mode():
    base = synthetic_data(seed=9, n=600)
    data = {m: [base[m][0]] * 3 for m in MATCHES}                   # three "modes"
    result = cd.calibrate(data, cd.FULL_FEATURES, step=15)
    assert result["folds"]["M1"]["n_train"] == 3 * 3 * 40
    assert result["all"]["n_train"] == 4 * 3 * 40


def test_fit_is_deterministic():
    data = synthetic_data(seed=11, n=1500)
    a = cd.calibrate(data, cd.FULL_FEATURES, step=1)
    b = cd.calibrate(data, cd.FULL_FEATURES, step=1)
    assert a == b


def test_fit_needs_both_classes_and_the_right_shape():
    X1 = np.zeros((5, 2))
    with pytest.raises(ValueError, match="both classes"):
        cd.fit_model(X1, np.zeros(5, dtype=bool), ["has_ball", "gap_exit"], 1.0)
    with pytest.raises(ValueError, match="shape"):
        cd.fit_model(X1, np.array([0, 1, 0, 1, 0], dtype=bool), ["has_ball"], 1.0)
    with pytest.raises(ValueError, match="two matches"):
        cd.calibrate({"M1": synthetic_data(n=50)["M1"]}, cd.FULL_FEATURES, 1)


def test_constant_column_is_not_scaled():
    rng = np.random.default_rng(0)
    X1 = np.column_stack([np.ones(200), rng.normal(size=200)])
    y = rng.random(200) < 0.5
    model = cd.fit_model(X1, y, ["has_ball", "mean_speed"], 1.0)
    assert model["std"]["has_ball"] == 1.0
    assert np.isfinite(cd.d_fit(X1, model, ["has_ball", "mean_speed"])).all()


def test_log_loss_and_auc_helpers():
    from sklearn.metrics import log_loss as sk_log_loss
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(2)
    y = rng.random(500) < 0.4
    logit = rng.normal(size=500) + y
    p = 1.0 / (1.0 + np.exp(-logit))
    assert cd.log_loss(y, logit) == pytest.approx(sk_log_loss(y, p))
    assert cd.auc(y, logit) == pytest.approx(roc_auc_score(y, logit))
    assert cd.auc(y, np.ones(500)) == pytest.approx(0.5)
    assert np.isnan(cd.auc(np.zeros(5, dtype=bool), np.arange(5.0)))


# -- robustness scenarios (item 4) ---------------------------------------------------------------
def test_scenarios_are_the_ones_of_the_task():
    X1 = cd.scenario_design(cd.FULL_FEATURES)
    assert list(X1.index) == ["R1", "R2", "R3"]
    r1 = X1.loc["R1"]
    assert r1["has_ball"] == 0 and r1["out_of_pitch"] == 0 and r1["gap_long"] == 1
    assert r1["gap_exit"] == 0 and r1["fast_in_pitch"] == 0
    assert r1["still_frac"] == 0.2 and r1["mean_speed"] == 2.5
    assert r1["log_gap"] == pytest.approx(np.log1p(50.0)) and r1["log_gap_missing"] == 0
    assert X1.loc["R2", "gap_exit"] == 1 and X1.loc["R2", "gap_long"] == 0
    assert X1.loc["R3", "still_frac"] == 0.45 and X1.loc["R3", "mean_speed"] == 1.8


def unit_model(features, weights, intercept=0.0, pi=0.5):
    return {"mean": {f: 0.0 for f in features}, "std": {f: 1.0 for f in features},
            "weights": {f: float(weights.get(f, 0.0)) for f in features},
            "intercept": intercept, "pi_d": pi, "n_train": 1}


def test_robustness_checks_r1_and_r2_on_every_model():
    feats = cd.FULL_FEATURES
    good = unit_model(feats, {"gap_exit": 2.0, "gap_long": -1.0})
    bad = unit_model(feats, {"gap_exit": 2.0, "gap_long": 1.0})
    check = cd.robustness({"all": good, "folds": {"M1": good, "M2": bad}}, feats)
    assert not check["r1_pass"] and check["r2_pass"]
    assert check["values"]["all"]["R1"] == pytest.approx(-1.0)
    assert check["values"]["fold M2"]["R1"] == pytest.approx(1.0)
    assert check["values"]["all"]["R2"] == pytest.approx(2.0)
    assert set(check["values"]) == {"all", "fold M1", "fold M2"}


SCEN_VALUES = {"R1": -1.0, "R2": 1.0, "R3": 0.0}


def fake_robustness_by_set(r1_by_set):
    """A robustness stub: R1 holds for the sets mapped to True; records the sets it sees."""
    seen = []

    def fake(calib, features, gap_log_clip_frames=cd.GAP_LOG_CLIP_FRAMES):
        name = next(n for n, f in cd.FEATURE_SETS.items() if tuple(f) == tuple(features))
        seen.append(name)
        return {"values": {"all": dict(SCEN_VALUES)}, "r1_pass": r1_by_set[name],
                "r2_pass": True}
    return fake, seen


def test_fallback_to_the_reduced_set_when_r1_fails_for_every_candidate(monkeypatch):
    fake, seen = fake_robustness_by_set({"full": False, "interactions": False, "reduced": True})
    monkeypatch.setattr(cd, "robustness", fake)
    result = cd.calibrate_with_fallback(synthetic_data(n=600), step=1)
    assert seen == ["full", "interactions", "reduced"]
    assert result["feature_set"] == "reduced" and result["features"] == list(cd.REDUCED_FEATURES)
    assert set(result["all"]["weights"]) == set(cd.REDUCED_FEATURES)
    assert [t["feature_set"] for t in result["tried"]] == ["full", "interactions", "reduced"]


def test_the_set_with_the_lowest_log_loss_among_r1_passes_is_kept(monkeypatch):
    # True model with an exit x speed interaction: the interactions set fits it better.
    weights = {**TRUE_WEIGHTS, "exit_x_speed": 1.5}
    data = synthetic_data(seed=21, n=3000, weights=weights)
    fake, seen = fake_robustness_by_set({"full": True, "interactions": True, "reduced": True})
    monkeypatch.setattr(cd, "robustness", fake)
    result = cd.calibrate_with_fallback(data, step=1)
    losses = {t["feature_set"]: t["cv_log_loss"] for t in result["tried"]}
    assert seen == ["full", "interactions"] and set(losses) == {"full", "interactions"}
    assert losses["interactions"] < losses["full"]
    assert result["feature_set"] == "interactions"
    assert result["features"] == list(cd.ALL_FEATURES)
    for t in result["tried"]:
        assert set(t["cv_log_loss_by_match"]) == set(MATCHES)
        assert t["cv_log_loss"] == pytest.approx(np.mean(list(t["cv_log_loss_by_match"].values())))
        assert t["scenarios_all"] == SCEN_VALUES
    # A candidate that fails R1 is never kept, whatever its log-loss.
    fake, _ = fake_robustness_by_set({"full": True, "interactions": False, "reduced": True})
    monkeypatch.setattr(cd, "robustness", fake)
    assert cd.calibrate_with_fallback(data, step=1)["feature_set"] == "full"


def test_ties_in_the_set_log_loss_go_to_the_simpler_set(monkeypatch):
    fake, _ = fake_robustness_by_set({"full": True, "interactions": True, "reduced": True})
    monkeypatch.setattr(cd, "robustness", fake)
    monkeypatch.setattr(cd, "log_loss", lambda y, logit: 1.0)
    result = cd.calibrate_with_fallback(synthetic_data(n=600), step=1)
    assert result["feature_set"] == "full" and len(result["tried"]) == 2


def test_r2_is_never_a_selection_criterion(monkeypatch):
    weights = {**TRUE_WEIGHTS, "exit_x_speed": 1.5}
    data = synthetic_data(seed=21, n=3000, weights=weights)

    def fake(calib, features, gap_log_clip_frames=cd.GAP_LOG_CLIP_FRAMES):
        r2 = tuple(features) == cd.FULL_FEATURES          # only the worse set passes R2
        return {"values": {"all": dict(SCEN_VALUES)}, "r1_pass": True, "r2_pass": r2}
    monkeypatch.setattr(cd, "robustness", fake)
    result = cd.calibrate_with_fallback(data, step=1)
    assert result["feature_set"] == "interactions" and not result["robustness"]["r2_pass"]


def test_interactions_are_recovered_on_synthetic_data():
    weights = {**TRUE_WEIGHTS, "exit_x_speed": 1.5, "speed_x_log_gap": -0.3}
    result = cd.calibrate(synthetic_data(seed=23, n=4000, weights=weights), cd.ALL_FEATURES,
                          step=1)
    w = result["all"]["weights"]
    assert w["exit_x_speed"] > 0 and w["speed_x_log_gap"] < 0
    for f, true_w in TRUE_WEIGHTS.items():
        if f != "mean_speed":                   # its weight alone is not its effect here
            assert np.sign(w[f]) == np.sign(true_w), f


# -- calibration file and the calibrated variant ---------------------------------------------------
def test_calibrated_kind_is_registered():
    variants.load_kinds()
    assert variants.DEAD_KINDS["calibrated"] is dead_logit_calibrated


def test_file_round_trip_and_validation(toy_calibration, tmp_path):
    result, path = toy_calibration
    doc = cd.load_calibration(path)
    assert doc["features"] == result["features"] and set(doc["folds"]) == set(MATCHES)
    assert doc["all"]["weights"] == pytest.approx(result["all"]["weights"])
    assert doc["feature_params"] == cd.feature_params(P)
    for bad, match in (
        ({**doc, "format_version": 99}, "format_version"),
        ({**doc, "features": ["nope"]}, "feature list"),
        ({**doc, "all": {**doc["all"], "pi_d": 1.0}}, "pi_d"),
        ({**doc, "all": {**doc["all"], "std": {**doc["all"]["std"],
                                               doc["features"][0]: 0.0}}}, "positive"),
        ({**doc, "all": {**doc["all"], "weights": {}}}, "one value per feature"),
        ({k: v for k, v in doc.items() if k != "folds"}, "folds"),
    ):
        broken = tmp_path / "broken.yaml"
        broken.write_text(yaml.safe_dump(bad), encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            cd.load_calibration(broken)


def test_holdout_selects_the_right_fold(toy_calibration):
    _, path = toy_calibration
    master, _ = toy_match(42, n_cycles=4)
    doc = cd.load_calibration(path)
    tags = dead_tag_prior(master, P)
    for holdout, model in [(None, doc["all"])] + [(m, doc["folds"][m]) for m in MATCHES]:
        d = dead_logit_calibrated(master, params_with(path), holdout=holdout)
        X1 = cd.design_matrix(cd.raw_features(master, P), doc["features"])
        expected = np.clip(cd.d_fit(X1, model, doc["features"]) + tags, -12, 12)
        np.testing.assert_allclose(d, expected.astype(np.float32), rtol=1e-6, atol=1e-6)
    outs = [dead_logit_calibrated(master, params_with(path), holdout=m) for m in MATCHES]
    assert all(not np.array_equal(outs[0], o) for o in outs[1:])
    with pytest.raises(ValueError, match="no fold"):
        dead_logit_calibrated(master, params_with(path), holdout="J03WMX")


def _one_row(**values):
    row = {"has_ball": 1.0, "out_of_pitch": 0.0, "db_gap_kind": 0.0, "gap_length": 0.0,
           "db_still_frac": 0.2, "db_mean_speed": 2.5, "db_fast_in_pitch": 0.0}
    row.update(values)
    return pd.DataFrame([row])


def test_design_matrix_fill_replaces_the_zero_and_keeps_the_indicator():
    raw = pd.concat([_one_row(db_mean_speed=np.nan, db_still_frac=np.nan, has_ball=0.0,
                              out_of_pitch=np.nan, db_gap_kind=GAP_EXIT, gap_length=50.0)])
    X = pd.DataFrame(cd.design_matrix(raw, cd.ALL_FEATURES,
                                      fill={"mean_speed": 2.0, "still_frac": 0.3}),
                     columns=cd.ALL_FEATURES)
    assert X["mean_speed"][0] == 2.0 and X["mean_speed_missing"][0] == 1.0
    assert X["still_frac"][0] == 0.3 and X["still_frac_missing"][0] == 1.0
    assert X["exit_x_speed"][0] == 2.0
    assert X["speed_x_log_gap"][0] == pytest.approx(2.0 * np.log1p(50.0))
    with pytest.raises(ValueError, match="fill only takes"):
        cd.design_matrix(raw, cd.ALL_FEATURES, fill={"log_gap": 1.0})
    with pytest.raises(ValueError, match="finite"):
        cd.design_matrix(raw, cd.ALL_FEATURES, fill={"mean_speed": np.nan})


def test_missing_input_never_seen_in_the_fit_is_no_evidence(toy_calibration):
    """Rule 4: a NaN player speed or still fraction must not read as 0 (everyone standing)."""
    _, path = toy_calibration
    doc = cd.load_calibration(path)
    features = doc["features"]
    for model in [doc["all"], *doc["folds"].values()]:
        assert model["mean"]["mean_speed_missing"] == 0.0     # the toy fit never sees a NaN
        fill = cd.inference_fill(model, features)
        assert fill == {"still_frac": model["mean"]["still_frac"],
                        "mean_speed": model["mean"]["mean_speed"]}
        for kind, gap in [(0.0, 0.0), (GAP_EXIT, 50.0), (GAP_LONG, 300.0)]:
            base = dict(db_gap_kind=kind, gap_length=gap, has_ball=float(gap == 0.0),
                        out_of_pitch=0.0 if gap == 0.0 else np.nan)
            at_mean = cd.d_fit(cd.design_matrix(_one_row(
                **base, db_mean_speed=fill["mean_speed"], db_still_frac=fill["still_frac"]),
                features), model, features)
            missing = cd.d_fit(cd.design_matrix(_one_row(
                **base, db_mean_speed=np.nan, db_still_frac=np.nan), features, fill=fill),
                model, features)
            np.testing.assert_allclose(missing, at_mean, atol=1e-9)


def test_missing_input_seen_in_the_fit_keeps_the_fitted_convention():
    data = synthetic_data(seed=3)                     # random_raw has 5% NaN speeds and fractions
    X, y = cd.training_rows(data, MATCHES, 1, cd._columns(cd.FULL_FEATURES))
    model = cd.fit_model(X, y, cd.FULL_FEATURES, 1.0)
    assert model["mean"]["mean_speed_missing"] > 0.0
    assert cd.inference_fill(model, cd.FULL_FEATURES) == {}
    reduced = [f for f in cd.FULL_FEATURES if f != "mean_speed_missing"]
    model = cd.fit_model(X[:, cd._columns(reduced)], y, reduced, 1.0)
    assert set(cd.inference_fill(model, reduced)) == {"mean_speed"}   # no indicator: never seen


def test_calibrated_kind_treats_missing_player_speed_as_no_evidence(toy_calibration):
    _, path = toy_calibration
    master, _ = toy_match(42, n_cycles=4)
    gone = slice(100, 300)                            # longer than smooth_frames: db_mean_speed NaN
    master.loc[master.index[gone], ["mean_speed_home", "mean_speed_away"]] = np.nan
    raw = cd.raw_features(master, P)
    nan_rows = ~np.isfinite(raw["db_mean_speed"].to_numpy())
    assert nan_rows.any()
    doc = cd.load_calibration(path)
    model = doc["all"]
    d = cd.calibrated_d_fit(master, params_with(path), doc, None)
    raw.loc[nan_rows, "db_mean_speed"] = model["mean"]["mean_speed"]
    expected = cd.d_fit(cd.design_matrix(raw, doc["features"]), model, doc["features"])
    np.testing.assert_allclose(d, expected, atol=1e-9)


def test_calibrated_logit_signs_on_the_toy_matches(toy_calibration):
    result, path = toy_calibration
    w = result["all"]["weights"]
    assert w["out_of_pitch"] > 0 and w["fast_in_pitch"] < 0
    assert cd.signs_ok(result["all"], result["features"])
    master, labels = toy_match(99, n_cycles=10)
    d = dead_logit_calibrated(master, params_with(path))
    assert d[labels == D].mean() > 0 > d[labels != D].mean()


def test_tag_prior_is_added_and_the_sum_is_clipped(toy_calibration):
    _, path = toy_calibration
    master, _ = toy_match(7, n_cycles=4)
    base = dead_logit_calibrated(master, params_with(path))
    tagged = master.copy()
    tagged["is_set_piece"] = True
    with_tag = dead_logit_calibrated(tagged, params_with(path))
    expected = np.clip(base.astype(np.float64) + P["weights"]["tag_set_piece"], -12, 12)
    np.testing.assert_allclose(with_tag, expected, atol=1e-5)
    big = {**params_with(path), "weights": {**P["weights"], "tag_set_piece": 40.0}}
    assert (dead_logit_calibrated(tagged, big) == np.float32(contracts.DEAD_LOGIT_MAX)).all()


def test_zero_outside_the_halves_and_c2_contract_on_the_mock(toy_calibration):
    _, path = toy_calibration
    mock = generate_mock_master(seed=42)
    half = mock["half"].to_numpy()
    d = dead_logit_calibrated(mock, params_with(path))
    contracts.validate_logits(np.zeros(len(mock), np.float32), d, len(mock), half=half)
    assert not np.signbit(d[half == 0]).any() and (d[half == 0] == 0).all()
    again = dead_logit_calibrated(mock, params_with(path))
    assert again.tobytes() == d.tobytes()


def test_feature_params_must_match_the_fit(toy_calibration):
    _, path = toy_calibration
    master, _ = toy_match(3, n_cycles=2)
    other = {**params_with(path), "smooth_frames": P["smooth_frames"] + 2}
    with pytest.raises(ValueError, match="smooth_frames"):
        dead_logit_calibrated(master, other)
    # The weights are not feature parameters: changing them is allowed.
    reweighted = {**params_with(path), "weights": {**P["weights"], "tag_sp_prox": 0.5}}
    dead_logit_calibrated(master, reweighted)


def test_missing_calibrated_file_or_columns_are_errors(toy_calibration, tmp_path):
    _, path = toy_calibration
    master, _ = toy_match(3, n_cycles=2)
    with pytest.raises(ValueError, match="calibrated_file"):
        dead_logit_calibrated(master, P)
    with pytest.raises(FileNotFoundError):
        dead_logit_calibrated(master, params_with(tmp_path / "nope.yaml"))
    with pytest.raises(ValueError, match="is_set_piece"):
        dead_logit_calibrated(master.drop(columns=["is_set_piece"]), params_with(path))


def test_predict_labels_with_a_calibrated_config(toy_calibration, tmp_path):
    _, path = toy_calibration
    raw = yaml.safe_load((paths.repo_root() / "configs" / "v1.yaml").read_text(encoding="utf-8"))
    raw["name"] = "v1_dead_calibrated"
    raw["dead"]["kind"] = "calibrated"
    raw["dead"]["calibrated_file"] = str(path)
    target = tmp_path / "v1_dead_calibrated.yaml"
    target.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = config_mod.load_config(target)
    mock = generate_mock_master(seed=42)
    labels = predict_labels(mock, cfg)
    contracts.validate_labels(labels, mock["half"].to_numpy())


# -- end to end (run) ------------------------------------------------------------------------------
def test_run_writes_the_four_files_and_the_report(tmp_path, monkeypatch):
    designs = {m: cd.match_design(*toy_match(i, n_cycles=12), P) for i, m in enumerate(MATCHES)}

    def fake_load_mode(matches, mode, params):
        shift = {"default": 0, "hidelo": 1, "hidehi": 2}[mode]
        out = {}
        for m in matches:
            X1, y = designs[m]
            out[m] = (X1[shift:], y[shift:])
        return out
    monkeypatch.setattr(cd, "load_mode", fake_load_mode)
    written = cd.run(P, list(MATCHES), tmp_path / "calib", tmp_path / "report.md")
    assert set(written) == {"dead", "dead_pooled", "dead_hidelo", "dead_hidehi", "report"}
    for name in ("dead", "dead_pooled", "dead_hidelo", "dead_hidehi"):
        doc = cd.load_calibration(written[name])
        assert set(doc["folds"]) == set(MATCHES)
    pooled = cd.load_calibration(written["dead_pooled"])
    assert pooled["train_modes"] == list(cd.POOLED_MODES)
    assert pooled["subsample"] == cd.POOLED_SUBSAMPLE
    assert cd.load_calibration(written["dead"])["train_modes"] == ["default"]
    report = written["report"].read_text(encoding="utf-8")
    for text in ("## `dead`", "## `dead_pooled`", "## `dead_hidelo`", "## `dead_hidehi`",
                 "R1", "default log-loss", "hidehi AUC", "AUC of each feature alone",
                 "Expected signs", "Empirical P(D)", "2.3-2.7 m/s", "Feature-set choice",
                 "Gaps of at most", "The scenarios against the data", "`exit_x_speed`"):
        assert text in report


def test_calibrated_d_fit_is_zero_outside_the_halves(toy_calibration):
    _, path = toy_calibration
    master, _ = toy_match(5, n_cycles=3)
    master.loc[:49, "half"] = 0
    values = cd.calibrated_d_fit(master, P, cd.load_calibration(path), None)
    assert values.dtype == np.float64
    assert (values[:50] == 0).all() and not np.signbit(values[:50]).any()
    assert (values[50:] != 0).any()


def test_dead_rate_by_speed_bins_and_gap_kinds():
    raw = pd.DataFrame({
        "has_ball": [0, 0, 0, 0, 1, 1, 0], "out_of_pitch": np.nan,
        "db_gap_kind": [GAP_EXIT, GAP_EXIT, GAP_EXIT, GAP_LONG, 0, 0, GAP_AIR],
        "gap_length": 50.0,
        "db_still_frac": 0.2,
        "db_mean_speed": [0.0, 1.0, 2.5, 2.5, 4.0, np.nan, 1.2],
        "db_fast_in_pitch": 0.0})
    X1 = cd.design_matrix(raw, cd.FULL_FEATURES)
    y = np.array([True, False, False, True, False, True, True])
    rates = cd.dead_rate_by_speed({"M1": (X1[:3], y[:3]), "M2": (X1[3:], y[3:])})
    assert list(rates) == ["exit gap", "long gap", "air gap", "no long gap"]
    labels = cd.speed_bin_labels()
    assert len(labels) == len(cd.SPEED_BIN_EDGES_MPS) + 1 and labels[0] == "<= 1.0"
    exit_row = dict(zip(labels, rates["exit gap"]))
    assert exit_row["<= 1.0"] == (0.5, 2)              # 0.0 and 1.0 (upper edge included)
    assert exit_row["2.3-2.7"] == (0.0, 1)
    assert dict(zip(labels, rates["long gap"]))["2.3-2.7"] == (1.0, 1)
    assert dict(zip(labels, rates["air gap"]))["1.0-1.5"] == (1.0, 1)
    no_gap = dict(zip(labels, rates["no long gap"]))
    assert no_gap["> 3.5"] == (0.0, 1)                  # the NaN-speed frame is left out
    assert sum(n for _, n in rates["no long gap"]) == 1
    assert np.isnan(rates["air gap"][0][0]) and rates["air gap"][0][1] == 0


# -- effective slopes, scenario cells -------------------------------------------------------------
def test_effective_slopes_add_the_interaction_terms():
    feats = cd.ALL_FEATURES
    model = unit_model(feats, {"mean_speed": -2.0, "exit_x_speed": 1.5, "speed_x_log_gap": -0.1,
                               "out_of_pitch": 1.0, "fast_in_pitch": -1.0})
    model["std"]["mean_speed"] = 2.0                     # slopes are per raw unit
    rows = {"exit": {f: 0.0 for f in feats}, "plain": {f: 0.0 for f in feats}}
    rows["exit"].update({"gap_exit": 1.0, "log_gap": 3.0})
    slopes = cd.effective_slopes(model, feats, rows)
    assert set(slopes) == set(cd.EXPECTED_SIGNS)
    assert slopes["mean_speed"]["plain"] == pytest.approx(-1.0)
    assert slopes["mean_speed"]["exit"] == pytest.approx(-1.0 + 1.5 - 0.3)
    assert slopes["out_of_pitch"] == {"exit": 1.0, "plain": 1.0}
    # Without interactions the slope is w / std at every row, so the old check is unchanged.
    base = unit_model(cd.FULL_FEATURES, {"mean_speed": -2.0, "out_of_pitch": 1.0,
                                         "fast_in_pitch": -1.0})
    assert cd.signs_ok(base, cd.FULL_FEATURES)
    assert not cd.signs_ok({**base, "weights": {**base["weights"], "fast_in_pitch": 0.5}},
                           cd.FULL_FEATURES)
    # An interaction that flips the speed slope at a scenario fails the check.
    flip = unit_model(feats, {"mean_speed": -1.0, "exit_x_speed": 2.0, "out_of_pitch": 1.0,
                              "fast_in_pitch": -1.0})
    assert not cd.signs_ok(flip, feats)


def test_dead_rate_by_speed_can_keep_only_short_gaps():
    raw = pd.DataFrame({
        "has_ball": [0, 0, 0, 1], "out_of_pitch": np.nan,
        "db_gap_kind": [GAP_EXIT, GAP_EXIT, GAP_LONG, 0],
        "gap_length": [50.0, 900.0, 80.0, 0.0], "db_still_frac": 0.2,
        "db_mean_speed": 2.5, "db_fast_in_pitch": 0.0})
    X1 = cd.design_matrix(raw, cd.ALL_FEATURES)
    y = np.array([True, False, False, True])
    labels = cd.speed_bin_labels()
    every = cd.dead_rate_by_speed({"M1": (X1, y)})
    short = cd.dead_rate_by_speed({"M1": (X1, y)}, max_gap_frames=100)
    assert dict(zip(labels, every["exit gap"]))["2.3-2.7"] == (0.5, 2)
    assert dict(zip(labels, short["exit gap"]))["2.3-2.7"] == (1.0, 1)
    assert dict(zip(labels, short["long gap"]))["2.3-2.7"] == (0.0, 1)
    np.testing.assert_equal(short["no long gap"], every["no long gap"])     # NaN-aware
    with pytest.raises(ValueError, match="below"):
        cd.dead_rate_by_speed({"M1": (X1, y)}, max_gap_frames=cd.GAP_LOG_CLIP_FRAMES)


def _gap_rows(kind, length, speed, n_dead, n_live):
    n = n_dead + n_live
    raw = pd.DataFrame({"has_ball": 0.0, "out_of_pitch": np.nan, "db_gap_kind": float(kind),
                        "gap_length": float(length), "db_still_frac": 0.2,
                        "db_mean_speed": speed, "db_fast_in_pitch": 0.0}, index=range(n))
    return cd.design_matrix(raw, cd.ALL_FEATURES), np.r_[np.ones(n_dead), np.zeros(n_live)] == 1


def _visible_rows(n):
    raw = pd.DataFrame({"has_ball": 1.0, "out_of_pitch": 0.0, "db_gap_kind": 0.0,
                        "gap_length": 0.0, "db_still_frac": 0.2, "db_mean_speed": 2.5,
                        "db_fast_in_pitch": 0.0}, index=range(n))
    return cd.design_matrix(raw, cd.ALL_FEATURES), np.zeros(n, dtype=bool)


def test_gap_runs_split_gaps_at_observed_frames_and_length_changes():
    parts = [_gap_rows(GAP_EXIT, 50, 2.5, 2, 1), _visible_rows(2),
             _gap_rows(GAP_EXIT, 50, 2.5, 1, 1), _gap_rows(GAP_LONG, 60, 2.5, 0, 2)]
    X1 = np.vstack([p[0] for p in parts])
    assert cd._gap_runs(X1).tolist() == [0, 0, 0, -1, -1, 1, 1, 2, 2]


def test_scenario_cells_estimate_d_and_resample_whole_gaps():
    parts = [_gap_rows(GAP_EXIT, 50, 2.5, 30, 10), _visible_rows(3),
             _gap_rows(GAP_EXIT, 60, 2.4, 20, 20), _visible_rows(3),
             _gap_rows(GAP_EXIT, 900, 2.5, 0, 50),          # too long: outside the cell
             _visible_rows(3),
             _gap_rows(GAP_EXIT, 50, 1.0, 40, 0),           # too slow: outside the cell
             _visible_rows(3),
             _gap_rows(GAP_LONG, 40, 2.6, 2, 38)]
    X1 = np.vstack([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    cells = cd.scenario_cells({"M1": (X1, y)}, pi_d=0.5)
    r2, r1 = cells["R2"], cells["R1"]
    assert (r2["n_frames"], r2["n_gaps"]) == (80, 2)
    assert r2["p_dead"] == pytest.approx(50 / 80)
    assert r2["d"] == pytest.approx(np.log(50 / 30))
    assert r2["low"] <= r2["d"] <= r2["high"]
    assert (r1["n_frames"], r1["n_gaps"]) == (40, 1) and r1["d"] < 0
    # One gap only: every bootstrap draw is that gap, so the interval collapses on d.
    assert r1["low"] == pytest.approx(r1["d"]) and r1["high"] == pytest.approx(r1["d"])
    again = cd.scenario_cells({"M1": (X1, y)}, pi_d=0.5)
    assert again == cells
    empty = cd.scenario_cells({"M1": _visible_rows(5)}, pi_d=0.5)
    assert empty["R2"]["n_frames"] == 0 and np.isnan(empty["R2"]["d"])


def test_files_with_interactions_validate_and_unknown_features_do_not(toy_calibration, tmp_path):
    _, path = toy_calibration
    doc = cd.load_calibration(path)
    model = unit_model(cd.ALL_FEATURES, {"exit_x_speed": 0.5})
    good = {**doc, "features": list(cd.ALL_FEATURES), "all": model,
            "folds": {m: model for m in MATCHES}}
    target = tmp_path / "inter.yaml"
    target.write_text(yaml.safe_dump(good), encoding="utf-8")
    assert cd.load_calibration(target)["features"] == list(cd.ALL_FEATURES)
    bad = {**good, "features": list(cd.ALL_FEATURES) + ["ball_x_speed"]}
    target.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="feature list"):
        cd.load_calibration(target)
