"""F1-08: possession-change hazard. Synthetic data only (never the match data)."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import yaml

from modeling import variants
from modeling.calibration import hazard as cal
from modeling.contracts import LABEL_CODES
from modeling.evidence import hazard as hz
from tests.modeling_fixtures import fuzz_master, swap_teams, team_params

H, A, D, X = (LABEL_CODES[k] for k in "HADX")
FEATURES = list(hz.RAW_FEATURES)
SETTINGS = {"target_w_frames": 3, "bridge_d_frames": 4, "window_frames": 3,
            "acc_clip_mps2": 100.0, "dist_clip_m": 30.0, "gap_clip_frames": 1500.0,
            "h_clip": 8.0}


def settings() -> dict:
    return {**SETTINGS, "team_params": team_params()}


def hazard_master(n: int, seed: int, half=None) -> pd.DataFrame:
    """fuzz_master plus the ball/gap columns the hazard reads (NaN patterns as in SCHEMA.md)."""
    rng = np.random.default_rng(seed)
    m = fuzz_master(n, seed)
    has_ball = np.isfinite(m["ball_x"].to_numpy())
    m["ball_acc"] = np.where(has_ball, rng.uniform(0, 150, n), np.nan)
    m["ball_dir_change"] = np.where(has_ball & (rng.uniform(size=n) < 0.7),
                                    rng.uniform(0, np.pi, n), np.nan)
    m["ball_speed"] = np.where(has_ball, rng.uniform(0, 20, n), np.nan)
    m["gap_length"] = np.where(has_ball, 0.0, rng.integers(1, 3000, n)).astype(float)
    m["since_last_obs"] = np.where(has_ball, 0.0, rng.integers(1, 3000, n)).astype(float)
    m["until_next_obs"] = np.where(has_ball, 0.0, rng.integers(1, 3000, n)).astype(float)
    m.loc[:4, "since_last_obs"] = np.where(has_ball[:5], 0.0, np.nan)
    if half is not None:
        m["half"] = np.asarray(half, dtype=np.int8)
        gap = m["half"].to_numpy() == 0
        for c in m.columns:
            if c != "half":
                m.loc[gap, c] = np.nan
    return m


def toy_model(columns, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    return {"window_frames": 3, "C": 1.0, "pi": 0.1, "intercept": -1.0,
            "mean": {c: float(rng.normal()) for c in columns},
            "std": {c: float(rng.uniform(0.5, 2.0)) for c in columns},
            "weight": {c: float(rng.normal()) for c in columns}}


def toy_calib(features=FEATURES, folds=("M1", "M2")) -> dict:
    columns = hz.design_columns(features)
    return {"features": list(features), "settings": settings(),
            "all": toy_model(columns, 0),
            "folds": {m: toy_model(columns, i + 1) for i, m in enumerate(folds)}}


def write_calib(tmp_path, calib) -> str:
    path = tmp_path / "hazard.yaml"
    path.write_text(yaml.safe_dump(calib), encoding="utf-8")
    return str(path)


# -- registry -----------------------------------------------------------------------------------

def test_registry_has_none_and_calibrated():
    variants.load_kinds()
    assert {"none", "calibrated"} <= set(variants.HAZARD_KINDS)
    m = hazard_master(10, 0)
    out = variants.HAZARD_KINDS["none"](m, {}, holdout=None)
    assert out.dtype == np.float32 and (out == 0).all()


def test_calibrated_needs_the_file():
    with pytest.raises(ValueError, match="calibrated_file"):
        hz.hazard_calibrated(hazard_master(10, 0), {})


# -- features -----------------------------------------------------------------------------------

def test_symmetry_under_team_swap_is_exact(tmp_path):
    half = np.r_[np.ones(300), np.zeros(20), np.full(280, 2)]
    m = hazard_master(len(half), 3, half=half)
    sw = swap_teams(m)
    raw, raw_sw = (hz.build_hazard_features(x, settings()) for x in (m, sw))
    pd.testing.assert_frame_equal(raw, raw_sw, check_exact=True)
    assert raw["near_flip"].sum() > 0 and (raw["dlogit"] > 0).sum() > 0
    params = {"calibrated_file": write_calib(tmp_path, toy_calib())}
    h = hz.hazard_calibrated(m, params)
    np.testing.assert_array_equal(h, hz.hazard_calibrated(sw, params))
    assert (h != 0).any()


def test_zero_on_halftime_and_float32_finite_clipped(tmp_path):
    half = np.r_[np.ones(100), np.zeros(30), np.full(100, 2)]
    m = hazard_master(len(half), 4, half=half)
    calib = toy_calib()
    for model in [calib["all"]] + list(calib["folds"].values()):
        model["weight"] = {k: 50 * v for k, v in model["weight"].items()}
    h = hz.hazard_calibrated(m, {"calibrated_file": write_calib(tmp_path, calib)})
    assert h.dtype == np.float32 and np.isfinite(h).all()
    assert (h[half == 0] == 0).all()
    assert np.abs(h).max() <= 8.0 and np.isclose(np.abs(h).max(), 8.0)
    assert hz.build_hazard_features(m, settings()).loc[half == 0].isna().all().all()


def test_windows_never_cross_a_half():
    """Changing the second half leaves the windowed features of the first half unchanged."""
    half = np.r_[np.ones(60), np.zeros(10), np.full(60, 2)]
    m = hazard_master(len(half), 5, half=half)
    m2 = m.copy()
    second = half == 2
    m2.loc[second, "d_diff"] = -m2.loc[second, "d_diff"]
    m2.loc[second, ["d_near_home", "d_near_away"]] = \
        m2.loc[second, ["d_near_away", "d_near_home"]].to_numpy()
    m2.loc[second, "n_home_r2"] = 0.0
    first = half == 1
    a = hz.build_hazard_features(m, settings()).loc[first]
    b = hz.build_hazard_features(m2, settings()).loc[first]
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_near_flip_and_dlogit_on_a_clear_switch():
    """Home closer for 20 frames, then away closer: the flip fires only around frame 20."""
    n = 40
    m = hazard_master(n, 6)
    m["d_near_home"] = np.r_[np.full(20, 1.0), np.full(20, 6.0)]
    m["d_near_away"] = np.r_[np.full(20, 6.0), np.full(20, 1.0)]
    m["d_diff"] = m["d_near_home"] - m["d_near_away"]
    m["ball_observed"], m["ball_filled"] = 1.0, 0.0
    m["ball_x"], m["ball_y"] = 50.0, 30.0
    m["speed_rel_home"], m["speed_rel_away"] = 1.0, 1.0
    m["n_home_r2"], m["n_away_r2"] = 1.0, 0.0
    raw = hz.build_hazard_features(m, settings())
    flip = np.flatnonzero(raw["near_flip"].to_numpy())
    assert flip.min() >= 20 - 3 and flip.max() <= 20 + 2 and 19 in flip and 20 in flip
    dl = raw["dlogit"].to_numpy()
    assert dl[19] > 0 and dl[0] == 0 and dl[-1] == 0 and dl[5] == 0


def test_design_matrix_nan_contributes_zero_with_indicator():
    raw = pd.DataFrame({"ball_acc": [1.0, np.nan], "near_flip": [1.0, 0.0]})
    x = hz.design_matrix(raw, ["ball_acc", "near_flip"])
    assert hz.design_columns(["ball_acc", "near_flip"]) == ["ball_acc", "has_ball_acc",
                                                            "near_flip"]
    np.testing.assert_array_equal(x, [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="unknown hazard feature"):
        hz.design_columns(["speed_home"])


# -- prior correction and holdout ---------------------------------------------------------------

def test_neutral_features_give_h_close_to_zero(tmp_path):
    """Weights 0 and intercept = logit(pi): the model says only the prior, so h == 0."""
    calib = toy_calib()
    for model in [calib["all"]] + list(calib["folds"].values()):
        model["weight"] = {k: 0.0 for k in model["weight"]}
        model["intercept"] = float(np.log(model["pi"] / (1 - model["pi"])))
    h = hz.hazard_calibrated(hazard_master(50, 7),
                             {"calibrated_file": write_calib(tmp_path, calib)})
    np.testing.assert_allclose(h, 0.0, atol=1e-6)


def test_prior_correction_on_a_fitted_model():
    """Features independent of the target: the fitted logit sits at logit(pi), so h ~ 0."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(20000, 3))
    y = (rng.uniform(size=20000) < 0.2).astype(int)
    model = cal.fit_model(x, y, ["a", "b", "c"], 1.0, 1000)
    h = hz.model_logit(x, model, ["a", "b", "c"]) - hz.logit(model["pi"])
    assert np.abs(h).mean() < 0.05 and np.abs(h).max() < 0.2


def test_holdout_selects_the_fold(tmp_path):
    calib = toy_calib()
    path = write_calib(tmp_path, calib)
    m = hazard_master(80, 8)
    by_holdout = {k: hz.hazard_calibrated(m, {"calibrated_file": path}, holdout=k)
                  for k in (None, "M1", "M2")}
    np.testing.assert_array_equal(by_holdout[None], hz.hazard_from_model(m, calib, calib["all"]))
    for k in ("M1", "M2"):
        np.testing.assert_array_equal(by_holdout[k],
                                      hz.hazard_from_model(m, calib, calib["folds"][k]))
    assert not np.array_equal(by_holdout["M1"], by_holdout["M2"])
    with pytest.raises(ValueError, match="folds"):
        hz.hazard_calibrated(m, {"calibrated_file": path}, holdout="M3")


def test_load_rejects_incomplete_models(tmp_path):
    calib = toy_calib()
    del calib["folds"]["M1"]["weight"]["dlogit"]
    with pytest.raises(ValueError, match="weight"):
        hz.load_hazard_file(write_calib(tmp_path, calib))


# -- target -------------------------------------------------------------------------------------

def test_change_targets_direct_bridged_and_not_changes():
    lab = np.array([H] * 10 + [A] * 10 + [D] * 2 + [H] * 10 + [D] * 6 + [A] * 10
                   + [D] * 2 + [A] * 10)
    half = np.ones(len(lab), dtype=np.int8)
    y, eligible = cal.change_targets(lab, half, target_w=3, bridge_d=4)
    expected = np.zeros(len(lab), dtype=np.int8)
    expected[7:13] = 1            # direct H->A at frame 10
    expected[17:25] = 1           # A, D(2) at 20..21, H: bridged -> [20-3, 22+3)
    # H, D(6), A is not a change (D too long); A, D(2), A is not a change (same team)
    np.testing.assert_array_equal(y, expected)
    np.testing.assert_array_equal(eligible, np.isin(lab, (H, A)))


def test_change_targets_do_not_cross_halftime():
    lab = np.array([H] * 5 + [X] * 3 + [A] * 5)
    half = np.array([1] * 5 + [0] * 3 + [2] * 5, dtype=np.int8)
    y, eligible = cal.change_targets(lab, half, target_w=3, bridge_d=10)
    assert y.sum() == 0 and not eligible[5:8].any()
    lab2 = np.array([H] * 5 + [A] * 5)
    y2, _ = cal.change_targets(lab2, np.array([1] * 3 + [2] * 7, dtype=np.int8), 2, 10)
    np.testing.assert_array_equal(y2, [0, 0, 0, 1, 1, 1, 1, 0, 0, 0])


def test_still_ball_mask_needs_more_than_min_frames():
    n = 30
    m = pd.DataFrame({"half": np.ones(n, dtype=np.int8),
                      "ball_observed": np.r_[np.ones(25), np.zeros(5)],
                      "ball_speed": np.r_[np.full(12, 0.1), [5.0], np.full(12, 0.1),
                                          np.full(5, np.nan)]})
    assert not cal.still_ball_mask(m, 0.3, 12).any()
    np.testing.assert_array_equal(cal.still_ball_mask(m, 0.3, 11),
                                  np.r_[np.ones(12), 0, np.ones(12), np.zeros(5)].astype(bool))


# -- fit ----------------------------------------------------------------------------------------

def test_fit_recovers_known_weights():
    rng = np.random.default_rng(2)
    n, w, b = 60000, np.array([1.5, -1.0, 0.0, 0.5]), -1.2
    x = rng.normal(size=(n, 4)) * [1.0, 2.0, 0.5, 3.0] + [0.0, 1.0, -1.0, 2.0]
    z = (x - x.mean(0)) / x.std(0)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(b + z @ w)))).astype(int)
    model = cal.fit_model(x, y, list("abcd"), 1.0, 1000)
    got = np.array([model["weight"][c] for c in "abcd"])
    np.testing.assert_allclose(got, w, atol=0.06)
    assert abs(model["intercept"] - b) < 0.06
    assert model["converged"]


def test_fit_is_deterministic_and_handles_constant_columns():
    rng = np.random.default_rng(3)
    x = np.column_stack([rng.normal(size=5000), np.ones(5000)])
    y = (rng.uniform(size=5000) < 0.3).astype(int)
    a = cal.fit_model(x, y, ["a", "k"], 0.1, 1000)
    b = cal.fit_model(x.copy(), y.copy(), ["a", "k"], 0.1, 1000)
    assert a == b
    assert a["std"]["k"] == 1.0 and np.isfinite(a["weight"]["k"])
    with pytest.raises(ValueError, match="single class"):
        cal.fit_model(x, np.zeros(5000, int), ["a", "k"], 0.1, 1000)


def _synthetic_data(seed, n_matches=4, n=3000):
    rng = np.random.default_rng(seed)
    data = {}
    for i in range(n_matches):
        x = rng.normal(size=(n, 2))
        y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(-1.0 + 1.2 * x[:, 0])))).astype(np.int64)
        sub = np.zeros(n, dtype=bool)
        sub[::2] = True
        data["M{}".format(i)] = {"x": x, "y": y, "still": rng.uniform(size=n) < 0.1, "sub": sub}
    return data


def _by_window(seed):
    """Two windows; window 7 has an extra noise column so the two give different log-losses."""
    base = _synthetic_data(seed)
    noisy = copy.deepcopy(base)
    rng = np.random.default_rng(seed + 100)
    for d in noisy.values():
        d["x"] = d["x"] + rng.normal(scale=2.0, size=d["x"].shape)
    return {3: base, 7: noisy}


def test_nested_folds_never_see_their_holdout():
    data = _by_window(4)
    matches = list(data[3])
    fit = {"c_grid": [0.01, 0.1, 1.0], "max_iter": 1000}
    res = cal.fit_all(data, matches, ["a", "b"], fit)
    assert set(res["folds"]) == set(matches)
    assert len(res["scores"]["all"]) == 6
    for held in matches:
        train = [m for m in matches if m != held]
        w, c, _ = cal.choose_params(data, train, ["a", "b"], fit["c_grid"], 1000)
        expected = cal.fit_model(*cal._stack(data[w], train, sub=True), ["a", "b"], c, 1000)
        assert res["folds"][held] == {"window_frames": w, **expected}
    assert res["all"]["window_frames"] == 3          # the clean window wins
    assert res["all"]["n_train"] == sum(d["sub"].sum() for d in data[3].values())


def test_choose_params_breaks_ties_towards_smaller_window_and_c():
    data = _synthetic_data(5)
    w, c, scores = cal.choose_params({3: data, 9: copy.deepcopy(data)}, list(data), ["a", "b"],
                                     [1e6, 1e7], 1000)
    assert w == 3
    assert scores["W=3 C=1000000.0"] == scores["W=9 C=1000000.0"]
    with pytest.raises(ValueError, match="at least 2"):
        cal.choose_params({3: data}, ["M0"], ["a", "b"], [1.0], 1000)


def test_model_window_drives_the_features(tmp_path):
    """Two folds identical except for window_frames give different h."""
    calib = toy_calib()
    calib["folds"]["M2"] = copy.deepcopy(calib["folds"]["M1"])
    calib["folds"]["M2"]["window_frames"] = 9
    path = write_calib(tmp_path, calib)
    m = hazard_master(200, 11)
    a = hz.hazard_calibrated(m, {"calibrated_file": path}, holdout="M1")
    b = hz.hazard_calibrated(m, {"calibrated_file": path}, holdout="M2")
    assert not np.array_equal(a, b)


def test_change_events_and_classification():
    lab = np.array([H] * 10 + [A] * 10 + [D] * 3 + [H] * 10 + [A] * 10)
    half = np.ones(len(lab), dtype=np.int8)
    true_ev = cal.change_events(lab, half, 0)
    assert [e[:3] for e in true_ev] == [(10, H, A), (33, H, A)]
    assert [e[:3] for e in cal.change_events(lab, half, 5)] == [(10, H, A), (21, A, H),
                                                                 (33, H, A)]
    pred = [(14, H, A, 0, 43), (60, H, A, 0, 100), (11, A, H, 0, 43)]
    assert cal.classify_changes(true_ev, pred, on_time=5, tol=20) == ["on_time", "missed"]
    pred2 = [(10, H, A, 0, 43), (45, H, A, 0, 43)]
    assert cal.classify_changes(true_ev, pred2, on_time=5, tol=20) == ["on_time", "off_time"]
    # one prediction never serves two true changes
    assert cal.classify_changes(true_ev, [(20, H, A, 0, 43)], 5, 20) == ["off_time", "missed"]


def test_window_peak_and_control_centres_stay_inside_the_half():
    h = np.arange(20, dtype=float)
    assert cal.window_peak(h, 5, 3, 0, 10) == 7.0
    assert cal.window_peak(h, 9, 3, 0, 10) == 9.0          # clipped at the block end
    half = np.r_[np.ones(20), np.zeros(4), np.full(20, 2)].astype(np.int8)
    y = np.zeros(44, dtype=np.int8)
    y[30:33] = 1
    eligible = half != 0
    centres = cal.control_centres(y, eligible, half, 3)
    assert all(start <= c - 3 and c + 3 <= stop for c, start, stop in centres)
    assert all(not y[c - 3:c + 3].any() for c, _, _ in centres)
    assert [c for c, _, _ in centres] == [3, 9, 15, 27, 39]


def test_summarize_peaks_auc_against_control():
    out = cal.summarize_peaks({"on_time": [3.0, 2.0], "off_time": [], "missed": [0.5, -1.0],
                               "control": [0.0, -2.0]})
    assert out["on_time"]["auc_vs_control"] == 1.0 and out["on_time"]["frac_gt1"] == 1.0
    assert out["missed"]["auc_vs_control"] == 0.75 and out["missed"]["frac_pos"] == 0.5
    assert np.isnan(out["off_time"]["median"]) and np.isnan(out["control"]["auc_vs_control"])


def test_artifact_checks_fire_on_large_gaps():
    fit = {"artifact_auc_margin": 0.1, "degrade_fit": "default",
           "still_ball_features": ["ball_acc"], "gap_features": ["gap_length", "since_last_obs"]}
    feats = ["ball_acc", "gap_length", "since_last_obs", "dlogit"]

    def fa(gap_auc, since_auc):
        return {"gap_length": {"all": gap_auc}, "since_last_obs": {"all": since_auc},
                "has_since_last_obs": {"all": 0.5}}

    quiet = cal.artifact_checks({"default": {"still": {"in": 0.70, "out": 0.65}}},
                                {"hidelo": fa(0.6, 0.6), "hidehi": fa(0.65, 0.62)}, fit, feats)
    assert quiet["drop"] == [] and not quiet["still_fires"] and not quiet["gap_fires"]
    loud = cal.artifact_checks({"default": {"still": {"in": 0.90, "out": 0.65}}},
                               {"hidelo": fa(0.6, 0.6), "hidehi": fa(0.75, 0.6)}, fit, feats)
    assert loud["still_fires"] and loud["gap_fires"]
    assert loud["drop"] == ["ball_acc", "gap_length", "since_last_obs"]
    nan_slice = cal.artifact_checks({"default": {"still": {"in": float("nan"), "out": 0.6}}},
                                    {}, fit, feats)
    assert nan_slice["drop"] == []


def test_fit_config_is_strict(tmp_path):
    from modeling import paths
    fit = cal.load_fit_config(paths.repo_root() / "configs" / "hazard_fit.yaml")
    assert fit["team_params"]["kappa"] == 3.0 and fit["c_grid"] == [0.01, 0.1, 1.0]
    assert fit["window_grid"] == [12, 25, 50] and fit["bridge_d_frames"] == 0
    raw = yaml.safe_load((paths.repo_root() / "configs" / "hazard_fit.yaml").read_text())
    for bad in ({"extra": 1}, {"features": ["nope"]}, {"c_grid": [0.1, 0.1]},
                {"window_grid": [0]}, {"on_time_frames": 60}):
        cfg = copy.deepcopy(raw)
        cfg.update(bad)
        cfg["team_config"] = str(paths.repo_root() / "configs" / "v1.yaml")
        path = tmp_path / "fit.yaml"
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        with pytest.raises(ValueError):
            cal.load_fit_config(path)


def test_match_data_and_payload_round_trip(tmp_path):
    """End to end on a synthetic master: fit, write, read back, predict with the variant."""
    from modeling import paths
    fit = cal.load_fit_config(paths.repo_root() / "configs" / "hazard_fit.yaml")
    fit.update(target_w_frames=3, bridge_d_frames=4, window_frames=3, subsample_every=2)
    half = np.r_[np.ones(400), np.zeros(10), np.full(400, 2)]
    m = hazard_master(len(half), 9, half=half)
    rng = np.random.default_rng(9)
    lab = np.where(half == 0, X, np.repeat(rng.choice([H, A, D], len(half) // 15 + 1),
                                           15)[:len(half)]).astype(np.int8)
    d = cal.match_data(m, lab, fit, fit["features"], 3)
    assert d["x"].shape == (int(np.isin(lab, (H, A)).sum()), len(hz.design_columns(
        fit["features"])))
    assert np.isfinite(d["x"]).all() and d["y"].sum() > 0
    names = ["M1", "M2", "M3"]
    data = {3: {k: copy.deepcopy(d) for k in names}}
    models = cal.fit_all(data, names, hz.design_columns(fit["features"]),
                         {"c_grid": [0.1], "max_iter": 1000})
    rounds = [{"features": fit["features"], "models": models}]
    out = cal.write_calib(cal.calib_payload(fit, rounds, names), tmp_path / "h.yaml")
    h = hz.hazard_calibrated(m, {"calibrated_file": str(out)}, holdout="M1")
    assert h.dtype == np.float32 and np.isfinite(h).all() and (h[half == 0] == 0).all()
