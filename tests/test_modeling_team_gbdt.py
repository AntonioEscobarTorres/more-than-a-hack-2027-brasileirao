"""Tests for the F1-07 GBDT team_logit, its window features and its calibration.

Synthetic data only.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import yaml

from modeling import contracts, variants
from modeling.calibration.team import calibrate_team, write_team_yaml
from modeling.calibration.team_gbdt import (
    MODEL_PARAMS,
    calibrate_team_gbdt,
    evaluate_lam_grid,
    fit_gbdt,
    match_samples,
    save_bundle,
    write_gbdt_yaml,
)
from modeling.evidence.team import team_logit_gbdt
from modeling.evidence.team_gbdt import (
    RAW_PAIRS,
    antisymmetric_logit,
    build_gbdt_features,
    gbdt_feature_layout,
    load_gbdt_bundle,
    swap_matrix,
    swap_permutation,
)
from modeling.evidence.windows import (
    build_window_features,
    rolling_window_stats,
    window_feature_roles,
)
from tests.modeling_fixtures import swap_teams, team_params
from tests.test_modeling_calibration_team import _rich_synthetic_master

SMALL_MODEL = {**MODEL_PARAMS, "max_iter": 15, "min_samples_leaf": 5}
F_PARAMS = {"bridge_median_frames": 7, "bridge_tau_frames": [15, 35], "r_touch_m": 1.5,
            "touch_min_frames": 5, "touch_win_frames": 9, "touch_dir_change_rad": 0.5,
            "touch_acc_mps2": 8.0, "touch_require_abrupt": False, "window_sizes": [3, 6],
            "include_bridge": True}


def _full_master(n: int = 300, seed: int = 11):
    """``_rich_synthetic_master`` plus every column the GBDT features read."""
    master, gt = _rich_synthetic_master(n, seed=seed)
    rng = np.random.default_rng(seed + 1)
    for home, away in RAW_PAIRS:
        for col in (home, away):
            if col not in master.columns:
                master[col] = rng.integers(0, 8, n).astype(np.float64)
    for col in ("ball_speed", "d_touchline", "d_goalline", "since_last_obs", "until_next_obs"):
        master[col] = rng.integers(0, 30, n).astype(np.float64)
    master["pre_kickoff"] = False
    half = master["half"].to_numpy()
    master.loc[half == 0, ["ball_speed", "since_last_obs", "until_next_obs"]] = np.nan
    return master, gt


# --- windows ---------------------------------------------------------------------------------

def test_windows_cover_exactly_the_specified_frames():
    x = np.arange(10, dtype=float)
    r = rolling_window_stats(x, np.ones(10, dtype=np.int8), 2)
    assert r["past_mean"][5] == 4.0 and r["past_min"][5] == 3.0      # [3, 5]
    assert r["fut_mean"][5] == 6.0 and r["fut_min"][5] == 5.0        # [5, 7]
    assert r["cent_mean"][5] == 5.0 and r["cent_min"][5] == 3.0      # [3, 7]


def test_windows_at_half_edges_use_only_existing_frames_and_never_cross():
    half = np.array([1] * 5 + [0] * 2 + [2] * 3, dtype=np.int8)
    x = np.array([1, 2, 3, 4, 5, 100, 100, 7, 8, 9], dtype=float)
    r = rolling_window_stats(x, half, 3)
    assert r["past_mean"][0] == 1.0                     # only frame 0 exists
    assert r["fut_mean"][4] == 5.0                      # end of half 1, gap not used
    assert r["cent_mean"][7] == 8.0                     # [7, 9], half 1 not used
    assert np.isnan(r["past_mean"][5:7]).all()          # half == 0


def test_windows_are_nan_without_finite_values():
    x = np.array([1.0, np.nan, np.nan, np.nan, 5.0])
    r = rolling_window_stats(x, np.ones(5, dtype=np.int8), 1)
    assert np.isnan(r["past_mean"][3]) and np.isnan(r["fut_min"][2])
    assert r["past_mean"][1] == 1.0                     # NaN skipped


def test_window_features_do_not_depend_on_the_other_half():
    m1, _ = _full_master(240, seed=1)
    m2 = m1.copy(deep=True)
    second = m2["half"] == 2
    m2.loc[second, ["d_near_home", "d_near_away", "pressure_home"]] = 33.0
    m2.loc[second, "d_diff"] = 0.0
    p = {"window_sizes": [3, 6, 50]}
    a, b = build_window_features(m1, p), build_window_features(m2, p)
    first = m1["half"] == 1
    pd.testing.assert_frame_equal(a.loc[first], b.loc[first])
    assert a.loc[m1["half"] == 0].isna().all().all()


def test_window_roles_cover_every_column():
    m, _ = _full_master(60)
    feats = build_window_features(m, {"window_sizes": [3, 6]})
    roles = window_feature_roles([3, 6])
    named = [c for p in roles["pairs"] for c in p] + roles["odd"] + roles["even"]
    assert sorted(named) == sorted(feats.columns)


# --- swap and antisymmetry -------------------------------------------------------------------

def test_excluding_one_side_drops_the_whole_pair_and_unknown_names_fail():
    full = gbdt_feature_layout(F_PARAMS)["features"]
    layout = gbdt_feature_layout({**F_PARAMS, "exclude_features": ["n_tracked_home",
                                                                   "ev_coll_spread"]})
    gone = set(full) - set(layout["features"])
    assert gone == {"n_tracked_home", "n_tracked_away", "ev_coll_spread"}
    perm, sign = swap_permutation(layout)
    X = np.random.default_rng(1).normal(size=(5, len(layout["features"])))
    np.testing.assert_array_equal(swap_matrix(swap_matrix(X, perm, sign), perm, sign), X)
    with pytest.raises(ValueError, match="unknown"):
        gbdt_feature_layout({**F_PARAMS, "exclude_features": ["nope"]})


def test_feature_swap_is_an_exact_involution():
    layout = gbdt_feature_layout(F_PARAMS)
    perm, sign = swap_permutation(layout)
    X = np.random.default_rng(0).normal(size=(50, len(layout["features"])))
    X[::7, 3] = np.nan
    back = swap_matrix(swap_matrix(X, perm, sign), perm, sign)
    np.testing.assert_array_equal(back, X)


def test_feature_swap_matches_features_of_the_swapped_master():
    m, _ = _full_master(300, seed=3)
    p = team_params()
    feats, valid = build_gbdt_features(m, p, F_PARAMS)
    feats_s, valid_s = build_gbdt_features(swap_teams(m), p, F_PARAMS)
    layout = gbdt_feature_layout(F_PARAMS)
    perm, sign = swap_permutation(layout)
    np.testing.assert_array_equal(valid, valid_s)
    expected = swap_matrix(feats.to_numpy(), perm, sign)
    got = feats_s.to_numpy()
    np.testing.assert_array_equal(np.isnan(expected), np.isnan(got))
    ok = ~np.isnan(got)
    np.testing.assert_allclose(got[ok], expected[ok], atol=1e-9)


def _trained(seed=5):
    m, gt = _full_master(400, seed=seed)
    s = match_samples(m, gt, team_params(), F_PARAMS, step=1)
    layout = gbdt_feature_layout(F_PARAMS)
    perm, sign = swap_permutation(layout)
    X, y, _ = s["ball"]
    return fit_gbdt(X, y, perm, sign, SMALL_MODEL), X, perm, sign, layout


def test_logit_is_exactly_antisymmetric_under_the_swap():
    clf, X, perm, sign, _ = _trained()
    t = antisymmetric_logit(clf, X, perm, sign)
    t_s = antisymmetric_logit(clf, swap_matrix(X, perm, sign), perm, sign)
    np.testing.assert_array_equal(t_s, -t)
    assert np.abs(t).max() > 0.1


def test_training_is_deterministic():
    clf1, X, perm, sign, _ = _trained()
    clf2, *_ = _trained()
    np.testing.assert_array_equal(clf1.decision_function(X), clf2.decision_function(X))


# --- calibration, files and the registered kind ----------------------------------------------

@pytest.fixture(scope="module")
def calibrated(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("gbdt")
    matches = {"m1": _full_master(300, seed=10), "m2": _full_master(300, seed=20),
               "m3": _full_master(300, seed=30)}
    load = lambda match, degrade: matches[match]  # noqa: E731
    cfg = {"name": "t", "team": {"kind": "rules_v1", "params": team_params(),
                                 "calibrated_file": None}}
    logistic = calibrate_team(cfg, variant="touch", matches=list(matches), load_fn=load,
                              touch_params={"r_touch_m": 1.5, "touch_min_frames": 5})
    logistic_path = write_team_yaml(logistic, tmp / "team.yaml")
    table = calibrate_team_gbdt(cfg, logistic_file=logistic_path, matches=list(matches),
                                step=1, window_sizes=[3, 6], model_params=SMALL_MODEL,
                                all_model_path=tmp / "all.joblib", fold_dir=tmp / "folds",
                                load_fn=load, log=lambda *_: None)
    path = write_gbdt_yaml(table, tmp / "team_gbdt.yaml")
    params = {**team_params(), "calibrated_file": str(path)}
    return {"tmp": tmp, "matches": matches, "table": table, "path": path, "params": params,
            "load": load, "logistic_path": logistic_path}


def test_gbdt_is_registered():
    variants.load_kinds()
    assert variants.TEAM_KINDS["gbdt"] is team_logit_gbdt


def test_calibration_writes_fold_and_all_models(calibrated):
    tmp, table = calibrated["tmp"], calibrated["table"]
    assert (tmp / "all.joblib").is_file()
    for m in ("m1", "m2", "m3"):
        assert (tmp / "folds" / "{}.joblib".format(m)).is_file()
        fm = table["meta"]["fold_metrics"][m]
        assert np.isfinite(fm["ball_log_loss"]) and np.isfinite(fm["ball_log_loss_logistic"])
    assert table["sklearn_version"] == __import__("sklearn").__version__


def test_team_logit_gbdt_contract_and_master_swap(calibrated):
    m, _ = calibrated["matches"]["m1"]
    params = calibrated["params"]
    t = team_logit_gbdt(m, params)
    assert t.dtype == np.float32 and len(t) == len(m)
    assert np.isfinite(t).all() and np.abs(t).max() <= contracts.TEAM_LOGIT_MAX
    assert (t[m["half"].to_numpy() == 0] == 0).all()
    assert np.abs(t).max() > 0.1
    t_s = team_logit_gbdt(swap_teams(m), params)
    np.testing.assert_array_equal(t_s, -t)


def test_holdout_uses_its_fold_model(calibrated):
    m, _ = calibrated["matches"]["m2"]
    params = calibrated["params"]
    t_all = team_logit_gbdt(m, params)
    t_m2 = team_logit_gbdt(m, params, holdout="m2")
    t_m3 = team_logit_gbdt(m, params, holdout="m3")
    assert not np.array_equal(t_all, t_m2) and not np.array_equal(t_m2, t_m3)
    # Same predictions as the fold bundle used directly.
    bundle = load_gbdt_bundle(calibrated["tmp"] / "folds" / "m2.joblib",
                              sklearn_version=calibrated["table"]["sklearn_version"],
                              features=calibrated["table"]["features"])
    feats, valid = build_gbdt_features(m, params, calibrated["table"]["feature_params"])
    perm, sign = swap_permutation(gbdt_feature_layout(calibrated["table"]["feature_params"]))
    direct = antisymmetric_logit(bundle["ball"], feats.to_numpy()[valid], perm, sign)
    np.testing.assert_array_equal(t_m2[valid], np.clip(direct, -8, 8).astype(np.float32))
    with pytest.raises(FileNotFoundError):
        team_logit_gbdt(m, params, holdout="unknown")


def test_save_and_load_preserve_predictions_bit_for_bit(calibrated, tmp_path):
    clf, X, perm, sign, layout = _trained()
    path = save_bundle(tmp_path / "b.joblib", {"ball": clf, "gap": clf}, layout["features"])
    import sklearn
    loaded = load_gbdt_bundle(path, sklearn_version=sklearn.__version__,
                              features=layout["features"])
    np.testing.assert_array_equal(loaded["ball"].decision_function(X), clf.decision_function(X))
    with pytest.raises(RuntimeError, match="scikit-learn"):
        load_gbdt_bundle(path, sklearn_version="0.0.0", features=layout["features"])
    with pytest.raises(ValueError, match="feature list"):
        load_gbdt_bundle(path, sklearn_version=sklearn.__version__, features=["x"])


def test_logistic_gap_model_uses_the_f1_05_gap(calibrated, tmp_path):
    m, _ = calibrated["matches"]["m1"]
    path = write_gbdt_yaml(calibrated["table"], tmp_path / "logistic_gap.yaml",
                           gap_model="logistic")
    raw = yaml.safe_load(path.read_text())
    raw["logistic_file"] = str(calibrated["logistic_path"])
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    params = {**team_params(), "calibrated_file": str(path)}
    t_log = team_logit_gbdt(m, params)
    t_gbdt = team_logit_gbdt(m, calibrated["params"])
    _, valid = build_gbdt_features(m, params, calibrated["table"]["feature_params"])
    np.testing.assert_array_equal(t_log[valid], t_gbdt[valid])

    from modeling.evidence.team import team_logit_calibrated
    t_cal = team_logit_calibrated(m, {**team_params(),
                                      "calibrated_file": str(calibrated["logistic_path"])})
    gap = ~valid & (m["half"].to_numpy() != 0)
    np.testing.assert_allclose(t_log[gap], t_cal[gap], atol=1e-6)


def test_calibration_is_deterministic_and_rejects_degrade_none(calibrated, tmp_path):
    cfg = {"name": "t", "team": {"kind": "rules_v1", "params": team_params(),
                                 "calibrated_file": None}}
    again = calibrate_team_gbdt(cfg, logistic_file=calibrated["logistic_path"],
                                matches=["m1", "m2", "m3"], step=1, window_sizes=[3, 6],
                                model_params=SMALL_MODEL, all_model_path=tmp_path / "all.joblib",
                                fold_dir=tmp_path / "folds", load_fn=calibrated["load"],
                                log=lambda *_: None)
    a, b = copy.deepcopy(again), copy.deepcopy(calibrated["table"])
    for t in (a, b):
        for k in ("model_file", "fold_dir"):
            t.pop(k)
        t["meta"].pop("all_model_bytes")
    assert a == b
    with pytest.raises(ValueError, match="forbidden"):
        calibrate_team_gbdt(cfg, degrade="none", matches=["m1", "m2"])


def test_evaluate_lam_grid_runs_leave_one_match_out(calibrated):
    from tests.modeling_fixtures import _DEAD, _DECODER, _HAZARD, _POST
    cfg = {"name": "g", "pipeline": "hmm", "ball_source": "tactical",
           "team": {"kind": "gbdt", "params": team_params(),
                    "calibrated_file": str(calibrated["path"])},
           "dead": copy.deepcopy(_DEAD), "decoder": copy.deepcopy(_DECODER),
           "postprocess": copy.deepcopy(_POST), "hazard": copy.deepcopy(_HAZARD)}
    df = evaluate_lam_grid({"g": cfg}, [0.2, 0.4], degrade="default",
                           matches=["m1", "m2"], load_fn=calibrated["load"])
    assert len(df) == 4 and df["spell_f1_mean"].between(0, 1).all()


# --- tools/gbdt_feature_shift.py -------------------------------------------------------------

def _tool_cfg(calibrated):
    path = calibrated["path"]
    raw = yaml.safe_load(path.read_text())
    raw["logistic_file"] = str(calibrated["logistic_path"])
    fixed = path.with_name("team_gbdt_tool.yaml")
    fixed.write_text(yaml.safe_dump(raw, sort_keys=False))
    return {"team": {"kind": "gbdt", "params": team_params(), "calibrated_file": str(fixed)}}


def _run_tool(monkeypatch, capsys, calibrated, ref, argv=()):
    from tools import gbdt_feature_shift as gfs
    cfg = _tool_cfg(calibrated)
    monkeypatch.setattr(gfs, "_n_frames", lambda: len(ref))
    monkeypatch.setattr(gfs, "load_config", lambda _p: cfg)
    monkeypatch.setattr(gfs.io, "load_master", lambda *_a, **_k: ref)
    monkeypatch.setattr(gfs, "load_external", lambda m, _d: calibrated["matches"][m])
    code = gfs.main(["--match", "m1", "m2", "m3", *argv])
    return code, capsys.readouterr().out


def test_tool_passes_on_a_master_from_the_training_distribution(monkeypatch, capsys,
                                                                 calibrated):
    ref, _ = _full_master(300, seed=10)
    code, out = _run_tool(monkeypatch, capsys, calibrated, ref)
    assert "PASS  (a)" in out and "PASS  (b)" in out
    assert code == 0 and out.strip().splitlines()[-1] == "PASS"
    # Default output: PASS/FAIL lines only, no numbers.
    assert not any(ch.isdigit() for line in out.splitlines() for ch in line
                   if not line.startswith(("PASS", "FAIL")))


def test_tool_fails_and_names_features_out_of_range(monkeypatch, capsys, calibrated):
    ref, _ = _full_master(300, seed=10)
    ref["d_touchline"] = 500.0
    ref["d_goalline"] = 500.0
    code, out = _run_tool(monkeypatch, capsys, calibrated, ref)
    assert code == 1 and "FAIL  (a)" in out
    line = next(l for l in out.splitlines() if "failing features" in l)
    assert "d_touchline" in line and "d_goalline" in line
    assert not any(ch.isdigit() for ch in line)  # names only
