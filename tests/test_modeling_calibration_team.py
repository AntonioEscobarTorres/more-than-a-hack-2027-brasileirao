"""Tests for team_logit calibration, pass-bridge features, touch feature and tag prior.

Synthetic data only.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from evaluation import possession_metrics as pm
from modeling import contracts, variants
from modeling.calibration.team import (
    VARIANT_NAMES,
    calibrate_team,
    write_team_yaml,
)
from modeling.evidence.team import (
    build_pass_bridge_features,
    build_team_calibration_features,
    build_touch_feature,
    team_logit_calibrated,
    team_tag_prior,
)
from tests.modeling_fixtures import PITCH_L, make_master, swap_teams, team_params


def _rich_synthetic_master(n: int = 300, seed: int = 11) -> tuple[pd.DataFrame, np.ndarray]:
    """Synthetic two-half match where ground-truth possession follows team proximity and advance."""
    rng = np.random.default_rng(seed)
    half = np.ones(n, dtype=np.int8)
    half[n // 2 - 10:n // 2 + 10] = 0
    half[n // 2 + 10:] = 2

    state = np.where((np.arange(n) // 40) % 2 == 0, 1.0, -1.0)  # +1 -> H, -1 -> A
    d_home = np.where(state > 0, rng.uniform(0.5, 2.0, n), rng.uniform(4.0, 9.0, n))
    d_away = np.where(state > 0, rng.uniform(4.0, 9.0, n), rng.uniform(0.5, 2.0, n))

    ball_obs = (rng.uniform(0, 1, n) > 0.25).astype(float)
    ball_obs[half == 0] = 0.0

    # Advancing centroids aligned with possession state
    t_idx = np.arange(n, dtype=float)
    c_home_x = 50.0 + 2.5 * state * np.sin(t_idx / 10.0)
    c_away_x = 50.0 + 2.5 * state * np.sin(t_idx / 10.0)

    m = make_master(
        n,
        half=half,
        ball_observed=ball_obs,
        ball_filled=0.0,
        d_near_home=d_home,
        d_near_away=d_away,
        n_home_r2=(state > 0).astype(float),
        n_away_r2=(state < 0).astype(float),
        n_home_r3=np.where(state > 0, 2.0, 0.0),
        n_away_r3=np.where(state < 0, 2.0, 0.0),
        pressure_home=np.where(state > 0, 1.5, 0.2),
        pressure_away=np.where(state < 0, 1.5, 0.2),
        speed_rel_home=rng.uniform(0.5, 4.0, n),
        speed_rel_away=rng.uniform(0.5, 4.0, n),
        centroid_home_x=c_home_x,
        centroid_away_x=c_away_x,
        spread_home_x=np.where(state > 0, 14.0, 10.0),
        spread_away_x=np.where(state < 0, 14.0, 10.0),
        spread_home_y=np.where(state > 0, 12.0, 9.0),
        spread_away_y=np.where(state < 0, 12.0, 9.0),
        mean_speed_home=np.where(state > 0, 4.5, 3.0),
        mean_speed_away=np.where(state < 0, 4.5, 3.0),
        ball_dir_change=rng.uniform(0.0, 1.2, n),
        ball_acc=rng.uniform(0.0, 12.0, n),
    )
    no_ball = (ball_obs == 0) | (half == 0)
    for col in ("d_near_home", "d_near_away", "d_diff", "n_home_r2", "n_away_r2",
                "n_home_r3", "n_away_r3", "pressure_home", "pressure_away",
                "speed_rel_home", "speed_rel_away", "ball_dir_change", "ball_acc"):
        m.loc[no_ball, col] = np.nan

    gt = np.where(half == 0, pm.X, np.where(state > 0, pm.H, pm.A)).astype(np.int8)
    return m, gt


def _swap_rich_teams(master: pd.DataFrame) -> pd.DataFrame:
    out = swap_teams(master)
    return out


def _dummy_cfg() -> dict:
    return {"name": "v1", "team": {"kind": "rules_v1", "params": team_params(), "calibrated_file": None}}


def test_calibrated_is_registered():
    variants.load_kinds()
    assert variants.TEAM_KINDS["calibrated"] is team_logit_calibrated


def test_new_features_never_cross_half_boundary():
    m1, _ = _rich_synthetic_master(240, seed=1)
    m2 = m1.copy(deep=True)
    second_half = m2["half"] == 2
    m2.loc[second_half, "d_near_home"] = 20.0
    m2.loc[second_half, "d_near_away"] = 0.2
    m2.loc[second_half, "d_diff"] = 19.8
    m2.loc[second_half, "ball_acc"] = 50.0

    p = team_params()
    f_a = build_team_calibration_features(m1, p)
    f_b = build_team_calibration_features(m2, p)

    first_half = m1["half"] == 1
    pd.testing.assert_frame_equal(f_a.loc[first_half], f_b.loc[first_half])


def test_swapping_teams_flips_all_candidate_features_exactly():
    master, _ = _rich_synthetic_master(220, seed=2)
    swapped = _swap_rich_teams(master)
    p = team_params()

    fa = build_team_calibration_features(master, p)
    fb = build_team_calibration_features(swapped, p)

    odd_cols = [c for c in fa.columns if c != "ev_ball_valid"]
    for col in odd_cols:
        a = fa[col].to_numpy(dtype=np.float64)
        b = fb[col].to_numpy(dtype=np.float64)
        assert np.array_equal(np.isnan(a), np.isnan(b)), f"NaN mismatch on {col}"
        ok = ~np.isnan(a)
        assert np.allclose(b[ok], -a[ok], atol=1e-6), f"antisymmetry failed on {col}"


def test_synthetic_pass_bridge_keeps_team_sign_during_flight_and_ignores_single_deflection():
    n = 90
    d_home = np.full(n, 12.0)
    d_away = np.full(n, 12.0)
    # Home player passes at frames 10..20, ball flies far at 21..49, Home receives at 50..65
    d_home[10:21] = 0.8
    d_away[10:21] = 6.0
    d_home[50:66] = 0.8
    d_away[50:66] = 6.0
    # Single-frame Away deflection in the middle of flight (filtered out by 7-frame median)
    d_home[35] = 6.0
    d_away[35] = 0.8

    master = make_master(n, d_near_home=d_home, d_near_away=d_away)
    bridge = build_pass_bridge_features(master, team_params())

    flight = slice(21, 50)
    assert (bridge["f10_15"].iloc[flight] > 0.0).all()
    assert (bridge["f10_35"].iloc[flight] > 0.0).all()
    assert (bridge["f11_15"].iloc[flight] > 0.0).all()


def test_touch_feature_requires_both_abrupt_change_and_persistence():
    n = 40
    # Home close for 10 frames (10..19), with abrupt acceleration at frame 14
    d_home = np.full(n, 8.0)
    d_away = np.full(n, 8.0)
    d_home[10:20] = 1.0
    d_away[10:20] = 5.0
    # Single-frame contact at frame 30 with abrupt change but no persistence
    d_home[30] = 1.0
    d_away[30] = 5.0

    acc = np.zeros(n)
    acc[14] = 15.0
    acc[30] = 15.0

    master = make_master(n, d_near_home=d_home, d_near_away=d_away,
                         ball_acc=acc, ball_dir_change=0.0)
    f9 = build_touch_feature(master, team_params()).to_numpy()
    assert f9[14] > 0.0
    assert f9[30] == 0.0


def test_calibration_recovers_signs_is_deterministic_and_uses_holdout(tmp_path):
    matches = {"m1": _rich_synthetic_master(300, seed=10),
               "m2": _rich_synthetic_master(300, seed=20),
               "m3": _rich_synthetic_master(300, seed=30)}

    def fake_load(match, degrade):
        return matches[match]

    cfg = _dummy_cfg()
    t1 = calibrate_team(cfg, variant="bridge_touch", matches=list(matches), load_fn=fake_load)
    t2 = calibrate_team(cfg, variant="bridge_touch", matches=list(matches), load_fn=fake_load)
    assert t1 == t2

    # Positive weights on primary proximity features f1..f6
    for feat in ("f1", "f2", "f3", "f4", "f5", "f6"):
        assert t1["ball"]["all"]["weights"][feat] > 0.0

    # Modify fold m1 distinctly to verify holdout routing
    table = copy.deepcopy(t1)
    for k in table["ball"]["folds"]["m1"]["weights"]:
        table["ball"]["folds"]["m1"]["weights"][k] *= 0.25

    yaml_path = write_team_yaml(table, tmp_path / "team.yaml")
    p = {**team_params(), "calibrated_file": str(yaml_path)}

    master, _ = matches["m1"]
    logit_all = team_logit_calibrated(master, p, holdout=None)
    logit_m1 = team_logit_calibrated(master, p, holdout="m1")
    assert not np.allclose(logit_all, logit_m1)

    # Antisymmetry under swap_teams
    logit_swapped = team_logit_calibrated(_swap_rich_teams(master), p, holdout=None)
    assert np.allclose(logit_swapped, -logit_all, atol=1e-5)

    # Contract C2 compliance
    half = master["half"].to_numpy()
    contracts.validate_logits(logit_all, np.zeros(len(master), dtype=np.float32),
                              len(master), half=half)


def test_calibrating_on_degrade_none_is_rejected():
    with pytest.raises(ValueError, match="degrade='none' is forbidden"):
        calibrate_team(_dummy_cfg(), degrade="none", matches=["m1", "m2"])


def test_tag_prior_is_antisymmetric_and_zero_by_default():
    master, _ = _rich_synthetic_master(60, seed=7)
    master["n_tags_favor_home"] = 2.0
    master["n_tags_favor_away"] = 0.0
    master["is_set_piece"] = False
    master.loc[5:10, "is_set_piece"] = True
    master["sp_team"] = ""
    master.loc[5:10, "sp_team"] = "H"
    master["frames_since_sp_end"] = np.nan
    master.loc[11:20, "frames_since_sp_end"] = np.arange(1, 11, dtype=float)

    assert np.all(team_tag_prior(master, team_params()) == 0.0)

    p_on = {**team_params(), "w_tag_phase": 0.25, "w_sp_team": 0.5, "tau_sp_team": 75.0}
    prior = team_tag_prior(master, p_on)
    assert prior[6] > 0.0 and prior[12] > 0.0

    # Swap tags and check sign flip
    swapped = _swap_rich_teams(master)
    swapped["sp_team"] = master["sp_team"].replace({"H": "A", "A": "H"})
    prior_swapped = team_tag_prior(swapped, p_on)
    assert np.allclose(prior_swapped, -prior, atol=1e-6)

# -- review fixes (2026-11-16) ---------------------------------------------------------------

def _three_matches():
    return {"m1": _rich_synthetic_master(300, seed=10),
            "m2": _rich_synthetic_master(300, seed=20),
            "m3": _rich_synthetic_master(300, seed=30)}


def _v1_raw() -> dict:
    import yaml
    from modeling import paths
    with open(paths.repo_root() / "configs" / "v1.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_config_accepts_bridge_tau_frames_list_and_calibration_uses_it(tmp_path):
    import yaml
    from modeling.config import ConfigError, load_config

    raw = _v1_raw()
    raw["name"] = "taus"
    raw["team"]["params"]["bridge_tau_frames"] = [10, 20]
    path = tmp_path / "taus.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    cfg = load_config(path)
    assert cfg["team"]["params"]["bridge_tau_frames"] == [10.0, 20.0]

    matches = _three_matches()
    table = calibrate_team(cfg, variant="bridge", matches=list(matches),
                           load_fn=lambda m, d: matches[m])
    assert table["feature_params"]["bridge_tau_frames"] == [10, 20]
    assert "f10_10" in table["ball"]["features"] and "f11_20" in table["gap"]["features"]

    for bad in ({"bridge_tau_1": 10}, {"bridge_tau_frames": []}, {"bridge_tau_frames": [0]}):
        raw_bad = _v1_raw()
        raw_bad["name"] = "bad"
        raw_bad["team"]["params"].update(bad)
        bad_path = tmp_path / "bad.yaml"
        bad_path.write_text(yaml.safe_dump(raw_bad, sort_keys=False), encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(bad_path)


def test_select_touch_params_is_deterministic_and_picks_from_the_grid():
    from modeling.calibration.team import select_touch_params

    matches = _three_matches()
    grid = {"r_touch_m": (1.5, 3.0), "touch_min_frames": (2, 5),
            "touch_acc_mps2": (8.0,), "touch_dir_change_rad": (0.5,)}
    kw = dict(matches=list(matches), grid=grid, load_fn=lambda m, d: matches[m])
    s1 = select_touch_params(_dummy_cfg(), **kw)
    s2 = select_touch_params(_dummy_cfg(), **kw)
    assert s1 == s2
    assert len(s1["results"]) == 4
    best = min(s1["results"], key=lambda r: r["log_loss"])
    for key, value in s1["selected"].items():
        assert value == best[key]
        assert value in grid[key]
    assert isinstance(s1["selected"]["touch_min_frames"], int)

    with pytest.raises(ValueError, match="forbidden"):
        select_touch_params(_dummy_cfg(), degrade="none", **kw)
    with pytest.raises(ValueError, match="exactly the keys"):
        select_touch_params(_dummy_cfg(), **{**kw, "grid": {"r_touch_m": (2.0,)}})


def test_touch_params_override_is_recorded_and_used():
    matches = _three_matches()
    kw = dict(variant="touch", matches=list(matches), load_fn=lambda m, d: matches[m])
    override = {"r_touch_m": 3.0, "touch_min_frames": 5,
                "touch_acc_mps2": 12.0, "touch_dir_change_rad": 0.8}
    t_default = calibrate_team(_dummy_cfg(), **kw)
    t_override = calibrate_team(_dummy_cfg(), touch_params=override, **kw)
    for key, value in override.items():
        assert t_override["feature_params"][key] == value
    assert t_override["ball"]["all"]["weights"]["f9"] != t_default["ball"]["all"]["weights"]["f9"]


def test_report_has_touch_selection_and_per_match_decoder_rows():
    from modeling.calibration.team import render_calibration_report

    matches = _three_matches()
    table = calibrate_team(_dummy_cfg(), variant="touch", matches=list(matches),
                           load_fn=lambda m, d: matches[m])
    df = pd.DataFrame(
        {"spell_f1_mean": [0.3, 0.4, 0.35, 0.05], "spell_count_ratio": [0.5, 0.6, 0.55, 0.05],
         "boundary_shift_frames": [-1.0, 2.0, 0.5, 1.0]},
        index=["m1", "m2", "mean", "se"])
    selection = {"grid": {"r_touch_m": [2.0], "touch_min_frames": [3],
                          "touch_acc_mps2": [8.0], "touch_dir_change_rad": [0.5]},
                 "selected": {"r_touch_m": 2.0, "touch_min_frames": 3,
                              "touch_acc_mps2": 8.0, "touch_dir_change_rad": 0.5},
                 "results": [{"r_touch_m": 2.0, "touch_min_frames": 3, "touch_acc_mps2": 8.0,
                              "touch_dir_change_rad": 0.5, "log_loss": 0.4}]}
    md = render_calibration_report({"touch": table}, {"touch": {"default": df}},
                                   touch_selection=selection, written_variant="touch")
    assert "variant written to the calibration file: `touch`" in md
    assert "## Touch thresholds" in md
    assert "| m2 | 0.4000 | 0.6000 | +2.00 |" in md


def test_touch_without_abrupt_condition_is_persistence_only():
    n = 40
    d_home = np.full(n, 8.0)
    d_away = np.full(n, 8.0)
    d_home[10:20] = 1.0
    d_away[10:20] = 5.0
    master = make_master(n, d_near_home=d_home, d_near_away=d_away,
                         ball_acc=0.0, ball_dir_change=0.0)
    with_abrupt = build_touch_feature(master, team_params()).to_numpy()
    without = build_touch_feature(master, {**team_params(), "touch_require_abrupt": False})
    assert np.all(np.nan_to_num(with_abrupt) == 0.0)
    assert without.to_numpy()[14] > 0.0 and without.to_numpy()[30] == 0.0


def test_touch_require_abrupt_is_recorded_and_shrinks_the_grid():
    from modeling.calibration.team import select_touch_params

    matches = _three_matches()
    cfg = _dummy_cfg()
    cfg["team"]["params"] = {**cfg["team"]["params"], "touch_require_abrupt": False}
    kw = dict(matches=list(matches), load_fn=lambda m, d: matches[m])
    table = calibrate_team(cfg, variant="touch", **kw)
    assert table["feature_params"]["touch_require_abrupt"] is False
    sel = select_touch_params(cfg, grid=None, **kw)
    assert sel["require_abrupt"] is False
    assert len(sel["grid"]["touch_acc_mps2"]) == 1 and len(sel["grid"]["touch_dir_change_rad"]) == 1
    assert len(sel["results"]) == 9
