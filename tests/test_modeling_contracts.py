"""Contracts C0-C2, C4, C6, C7 and C9 of the modeling package (task F5-01).

Synthetic data only: the master is preprocessing.make_mock_master and every file lives in
tmp_path. No test reads MTAH_DATA_DIR.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import games
from evaluation import possession_metrics as pm
from modeling import config as mcfg
from modeling import contracts, io, paths, variants
from modeling.external import files as ext
from preprocessing.make_mock_master import generate_mock_master

N_GAME = games.GAMES[games.GAME]["n_video_frames"]


# -- fixtures -----------------------------------------------------------------------------------

@pytest.fixture(scope="session")
def mock_master() -> pd.DataFrame:
    return generate_mock_master(seed=42)


@pytest.fixture
def data_root(tmp_path, monkeypatch) -> Path:
    """Redirect the data directory to tmp_path (both the env var and the loaded constant)."""
    root = tmp_path / "data"
    monkeypatch.setenv("MTAH_DATA_DIR", str(root))
    monkeypatch.setattr(games, "DATA_DIR", str(root))
    return root


@pytest.fixture
def ext_root(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "external"
    monkeypatch.setenv("MTAH_EXT_DIR", str(root))
    return root


def _half(n_first=5, n_gap=3, n_second=4) -> np.ndarray:
    return np.r_[np.full(n_first, 1), np.zeros(n_gap), np.full(n_second, 2)].astype(np.int8)


def _logits(n=12, team=0.5, dead=-0.5):
    return np.full(n, team, np.float32), np.full(n, dead, np.float32)


HMM_CFG = {
    "name": "v1",
    "pipeline": "hmm",
    "ball_source": "tactical",
    "team": {
        "kind": "rules_v1",
        "params": {
            "pitch_l_m": 105.3, "pitch_w_m": 68.0, "kappa": 3.0, "sigma_d": 2.0, "r_ctrl": 2.0,
            "l_prox": 3.0, "v_rel": 6.0, "ball_conf_filled": 0.6, "contest_factor": 0.5,
            "rel_w_when_nan": 0.5,
            "collective": {"win_frames": 25, "w_adv": 0.0, "w_spread": 0.0, "w_speed": 0.0},
        },
        "calibrated_file": None,
    },
    "dead": {
        "kind": "rules_v1",
        "params": {
            "long_gap_frames": 12, "exit_line_m": 3.0, "exit_speed_mps": 0.5,
            "restart_line_m": 1.5, "goal_area_depth_m": 6.0, "goal_area_half_width_m": 9.2,
            "pitch_w_m": 68.0, "air_speed_mps": 8.0, "air_max_frames": 100, "smooth_frames": 25,
            "rest_speed_mps": 0.3, "rest_min_frames": 25, "fast_speed_mps": 6.0,
            "sp_prox_tau_frames": 250, "sp_after_tau_frames": 75, "goal_window_frames": 1500,
            "weights": {
                "out_of_pitch": 4.0, "gap_exit": 3.0, "gap_air": -1.0, "gap_long": 0.5,
                "still": 3.0, "speed_high": -2.0, "speed_low": 1.5, "ball_rest": 2.0,
                "fast_in_pitch": -3.0, "tag_set_piece": 1.0, "tag_sp_prox": 1.0,
                "tag_sp_after": -1.0, "tag_goal_window": 2.0,
            },
        },
        "calibrated_file": None,
    },
    "decoder": {
        "lam_team": 0.2,
        "lam_dead": 0.2,
        "mean_duration_s": {"H": 15.0, "A": 15.0, "D": 20.0},
        "exit_split": {"H": {"A": 0.5, "D": 0.5}, "A": {"H": 0.5, "D": 0.5},
                       "D": {"H": 0.5, "A": 0.5}},
        "min_dwell_frames": {"H": 12, "A": 12, "D": 25},
    },
    "postprocess": {
        "midpoint_neutral_switch": False,
        "boundary_shift_file": None,
        "d_bridge_frames": 0,
        "d_cross_frames": 0,
        "min_spell_frames": 0,
    },
    "hazard": {"kind": "none", "calibrated_file": None, "beta": 0.0},
}


def _write_cfg(tmp_path, cfg, name=None) -> Path:
    path = tmp_path / "{}.yaml".format(name or cfg["name"])
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def _external_match(n_first=30, n_second=20, match_id="SYN001", seed=0) -> ext.ExternalMatch:
    rng = np.random.default_rng(seed)
    half = ext.expected_half(n_first, n_second)
    n = half.size
    gap = half == 0
    players = rng.uniform(0, 1, (n, 46, 3)).astype(np.float32)
    ball = rng.uniform(0, 1, (n, 3)).astype(np.float32)
    ball_z = rng.uniform(0, 3, n).astype(np.float32)
    players[gap] = np.nan
    ball[gap] = np.nan
    ball_z[gap] = np.nan
    labels = rng.integers(0, 3, n).astype(np.int8)
    labels[gap] = contracts.LABEL_CODES["X"]
    slots = np.r_[np.zeros(24), np.ones(22)].astype(np.int8)
    return ext.ExternalMatch(match_id=match_id, players=players, ball=ball, ball_z=ball_z,
                             labels=labels, half=half, slot_team_ids=slots,
                             pitch=np.array([105.0, 68.0]))


# -- C0 -----------------------------------------------------------------------------------------

def test_label_codes_follow_official_order():
    assert tuple(contracts.LABEL_CODES) == tuple(pm.LABELS)
    assert list(contracts.LABEL_CODES.values()) == [0, 1, 2, 3]
    assert contracts.STATES == ("H", "A", "D")


# -- C2: validate_logits ------------------------------------------------------------------------

def test_validate_logits_accepts_valid_input():
    team, dead = _logits()
    team[0], dead[0] = contracts.TEAM_LOGIT_MAX, -contracts.DEAD_LOGIT_MAX
    contracts.validate_logits(team, dead, 12)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("which", ["team", "dead"])
def test_validate_logits_rejects_non_finite(bad, which):
    team, dead = _logits()
    (team if which == "team" else dead)[3] = bad
    with pytest.raises(ValueError, match="non-finite"):
        contracts.validate_logits(team, dead, 12)


@pytest.mark.parametrize("which", ["team", "dead"])
def test_validate_logits_rejects_wrong_dtype(which):
    team, dead = _logits()
    if which == "team":
        team = team.astype(np.float64)
    else:
        dead = dead.astype(np.float64)
    with pytest.raises(ValueError, match="float32"):
        contracts.validate_logits(team, dead, 12)


@pytest.mark.parametrize("shape", [(11,), (13,), (12, 1)])
def test_validate_logits_rejects_wrong_shape(shape):
    team, dead = _logits()
    with pytest.raises(ValueError, match="shape"):
        contracts.validate_logits(np.zeros(shape, np.float32), dead, 12)
    with pytest.raises(ValueError, match="shape"):
        contracts.validate_logits(team, np.zeros(shape, np.float32), 12)


def test_validate_logits_rejects_lists():
    team, dead = _logits()
    with pytest.raises(ValueError, match="numpy array"):
        contracts.validate_logits(team.tolist(), dead, 12)


@pytest.mark.parametrize("which,value", [("team", 8.001), ("team", -8.5), ("dead", 12.01),
                                         ("dead", -13.0)])
def test_validate_logits_rejects_out_of_bounds(which, value):
    team, dead = _logits()
    (team if which == "team" else dead)[5] = value
    with pytest.raises(ValueError, match=r"\|x\|"):
        contracts.validate_logits(team, dead, 12)


def test_validate_logits_checks_zero_on_gap_when_half_given():
    half = _half()
    team, dead = _logits()
    with pytest.raises(ValueError, match="half == 0"):
        contracts.validate_logits(team, dead, 12, half=half)
    team[half == 0] = 0
    dead[half == 0] = 0
    contracts.validate_logits(team, dead, 12, half=half)


# -- C4: validate_force_d -----------------------------------------------------------------------

def test_validate_force_d():
    half = _half()
    mask = np.zeros(12, bool)
    mask[:2] = True
    contracts.validate_force_d(mask, half)
    with pytest.raises(ValueError, match="half == 0"):
        bad = mask.copy()
        bad[np.flatnonzero(half == 0)[0]] = True
        contracts.validate_force_d(bad, half)
    with pytest.raises(ValueError, match="bool"):
        contracts.validate_force_d(mask.astype(np.int8), half)
    with pytest.raises(ValueError, match="shape"):
        contracts.validate_force_d(mask[:-1], half)


# -- C0/C5: validate_labels ---------------------------------------------------------------------

def _labels_for(half) -> np.ndarray:
    labels = np.where(half == 0, "X", "H").astype("<U1")
    labels[half == 2] = "D"
    return labels


def test_validate_labels_accepts_valid_input():
    half = _half()
    contracts.validate_labels(_labels_for(half), half)


def test_validate_labels_rejects_x_inside_half():
    half = _half()
    labels = _labels_for(half)
    labels[1] = "X"
    with pytest.raises(ValueError, match="X inside"):
        contracts.validate_labels(labels, half)


def test_validate_labels_rejects_non_x_on_gap():
    half = _half()
    labels = _labels_for(half)
    labels[np.flatnonzero(half == 0)[0]] = "H"
    with pytest.raises(ValueError, match="other than X"):
        contracts.validate_labels(labels, half)


def test_validate_labels_rejects_bad_dtype_values_and_shape():
    half = _half()
    labels = _labels_for(half)
    with pytest.raises(ValueError, match="<U1"):
        contracts.validate_labels(labels.astype(object), half)
    with pytest.raises(ValueError, match="<U1"):
        contracts.validate_labels(labels.astype("<U2"), half)
    bad = labels.copy()
    bad[0] = "Z"
    with pytest.raises(ValueError, match="outside"):
        contracts.validate_labels(bad, half)
    with pytest.raises(ValueError, match="shape"):
        contracts.validate_labels(labels[:-1], half)


# -- C1: master ---------------------------------------------------------------------------------

def test_validate_master_accepts_mock_with_and_without_reproj_error(mock_master):
    io.validate_master(mock_master, n_frames=len(mock_master))
    io.validate_master(mock_master.drop(columns=["reproj_error"]))


def test_validate_master_rejections(mock_master):
    small = mock_master.iloc[:1000].copy()
    io.validate_master(small)
    with pytest.raises(ValueError, match="missing"):
        io.validate_master(small.drop(columns=["d_diff"]))
    with pytest.raises(ValueError, match="outside the schema"):
        io.validate_master(small.assign(team_logit=np.float32(0)))
    with pytest.raises(ValueError, match="dtype"):
        io.validate_master(small.astype({"d_diff": "float64"}))
    with pytest.raises(ValueError, match="rows"):
        io.validate_master(small, n_frames=999)
    shifted = small.copy()
    shifted["video_frame"] = shifted["video_frame"] + 1  # 1-based is a bug
    with pytest.raises(ValueError, match="video_frame"):
        io.validate_master(shifted)


def test_load_master_default_path_and_length(mock_master, data_root):
    assert paths.processed_dir() == data_root / "processed"
    path = data_root / "processed" / io.MASTER_FILENAME
    path.parent.mkdir(parents=True)
    mock_master.to_parquet(path, index=False)
    loaded = io.load_master()
    assert len(loaded) == N_GAME
    pd.testing.assert_frame_equal(loaded, mock_master)


def test_load_master_explicit_path_skips_length_check(mock_master, tmp_path):
    path = tmp_path / "ext_master.parquet"
    mock_master.iloc[:500].drop(columns=["reproj_error"]).to_parquet(path, index=False)
    assert len(io.load_master(path)) == 500
    with pytest.raises(ValueError, match="rows"):
        io.load_master(path, n_frames=501)
    with pytest.raises(FileNotFoundError):
        io.load_master(tmp_path / "missing.parquet")


# -- C2 persistence -----------------------------------------------------------------------------

def test_logits_round_trip(data_root):
    rng = np.random.default_rng(0)
    team = rng.uniform(-8, 8, 1000).astype(np.float32)
    dead = rng.uniform(-12, 12, 1000).astype(np.float32)
    path = io.save_logits(team, dead, "v1")
    assert path == data_root / "processed" / "logits_v1.parquet"
    team2, dead2 = io.load_logits("v1")
    assert team2.dtype == np.float32 and dead2.dtype == np.float32
    np.testing.assert_array_equal(team2, team)
    np.testing.assert_array_equal(dead2, dead)
    assert list(pd.read_parquet(path).columns) == ["video_frame", "team_logit", "dead_logit"]


def test_save_logits_validates_and_rejects_bad_names(data_root):
    team, dead = _logits()
    team[0] = np.nan
    with pytest.raises(ValueError):
        io.save_logits(team, dead, "v1")
    team, dead = _logits()
    for name in ("../evil", "", "a/b"):
        with pytest.raises(ValueError, match="invalid name"):
            io.save_logits(team, dead, name)


# -- C6: submission -----------------------------------------------------------------------------

def test_write_submission_passes_official_validator(tmp_path):
    n = 1000
    labels = np.array(["H", "A", "D", "X"] * (n // 4), dtype="<U1")
    path = io.write_submission(labels, tmp_path / "out" / "sub.csv", n_expected=n)
    raw = path.read_bytes()
    assert raw.startswith(b"frame,label\n0,H\n1,A\n")
    assert b"\r" not in raw
    frame = pm.read_label_csv(path)
    assert pm.validate_submission(frame, n) is None
    assert list(frame["label"]) == labels.tolist()


def test_write_submission_rejects_wrong_n(tmp_path):
    labels = np.array(["H"] * 100, dtype="<U1")
    target = tmp_path / "sub.csv"
    with pytest.raises(ValueError, match="row count"):
        io.write_submission(labels, target, n_expected=101)
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []  # no temporary file left behind


def test_write_submission_rejects_bad_labels(tmp_path):
    labels = np.array(["H", "Q", "A"], dtype="<U1")
    with pytest.raises(ValueError, match="invalid submission"):
        io.write_submission(labels, tmp_path / "sub.csv", n_expected=3)


def test_write_submission_default_n_is_the_game(tmp_path):
    with pytest.raises(ValueError, match="!= {}".format(N_GAME)):
        io.write_submission(np.array(["H"] * 10, dtype="<U1"), tmp_path / "sub.csv")


# -- C7: config ---------------------------------------------------------------------------------

def test_baseline_config_in_repo_is_valid():
    cfg = mcfg.load_config(paths.repo_root() / "configs" / "baseline.yaml")
    assert cfg == {"name": "baseline", "pipeline": "baseline", "ball_source": "tactical",
                   "baseline": {"d_gap_frames": 100, "slow_speed_mps": 1.5,
                                "min_spell_frames": 25}}


def test_hmm_config_from_appendix_is_valid(tmp_path):
    cfg = mcfg.load_config(_write_cfg(tmp_path, HMM_CFG))
    assert cfg == HMM_CFG
    assert isinstance(cfg["dead"]["params"]["sp_prox_tau_frames"], float)


@pytest.mark.parametrize("where", [(), ("team",), ("team", "params", "collective"),
                                   ("decoder", "exit_split", "H"), ("postprocess",)])
def test_load_config_rejects_unknown_key(tmp_path, where):
    cfg = copy.deepcopy(HMM_CFG)
    node = cfg
    for key in where:
        node = node[key]
    node["surprise"] = 1
    with pytest.raises(mcfg.ConfigError, match="unknown keys"):
        mcfg.load_config(_write_cfg(tmp_path, cfg))


@pytest.mark.parametrize("path", [("ball_source",), ("pipeline",), ("team", "calibrated_file"),
                                  ("dead", "params", "weights", "still"),
                                  ("decoder", "min_dwell_frames", "D")])
def test_load_config_rejects_missing_key(tmp_path, path):
    cfg = copy.deepcopy(HMM_CFG)
    node = cfg
    for key in path[:-1]:
        node = node[key]
    del node[path[-1]]
    with pytest.raises(mcfg.ConfigError, match="missing keys"):
        mcfg.load_config(_write_cfg(tmp_path, cfg))


def test_load_config_section_presence_follows_pipeline(tmp_path):
    cfg = copy.deepcopy(HMM_CFG)
    cfg["baseline"] = {"d_gap_frames": 100, "slow_speed_mps": 1.5, "min_spell_frames": 25}
    with pytest.raises(mcfg.ConfigError, match="unknown keys"):
        mcfg.load_config(_write_cfg(tmp_path, cfg))
    base = mcfg.load_config(paths.repo_root() / "configs" / "baseline.yaml")
    base["decoder"] = HMM_CFG["decoder"]
    with pytest.raises(mcfg.ConfigError, match="unknown keys"):
        mcfg.load_config(_write_cfg(tmp_path, base))


@pytest.mark.parametrize("path,value,match", [
    (("decoder", "lam_team"), 0.0, "> 0"),
    (("decoder", "lam_dead"), -0.1, "> 0"),
    (("decoder", "exit_split", "H", "A"), 0.6, "sum to 1"),
    (("decoder", "mean_duration_s", "H"), 0.4, "must exceed"),   # 10 frames <= 12
    (("decoder", "min_dwell_frames", "D"), 500, "must exceed"),
    (("decoder", "min_dwell_frames", "H"), 12.0, "integer"),
    (("decoder", "lam_team"), True, "number"),
    (("decoder", "lam_team"), "0.2", "number"),
    (("team", "kind"), "magic", "one of"),
    (("team", "kind"), "calibrated", "calibrated_file is required"),
    (("team", "params", "contest_factor"), 1.5, "<= 1"),
    (("postprocess", "midpoint_neutral_switch"), 0, "true/false"),
    (("ball_source",), "broadcast", "one of"),
    (("pipeline",), "magic", "one of"),
])
def test_load_config_rejects_bad_values(tmp_path, path, value, match):
    cfg = copy.deepcopy(HMM_CFG)
    node = cfg
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(mcfg.ConfigError, match=match):
        mcfg.load_config(_write_cfg(tmp_path, cfg))


def test_load_config_exit_split_tolerance(tmp_path):
    cfg = copy.deepcopy(HMM_CFG)
    cfg["decoder"]["exit_split"]["H"] = {"A": 0.1, "D": 0.9}
    cfg["decoder"]["exit_split"]["A"] = {"H": 1 / 3, "D": 2 / 3}
    mcfg.load_config(_write_cfg(tmp_path, cfg))


def test_load_config_name_must_match_file(tmp_path):
    with pytest.raises(mcfg.ConfigError, match="file name"):
        mcfg.load_config(_write_cfg(tmp_path, HMM_CFG, name="other"))


def test_load_config_rejects_duplicated_keys(tmp_path):
    path = tmp_path / "baseline.yaml"
    text = (paths.repo_root() / "configs" / "baseline.yaml").read_text(encoding="utf-8")
    path.write_text(text + "ball_source: fused\n", encoding="utf-8")
    with pytest.raises(mcfg.ConfigError, match="duplicated key"):
        mcfg.load_config(path)


def test_config_hash_ignores_key_order():
    shuffled = {k: HMM_CFG[k] for k in reversed(list(HMM_CFG))}
    shuffled["decoder"] = {k: HMM_CFG["decoder"][k] for k in reversed(list(HMM_CFG["decoder"]))}
    assert mcfg.config_hash(shuffled) == mcfg.config_hash(HMM_CFG)
    assert len(mcfg.config_hash(HMM_CFG)) == 64
    changed = copy.deepcopy(HMM_CFG)
    changed["decoder"]["lam_team"] = 0.3
    assert mcfg.config_hash(changed) != mcfg.config_hash(HMM_CFG)


def test_config_hash_same_for_int_and_float_spelling(tmp_path):
    cfg = copy.deepcopy(HMM_CFG)
    cfg["decoder"]["mean_duration_s"]["H"] = 15   # int in YAML, normalised to 15.0
    loaded = mcfg.load_config(_write_cfg(tmp_path, cfg))
    reference = mcfg.load_config(_write_cfg(tmp_path, HMM_CFG))
    assert mcfg.config_hash(loaded) == mcfg.config_hash(reference)


# -- variants -----------------------------------------------------------------------------------

def test_none_variants_return_float32_zeros(mock_master):
    small = mock_master.iloc[:50]
    for registry in (variants.TEAM_KINDS, variants.DEAD_KINDS):
        out = registry["none"](small, {}, holdout=None)
        assert out.dtype == np.float32 and out.shape == (50,) and not out.any()


def test_register_rejects_duplicates(monkeypatch):
    monkeypatch.setattr(variants, "TEAM_KINDS", dict(variants.TEAM_KINDS))

    @variants.register_team("toy")
    def toy(master, params, *, holdout=None):
        return np.zeros(len(master), np.float32)

    assert variants.TEAM_KINDS["toy"] is toy
    variants.register_team("toy")(toy)  # re-registering the same function is harmless
    with pytest.raises(ValueError, match="already registered"):
        variants.register_team("toy")(lambda master, params, *, holdout=None: None)


def test_load_kinds_tolerates_missing_modules():
    variants.load_kinds()
    assert "none" in variants.TEAM_KINDS and "none" in variants.DEAD_KINDS


# -- paths --------------------------------------------------------------------------------------

def test_ext_dir_inside_repo_is_rejected(monkeypatch):
    monkeypatch.setenv("MTAH_EXT_DIR", str(paths.repo_root() / "external"))
    with pytest.raises(ValueError, match="inside the repository"):
        paths.ext_dir()
    monkeypatch.setenv("MTAH_EXT_DIR", str(paths.repo_root()))
    with pytest.raises(ValueError, match="inside the repository"):
        paths.ext_dir()


def test_ext_dir_outside_repo(ext_root):
    assert paths.ext_dir() == ext_root.resolve()


def test_outputs_dir_is_git_ignored():
    assert paths.outputs_dir() == paths.repo_root() / "outputs"
    lines = (paths.repo_root() / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "outputs/" in lines


# -- C9: external matches -----------------------------------------------------------------------

def test_external_match_round_trip(ext_root):
    match = _external_match()
    path = match.save()
    assert path == ext_root.resolve() / "arrays" / "SYN001.npz"
    loaded = ext.ExternalMatch.load(path)
    assert loaded.match_id == "SYN001"
    for field, (dtype, _) in ext.ARRAY_FIELDS.items():
        got, want = getattr(loaded, field), getattr(match, field)
        assert got.dtype == dtype
        np.testing.assert_array_equal(got, want)
    assert loaded.blocks() == [(1, 0, 29), (2, 30 + ext.GAP_FRAMES, 49 + ext.GAP_FRAMES)]


def test_external_match_validation():
    good = _external_match()
    fields = {f: getattr(good, f) for f in ("match_id",) + tuple(ext.ARRAY_FIELDS)}

    def build(**over):
        return ext.ExternalMatch(**{**fields, **over})

    with pytest.raises(ValueError, match="dtype"):
        build(players=good.players.astype(np.float64))
    with pytest.raises(ValueError, match="shape"):
        build(ball=good.ball[:-1])
    with pytest.raises(ValueError, match="halftime gap|1s, then"):
        build(half=np.roll(good.half, 1))
    labels = good.labels.copy()
    labels[0] = 4
    with pytest.raises(ValueError, match="C0"):
        build(labels=labels)
    players = good.players.copy()
    players[good.half == 0] = 0.5
    with pytest.raises(ValueError, match="NaN"):
        build(players=players)
    with pytest.raises(ValueError, match="slot_team_ids"):
        build(slot_team_ids=good.slot_team_ids[::-1].copy())
    with pytest.raises(ValueError, match="match_id"):
        build(match_id="../x")


def test_external_match_load_rejects_extra_keys(tmp_path):
    match = _external_match()
    path = tmp_path / "bad.npz"
    arrays = {f: getattr(match, f) for f in ext.ARRAY_FIELDS}
    np.savez_compressed(path, match_id=np.array("SYN001"), extra=np.zeros(1), **arrays)
    with pytest.raises(ValueError, match="keys"):
        ext.ExternalMatch.load(path)


def test_external_paths_and_load_external(mock_master, ext_root):
    assert ext.IDSSE_MATCHES == ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ",
                                 "J03WR9")
    base = ext_root.resolve() / "processed"
    assert ext.master_path("J03WMX", "none") == base / "J03WMX__none.parquet"
    assert ext.labels_path("J03WMX", "none") == base / "J03WMX__none_labels.npy"
    with pytest.raises(ValueError, match="unknown degrade"):
        ext.master_path("J03WMX", "does_not_exist")

    n = 2000
    master = mock_master.iloc[:n].drop(columns=["reproj_error"]).reset_index(drop=True)
    labels = np.where(master["half"].to_numpy() == 0, 3, 0).astype(np.int8)
    base.mkdir(parents=True)
    master.to_parquet(ext.master_path("SYN001", "none"), index=False)
    np.save(ext.labels_path("SYN001", "none"), labels)
    got_master, got_labels = ext.load_external("SYN001", "none")
    assert len(got_master) == n and len(got_master.columns) == 59
    np.testing.assert_array_equal(got_labels, labels)

    np.save(ext.labels_path("SYN001", "none"), labels.astype(np.int64))
    with pytest.raises(ValueError, match="int8"):
        ext.load_external("SYN001", "none")
