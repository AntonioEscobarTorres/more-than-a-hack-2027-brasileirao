"""
Contract and schema tests for master_features (M2 / D3).
Validates either the synthetic mock (default) or any Parquet via `pytest --master <path>`.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from games import GAME, GAMES, half_video_blocks
from preprocessing.make_mock_master import default_mock_path, generate_mock_master, main as cli_main, save_mock_master
from preprocessing.params import (
    KICKOFF_FALLBACK,
    MAX_GAP_FRAMES,
    MIN_BALL_SPEED_DIR,
    N_FRAMES,
    PITCH_L,
    PITCH_MARGIN,
    PITCH_W,
    SCHEMA_VERSION,
)
from preprocessing.schema import (
    INTERMEDIATE_FILES,
    MASTER_GROUPS,
    N_SLOTS,
    OPTIONAL_COLUMNS,
    SLOT_TEAMS_COLUMNS,
    master_column_names,
    master_columns,
    slot_columns,
)

VALID_SP_TYPES = {
    "throw-in",
    "corner kick",
    "goal kick",
    "free kick",
    "kick off (after goal)",
    "kick off (start)",
}


@pytest.fixture(scope="module")
def master_path(request: pytest.FixtureRequest) -> str | None:
    """Return the path passed via --master, or None to use the in-memory synthetic mock."""
    return request.config.getoption("--master")


@pytest.fixture(scope="module")
def mock_df(master_path: str | None) -> pd.DataFrame:
    """Load the Parquet pointed by --master, or generate the default synthetic mock DataFrame."""
    if master_path is not None:
        if not os.path.isfile(master_path):
            pytest.fail(f"Arquivo especificado em --master nao encontrado: {master_path}")
        return pd.read_parquet(master_path)
    return generate_mock_master(game=GAME, seed=42, include_optional=True)


def test_schema_definitions_and_intermediate_files_contract():
    """Validate the internal integrity of MASTER_GROUPS, INTERMEDIATE_FILES and slot_columns in schema.py."""
    all_cols = master_columns(include_optional=True)
    req_cols = master_columns(include_optional=False)
    assert len(all_cols) == 60
    assert len(req_cols) == 59

    for group_name, groups_list in INTERMEDIATE_FILES.items():
        assert group_name.endswith(".parquet")
        for g in groups_list:
            assert g in MASTER_GROUPS, f"Grupo {g} de {group_name} inexistente em MASTER_GROUPS"

    assert SLOT_TEAMS_COLUMNS == [("slot", "int8"), ("is_home", "bool"), ("is_field_player", "bool")]
    s_cols = slot_columns(N_SLOTS)
    assert len(s_cols) == N_SLOTS * 5
    assert s_cols[0] == "s01_x" and s_cols[-1] == f"s{N_SLOTS:02d}_break"


def test_schema_column_names_and_order(mock_df: pd.DataFrame, master_path: str | None):
    """Ensure columns are present in the exact order defined by master_columns(), with no duplicates."""
    assert len(set(mock_df.columns)) == len(mock_df.columns), "Ha colunas duplicadas no DataFrame"

    expected_with_opt = master_column_names(include_optional=True)
    expected_no_opt = master_column_names(include_optional=False)

    if master_path is None:
        assert len(mock_df.columns) == 60
        assert list(mock_df.columns) == expected_with_opt
    else:
        assert list(mock_df.columns) in (expected_with_opt, expected_no_opt)


def test_schema_column_dtypes(mock_df: pd.DataFrame):
    """Ensure every present column has the exact dtype specified by the schema.py contract."""
    expected_types = dict(master_columns(include_optional=True))
    for col in mock_df.columns:
        exp_type = expected_types[col]
        actual_type = mock_df[col].dtype.name
        if exp_type == "string":
            assert actual_type in ("string", "object", "str"), f"Coluna {col} deveria ser string, obteve {actual_type}"
        elif exp_type == "bool":
            assert actual_type == "bool", f"Coluna {col} deveria ser bool, obteve {actual_type}"
        else:
            assert actual_type == exp_type, f"Coluna {col} deveria ser {exp_type}, obteve {actual_type}"


def test_master_file_metadata_when_path_given(master_path: str | None):
    """When --master is provided, validate schema_version = '1.0' directly in the Parquet metadata."""
    if master_path is None:
        return
    parquet_file = pq.ParquetFile(master_path)
    meta = parquet_file.schema_arrow.metadata or {}
    assert b"schema_version" in meta, "Metadado schema_version ausente no arquivo Parquet"
    assert meta[b"schema_version"].decode("utf-8") == SCHEMA_VERSION


def test_no_optional_columns():
    """Ensure include_optional=False yields exactly the 59 required columns."""
    df_no_opt = generate_mock_master(game=GAME, seed=42, include_optional=False)
    expected_cols = master_column_names(include_optional=False)
    assert len(df_no_opt.columns) == 59
    for opt_col in OPTIONAL_COLUMNS:
        assert opt_col not in df_no_opt.columns
    assert list(df_no_opt.columns) == expected_cols


def test_row_count_and_video_frame_index(mock_df: pd.DataFrame):
    """Validate 199,319 rows and a contiguous video_frame from 0 to 199,318."""
    assert len(mock_df) == N_FRAMES
    assert (mock_df["video_frame"].to_numpy() == np.arange(N_FRAMES, dtype=np.int32)).all()


def test_half_and_graded_blocks(mock_df: pd.DataFrame):
    """Validate half blocks from half_video_blocks('BAR-ATH') and counts 79,467 and 79,390."""
    blocks = half_video_blocks(GAME)
    # [(1, 11185, 90651), (2, 110808, 190197)]
    assert (mock_df["half"] == 1).sum() == 79_467
    assert (mock_df["half"] == 2).sum() == 79_390

    for h, f0, f1 in blocks:
        assert (mock_df.loc[f0:f1, "half"] == h).all()
        assert (mock_df.loc[f0:f1, "is_graded"] == True).all()
        expected_fih = np.arange(f1 - f0 + 1, dtype=np.float32)
        actual_fih = mock_df.loc[f0:f1, "frame_in_half"].to_numpy()
        np.testing.assert_array_equal(actual_fih, expected_fih)

    # Outside the halves (half == 0)
    outside_mask = mock_df["half"] == 0
    assert outside_mask.sum() == N_FRAMES - (79_467 + 79_390)
    assert (mock_df.loc[outside_mask, "is_graded"] == False).all()
    assert mock_df.loc[outside_mask, "frame_in_half"].isna().all()


def test_outside_halves_all_tracking_and_derived_are_nan(mock_df: pd.DataFrame):
    """Outside the halves (half == 0), tracking-derived columns and SP distances must be NaN (flags 0)."""
    outside = mock_df["half"] == 0

    # Confidence, phase and kickoff flags must be 0 / False / empty string
    assert (mock_df.loc[outside, "ball_observed"] == 0).all()
    assert (mock_df.loc[outside, "ball_jump_removed"] == 0).all()
    assert (mock_df.loc[outside, "ball_filled"] == 0).all()
    assert (mock_df.loc[outside, "is_set_piece"] == False).all()
    assert (mock_df.loc[outside, "sp_type"] == "").all()
    assert (mock_df.loc[outside, "sp_team"] == "").all()
    assert (mock_df.loc[outside, "next_sp_type"] == "").all()
    assert (mock_df.loc[outside, "prev_sp_type"] == "").all()
    assert (mock_df.loc[outside, "n_tags_favor_home"] == 0).all()
    assert (mock_df.loc[outside, "n_tags_favor_away"] == 0).all()
    assert (mock_df.loc[outside, "pre_kickoff"] == False).all()

    # float32 columns outside the halves must be NaN
    nan_cols = [
        "frame_in_half", "ball_x", "ball_y", "ball_vx", "ball_vy", "ball_speed", "ball_acc",
        "ball_dir_change", "gap_length", "since_last_obs", "until_next_obs",
        "centroid_home_x", "centroid_home_y", "centroid_away_x", "centroid_away_y",
        "spread_home_x", "spread_home_y", "spread_away_x", "spread_away_y",
        "mean_speed_home", "mean_speed_away", "n_still_home", "n_still_away",
        "n_tracked_home", "n_tracked_away", "d_near_home", "d_near_away", "d_diff",
        "n_home_r2", "n_home_r3", "n_home_r5", "n_away_r2", "n_away_r3", "n_away_r5",
        "pressure_home", "pressure_away", "speed_rel_home", "speed_rel_away",
        "speed_radial_home", "speed_radial_away", "out_of_pitch", "d_touchline", "d_goalline",
        "frames_to_next_sp", "frames_since_sp_end",
    ]
    if "reproj_error" in mock_df.columns:
        nan_cols.append("reproj_error")

    for col in nan_cols:
        assert mock_df.loc[outside, col].isna().all(), f"Coluna {col} deveria ser NaN fora dos tempos"


def test_ball_missing_ratio_and_gaps(mock_df: pd.DataFrame):
    """Validate ~41% missing ball on graded frames and the gap_length rules."""
    graded = mock_df["is_graded"]
    evaluated_df = mock_df[graded]

    missing_ratio = (evaluated_df["ball_observed"] == 0).mean()
    assert 0.38 <= missing_ratio <= 0.45, f"Bola ausente {missing_ratio:.2%} fora da faixa esperada (~41%)"

    # When ball_observed == 1, gap_length must be 0
    obs_mask = evaluated_df["ball_observed"] == 1
    assert (evaluated_df.loc[obs_mask, "gap_length"] == 0.0).all()

    # When ball_observed == 0, gap_length must be > 0
    unobs_mask = evaluated_df["ball_observed"] == 0
    assert (evaluated_df.loc[unobs_mask, "gap_length"] > 0).all()


def test_ball_filled_coherence(mock_df: pd.DataFrame):
    """Validate the core invariant: ball_filled == 1 => gap_length <= MAX_GAP_FRAMES and ball_observed == 0."""
    graded = mock_df["is_graded"]
    evaluated_df = mock_df[graded]

    filled = evaluated_df["ball_filled"] == 1
    assert filled.any(), "Deve haver lacunas curtas interpoladas (ball_filled == 1)"
    assert (evaluated_df.loc[filled, "ball_observed"] == 0).all()
    assert (evaluated_df.loc[filled, "gap_length"] <= MAX_GAP_FRAMES).all()
    assert (evaluated_df.loc[filled, "gap_length"] >= 1.0).all()


def test_since_and_until_last_obs(mock_df: pd.DataFrame):
    """Validate since_last_obs and until_next_obs within each half."""
    blocks = half_video_blocks(GAME)
    for _, f0, f1 in blocks:
        sub = mock_df.loc[f0:f1]
        obs = sub["ball_observed"] == 1
        first_obs_idx = sub.index[obs][0]
        last_obs_idx = sub.index[obs][-1]

        # since_last_obs is NaN strictly before the first observation of the half
        assert sub.loc[:first_obs_idx - 1, "since_last_obs"].isna().all()
        assert sub.loc[first_obs_idx:, "since_last_obs"].notna().all()

        # until_next_obs is NaN strictly after the last observation of the half
        assert sub.loc[last_obs_idx + 1:, "until_next_obs"].isna().all()
        assert sub.loc[:last_obs_idx, "until_next_obs"].notna().all()

        # Where there is an observation, both are 0
        assert (sub.loc[obs, "since_last_obs"] == 0.0).all()
        assert (sub.loc[obs, "until_next_obs"] == 0.0).all()


def test_ball_coordinates_and_kinematics_contract(mock_df: pd.DataFrame):
    """Validate ball_x/y exist iff (ball_observed == 1 or ball_filled == 1), and derivatives only on observed frames."""
    has_ball = (mock_df["ball_observed"] == 1) | (mock_df["ball_filled"] == 1)
    assert mock_df["ball_x"].notna().equals(has_ball)
    assert mock_df["ball_y"].notna().equals(has_ball)

    # Where ball_observed == 0 (includes ball_filled == 1 and no ball), kinematic derivatives must be NaN
    unobs = mock_df["ball_observed"] == 0
    for col in ("ball_vx", "ball_vy", "ball_speed", "ball_acc", "ball_dir_change"):
        assert mock_df.loc[unobs, col].isna().all(), f"Coluna {col} deve ser NaN onde ball_observed == 0"

    # Where velocity is finite, ball_speed == hypot(vx, vy)
    valid_speed = mock_df["ball_speed"].notna()
    expected_speed = np.hypot(mock_df.loc[valid_speed, "ball_vx"], mock_df.loc[valid_speed, "ball_vy"])
    np.testing.assert_allclose(mock_df.loc[valid_speed, "ball_speed"], expected_speed, rtol=1e-5)

    # Where ball_dir_change is finite, it must lie in [0, pi]
    valid_dc = mock_df["ball_dir_change"].notna()
    dc_vals = mock_df.loc[valid_dc, "ball_dir_change"]
    assert (dc_vals >= 0.0).all()
    assert (dc_vals <= np.pi + 1e-4).all()

    # If ball_speed < MIN_BALL_SPEED_DIR (1.0 m/s), dir_change must be NaN
    slow = valid_speed & (mock_df["ball_speed"] < MIN_BALL_SPEED_DIR)
    assert mock_df.loc[slow, "ball_dir_change"].isna().all()


def test_proximity_relative_speed_and_pitch_contract(mock_df: pd.DataFrame, master_path: str | None):
    """Validate NaN without a ball, geometric monotonicity of n_*_r*, relative speed and out_of_pitch."""
    has_ball = (mock_df["ball_observed"] == 1) | (mock_df["ball_filled"] == 1)

    prox_cols = [
        "d_near_home", "d_near_away", "d_diff",
        "n_home_r2", "n_home_r3", "n_home_r5", "n_away_r2", "n_away_r3", "n_away_r5",
        "pressure_home", "pressure_away", "out_of_pitch", "d_touchline", "d_goalline",
    ]
    for col in prox_cols:
        assert mock_df.loc[~has_ball, col].isna().all(), f"Coluna {col} deveria ser NaN sem bola"

    # Relative and radial speeds require observed ball velocity (NaN without a ball or when ball_filled == 1)
    no_ball_vel = mock_df["ball_vx"].isna() | mock_df["ball_vy"].isna()
    for col in ("speed_rel_home", "speed_rel_away", "speed_radial_home", "speed_radial_away"):
        assert mock_df.loc[no_ball_vel, col].isna().all(), f"Coluna {col} deve ser NaN sem velocidade observada da bola"

    # Coerencia geometrica das contagens por raio onde ha bola e jogadores rastreados
    for team in ("home", "away"):
        valid_prox = mock_df[f"d_near_{team}"].notna()
        d_near = mock_df.loc[valid_prox, f"d_near_{team}"]
        r2 = mock_df.loc[valid_prox, f"n_{team}_r2"]
        r3 = mock_df.loc[valid_prox, f"n_{team}_r3"]
        r5 = mock_df.loc[valid_prox, f"n_{team}_r5"]

        assert (r2 >= 0.0).all()
        assert (r2 <= r3).all(), f"Violacao n_{team}_r2 <= n_{team}_r3"
        assert (r3 <= r5).all(), f"Violacao n_{team}_r3 <= n_{team}_r5"
        assert ((r2 > 0) == (d_near < 2.0)).all(), f"Inconsistencia entre n_{team}_r2 e d_near_{team} < 2.0"
        assert ((r3 > 0) == (d_near < 3.0)).all(), f"Inconsistencia entre n_{team}_r3 e d_near_{team} < 3.0"
        assert ((r5 > 0) == (d_near < 5.0)).all(), f"Inconsistencia entre n_{team}_r5 e d_near_{team} < 5.0"

    # d_diff == d_near_home - d_near_away
    valid_diff = mock_df["d_diff"].notna()
    np.testing.assert_allclose(
        mock_df.loc[valid_diff, "d_diff"],
        mock_df.loc[valid_diff, "d_near_home"] - mock_df.loc[valid_diff, "d_near_away"],
        rtol=1e-5,
        atol=1e-5,
    )

    # out_of_pitch belongs to {0, 1, NaN} and matches d_touchline / d_goalline
    oop_valid = mock_df.loc[has_ball, "out_of_pitch"]
    assert set(oop_valid.unique()).issubset({0.0, 1.0})
    if master_path is None:
        assert set(oop_valid.unique()) == {0.0, 1.0}, "Mock deve exercitar tanto out_of_pitch == 0 quanto == 1"

    pitch_l, pitch_w = GAMES.get(GAME, {}).get("pitch", (PITCH_L, PITCH_W))
    bx = mock_df.loc[has_ball, "ball_x"]
    by = mock_df.loc[has_ball, "ball_y"]
    expected_oop = (
        (bx < -PITCH_MARGIN) | (bx > pitch_l + PITCH_MARGIN) |
        (by < -PITCH_MARGIN) | (by > pitch_w + PITCH_MARGIN)
    ).astype(np.float32)
    np.testing.assert_array_equal(oop_valid.to_numpy(), expected_oop.to_numpy())


def test_set_pieces_contract(mock_df: pd.DataFrame, master_path: str | None):
    """Validate the set-piece contract, sp_team ('H', 'A', 'U'), and distances scoped to each half."""
    is_sp = mock_df["is_set_piece"]

    # When is_set_piece == True, sp_team must be 'H', 'A' or 'U', and sp_type a valid XML type
    assert set(mock_df.loc[is_sp, "sp_type"].unique()).issubset(VALID_SP_TYPES)
    if master_path is None:
        assert set(mock_df.loc[is_sp, "sp_type"].unique()) == VALID_SP_TYPES
        for h, kick_f in KICKOFF_FALLBACK.items():
            assert mock_df.loc[kick_f, "sp_type"] == "kick off (start)", f"Kickoff do {h}o tempo fora da janela da tag"
    assert set(mock_df.loc[is_sp, "sp_team"].unique()).issubset({"H", "A", "U"})

    # When is_set_piece == False, sp_team and sp_type must be empty
    assert (mock_df.loc[~is_sp, "sp_type"] == "").all()
    assert (mock_df.loc[~is_sp, "sp_team"] == "").all()

    # Validate next_sp_type / prev_sp_type against frames_to_next_sp / frames_since_sp_end
    has_next = mock_df["frames_to_next_sp"].notna()
    assert set(mock_df.loc[has_next, "next_sp_type"].unique()).issubset(VALID_SP_TYPES)
    assert (mock_df.loc[~has_next, "next_sp_type"] == "").all()
    assert (mock_df.loc[has_next, "frames_to_next_sp"] >= 0.0).all()

    has_prev = mock_df["frames_since_sp_end"].notna()
    assert set(mock_df.loc[has_prev, "prev_sp_type"].unique()).issubset(VALID_SP_TYPES)
    assert (mock_df.loc[~has_prev, "prev_sp_type"] == "").all()
    assert (mock_df.loc[has_prev, "frames_since_sp_end"] >= 0.0).all()

    # Per-half isolation and no off-by-one at the end of each set piece
    for _, f0, f1 in half_video_blocks(GAME):
        sub = mock_df.loc[f0:f1]
        sp_mask = sub["is_set_piece"].to_numpy()
        edges = np.flatnonzero(np.diff(np.r_[False, sp_mask, False].astype(np.int8)))
        starts = f0 + edges[::2]
        ends = f0 + edges[1::2] - 1  # ultimo frame ativo (inclusivo)

        if len(starts) > 0:
            first_start = int(starts[0])
            last_start = int(starts[-1])
            if first_start > f0:
                assert sub.loc[f0:first_start - 1, "frames_since_sp_end"].isna().all()
                assert (sub.loc[f0:first_start - 1, "prev_sp_type"] == "").all()
            assert sub.loc[first_start, "frames_to_next_sp"] == 0.0
            if last_start < f1:
                assert sub.loc[last_start + 1:f1, "frames_to_next_sp"].isna().all()
                assert (sub.loc[last_start + 1:f1, "next_sp_type"] == "").all()

            for end_f in ends:
                end_f = int(end_f)
                assert sub.loc[end_f, "frames_since_sp_end"] == 0.0
                if end_f + 1 <= f1 and not sub.loc[end_f + 1, "is_set_piece"]:
                    assert sub.loc[end_f + 1, "frames_since_sp_end"] == 1.0


def test_pre_kickoff_segment(mock_df: pd.DataFrame):
    """Validate the pre_kickoff segment (2,934 frames before the 2nd-half kickoff: 110808..113741)."""
    blocks = half_video_blocks(GAME)
    f0_2 = blocks[1][1]  # 110808
    kickoff_2 = KICKOFF_FALLBACK[2]  # 113742

    # 110808..113741 must be True
    pre_sub = mock_df.loc[f0_2:kickoff_2 - 1, "pre_kickoff"]
    assert len(pre_sub) == 2934
    assert pre_sub.all()

    # During pre_kickoff there must be no set pieces beyond the kick-off (start) window opening
    assert set(mock_df.loc[f0_2:kickoff_2 - 1, "sp_type"].unique()).issubset({"", "kick off (start)"})

    # After the 2nd-half kickoff it must be False
    assert (mock_df.loc[kickoff_2:blocks[1][2], "pre_kickoff"] == False).all()

    # In the 1st half it must be False (kickoff coincides with f0 = 11185)
    assert (mock_df.loc[blocks[0][1]:blocks[0][2], "pre_kickoff"] == False).all()


def test_determinism_and_parquet_hash(tmp_path):
    """Validate that two runs with the same seed produce DataFrames and Parquets with identical SHA-256 hashes."""
    p1 = str(tmp_path / "run1.parquet")
    p2 = str(tmp_path / "run2.parquet")

    assert cli_main(["--output", p1, "--seed", "42"]) == 0
    assert cli_main(["--output", p2, "--seed", "42"]) == 0

    with open(p1, "rb") as f1, open(p2, "rb") as f2:
        h1 = hashlib.sha256(f1.read()).hexdigest()
        h2 = hashlib.sha256(f2.read()).hexdigest()
    assert h1 == h2, "Hashes SHA-256 dos arquivos Parquet deveriam ser identicos"

    df1 = pd.read_parquet(p1)
    df2 = pd.read_parquet(p2)
    pd.testing.assert_frame_equal(df1, df2)

    # Seeds distintas produzem DataFrames diferentes
    df3 = generate_mock_master(game=GAME, seed=99, include_optional=True)
    assert not df1["ball_x"].equals(df3["ball_x"])


def test_parquet_metadata_roundtrip(tmp_path):
    """Validate Parquet save with schema_version = '1.0' metadata and a faithful reload."""
    out_file = str(tmp_path / "mock_test.parquet")
    df = generate_mock_master(game=GAME, seed=42, include_optional=True)

    save_mock_master(df, out_file, schema_version=SCHEMA_VERSION)
    assert os.path.exists(out_file)

    # Read pyarrow metadata
    parquet_file = pq.ParquetFile(out_file)
    meta = parquet_file.schema_arrow.metadata or {}
    assert b"schema_version" in meta
    assert meta[b"schema_version"].decode("utf-8") == SCHEMA_VERSION

    # Le tabela pandas
    df_read = pd.read_parquet(out_file)
    assert len(df_read) == N_FRAMES
    assert len(df_read.columns) == 60
    assert list(df_read.columns) == list(df.columns)
    for c in df.columns:
        assert df_read[c].dtype.name == df[c].dtype.name
    assert default_mock_path(str(tmp_path)).endswith("processed/master_features_mock.parquet")
