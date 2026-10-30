"""
Testes de contrato e schema do master_features (M2 / D3).
Usa apenas dados sinteticos gerados por preprocessing/make_mock_master.py.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from games import GAME, half_video_blocks
from preprocessing.make_mock_master import default_mock_path, generate_mock_master, save_mock_master
from preprocessing.params import (
    KICKOFF_FALLBACK,
    MAX_GAP_FRAMES,
    MIN_BALL_SPEED_DIR,
    N_FRAMES,
    SCHEMA_VERSION,
)
from preprocessing.schema import MASTER_GROUPS, master_column_names, master_columns


@pytest.fixture(scope="module")
def mock_df() -> pd.DataFrame:
    """Gera um DataFrame mock completo para reutilizar entre os testes de leitura."""
    return generate_mock_master(game=GAME, seed=42, include_optional=True)


def test_schema_column_names_and_order(mock_df: pd.DataFrame):
    """Garante que as 60 colunas estao presentes na ordem exata definida em master_columns()."""
    expected_cols = master_column_names(include_optional=True)
    assert len(mock_df.columns) == 60
    assert list(mock_df.columns) == expected_cols
    assert len(set(mock_df.columns)) == 60, "Ha colunas duplicadas no DataFrame"


def test_schema_column_dtypes(mock_df: pd.DataFrame):
    """Garante que cada coluna tem o dtype exato especificado no contrato do schema.py."""
    expected_types = dict(master_columns(include_optional=True))
    for col, exp_type in expected_types.items():
        actual_type = mock_df[col].dtype.name
        if exp_type == "string":
            assert actual_type == "string", f"Coluna {col} deveria ser string, obteve {actual_type}"
        elif exp_type == "bool":
            assert actual_type == "bool", f"Coluna {col} deveria ser bool, obteve {actual_type}"
        else:
            assert actual_type == exp_type, f"Coluna {col} deveria ser {exp_type}, obteve {actual_type}"


def test_no_optional_columns():
    """Garante que include_optional=False gera exatamente as 59 colunas obrigatorias."""
    df_no_opt = generate_mock_master(game=GAME, seed=42, include_optional=False)
    expected_cols = master_column_names(include_optional=False)
    assert len(df_no_opt.columns) == 59
    assert "reproj_error" not in df_no_opt.columns
    assert list(df_no_opt.columns) == expected_cols


def test_row_count_and_video_frame_index(mock_df: pd.DataFrame):
    """Valida 199.319 linhas e video_frame continuo de 0 a 199.318."""
    assert len(mock_df) == N_FRAMES
    assert (mock_df["video_frame"].to_numpy() == np.arange(N_FRAMES, dtype=np.int32)).all()


def test_half_and_graded_blocks(mock_df: pd.DataFrame):
    """Valida blocos de tempo conforme half_video_blocks('BAR-ATH')."""
    blocks = half_video_blocks(GAME)
    # [(1, 11185, 90651), (2, 110808, 190197)]
    for h, f0, f1 in blocks:
        assert (mock_df.loc[f0:f1, "half"] == h).all()
        assert (mock_df.loc[f0:f1, "is_graded"] == True).all()
        expected_fih = np.arange(f1 - f0 + 1, dtype=np.float32)
        actual_fih = mock_df.loc[f0:f1, "frame_in_half"].to_numpy()
        np.testing.assert_array_equal(actual_fih, expected_fih)

    # Fora dos tempos (half == 0)
    outside_mask = mock_df["half"] == 0
    assert outside_mask.sum() == N_FRAMES - sum(f1 - f0 + 1 for _, f0, f1 in blocks)
    assert (mock_df.loc[outside_mask, "is_graded"] == False).all()
    assert mock_df.loc[outside_mask, "frame_in_half"].isna().all()


def test_outside_halves_all_tracking_and_derived_are_nan(mock_df: pd.DataFrame):
    """Fora dos tempos (half == 0), colunas derivadas do tracking e distancias de SP devem ser NaN (flags 0)."""
    outside = mock_df["half"] == 0

    # Flags de confianca devem ser 0
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

    # Float columns outside halves must be NaN
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
        "frames_to_next_sp", "frames_since_sp_end", "reproj_error"
    ]
    for col in nan_cols:
        assert mock_df.loc[outside, col].isna().all(), f"Coluna {col} deveria ser NaN fora dos tempos"


def test_ball_missing_ratio_and_gaps(mock_df: pd.DataFrame):
    """Valida bola ausente em ~41% dos frames avaliados e regras de gap_length."""
    graded = mock_df["is_graded"]
    evaluated_df = mock_df[graded]

    missing_ratio = (evaluated_df["ball_observed"] == 0).mean()
    assert 0.38 <= missing_ratio <= 0.44, f"Bola ausente {missing_ratio:.2%} fora da faixa esperada (~41%)"

    # Quando ball_observed == 1, gap_length deve ser 0
    obs_mask = evaluated_df["ball_observed"] == 1
    assert (evaluated_df.loc[obs_mask, "gap_length"] == 0.0).all()

    # Quando ball_observed == 0, gap_length deve ser > 0
    unobs_mask = evaluated_df["ball_observed"] == 0
    assert (evaluated_df.loc[unobs_mask, "gap_length"] > 0).all()


def test_ball_filled_coherence(mock_df: pd.DataFrame):
    """Valida invariante fundamental: ball_filled == 1 => gap_length <= 12 e ball_observed == 0."""
    graded = mock_df["is_graded"]
    evaluated_df = mock_df[graded]

    filled = evaluated_df["ball_filled"] == 1
    # Se filled == 1, nao pode ser observed
    assert (evaluated_df.loc[filled, "ball_observed"] == 0).all()

    # Se filled == 1, a lacuna precisa ter tamanho <= MAX_GAP_FRAMES (12)
    assert (evaluated_df.loc[filled, "gap_length"] <= MAX_GAP_FRAMES).all()
    assert (evaluated_df.loc[filled, "gap_length"] >= 1.0).all()


def test_since_and_until_last_obs(mock_df: pd.DataFrame):
    """Valida since_last_obs e until_next_obs dentro de cada tempo."""
    blocks = half_video_blocks(GAME)
    for _, f0, f1 in blocks:
        sub = mock_df.loc[f0:f1]
        obs = sub["ball_observed"] == 1
        first_obs_idx = sub.index[obs][0]
        last_obs_idx = sub.index[obs][-1]

        # since_last_obs e NaN estritamente antes da 1a observacao do tempo
        assert sub.loc[:first_obs_idx - 1, "since_last_obs"].isna().all()
        assert sub.loc[first_obs_idx:, "since_last_obs"].notna().all()

        # until_next_obs e NaN estritamente depois da ultima observacao do tempo
        assert sub.loc[last_obs_idx + 1:, "until_next_obs"].isna().all()
        assert sub.loc[:last_obs_idx, "until_next_obs"].notna().all()

        # Onde ha observacao, ambos sao 0
        assert (sub.loc[obs, "since_last_obs"] == 0.0).all()
        assert (sub.loc[obs, "until_next_obs"] == 0.0).all()


def test_ball_coordinates_and_kinematics_contract(mock_df: pd.DataFrame):
    """Valida que ball_x/y existem sse (ball_observed == 1 ou ball_filled == 1), e derivadas apenas em observados."""
    has_ball = (mock_df["ball_observed"] == 1) | (mock_df["ball_filled"] == 1)
    assert mock_df["ball_x"].notna().equals(has_ball)
    assert mock_df["ball_y"].notna().equals(has_ball)

    # Onde ball_filled == 1, derivadas cinematicas devem ser NaN
    filled = mock_df["ball_filled"] == 1
    for col in ("ball_vx", "ball_vy", "ball_speed", "ball_acc"):
        assert mock_df.loc[filled, col].isna().all(), f"Coluna {col} nao deveria ser calculada onde ball_filled == 1"

    # Onde ha velocidade finita, ball_speed == hypot(vx, vy)
    valid_speed = mock_df["ball_speed"].notna()
    expected_speed = np.hypot(mock_df.loc[valid_speed, "ball_vx"], mock_df.loc[valid_speed, "ball_vy"])
    np.testing.assert_allclose(mock_df.loc[valid_speed, "ball_speed"], expected_speed, rtol=1e-5)

    # Onde ha ball_dir_change finito, deve estar no intervalo [0, pi]
    valid_dc = mock_df["ball_dir_change"].notna()
    dc_vals = mock_df.loc[valid_dc, "ball_dir_change"]
    assert (dc_vals >= 0.0).all()
    assert (dc_vals <= np.pi + 1e-4).all()

    # Se ball_speed < MIN_BALL_SPEED_DIR (1.0 m/s), dir_change deve ser NaN
    slow = valid_speed & (mock_df["ball_speed"] < MIN_BALL_SPEED_DIR)
    assert mock_df.loc[slow, "ball_dir_change"].isna().all()


def test_proximity_features_nan_contract(mock_df: pd.DataFrame):
    """Proximidades e contagens n_*_r* devem ser NaN estritamente sem bola (nunca zero)."""
    has_ball = (mock_df["ball_observed"] == 1) | (mock_df["ball_filled"] == 1)

    prox_cols = [
        "d_near_home", "d_near_away", "d_diff",
        "n_home_r2", "n_home_r3", "n_home_r5", "n_away_r2", "n_away_r3", "n_away_r5",
        "pressure_home", "pressure_away", "out_of_pitch", "d_touchline", "d_goalline"
    ]
    for col in prox_cols:
        assert mock_df.loc[~has_ball, col].isna().all(), f"Coluna {col} deveria ser NaN sem bola"
        assert mock_df.loc[has_ball, col].notna().all(), f"Coluna {col} deveria ter valor onde ha bola"


def test_set_pieces_contract(mock_df: pd.DataFrame):
    """Valida contrato de set pieces, sp_team ('H', 'A', 'U') e distancias limitadas ao tempo."""
    is_sp = mock_df["is_set_piece"]

    # Quando is_set_piece == True, sp_team deve ser 'H', 'A' ou 'U', e sp_type nao vazio
    assert (mock_df.loc[is_sp, "sp_type"] != "").all()
    assert set(mock_df.loc[is_sp, "sp_team"].unique()).issubset({"H", "A", "U"})

    # Quando is_set_piece == False, sp_team e sp_type devem ser vazios
    assert (mock_df.loc[~is_sp, "sp_type"] == "").all()
    assert (mock_df.loc[~is_sp, "sp_team"] == "").all()

    # Valida next_sp_type / prev_sp_type em relacao a frames_to_next_sp / frames_since_sp_end
    has_next = mock_df["frames_to_next_sp"].notna()
    assert (mock_df.loc[has_next, "next_sp_type"] != "").all()
    assert (mock_df.loc[~has_next, "next_sp_type"] == "").all()

    has_prev = mock_df["frames_since_sp_end"].notna()
    assert (mock_df.loc[has_prev, "prev_sp_type"] != "").all()
    assert (mock_df.loc[~has_prev, "prev_sp_type"] == "").all()


def test_pre_kickoff_segment(mock_df: pd.DataFrame):
    """Valida segmento pre_kickoff (2.934 frames antes do kickoff do 2o tempo: 110808..113741)."""
    blocks = half_video_blocks(GAME)
    f0_2 = blocks[1][1] # 110808
    kickoff_2 = KICKOFF_FALLBACK[2] # 113742

    # 110808..113741 deve ser True
    pre_sub = mock_df.loc[f0_2:kickoff_2 - 1, "pre_kickoff"]
    assert len(pre_sub) == 2934
    assert pre_sub.all()

    # Apos o pontape inicial no 2o tempo deve ser False
    assert (mock_df.loc[kickoff_2:blocks[1][2], "pre_kickoff"] == False).all()

    # No 1o tempo deve ser False (kickoff coincide com f0 = 11185)
    assert (mock_df.loc[blocks[0][1]:blocks[0][2], "pre_kickoff"] == False).all()


def test_determinism():
    """Valida que duas execucoes com a mesma seed produzem DataFrames 100% identicos."""
    df1 = generate_mock_master(game=GAME, seed=42, include_optional=True)
    df2 = generate_mock_master(game=GAME, seed=42, include_optional=True)
    pd.testing.assert_frame_equal(df1, df2)

    # Seeds distintas produzem DataFrames diferentes
    df3 = generate_mock_master(game=GAME, seed=99, include_optional=True)
    assert not df1["ball_x"].equals(df3["ball_x"])


def test_parquet_metadata_roundtrip(tmp_path):
    """Valida salvamento em Parquet com metadata schema_version = '1.0' e recarga integra."""
    out_file = str(tmp_path / "mock_test.parquet")
    df = generate_mock_master(game=GAME, seed=42, include_optional=True)

    save_mock_master(df, out_file, schema_version=SCHEMA_VERSION)
    assert os.path.exists(out_file)

    # Le metadados pyarrow
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
