"""Contrato legivel por maquina do master_features.parquet. Espelha SCHEMA.md (versao 1.0).

Use este modulo no mock (make_mock_master.py), nos testes (tests/test_master_schema.py) e no
build_master_table.py. Mudou o contrato? Atualize SCHEMA.md e este arquivo no mesmo PR.
Somente biblioteca padrao (sem numpy/pandas).
"""
from collections import OrderedDict

from .params import N_FRAMES, SCHEMA_VERSION

# grupo -> (dono, [(coluna, dtype)])   dtype = nome de dtype numpy/pandas
MASTER_GROUPS = OrderedDict([
    ("indice", ("D", [("video_frame", "int32"), ("half", "int8"), ("is_graded", "bool"),
                      ("frame_in_half", "float32")])),
    ("bola", ("A", [(c, "float32") for c in
                    ("ball_x", "ball_y", "ball_vx", "ball_vy", "ball_speed", "ball_acc")])),
    ("confianca", ("A", [("ball_observed", "int8"), ("ball_jump_removed", "int8"),
                         ("ball_filled", "int8"), ("gap_length", "float32"),
                         ("since_last_obs", "float32"), ("until_next_obs", "float32")])),
    ("times", ("A", [(c, "float32") for c in (
        "centroid_home_x", "centroid_home_y", "centroid_away_x", "centroid_away_y",
        "spread_home_x", "spread_home_y", "spread_away_x", "spread_away_y",
        "mean_speed_home", "mean_speed_away", "n_still_home", "n_still_away",
        "n_tracked_home", "n_tracked_away")])),
    ("proximidade", ("B", [(c, "float32") for c in (
        "d_near_home", "d_near_away", "d_diff",
        "n_home_r2", "n_home_r3", "n_home_r5", "n_away_r2", "n_away_r3", "n_away_r5")])),
    ("velocidade_relativa", ("B", [("speed_rel_home", "float32"), ("speed_rel_away", "float32")])),
    ("campo", ("B", [("out_of_pitch", "float32"), ("d_touchline", "float32"),
                     ("d_goalline", "float32")])),
    ("set_pieces", ("C", [("is_set_piece", "bool"), ("sp_type", "string"), ("sp_team", "string"),
                          ("frames_to_next_sp", "float32"), ("frames_since_sp_end", "float32")])),
    ("fases", ("C", [("n_tags_favor_home", "int16"), ("n_tags_favor_away", "int16")])),
    ("pontape", ("C", [("pre_kickoff", "bool")])),
    ("homografia", ("C", [("reproj_error", "float32")])),   # opcional
])

OPTIONAL_COLUMNS = ("reproj_error",)

# colunas que cada arquivo intermediario entrega (alem de video_frame e half)
INTERMEDIATE_FILES = OrderedDict([
    ("proximities.parquet", ["proximidade", "velocidade_relativa", "campo"]),
    ("tagging.parquet", ["set_pieces", "fases", "pontape"]),
    ("homography.parquet", ["homografia"]),
])

SLOT_TEAMS_COLUMNS = [("slot", "int8"), ("is_home", "bool"), ("is_field_player", "bool")]
N_SLOTS = 46


def master_columns(include_optional=True):
    """Lista ordenada de (coluna, dtype) do master."""
    cols = [c for _, (_, cs) in MASTER_GROUPS.items() for c in cs]
    if not include_optional:
        cols = [c for c in cols if c[0] not in OPTIONAL_COLUMNS]
    return cols


def master_column_names(include_optional=True):
    return [c for c, _ in master_columns(include_optional)]


def slot_columns(n_slots=N_SLOTS):
    """Colunas por slot do tracking_clean.parquet: s01_x, s01_y, s01_vx, s01_vy, s01_break ..."""
    return ["s%02d_%s" % (i, k) for i in range(1, n_slots + 1) for k in ("x", "y", "vx", "vy", "break")]


__all__ = ["MASTER_GROUPS", "INTERMEDIATE_FILES", "SLOT_TEAMS_COLUMNS", "N_FRAMES",
           "SCHEMA_VERSION", "master_columns", "master_column_names", "slot_columns"]
