"""Machine-readable contract for master_features.parquet. Mirrors SCHEMA.md (version 1.0).

Use this module in the mock (make_mock_master.py), the tests (tests/test_master_schema.py), and
build_master_table.py. Changed the contract? Update SCHEMA.md and this file in the same PR.
Standard library only (no numpy/pandas).
"""
from collections import OrderedDict

from .params import N_FRAMES, SCHEMA_VERSION

# group -> (owner, [(column, dtype)])   dtype = numpy/pandas dtype name
MASTER_GROUPS = OrderedDict([
    ("indice", ("D", [("video_frame", "int32"), ("half", "int8"), ("is_graded", "bool"),
                      ("frame_in_half", "float32")])),
    ("bola", ("A", [(c, "float32") for c in
                    ("ball_x", "ball_y", "ball_vx", "ball_vy", "ball_speed", "ball_acc",
                     "ball_dir_change")])),
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
        "n_home_r2", "n_home_r3", "n_home_r5", "n_away_r2", "n_away_r3", "n_away_r5",
        "pressure_home", "pressure_away")])),
    ("velocidade_relativa", ("B", [("speed_rel_home", "float32"), ("speed_rel_away", "float32"),
                                   ("speed_radial_home", "float32"), ("speed_radial_away", "float32")])),
    ("campo", ("B", [("out_of_pitch", "float32"), ("d_touchline", "float32"),
                     ("d_goalline", "float32")])),
    ("set_pieces", ("C", [("is_set_piece", "bool"), ("sp_type", "string"), ("sp_team", "string"),
                          ("frames_to_next_sp", "float32"), ("frames_since_sp_end", "float32"),
                          ("next_sp_type", "string"), ("prev_sp_type", "string")])),
    ("fases", ("C", [("n_tags_favor_home", "int16"), ("n_tags_favor_away", "int16")])),
    ("pontape", ("C", [("pre_kickoff", "bool")])),
    ("homografia", ("C", [("reproj_error", "float32")])),   # optional
])

OPTIONAL_COLUMNS = ("reproj_error",)

# columns each intermediate file delivers (besides video_frame and half)
INTERMEDIATE_FILES = OrderedDict([
    ("proximities.parquet", ["proximidade", "velocidade_relativa", "campo"]),
    ("tagging.parquet", ["set_pieces", "fases", "pontape"]),
    ("homography.parquet", ["homografia"]),
])

SLOT_TEAMS_COLUMNS = [("slot", "int8"), ("is_home", "bool"), ("is_field_player", "bool")]
N_SLOTS = 46


def master_columns(include_optional=True):
    """Ordered list of (column, dtype) for the master table."""
    cols = [c for _, (_, cs) in MASTER_GROUPS.items() for c in cs]
    if not include_optional:
        cols = [c for c in cols if c[0] not in OPTIONAL_COLUMNS]
    return cols


def master_column_names(include_optional=True):
    return [c for c, _ in master_columns(include_optional)]


def slot_columns(n_slots=N_SLOTS):
    """Per-slot columns of tracking_clean.parquet: s01_x, s01_y, s01_vx, s01_vy, s01_break ..."""
    return ["s%02d_%s" % (i, k) for i in range(1, n_slots + 1) for k in ("x", "y", "vx", "vy", "break")]


__all__ = ["MASTER_GROUPS", "INTERMEDIATE_FILES", "SLOT_TEAMS_COLUMNS", "N_FRAMES",
           "SCHEMA_VERSION", "master_columns", "master_column_names", "slot_columns"]
