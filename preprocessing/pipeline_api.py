"""In-memory entry points for preprocessing arrays from another match."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .build_master_table import assemble_master
from .export_tagging import empty_tagging_dataframe
from .params import PITCH_L, PITCH_W
from .prep_proximities import assemble_inputs, build_proximities_table
from .prep_tracking import prepare_tracking


def build_master_from_arrays(
    players: np.ndarray,
    ball: np.ndarray,
    slot_team_ids: np.ndarray,
    blocks: list[tuple[int, int, int]],
    pitch: tuple[float, float] = (PITCH_L, PITCH_W),
) -> pd.DataFrame:
    """Build a schema-compatible master DataFrame from normalized tracking arrays."""
    players = np.asarray(players)
    ball = np.asarray(ball)
    n_frames = players.shape[0]
    tracking, slot_teams = prepare_tracking(
        players, ball, slot_team_ids, blocks=blocks, pitch=pitch,
        kickoffs={half: first for half, first, _ in blocks}, stage="V1")
    inputs = assemble_inputs(
        tracking, slot_teams, "V1", n_frames=n_frames, n_slots=players.shape[1])
    proximities, *_ = build_proximities_table(inputs, pitch=pitch)
    tagging = empty_tagging_dataframe(blocks, n_frames)
    return assemble_master(
        tracking, proximities, tagging, homography=None,
        blocks=blocks, n_frames=n_frames)
