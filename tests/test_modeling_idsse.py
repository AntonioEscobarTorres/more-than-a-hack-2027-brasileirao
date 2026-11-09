"""Synthetic tests for the IDSSE converter; network access is opt-in."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from modeling.contracts import LABEL_CODES
from modeling.external.files import ExternalMatch, GAP_FRAMES
from kloppy.sportec import load_tracking

from modeling.external.idsse import download, frames_to_arrays, parse_ball_labels


def test_parse_ball_labels_keeps_invalid_live_possession_as_x():
    fixture = Path(__file__).parent / "fixtures" / "idsse_mini_tracking.xml"
    labels = parse_ball_labels(fixture)
    assert len(labels) == 20
    assert labels[(1, 10)] == LABEL_CODES["H"]
    assert labels[(1, 11)] == LABEL_CODES["X"]
    assert labels[(1, 12)] == LABEL_CODES["D"]
    assert labels[(1, 14)] == LABEL_CODES["X"]
    assert labels[(2, 20)] == LABEL_CODES["A"]
    assert labels[(2, 23)] == LABEL_CODES["X"]


def _record(half, frame_id, *, label=0, players=None, ball=None):
    return {"half": half, "frame_id": frame_id, "label": label,
            "players": players or {}, "ball": ball}


def test_frames_to_arrays_normalizes_slots_and_fills_frame_gaps():
    records = [
        _record(1, 5, label=LABEL_CODES["H"], players={
            "home-1": {"team_id": 0, "x": 0.0, "y": 0.0, "speed": 4.0},
            "away-1": {"team_id": 1, "x": 52.5, "y": 34.0, "speed": 2.0}},
            ball={"x": -52.5, "y": 34.0, "z": 1.2, "speed": 8.0}),
        _record(1, 7, label=LABEL_CODES["A"]),
        _record(2, 20, label=LABEL_CODES["D"], players={
            "home-1": {"team_id": 0, "x": 52.5, "y": -34.0, "speed": 0.0}},
            ball={"x": 52.5, "y": -34.0, "z": 0.0, "speed": 0.0}),
    ]
    match = frames_to_arrays(records, match_id="SYNTH", pitch=(105.0, 68.0))
    assert isinstance(match, ExternalMatch)
    assert match.players.dtype == np.float32
    assert match.ball.dtype == np.float32
    assert match.ball_z.dtype == np.float32
    assert match.labels.dtype == np.int8
    assert match.half.dtype == np.int8
    assert match.slot_team_ids.dtype == np.int8
    assert match.pitch.dtype == np.float64
    np.testing.assert_allclose(match.players[0, 0], [0.5, 0.5, 4.0])
    np.testing.assert_allclose(match.players[0, 24], [1.0, 0.0, 2.0])
    np.testing.assert_allclose(match.ball[0], [0.0, 0.0, 8.0])
    assert match.ball_z[0] == pytest.approx(1.2)
    assert np.isnan(match.players[1]).all()
    assert np.isnan(match.ball[1]).all()
    assert match.labels[1] == LABEL_CODES["X"]
    second_half_start = 3 + GAP_FRAMES
    assert match.half[second_half_start] == 2
    np.testing.assert_allclose(match.players[second_half_start, 0], [1.0, 1.0, 0.0])
    halftime_start = 3
    assert match.labels[halftime_start:second_half_start].tolist() == [LABEL_CODES["X"]] * GAP_FRAMES
    assert np.isnan(match.players[halftime_start:second_half_start]).all()
    assert match.blocks() == [(1, 0, 2), (2, second_half_start, second_half_start)]


def test_frames_to_arrays_rejects_roster_overflow():
    players = {"home-{}".format(i): {"team_id": 0, "x": 0, "y": 0, "speed": 0}
               for i in range(25)}
    records = [_record(1, 1, players=players), _record(2, 2)]
    with pytest.raises(ValueError, match="24 player limit"):
        frames_to_arrays(records, match_id="SYNTH", pitch=(100, 60))


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("IDSSE_NETWORK_TESTS") != "1",
                    reason="set IDSSE_NETWORK_TESTS=1 to run the public IDSSE smoke test")
def test_real_idsse_smoke_loads_500_frames():
    meta_path, tracking_path = download("J03WMX")
    dataset = load_tracking(meta_data=meta_path, raw_data=tracking_path,
                            only_alive=False, limit=500)
    assert len(dataset.frames) == 500
