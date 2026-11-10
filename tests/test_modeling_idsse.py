"""Synthetic tests for the IDSSE converter; network access is opt-in."""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from modeling.contracts import LABEL_CODES
from modeling.external.files import ExternalMatch, GAP_FRAMES
from kloppy.domain import DatasetType, Provider, build_coordinate_system
from kloppy.domain.models.common import BallState

from modeling.external import idsse
from modeling.external.idsse import frames_to_arrays, load_match, parse_ball_labels


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
    meta_path, tracking_path = idsse.download("J03WMX")
    dataset = idsse.load_dataset(meta_path, tracking_path, limit=500)
    assert len(dataset.frames) == 500
    records = idsse._records_from_dataset(dataset, parse_ball_labels(tracking_path))
    # limit=500 only reaches the first half; mirror it as half 2 to run the C9 conversion.
    records = records + [dict(record, half=2) for record in records]
    dims = dataset.metadata.pitch_dimensions
    match = frames_to_arrays(records, match_id="J03WMX",
                             pitch=(dims.pitch_length, dims.pitch_width))
    observed = match.half != 0
    x = match.players[observed, :, 0]
    y = match.players[observed, :, 1]
    # Players spread over the pitch: a collapsed (double-normalised) layout fails this.
    assert np.nanmax(x) - np.nanmin(x) > 0.3
    assert np.nanmax(y) - np.nanmin(y) > 0.3
    assert np.nanmin(x) > -0.1 and np.nanmax(x) < 1.1
    # Speeds are m/s: nobody runs faster than ~12 m/s.
    assert np.nanmax(match.players[observed, :, 2]) < 12.5


# -- coordinate system, units and labels of the kloppy adapter ------------------------------------

def _coordinate_system(provider):
    return build_coordinate_system(provider, dataset_type=DatasetType.TRACKING,
                                   pitch_length=105.0, pitch_width=68.0)


def test_check_coordinate_system_accepts_sportec_and_rejects_kloppy_default():
    idsse.check_coordinate_system(_coordinate_system(Provider.SPORTEC))
    with pytest.raises(ValueError, match="centred metres"):
        idsse.check_coordinate_system(_coordinate_system(Provider.KLOPPY))


def test_load_match_requests_sportec_coordinates(monkeypatch, tmp_path):
    calls = {}

    class Stop(Exception):
        pass

    def fake_load_tracking(**kwargs):
        calls.update(kwargs)
        raise Stop

    monkeypatch.setattr(idsse, "download", lambda match_id: (tmp_path / "m.xml", tmp_path / "t.xml"))
    monkeypatch.setattr(idsse, "parse_ball_labels", lambda path: {})
    monkeypatch.setattr(idsse, "load_tracking", fake_load_tracking)  # used by load_dataset
    with pytest.raises(Stop):
        load_match("J03WMX", limit=10)
    assert calls["coordinates"] == "sportec"
    assert calls["only_alive"] is False


class _Player:
    """Hashable stand-in for kloppy's Player (only player_id is used)."""

    def __init__(self, player_id):
        self.player_id = player_id


def _fake_dataset(coordinate_system):
    p_home, p_away = _Player("p_home"), _Player("p_away")
    home, away = SimpleNamespace(players=[p_home]), SimpleNamespace(players=[p_away])

    def frame(frame_id, state, owner):
        return SimpleNamespace(
            period=SimpleNamespace(id=1), frame_id=frame_id, ball_state=state,
            ball_owning_team=owner, ball_speed=36.0,
            ball_coordinates=SimpleNamespace(x=0.0, y=0.0, z=0.1),
            players_data={
                p_home: SimpleNamespace(coordinates=SimpleNamespace(x=-10.0, y=5.0), speed=18.0),
                p_away: SimpleNamespace(coordinates=SimpleNamespace(x=10.0, y=-5.0), speed=7.2),
            })

    metadata = SimpleNamespace(teams=[home, away], coordinate_system=coordinate_system)
    frames = [frame(1, BallState.ALIVE, home), frame(2, BallState.DEAD, away)]
    return SimpleNamespace(metadata=metadata, frames=frames)


def test_records_convert_speed_to_mps_and_keep_labels():
    ds = _fake_dataset(_coordinate_system(Provider.SPORTEC))
    xml = {(1, 1): LABEL_CODES["H"], (1, 2): LABEL_CODES["D"]}
    records = idsse._records_from_dataset(ds, xml)
    assert [r["label"] for r in records] == [LABEL_CODES["H"], LABEL_CODES["D"]]
    assert records[0]["ball"]["speed"] == pytest.approx(10.0)
    assert records[0]["players"]["p_home"]["speed"] == pytest.approx(5.0)
    assert records[0]["players"]["p_away"]["speed"] == pytest.approx(2.0)


def test_records_reject_kloppy_coordinates_and_label_mismatch():
    with pytest.raises(ValueError, match="centred metres"):
        idsse._records_from_dataset(_fake_dataset(_coordinate_system(Provider.KLOPPY)),
                                    {(1, 1): LABEL_CODES["H"], (1, 2): LABEL_CODES["D"]})
    with pytest.raises(ValueError, match="label mismatch"):
        idsse._records_from_dataset(_fake_dataset(_coordinate_system(Provider.SPORTEC)),
                                    {(1, 1): LABEL_CODES["A"], (1, 2): LABEL_CODES["D"]})


def test_summary_excludes_missing_frames_from_percentages():
    half = np.r_[np.ones(4, np.int8), np.zeros(GAP_FRAMES, np.int8), np.full(2, 2, np.int8)]
    match = SimpleNamespace(half=half, pitch=np.array([105.0, 68.0]))
    # Half 1 has frames 10..13 in the arrays but frame 12 is missing from the XML.
    xml = {(1, 10): LABEL_CODES["H"], (1, 11): LABEL_CODES["D"], (1, 13): LABEL_CODES["X"],
           (2, 5): LABEL_CODES["A"], (2, 6): LABEL_CODES["A"]}
    row = idsse._summary("J03WMX", match, xml)
    assert row["missing"] == 1
    assert row["alive"] == pytest.approx(4 / 5)
    assert row["d"] == pytest.approx(1 / 5)
    assert row["home"] == pytest.approx(1 / 5) and row["away"] == pytest.approx(2 / 5)
    assert row["x"] == 1
