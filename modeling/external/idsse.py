"""Download and convert the public IDSSE Sportec tracking matches."""
from __future__ import annotations

import argparse
import shutil
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
from lxml import etree

from kloppy.domain.models.common import BallState
from kloppy.sportec import load_tracking

from .files import (GAP_FRAMES, IDSSE_MATCHES, N_HOME_SLOTS, N_SLOTS,
                    ExternalMatch, arrays_path)
from .. import paths
from ..contracts import LABEL_CODES

REVISION = "a715a38dfbaf5f58e431727c2b78d174101a703c"
DATASET_URL = "https://huggingface.co/datasets/pysport/idsse-data/resolve/{}"
_COMPETITION_BY_MATCH = {"J03WMX": "000001", "J03WN1": "000001"}
_ALIVE_STATUS = "1"
_SOURCE_BYTES = {
    "J03WMX": (11930, 418524833), "J03WN1": (11722, 391722803),
    "J03WOH": (11745, 348945715), "J03WOY": (11952, 362567859),
    "J03WPY": (11922, 371688499), "J03WQQ": (11932, 356219423),
    "J03WR9": (11823, 372934140),
}


def _file_names(match_id: str) -> tuple[str, str]:
    if match_id not in IDSSE_MATCHES:
        raise ValueError("unknown IDSSE match {!r}".format(match_id))
    competition = _COMPETITION_BY_MATCH.get(match_id, "000002")
    base = "DFL-COM-{}_DFL-MAT-{}".format(competition, match_id)
    return ("DFL_02_01_matchinformation_{}.xml".format(base),
            "DFL_04_03_positions_raw_observed_{}.xml".format(base))


def _is_xml(path: Path) -> bool:
    with path.open("rb") as stream:
        return stream.read(5) == b"<?xml"


def download(match_id: str) -> tuple[Path, Path]:
    """Download one metadata/tracking pair from the pinned Hugging Face revision."""
    raw_dir = paths.ext_dir() / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    result = []
    for filename in _file_names(match_id):
        destination = raw_dir / filename
        if not destination.exists():
            url = DATASET_URL.format(REVISION) + "/" + filename
            urllib.request.urlretrieve(url, destination)
        if not _is_xml(destination):
            destination.unlink(missing_ok=True)
            raise ValueError("downloaded file is not XML: {}".format(destination))
        result.append(destination)
    return result[0], result[1]


def _half_from_section(section: str | None) -> int | None:
    if section is None:
        return None
    normalized = section.lower().replace("_", "").replace(" ", "")
    if normalized in {"firsthalf", "1sthalf"}:
        return 1
    if normalized in {"secondhalf", "2ndhalf"}:
        return 2
    return None


def parse_ball_labels(tracking_path: str | Path) -> dict[tuple[int, int], int]:
    """Read ball labels by (half, frame id), preserving invalid live possession as X."""
    labels: dict[tuple[int, int], int] = {}
    parser = etree.iterparse(str(tracking_path), events=("start", "end"), huge_tree=True)
    current_half = None
    in_ball_set = False
    for event, elem in parser:
        if event == "start" and elem.tag == "FrameSet":
            in_ball_set = elem.get("TeamId") == "BALL"
            current_half = _half_from_section(elem.get("GameSection")) if in_ball_set else None
        elif event == "start" and elem.tag == "Frame" and in_ball_set and current_half:
            frame_id = elem.get("N")
            if frame_id is None:
                raise ValueError("ball Frame is missing N")
            status = elem.get("BallStatus")
            possession = elem.get("BallPossession")
            label = (LABEL_CODES["D"] if status != _ALIVE_STATUS else
                     LABEL_CODES["H"] if possession == "1" else
                     LABEL_CODES["A"] if possession == "2" else LABEL_CODES["X"])
            key = (current_half, int(frame_id))
            if key in labels:
                raise ValueError("duplicate ball frame {}".format(key))
            labels[key] = label
        elif event == "end" and elem.tag in ("Frame", "FrameSet"):
            elem.clear()
            parent = elem.getparent()
            if parent is not None:
                while elem.getprevious() is not None:
                    del parent[0]
            if elem.tag == "FrameSet":
                in_ball_set = False
                current_half = None
    if not labels:
        raise ValueError("no BALL frames found in {}".format(tracking_path))
    return labels


def frames_to_arrays(records, *, match_id: str, pitch, slot_ids=None) -> ExternalMatch:
    """Convert flat frame records into the C9 arrays, inserting per-half frame gaps.

    Each record has ``half``, ``frame_id``, ``players`` (id to x/y/speed), ``ball``
    (x/y/z/speed), and C0 ``label``. Coordinates are centred Sportec metres.
    """
    length, width = map(float, pitch)
    if not np.isfinite([length, width]).all() or length <= 0 or width <= 0:
        raise ValueError("pitch dimensions must be positive finite metres")
    rows = defaultdict(dict)
    player_order = [[], []]
    team_by_player = {}
    for record in records:
        half = int(record["half"])
        frame_id = int(record["frame_id"])
        if half not in (1, 2):
            raise ValueError("record half must be 1 or 2")
        if frame_id in rows[half]:
            raise ValueError("duplicate frame {} in half {}".format(frame_id, half))
        rows[half][frame_id] = record
        for player_id, player in record.get("players", {}).items():
            team_id = int(player["team_id"])
            if team_id not in (0, 1):
                raise ValueError("player team_id must be 0 or 1")
            if player_id not in team_by_player:
                team_by_player[player_id] = team_id
                player_order[team_id].append(player_id)
            elif team_by_player[player_id] != team_id:
                raise ValueError("player changed teams: {}".format(player_id))
    if any(len(roster) > N_HOME_SLOTS for roster in player_order):
        raise ValueError("a team exceeds its 24 player limit")
    if len(player_order[1]) > N_SLOTS - N_HOME_SLOTS:
        raise ValueError("the away team exceeds its available player slots")
    if slot_ids is None:
        slot_ids = np.r_[np.zeros(N_HOME_SLOTS, np.int8),
                         np.ones(N_SLOTS - N_HOME_SLOTS, np.int8)]
    slot_ids = np.asarray(slot_ids, dtype=np.int8)
    if slot_ids.shape != (N_SLOTS,):
        raise ValueError("slot_ids must have shape ({})".format(N_SLOTS))
    slots = {}
    for team_id, roster in enumerate(player_order):
        for index, player_id in enumerate(roster):
            slots[player_id] = index if team_id == 0 else 24 + index

    half_arrays = []
    labels_arrays = []
    player_arrays = []
    ball_arrays = []
    ball_z_arrays = []
    for half in (1, 2):
        frame_ids = sorted(rows[half])
        if not frame_ids:
            raise ValueError("both halves must contain at least one frame")
        frame_range = range(frame_ids[0], frame_ids[-1] + 1)
        count = len(frame_range)
        players = np.full((count, N_SLOTS, 3), np.nan, np.float32)
        ball = np.full((count, 3), np.nan, np.float32)
        ball_z = np.full(count, np.nan, np.float32)
        labels = np.full(count, LABEL_CODES["X"], np.int8)
        for offset, frame_id in enumerate(frame_range):
            record = rows[half].get(frame_id)
            if record is None:
                continue
            labels[offset] = np.int8(record["label"])
            for player_id, player in record.get("players", {}).items():
                slot = slots[player_id]
                x, y = float(player["x"]), float(player["y"])
                players[offset, slot] = (x / length + 0.5, 0.5 - y / width,
                                         float(player.get("speed", np.nan)))
            if record.get("ball") is not None:
                item = record["ball"]
                ball[offset] = (float(item["x"]) / length + 0.5,
                                0.5 - float(item["y"]) / width,
                                float(item.get("speed", np.nan)))
                ball_z[offset] = float(item.get("z", np.nan))
        half_arrays.append(np.full(count, half, np.int8))
        labels_arrays.append(labels)
        player_arrays.append(players)
        ball_arrays.append(ball)
        ball_z_arrays.append(ball_z)

    empty_players = np.full((GAP_FRAMES, N_SLOTS, 3), np.nan, np.float32)
    empty_ball = np.full((GAP_FRAMES, 3), np.nan, np.float32)
    empty_z = np.full(GAP_FRAMES, np.nan, np.float32)
    arrays = ExternalMatch(
        match_id=match_id,
        players=np.concatenate([player_arrays[0], empty_players, player_arrays[1]]),
        ball=np.concatenate([ball_arrays[0], empty_ball, ball_arrays[1]]),
        ball_z=np.concatenate([ball_z_arrays[0], empty_z, ball_z_arrays[1]]),
        labels=np.concatenate([labels_arrays[0], np.full(GAP_FRAMES, LABEL_CODES["X"], np.int8), labels_arrays[1]]),
        half=np.concatenate([half_arrays[0], np.zeros(GAP_FRAMES, np.int8), half_arrays[1]]),
        slot_team_ids=slot_ids.copy(), pitch=np.asarray([length, width], dtype=np.float64))
    return arrays


def _records_from_dataset(ds, xml_labels):
    teams = ds.metadata.teams
    player_team = {}
    for team_index, team in enumerate(teams):
        for player in team.players:
            player_team[player] = team_index
    records = []
    for frame in ds.frames:
        half = int(frame.period.id)
        frame_id = int(frame.frame_id)
        ball = {"x": frame.ball_coordinates.x, "y": frame.ball_coordinates.y,
                "z": frame.ball_coordinates.z, "speed": frame.ball_speed}
        players = {}
        for player, pdata in frame.players_data.items():
            if player not in player_team:
                continue
            players[player.player_id] = {"team_id": player_team[player],
                                         "x": pdata.coordinates.x,
                                         "y": pdata.coordinates.y,
                                         "speed": pdata.speed}
        xml_label = xml_labels.get((half, frame_id))
        if xml_label is None:
            raise ValueError("Kloppy frame missing from XML labels: {}".format((half, frame_id)))
        if frame.ball_state == BallState.DEAD:
            kloppy_label = LABEL_CODES["D"]
        else:
            owning = frame.ball_owning_team
            kloppy_label = LABEL_CODES["H"] if owning == teams[0] else LABEL_CODES["A"]
        if xml_label != LABEL_CODES["X"]:
            assert kloppy_label == xml_label, "Kloppy/XML label mismatch at {}".format((half, frame_id))
        records.append({"half": half, "frame_id": frame_id, "players": players,
                        "ball": ball, "label": xml_label})
    return records


def load_match(match_id: str, *, limit: int | None = None) -> ExternalMatch:
    """Load one IDSSE match, validate labels against XML, and convert to C9."""
    meta_path, tracking_path = download(match_id)
    xml_labels = parse_ball_labels(tracking_path)
    dataset = load_tracking(meta_data=meta_path, raw_data=tracking_path,
                            only_alive=False, limit=limit)
    records = _records_from_dataset(dataset, xml_labels)
    dimensions = dataset.metadata.pitch_dimensions
    match = frames_to_arrays(
        records, match_id=match_id,
        pitch=(dimensions.pitch_length, dimensions.pitch_width))
    return match


def _write_reports(match_summaries):
    reports_dir = paths.repo_root() / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    rows = ["# IDSSE overview", "", "Generated from the public IDSSE matches.", "",
            "| Match | Half 1 frames | Half 2 frames | Missing frames | Alive | Dead | H | A | D | X (invalid possession) | Pitch (m) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for item in match_summaries:
        rows.append("| {match} | {h1} | {h2} | {missing} | {alive:.2%} | {dead:.2%} | {home:.2%} | {away:.2%} | {d:.2%} | {x} | {length:g} × {width:g} |".format(**item))
    (reports_dir / "idsse_overview.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _check_disk_space(match_id: str) -> None:
    if match_id not in IDSSE_MATCHES:
        raise ValueError("unknown IDSSE match {!r}".format(match_id))
    raw_dir = paths.ext_dir() / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    filenames = _file_names(match_id)
    required = sum(size for filename, size in zip(filenames, _SOURCE_BYTES[match_id])
                   if not (raw_dir / filename).exists())
    if shutil.disk_usage(paths.ext_dir()).free < required:
        raise OSError("insufficient free disk space to download and convert {}".format(match_id))


def process_match(match_id: str, *, delete_raw: bool = False, limit: int | None = None):
    _check_disk_space(match_id)
    output_path = arrays_path(match_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path, tracking_path = download(match_id)
    match = load_match(match_id, limit=limit)
    output = match.save(arrays_path(match_id))
    if delete_raw:
        meta_path.unlink(missing_ok=True)
        tracking_path.unlink(missing_ok=True)
    return output, match


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", required=True, help="all or comma-separated IDSSE match IDs")
    parser.add_argument("--delete-raw", action="store_true", help="delete XML files after conversion")
    args = parser.parse_args(argv)
    matches = IDSSE_MATCHES if args.matches == "all" else tuple(x.strip() for x in args.matches.split(",") if x.strip())
    summaries = []
    for match_id in matches:
        _check_disk_space(match_id)
        _, tracking_path = download(match_id)
        xml_labels = parse_ball_labels(tracking_path)
        output, match = process_match(match_id, delete_raw=args.delete_raw)
        observed = match.half != 0
        dead = match.labels[observed] == LABEL_CODES["D"]
        alive = ~dead
        denom = int(observed.sum())
        n1 = int(np.count_nonzero(match.half == 1))
        n2 = int(np.count_nonzero(match.half == 2))
        xml_h1 = sum(half == 1 for half, _ in xml_labels)
        xml_h2 = sum(half == 2 for half, _ in xml_labels)
        summaries.append({"match": match_id, "h1": n1, "h2": n2,
                          "missing": (n1 - xml_h1) + (n2 - xml_h2),
                          "alive": float(alive.sum() / denom), "dead": float(dead.sum() / denom),
                          "home": float(np.count_nonzero(match.labels[observed] == LABEL_CODES["H"]) / denom),
                          "away": float(np.count_nonzero(match.labels[observed] == LABEL_CODES["A"]) / denom),
                          "d": float(dead.sum() / denom),
                          "x": sum(label == LABEL_CODES["X"] for label in xml_labels.values()),
                          "length": match.pitch[0], "width": match.pitch[1]})
        print("{}: {}".format(match_id, output))
    _write_reports(summaries)


if __name__ == "__main__":
    main()
