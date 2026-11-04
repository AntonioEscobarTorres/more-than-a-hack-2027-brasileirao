"""A4: ATD format check + synthetic calibration of the reacquisition predictor.

Run from the repository root:  python analysis/slot_team_format_check.py

RULES: aggregated output only (structure, counts, proportions). No raw
TXT values, player IDs, or coordinates are printed. Does not modify the repository.
Fixed seeds.

Parts
  A. TXT: field count per group (slots and ball), groups per line, empty groups.
  B. XML: channel structure, <Player> attributes, channel order vs <Player>.
  C. kloppy vs team_of_each_track: how many slots diverge.
  D. Synthetic: expected accuracy of the reacquisition predictor with CORRECT labels
     under individual marking and ~45% coverage (reference for the real 66.4%).
"""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np

from games import asset_paths

SEED = 0
N_SAMPLE_LINES = 2000
FIELD_SEP, GROUP_SEP = ":", ";"


def _strip(tag: str) -> str:
    return tag.split("}")[-1]


def _shape(text: str) -> str:
    """Generalize an identifier (digits -> #) so IDs are not exposed."""
    return re.sub(r"\d+", "#", str(text))


# ----------------------------------------------------------------------------
# A. Estrutura do TXT
# ----------------------------------------------------------------------------
def check_txt(raw_path) -> dict:
    with open(raw_path, encoding="utf-8") as fh:
        lines = [ln for ln in fh if ln.strip()]
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(len(lines), size=min(N_SAMPLE_LINES, len(lines)), replace=False))
    # ensure start, middle and end are in the sample
    idx = np.unique(np.r_[0, len(lines) // 2, len(lines) - 1, idx])

    top_fields = Counter()
    groups_per_line = Counter()
    slot_field_counts, ball_field_counts = Counter(), Counter()
    empty_slot_groups = empty_ball_groups = total_slot_groups = 0
    separators_ok = True

    for i in idx:
        parts = lines[i].rstrip("\r\n").split(FIELD_SEP)
        top_fields[len(parts)] += 1
        if len(parts) != 3:
            separators_ok = False
            continue
        groups = parts[1].split(GROUP_SEP)
        groups_per_line[len(groups)] += 1
        for g in groups:
            total_slot_groups += 1
            if not g or g[0] in "Nn":
                empty_slot_groups += 1
                continue
            slot_field_counts[len(g.split(","))] += 1
        b = parts[2]
        if not b or b[0] in "Nn":
            empty_ball_groups += 1
        else:
            ball_field_counts[len(b.split(","))] += 1

    return {
        "total_nonblank_lines": len(lines),
        "sampled_lines": int(len(idx)),
        "colon_separated_parts_per_line": dict(top_fields),
        "groups_per_line": dict(groups_per_line),
        "values_per_slot_group": dict(slot_field_counts),
        "values_per_ball_group": dict(ball_field_counts),
        "empty_slot_groups_fraction": empty_slot_groups / max(1, total_slot_groups),
        "empty_ball_groups_in_sample": empty_ball_groups,
        "separators_consistent": separators_ok,
        "extra_field_beyond_3_exists": any(k > 3 for k in slot_field_counts),
    }


# ----------------------------------------------------------------------------
# B. Estrutura do XML
# ----------------------------------------------------------------------------
def check_xml(meta_path) -> dict:
    root = ET.parse(meta_path).getroot()
    tags = Counter(_strip(e.tag) for e in root.iter())

    players = [e for e in root.iter() if _strip(e.tag) == "Player"]
    player_attr_names = sorted({a for p in players for a in p.attrib})
    teams = [e for e in root.iter() if _strip(e.tag) == "Team"]

    # declared channels (structure only)
    channel_elems = [e for e in root.iter() if "Channel" in _strip(e.tag)]
    channel_tag_counts = Counter(_strip(e.tag) for e in channel_elems)
    channel_attr_names = sorted({a for e in channel_elems for a in e.attrib})
    channel_shapes = Counter(
        _shape(e.attrib.get("id") or e.attrib.get("channelId") or e.attrib.get("name") or "")
        for e in channel_elems
    )

    # appearance order of player numbers in the channels (e.g. player12_x)
    order_in_channels, seen = [], set()
    for e in channel_elems:
        for v in e.attrib.values():
            m = re.search(r"(\d+)", str(v))
            if m and re.search(r"player|track|p\d", str(v), re.I):
                n = int(m.group(1))
                if n not in seen:
                    seen.add(n)
                    order_in_channels.append(n)

    # ids dos <Player> na ordem do XML
    id_key = next((k for k in ("id", "Id", "ID", "shirtNumber", "ShirtNumber") if k in player_attr_names), None)
    player_ids = []
    for p in players:
        raw = p.attrib.get(id_key) if id_key else None
        m = re.search(r"(\d+)", str(raw)) if raw is not None else None
        player_ids.append(int(m.group(1)) if m else None)
    numeric_ids = [i for i in player_ids if i is not None]

    team_id_key = next((k for k in ("teamId", "TeamId", "teamID") if k in player_attr_names), None)
    teams_per_player_id = Counter(numeric_ids)

    result = {
        "xml_tag_counts": dict(tags),
        "n_teams": len(teams),
        "n_players": len(players),
        "player_attribute_names": player_attr_names,
        "player_id_attribute_used": id_key,
        "team_attribute_used": team_id_key,
        "player_ids_complete_1_to_46": sorted(numeric_ids) == list(range(1, 47)),
        "player_ids_unique": len(numeric_ids) == len(set(numeric_ids)),
        "player_xml_order_sorted_by_id": numeric_ids == sorted(numeric_ids),
        "players_with_repeated_id": int(sum(1 for c in teams_per_player_id.values() if c > 1)),
        "has_time_validity_attrs": any(
            re.search(r"start|end|from|to|frame|period", a, re.I) for a in player_attr_names
        ),
        "channel_tag_counts": dict(channel_tag_counts),
        "channel_attribute_names": channel_attr_names,
        "channel_id_shapes": dict(channel_shapes.most_common(10)),
        "channel_order_players_found": len(order_in_channels),
    }
    if order_in_channels and numeric_ids:
        n = min(len(order_in_channels), len(numeric_ids))
        result["channel_order_matches_player_order"] = order_in_channels[:n] == numeric_ids[:n]
        result["slots_with_different_position"] = int(
            sum(1 for a, b in zip(order_in_channels[:n], numeric_ids[:n]) if a != b)
        )
    else:
        result["channel_order_matches_player_order"] = None  # not determinable by this parser
    return result


# ----------------------------------------------------------------------------
# C. kloppy vs team_of_each_track
# ----------------------------------------------------------------------------
def check_kloppy(meta_path, raw_path) -> dict:
    out = {}
    try:
        from kloppy import metrica
        from starter.baseline import team_of_each_track

        ds = metrica.load_tracking_epts(meta_data=str(meta_path), raw_data=str(raw_path))
        ours = np.asarray(team_of_each_track(meta_path))  # 0 home, 1 away, ordem do XML

        meta_teams = ds.metadata.teams  # [home, away]
        pid_to_team = {}
        for t_idx, team in enumerate(meta_teams):
            for pl in team.players:
                pid_to_team[str(pl.player_id)] = t_idx

        # slot order as kloppy exposes it in the DataFrame
        cols = [c for c in ds.to_df().columns if c.endswith("_x") and c != "ball_x"]
        slot_ids = [c[:-2] for c in cols]
        kl = np.array([pid_to_team.get(s, -1) for s in slot_ids])

        out["kloppy_slots"] = int(len(kl))
        out["our_slots"] = int(len(ours))
        out["kloppy_unmapped_slots"] = int((kl < 0).sum())
        if len(kl) == len(ours):
            out["slots_diverging"] = int((kl != ours).sum())
        else:
            out["slots_diverging"] = None
        out["kloppy_df_extra_columns_beyond_x_y_s"] = sorted(
            {re.sub(r"^.*?_", "", c) for c in ds.to_df().columns
             if c not in ("frame_id", "timestamp", "period_id", "ball_state", "ball_owning_team")}
            - {"x", "y", "s", "z", "d"}
        )[:10]
    except Exception as exc:  # diagnostics: report type, not content
        out["error"] = type(exc).__name__
        out["error_msg_head"] = str(exc)[:120]
    return out


# ----------------------------------------------------------------------------
# D. Synthetic calibration of the reacquisition predictor
# ----------------------------------------------------------------------------
def _simulate_frame(rng, coverage=0.45, mark_dist=(1.5, 4.0)):
    """Two teams of 11 (22 slots), individual marking, partial coverage."""
    # time A (0) e B (1); cada B marca um A a 1.5-4 m
    ax = rng.uniform(15, 80, 11)
    ay = rng.uniform(5, 63, 11)
    ang = rng.uniform(0, 2 * np.pi, 11)
    r = rng.uniform(*mark_dist, 11)
    bx, by = ax + r * np.cos(ang), ay + r * np.sin(ang)
    x = np.r_[ax, bx]
    y = np.r_[ay, by]
    team = np.r_[np.zeros(11, int), np.ones(11, int)]
    obs = rng.random(22) < coverage * 22 / 20.5  # ~20 observed objects per frame in 46 slots
    return x, y, team, obs


def _predict(x, y, team, obs, s, exclude_self):
    """Mesmo preditor do script real: mediana dos pares por time; mais perto vence."""
    cents = []
    for t in (0, 1):
        peers = obs & (team == t)
        if exclude_self:
            peers = peers.copy()
            peers[s] = False
        if peers.sum() < 3:
            return None
        cents.append(np.array([np.median(x[peers]), np.median(y[peers])]))
    p = np.array([x[s], y[s]])
    return 0 if np.linalg.norm(p - cents[0]) < np.linalg.norm(p - cents[1]) else 1


def synthetic_calibration(n_frames=4000) -> dict:
    rng = np.random.default_rng(SEED)
    res = {}
    for exclude_self in (False, True):
        for label_mode in ("correct", "shuffled"):
            ok = tot = 0
            for _ in range(n_frames):
                x, y, team, obs = _simulate_frame(rng)
                labels = team.copy() if label_mode == "correct" else rng.permutation(team)
                for s in np.flatnonzero(obs):
                    pred = _predict(x, y, labels, obs, s, exclude_self)
                    if pred is None:
                        continue
                    tot += 1
                    ok += int(pred == labels[s])
            res[f"exclude_self={exclude_self}|labels={label_mode}"] = {
                "evaluable": tot, "accuracy": round(ok / max(1, tot), 4)}
    return res


def main():
    paths = asset_paths(source="tactical")
    report = {
        "A_txt": check_txt(paths["atd_raw"]),
        "B_xml": check_xml(paths["atd_meta"]),
        "C_kloppy": check_kloppy(paths["atd_meta"], paths["atd_raw"]),
        "D_synthetic_reacquisition_calibration": synthetic_calibration(),
        "note": ("Compare the real accuracy (66.4%) with 'labels=correct' (same setup as the "
                 "real script: exclude_self=False). If synthetic accuracy with correct labels "
                 "is close to the real one, the XML is consistent with being correct. exclude_self=False "
                 "inflates accuracy (the slot itself enters its team median)."),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()