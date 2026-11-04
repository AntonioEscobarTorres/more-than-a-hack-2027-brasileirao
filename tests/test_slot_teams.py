"""slot_teams tests (A7 / issue #14). Synthetic data only, fixed seed."""
import numpy as np
import pytest

from preprocessing import prep_tracking
from preprocessing.prep_tracking import prepare_tracking, read_atd
from preprocessing.schema import N_SLOTS

T = 200
BLOCKS = [(1, 0, 99), (2, 100, 199)]
KICKOFFS = {1: 0, 2: 100}


def _team_ids(n_home=24):
    return np.array([0] * n_home + [1] * (N_SLOTS - n_home))


def _synthetic(team_ids, seed=0):
    """Teams on opposite halves; sides swap in the 2nd half; ball missing."""
    rng = np.random.default_rng(seed)
    players = np.full((T, N_SLOTS, 3), np.nan, dtype=np.float32)
    for s in range(N_SLOTS):
        home = team_ids[s] == 0
        jitter = 0.02 * rng.random()
        y = (s + 1) / (N_SLOTS + 1)
        players[0:100, s, 0] = (0.20 if home else 0.40) + jitter
        players[100:200, s, 0] = (0.80 if home else 0.60) + jitter
        players[:, s, 1] = y
        players[:, s, 2] = 0.0
    ball = np.full((T, 3), np.nan, dtype=np.float32)
    return players, ball


def _run(team_ids, stage="V0"):
    players, ball = _synthetic(team_ids)
    return prepare_tracking(players, ball, team_ids, blocks=BLOCKS, n_frames=T,
                            kickoffs=KICKOFFS, stage=stage)


@pytest.mark.parametrize("stage", ["V0", "V1"])
def test_schema_and_dtypes(stage):
    _, st = _run(_team_ids(), stage)
    assert list(st.columns) == ["slot", "is_home", "is_field_player"]
    assert str(st["slot"].dtype) == "int8"
    assert st["is_home"].dtype == bool
    assert st["is_field_player"].dtype == bool
    assert len(st) == N_SLOTS
    assert st["slot"].tolist() == list(range(1, N_SLOTS + 1))


def test_is_home_matches_team_ids():
    ids = _team_ids(24)
    _, st = _run(ids)
    assert st["is_home"].to_numpy().tolist() == (ids == 0).tolist()
    assert int(st["is_home"].sum()) == 24
    assert int((~st["is_home"]).sum()) == 22


def test_is_home_follows_non_contiguous_ids():
    """Mapping must follow the ids; do not assume contiguous blocks."""
    rng = np.random.default_rng(1)
    ids = rng.permutation(_team_ids(23))
    _, st = _run(ids)
    assert st["is_home"].to_numpy().tolist() == (ids == 0).tolist()


def test_all_slots_are_field_players():
    _, st = _run(_team_ids())
    assert st["is_field_player"].all()


def test_halves_are_oriented_oppositely():
    frame, _ = _run(_team_ids())
    flips = frame.groupby("half")["flip"].first()
    assert flips[1] != flips[2]


def test_home_ends_on_low_x_at_kickoff_in_both_halves():
    ids = _team_ids()
    frame, _ = _run(ids)
    home_cols = [f"s{i + 1:02d}_x" for i in np.flatnonzero(ids == 0)]
    away_cols = [f"s{i + 1:02d}_x" for i in np.flatnonzero(ids == 1)]
    for kick in KICKOFFS.values():
        row = frame.loc[frame.video_frame == kick]
        assert row[home_cols].to_numpy().mean() < row[away_cols].to_numpy().mean()


@pytest.mark.parametrize("bad", [
    np.array([0] * 45),                 # tamanho errado
    np.array([0] * 23 + [2] * 23),      # valor fora de {0, 1}
    np.array([0] * 23 + [-1] * 23),
])
def test_invalid_slot_team_ids_raise(bad):
    players, ball = _synthetic(_team_ids())
    with pytest.raises(ValueError):
        prepare_tracking(players, ball, bad, blocks=BLOCKS, n_frames=T,
                         kickoffs=KICKOFFS, stage="V0")


def test_read_atd_rejects_wrong_slot_count(tmp_path, monkeypatch):
    monkeypatch.setattr(prep_tracking, "team_of_each_track",
                        lambda _path: np.zeros(N_SLOTS - 1, dtype=np.int8))
    raw = tmp_path / "atd.txt"
    raw.write_text("1:" + ";".join(["N"] * (N_SLOTS - 1)) + ":N\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_atd(str(raw), str(tmp_path / "meta.xml"), n_frames=1)


def test_read_atd_rejects_wrong_group_count(tmp_path, monkeypatch):
    monkeypatch.setattr(prep_tracking, "team_of_each_track",
                        lambda _path: np.zeros(N_SLOTS, dtype=np.int8))
    raw = tmp_path / "atd.txt"
    raw.write_text("1:" + ";".join(["N"] * (N_SLOTS - 1)) + ":N\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_atd(str(raw), str(tmp_path / "meta.xml"), n_frames=1)


def test_read_atd_ignores_blank_lines_and_keeps_three_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(prep_tracking, "team_of_each_track",
                        lambda _path: np.zeros(N_SLOTS, dtype=np.int8))
    groups = ["0.5,0.5,1.0"] + ["N"] * (N_SLOTS - 1)
    raw = tmp_path / "atd.txt"
    raw.write_text("1:" + ";".join(groups) + ":N\n\n", encoding="utf-8")
    players, ball, teams = read_atd(str(raw), str(tmp_path / "meta.xml"), n_frames=1)
    assert players.shape == (1, N_SLOTS, 3)
    assert np.isfinite(players[0, 0]).all()
    assert np.isnan(players[0, 1:]).all()
    assert len(teams) == N_SLOTS


def test_team_of_each_track_reads_teamid_from_xml(tmp_path):
    """Synthetic XML: 1st <Team> = Home (0), 2nd = Away (1)."""
    players = []
    for i in range(1, N_SLOTS + 1):
        team = "T1" if i <= 24 else "T2"
        players.append(f'<Player id="{i}" teamId="{team}"><ShirtNumber>{i}</ShirtNumber></Player>')
    xml = ('<main><Metadata><Sessions><Session><Teams>'
           '<Team id="T1"/><Team id="T2"/></Teams></Session></Sessions>'
           f'<Players>{"".join(players)}</Players></Metadata></main>')
    path = tmp_path / "meta.xml"
    path.write_text(xml, encoding="utf-8")
    from starter.baseline import team_of_each_track
    codes = np.asarray(team_of_each_track(str(path)))
    assert len(codes) == N_SLOTS
    assert int((codes == 0).sum()) == 24
    assert int((codes == 1).sum()) == 22