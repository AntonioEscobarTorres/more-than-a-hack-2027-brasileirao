"""Synthetic tests for the streaming ATD load stage; no match assets are read."""
import numpy as np
import pytest

from preprocessing import prep_tracking as pt


def _write_atd(path, frames):
    rows = []
    for frame_id, slot_groups, ball in frames:
        rows.append(f"{frame_id}:{';'.join(slot_groups)}:{ball}\n")
    path.write_text("".join(rows), encoding="utf-8")


def test_read_atd_maps_frames_slots_ball_and_missing_to_nan(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, "team_of_each_track", lambda _: np.arange(pt.N_SLOTS) % 2)
    empty = ["NaN"] * pt.N_SLOTS
    first = empty.copy()
    first[0] = "0.1,0.2,3.5"
    _write_atd(tmp_path / "raw.txt", [
        (1, first, "0.3,0.4,0"),
        (2, empty, "NaN"),
    ])

    players, ball, teams = pt.read_atd(tmp_path / "raw.txt", "unused.xml", n_frames=2)

    assert players.shape == (2, pt.N_SLOTS, 3)
    np.testing.assert_allclose(players[0, 0], [0.1, 0.2, 3.5])
    assert np.isnan(players[0, 1]).all() and np.isnan(players[1]).all()
    np.testing.assert_allclose(ball[0], [0.3, 0.4, 0.0])
    assert np.isnan(ball[1]).all()
    np.testing.assert_array_equal(teams, np.arange(pt.N_SLOTS) % 2)


@pytest.mark.parametrize("frames", [
    [(2, ["NaN"] * pt.N_SLOTS, "NaN"), (1, ["NaN"] * pt.N_SLOTS, "NaN")],
    [(1, ["NaN"] * pt.N_SLOTS, "NaN"), (1, ["NaN"] * pt.N_SLOTS, "NaN")],
])
def test_read_atd_rejects_out_of_order_and_duplicate_frames(tmp_path, monkeypatch, frames):
    monkeypatch.setattr(pt, "team_of_each_track", lambda _: np.zeros(pt.N_SLOTS))
    _write_atd(tmp_path / "raw.txt", frames)

    with pytest.raises(ValueError, match="out of order|duplicate"):
        pt.read_atd(tmp_path / "raw.txt", "unused.xml", n_frames=2)


def test_build_uses_asset_paths_and_returns_raw_arrays(tmp_path, monkeypatch):
    raw, meta = tmp_path / "atd.txt", tmp_path / "atd.xml"
    raw.touch()
    meta.touch()
    monkeypatch.setattr(pt, "asset_paths", lambda **kwargs: {"atd_raw": str(raw), "atd_meta": str(meta)})
    expected = (np.zeros((2, pt.N_SLOTS, 3)), np.zeros((2, 3)), np.zeros(pt.N_SLOTS))
    monkeypatch.setattr(pt, "read_atd", lambda raw_path, meta_path: expected)

    players, ball, teams = pt.build(data_dir=str(tmp_path))

    assert players.shape == (2, pt.N_SLOTS, 3)
    assert ball.shape == (2, 3)
    assert teams.shape == (pt.N_SLOTS,)


def test_prepare_tracking_keeps_halves_independent_and_outside_empty(monkeypatch):
    # Adjacent blocks make any accidental full-array interpolation/gradient visible.
    blocks = [(1, 1, 3), (2, 4, 6)]
    players = np.full((8, pt.N_SLOTS, 3), np.nan, dtype=np.float32)
    teams = np.arange(pt.N_SLOTS) % 2
    players[1:4, 0, :2] = (0.2, 0.5)
    players[4:7, 0, :2] = (0.7, 0.5)
    ball = np.full((8, 3), np.nan, dtype=np.float32)
    ball[1, :2] = (0.2, 0.5)
    ball[3, :2] = (0.2, 0.5)
    ball[4, :2] = (0.8, 0.5)
    ball[6, :2] = (0.8, 0.5)
    monkeypatch.setattr(pt, "MAX_GAP_FRAMES", 3)

    frame, _ = pt.prepare_tracking(players, ball, teams, blocks=blocks, n_frames=8)

    assert frame["half"].tolist() == [0, 1, 1, 1, 2, 2, 2, 0]
    assert frame.loc[[0, 7], "ball_observed"].tolist() == [0, 0]
    assert frame.loc[[0, 7], "ball_filled"].tolist() == [0, 0]
    derived = ["ball_x", "ball_vx", "ball_acc", "gap_length",
               "centroid_home_x", "mean_speed_home"]
    assert frame.loc[[0, 7], derived].isna().all().all()
    assert frame.loc[2, "ball_filled"] == 1
    assert frame.loc[5, "ball_filled"] == 1
    # Constant trajectories in each half have zero speed; the boundary is never used.
    np.testing.assert_allclose(frame.loc[1:3, "s01_vx"], 0)
    np.testing.assert_allclose(frame.loc[4:6, "s01_vx"], 0)


def test_flip_orients_coordinates_and_velocity_and_halves_oppose():
    blocks = [(1, 0, 2), (2, 3, 5)]
    teams = np.arange(pt.N_SLOTS) % 2
    players = np.full((6, pt.N_SLOTS, 3), np.nan, np.float32)
    # Home moves toward low x in half 1, high x in half 2: only half 2 is reflected.
    players[0:3, 0, :2] = np.array([[0.30, .40], [.31, .40], [.32, .40]])
    players[3:6, 0, :2] = np.array([[.70, .40], [.69, .40], [.68, .40]])
    ball = np.full((6, 3), np.nan, np.float32)
    ball[0:3, :2] = np.array([[.2, .5], [.21, .5], [.22, .5]])
    ball[3:6, :2] = np.array([[.8, .5], [.79, .5], [.78, .5]])
    frame, _ = pt.prepare_tracking(players, ball, teams, blocks=blocks,
                                   kickoffs={1: 0, 2: 3}, n_frames=6)
    assert frame.groupby("half")["flip"].first().tolist() == [0, 1]
    assert frame["flip"].dtype == np.int8
    assert frame.loc[0, "s01_vx"] > 0
    assert frame.loc[3, "s01_vx"] > 0  # original negative velocity was reflected
    assert frame.loc[0, "ball_vx"] > 0
    assert frame.loc[3, "ball_vx"] > 0
"""Testes complementares para prep_tracking alinhados à implementação real."""
import numpy as np
import pytest
from preprocessing import prep_tracking as pt


def test_kickoff_fallback_when_file_missing(tmp_path):
    """Garante fallback para KICKOFF_FALLBACK quando kickoffs.json não existe no data_dir."""
    # Passa um diretório vazio
    kickoffs = pt._kickoff_frames(data_dir=str(tmp_path))
    assert kickoffs[1] == pt.KICKOFF_FALLBACK[1]
    assert kickoffs[2] == pt.KICKOFF_FALLBACK[2]


def test_kickoff_loads_from_valid_json(tmp_path):
    """Garante leitura correta quando kickoffs.json existe."""
    import json
    proc_dir = tmp_path / "processed"
    proc_dir.mkdir(parents=True)
    payload = {"1": 12000, "2": 115000}
    (proc_dir / "kickoffs.json").write_text(json.dumps(payload), encoding="utf-8")

    kickoffs = pt._kickoff_frames(data_dir=str(tmp_path))
    assert kickoffs[1] == 12000
    assert kickoffs[2] == 115000


def test_coordinates_and_velocities_scaled_to_meters():
    """Garante que as saídas do prepare_tracking estão em metros (PITCH_L x PITCH_W)."""
    blocks = [(1, 0, 1)]
    teams = np.arange(pt.N_SLOTS) % 2
    players = np.full((2, pt.N_SLOTS, 3), np.nan, dtype=np.float32)
    # Jogador 0 no centro normalizado (0.5, 0.5)
    players[:, 0, :2] = [0.5, 0.5]
    ball = np.full((2, 3), np.nan, dtype=np.float32)
    ball[:, :2] = [0.5, 0.5]

    frame, _ = pt.prepare_tracking(
        players, ball, teams, blocks=blocks, pitch=(105.3, 68.0), kickoffs={1: 0}, n_frames=2
    )

    # 0.5 * 105.3 = 52.65m; 0.5 * 68.0 = 34.0m
    np.testing.assert_allclose(frame.loc[0, "s01_x"], 52.65, rtol=1e-3)
    np.testing.assert_allclose(frame.loc[0, "s01_y"], 34.00, rtol=1e-3)
    np.testing.assert_allclose(frame.loc[0, "ball_x"], 52.65, rtol=1e-3)
    np.testing.assert_allclose(frame.loc[0, "ball_y"], 34.00, rtol=1e-3)


def test_orient_half_reflects_x_y_and_vectors():
    """Valida reflexão direta da função _orient_half em coordenadas e velocidades."""
    length, width = 105.3, 68.0
    players = np.full((5, pt.N_SLOTS, 3), np.nan, dtype=np.float32)
    ball = np.full((5, 3), np.nan, dtype=np.float32)
    teams = [0] + [1] * (pt.N_SLOTS - 1)  # slot 0 é time da casa (0)

    # Time da casa em x alto (> length/2) -> Deve forçar flip=True
    players[:, 0, :2] = [80.0, 20.0]
    players[:, 0, 2] = 5.0  # speed escalar bruto
    ball[:, :2] = [70.0, 15.0]

    flipped = pt._orient_half(players, ball, teams, start=0, end=5, kick_frame=0, pitch=(length, width))

    assert flipped is True or flipped == 1
    # Posições refletidas: L - x e W - y
    # Inputs and reflected arrays use float32, so allow their rounding error.
    np.testing.assert_allclose(players[0, 0, 0], length - 80.0, atol=1e-5)
    np.testing.assert_allclose(players[0, 0, 1], width - 20.0, atol=1e-5)
    np.testing.assert_allclose(ball[0, 0], length - 70.0, atol=1e-5)
    np.testing.assert_allclose(ball[0, 1], width - 15.0, atol=1e-5)
    # Inversão do componente escalar/vetorial bruto
    np.testing.assert_allclose(players[0, 0, 2], -5.0)


def test_remove_ball_jump_outliers_bridges_long_phantom_block_exceeding_k():
    """Garante que um bloco de >K (40 > 30) frames fantasmas é removido sem quebrar a cadeia real."""
    x = np.full(110, np.nan, dtype=np.float32)
    y = np.full(110, 34.0, dtype=np.float32)
    # Bola real antes (35 frames) e depois (35 frames), movendo-se a 2.5 m/s (0.1 m/frame)
    x[0:35] = 10.0 + 0.1 * np.arange(35, dtype=np.float32)
    # Bloco fantasma de 40 frames (> BALL_K = 30) a ~35 m de distância
    x[35:75] = 48.0
    y[35:75] = 62.0
    # Bola real reaparece na continuidade natural da trajetória original
    x[75:110] = 10.0 + 0.1 * np.arange(75, 110, dtype=np.float32)

    xo, yo = pt.remove_ball_jump_outliers(x, y, k=30)

    assert np.isfinite(xo[0:35]).all() and np.isfinite(yo[0:35]).all()
    assert np.isnan(xo[35:75]).all() and np.isnan(yo[35:75]).all()
    assert np.isfinite(xo[75:110]).all() and np.isfinite(yo[75:110]).all()


def test_remove_ball_jump_outliers_keeps_disjoint_long_chains_after_multiball_jump():
    """Garante que duas cadeias longas disjuntas (ex.: reposição multi-bolas) são preservadas."""
    x = np.full(80, np.nan, dtype=np.float32)
    y = np.full(80, 34.0, dtype=np.float32)
    x[0:35] = 10.0 + 0.1 * np.arange(35, dtype=np.float32)
    x[35:75] = 80.0 - 0.1 * np.arange(40, dtype=np.float32)  # 2a bola a ~67 m em 1 frame
    x[75:77] = 5.0  # pico fantasma curto de 2 frames no fim

    xo, _ = pt.remove_ball_jump_outliers(x, y, k=30, min_chain=30)

    assert np.isfinite(xo[0:35]).all()
    assert np.isfinite(xo[35:75]).all()
    assert np.isnan(xo[75:77]).all()


def test_ball_derivatives_masked_when_filled_and_smoothed_against_jitter():
    """Garante que velocidades/aceleração ficam NaN onde ball_filled == 1 e que o Savitzky-Golay atenua jitter."""
    n = 20
    blocks = [(1, 0, n - 1)]
    teams = np.arange(pt.N_SLOTS) % 2
    players = np.full((n, pt.N_SLOTS, 3), np.nan, dtype=np.float32)
    players[:, 0, :2] = [0.2, 0.5]
    ball = np.full((n, 3), np.nan, dtype=np.float32)

    # Trajetória linear com jitter de alta frequência nos primeiros 10 frames
    t_obs = np.arange(10, dtype=np.float32)
    jitter = (0.002 * np.sin(1.8 * t_obs)).astype(np.float32)
    ball[0:10, 0] = 0.2 + 0.002 * t_obs + jitter
    ball[0:10, 1] = 0.5
    # Lacuna curta de 2 frames (frames 10 e 11 -> ball_filled == 1), seguida de observações em 12..19
    ball[12:20, 0] = 0.2 + 0.002 * np.arange(12, 20, dtype=np.float32)
    ball[12:20, 1] = 0.5

    frame, _ = pt.prepare_tracking(players, ball, teams, blocks=blocks, kickoffs={1: 0}, n_frames=n)

    # Posição bruta observada (em metros) não é alterada pelo filtro de derivada
    np.testing.assert_allclose(frame.loc[0:9, "ball_x"], frame.loc[0:9, "ball_x_raw"], rtol=1e-6)
    # Nos frames interpolados (10 e 11), posição existe mas velocidades e aceleração são NaN
    assert (frame.loc[10:11, "ball_filled"] == 1).all()
    assert frame.loc[10:11, ["ball_vx", "ball_vy", "ball_speed", "ball_acc"]].isna().all().all()
    # Nos frames observados (12..19), velocidades e aceleração existem
    assert frame.loc[12:19, ["ball_vx", "ball_speed", "ball_acc"]].notna().all().all()
    # Aceleração suavizada no trecho com jitter é muito menor que a 2a derivada crua
    raw_vx = np.gradient(frame.loc[0:9, "ball_x_raw"].to_numpy()) * pt.FPS
    raw_acc = np.abs(np.gradient(raw_vx) * pt.FPS)
    assert frame.loc[2:7, "ball_acc"].mean() < 0.5 * raw_acc[2:8].mean()


def test_ball_dir_change_detects_turn_and_masks_slow_ball():
    """Garante que ball_dir_change vale ~0 em linha reta, >0 em curva rápida e NaN em bola lenta."""
    n = 14
    blocks = [(1, 0, n - 1)]
    teams = np.arange(pt.N_SLOTS) % 2
    players = np.full((n, pt.N_SLOTS, 3), np.nan, dtype=np.float32)
    players[:, 0, :2] = [0.2, 0.5]
    ball = np.full((n, 3), np.nan, dtype=np.float32)

    # Frames 0..6: bola rápida (10.5 m/s) em linha reta ao longo de +x -> dir_change == 0 em 1..4
    for f in range(7):
        ball[f, :2] = [0.2 + 0.004 * f, 0.5]
    # Frames 7..10: curva de 90 graus movendo-se ao longo de +y (6.8 m/s)
    for f in range(7, 11):
        ball[f, :2] = [0.2 + 0.004 * 6, 0.5 + 0.004 * (f - 6)]
    # Frames 11..13: bola praticamente parada (< 1.0 m/s) -> dir_change deve ser NaN
    for f in range(11, 14):
        ball[f, :2] = [0.2 + 0.004 * 6, 0.5 + 0.004 * 4]

    frame, _ = pt.prepare_tracking(players, ball, teams, blocks=blocks, kickoffs={1: 0}, n_frames=n)

    np.testing.assert_allclose(frame.loc[1:4, "ball_dir_change"], 0.0, atol=1e-5)
    assert frame.loc[7, "ball_dir_change"] > 0.4  # mudança brusca de direção na quina
    assert frame.loc[12:13, "ball_dir_change"].isna().all()
