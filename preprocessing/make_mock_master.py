"""Gerador do master_features_mock.parquet (100% sintetico).

Gera um DataFrame deterministico respeitando integralmente o contrato congelado
v1.0 (SCHEMA.md e preprocessing/schema.py: master_columns()), para permitir
desenvolvimento, validacao e testes dos modelos antes da entrega do master real.

Saida padrao: data/processed/master_features_mock.parquet
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from games import GAME, GAMES, half_video_blocks

try:
    from .params import (
        FPS,
        KICKOFF_FALLBACK,
        MAX_GAP_FRAMES,
        MIN_BALL_SPEED_DIR,
        N_FRAMES,
        PITCH_L,
        PITCH_MARGIN,
        PITCH_W,
        PRESSURE_TAU,
        SCHEMA_VERSION,
    )
    from .schema import master_columns
except ImportError:
    from preprocessing.params import (
        FPS,
        KICKOFF_FALLBACK,
        MAX_GAP_FRAMES,
        MIN_BALL_SPEED_DIR,
        N_FRAMES,
        PITCH_L,
        PITCH_MARGIN,
        PITCH_W,
        PRESSURE_TAU,
        SCHEMA_VERSION,
    )
    from preprocessing.schema import master_columns


def _gap_features(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula gap_length, since_last_obs e until_next_obs para um vetor booleano de observacoes."""
    n = len(obs)
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(obs, idx, -1))
    nxt = np.minimum.accumulate(np.where(obs, idx, n)[::-1])[::-1]

    since = np.where(last >= 0, idx - last, np.nan).astype(np.float32)
    until = np.where(nxt < n, nxt - idx, np.nan).astype(np.float32)

    gap = np.zeros(n, dtype=np.float32)
    changes = np.flatnonzero(np.diff(np.r_[True, obs, True].astype(np.int8)))
    for a, b in zip(changes[::2], changes[1::2]):
        gap[a:b] = b - a

    return gap, since, until


def _generate_ball_mask(half_len: int, rng: np.random.Generator) -> np.ndarray:
    """Gera mascara booleana de bola observada (~41% ausente em lacunas realistas)."""
    obs = np.ones(half_len, dtype=bool)
    pos = int(rng.integers(15, 45))
    while pos < half_len - 50:
        obs_len = int(rng.integers(15, 52))
        pos += obs_len
        if pos >= half_len - 50:
            break
        # 60% lacunas curtas (1..12 frames), 40% lacunas longas (13..84 frames)
        if rng.random() < 0.60:
            gap_len = int(rng.integers(1, 13))
        else:
            gap_len = int(rng.integers(13, 85))
        end = min(pos + gap_len, half_len)
        obs[pos:end] = False
        pos = end
    return obs


def _smooth_trajectory(n: int, rng: np.random.Generator, x_bounds: tuple[float, float],
                       y_bounds: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Gera trajetoria suave em 2D contida nos limites informados."""
    steps_x = rng.normal(0.0, 0.35, size=n)
    steps_y = rng.normal(0.0, 0.30, size=n)

    # Adiciona tendencia leve para o centro do campo
    cx = (x_bounds[0] + x_bounds[1]) / 2.0
    cy = (y_bounds[0] + y_bounds[1]) / 2.0

    x = np.empty(n, dtype=np.float32)
    y = np.empty(n, dtype=np.float32)
    cur_x, cur_y = cx, cy

    for i in range(n):
        cur_x += float(steps_x[i]) + 0.005 * (cx - cur_x)
        cur_y += float(steps_y[i]) + 0.005 * (cy - cur_y)
        cur_x = float(np.clip(cur_x, x_bounds[0], x_bounds[1]))
        cur_y = float(np.clip(cur_y, y_bounds[0], y_bounds[1]))
        x[i] = cur_x
        y[i] = cur_y

    return x, y


def generate_mock_master(
    game: str = GAME,
    seed: int = 42,
    include_optional: bool = True,
) -> pd.DataFrame:
    """Gera o master_features 100% sintetico e deterministico conforme contrato do schema.py."""
    rng = np.random.default_rng(seed)

    game_cfg = GAMES.get(game, {})
    n_frames = game_cfg.get("n_video_frames", N_FRAMES)
    fps = game_cfg.get("fps", FPS)
    pitch_l, pitch_w = game_cfg.get("pitch", (PITCH_L, PITCH_W))
    blocks = half_video_blocks(game) if game in GAMES else [(1, 11185, 90651), (2, 110808, 190197)]

    # Colunas de indice
    video_frame = np.arange(n_frames, dtype=np.int32)
    half = np.zeros(n_frames, dtype=np.int8)
    frame_in_half = np.full(n_frames, np.nan, dtype=np.float32)

    for h, f0, f1 in blocks:
        half[f0:f1 + 1] = h
        frame_in_half[f0:f1 + 1] = np.arange(f1 - f0 + 1, dtype=np.float32)

    is_graded = half != 0

    # Inicializa arrays da bola e confianca
    ball_observed = np.zeros(n_frames, dtype=np.int8)
    ball_jump_removed = np.zeros(n_frames, dtype=np.int8)
    ball_filled = np.zeros(n_frames, dtype=np.int8)
    gap_length = np.full(n_frames, np.nan, dtype=np.float32)
    since_last_obs = np.full(n_frames, np.nan, dtype=np.float32)
    until_next_obs = np.full(n_frames, np.nan, dtype=np.float32)

    ball_x = np.full(n_frames, np.nan, dtype=np.float32)
    ball_y = np.full(n_frames, np.nan, dtype=np.float32)
    ball_vx = np.full(n_frames, np.nan, dtype=np.float32)
    ball_vy = np.full(n_frames, np.nan, dtype=np.float32)
    ball_speed = np.full(n_frames, np.nan, dtype=np.float32)
    ball_acc = np.full(n_frames, np.nan, dtype=np.float32)
    ball_dir_change = np.full(n_frames, np.nan, dtype=np.float32)

    # Inicializa arrays de times
    centroid_home_x = np.full(n_frames, np.nan, dtype=np.float32)
    centroid_home_y = np.full(n_frames, np.nan, dtype=np.float32)
    centroid_away_x = np.full(n_frames, np.nan, dtype=np.float32)
    centroid_away_y = np.full(n_frames, np.nan, dtype=np.float32)
    spread_home_x = np.full(n_frames, np.nan, dtype=np.float32)
    spread_home_y = np.full(n_frames, np.nan, dtype=np.float32)
    spread_away_x = np.full(n_frames, np.nan, dtype=np.float32)
    spread_away_y = np.full(n_frames, np.nan, dtype=np.float32)
    mean_speed_home = np.full(n_frames, np.nan, dtype=np.float32)
    mean_speed_away = np.full(n_frames, np.nan, dtype=np.float32)
    n_still_home = np.full(n_frames, np.nan, dtype=np.float32)
    n_still_away = np.full(n_frames, np.nan, dtype=np.float32)
    n_tracked_home = np.full(n_frames, np.nan, dtype=np.float32)
    n_tracked_away = np.full(n_frames, np.nan, dtype=np.float32)

    # Proximidade
    d_near_home = np.full(n_frames, np.nan, dtype=np.float32)
    d_near_away = np.full(n_frames, np.nan, dtype=np.float32)
    d_diff = np.full(n_frames, np.nan, dtype=np.float32)
    n_home_r2 = np.full(n_frames, np.nan, dtype=np.float32)
    n_home_r3 = np.full(n_frames, np.nan, dtype=np.float32)
    n_home_r5 = np.full(n_frames, np.nan, dtype=np.float32)
    n_away_r2 = np.full(n_frames, np.nan, dtype=np.float32)
    n_away_r3 = np.full(n_frames, np.nan, dtype=np.float32)
    n_away_r5 = np.full(n_frames, np.nan, dtype=np.float32)
    pressure_home = np.full(n_frames, np.nan, dtype=np.float32)
    pressure_away = np.full(n_frames, np.nan, dtype=np.float32)

    # Velocidade relativa
    speed_rel_home = np.full(n_frames, np.nan, dtype=np.float32)
    speed_rel_away = np.full(n_frames, np.nan, dtype=np.float32)
    speed_radial_home = np.full(n_frames, np.nan, dtype=np.float32)
    speed_radial_away = np.full(n_frames, np.nan, dtype=np.float32)

    # Campo
    out_of_pitch = np.full(n_frames, np.nan, dtype=np.float32)
    d_touchline = np.full(n_frames, np.nan, dtype=np.float32)
    d_goalline = np.full(n_frames, np.nan, dtype=np.float32)

    # Set pieces e fases
    is_set_piece = np.zeros(n_frames, dtype=bool)
    sp_type = np.full(n_frames, "", dtype=object)
    sp_team = np.full(n_frames, "", dtype=object)
    frames_to_next_sp = np.full(n_frames, np.nan, dtype=np.float32)
    frames_since_sp_end = np.full(n_frames, np.nan, dtype=np.float32)
    next_sp_type = np.full(n_frames, "", dtype=object)
    prev_sp_type = np.full(n_frames, "", dtype=object)

    n_tags_favor_home = np.zeros(n_frames, dtype=np.int16)
    n_tags_favor_away = np.zeros(n_frames, dtype=np.int16)

    pre_kickoff = np.zeros(n_frames, dtype=bool)
    reproj_error = np.full(n_frames, np.nan, dtype=np.float32)

    # Set piece types pool
    sp_pool = [
        "throw-in", "corner kick", "goal kick", "free kick",
        "throw-in", "throw-in", "corner kick", "goal kick"
    ]
    sp_team_pool = ["H", "A", "U", "H", "A"]

    # Processa cada tempo independentemente
    for h, f0, f1 in blocks:
        half_len = f1 - f0 + 1
        sl = slice(f0, f1 + 1)

        # 1. Bola e confianca
        obs = _generate_ball_mask(half_len, rng)
        gl, st, un = _gap_features(obs)

        gap_length[sl] = gl
        since_last_obs[sl] = st
        until_next_obs[sl] = un
        ball_observed[sl] = obs.astype(np.int8)

        # Lacunas curtas interpoladas (MAX_GAP_FRAMES = 12)
        filled = np.zeros(half_len, dtype=np.int8)
        # Interpola apenas lacunas internas com gap <= MAX_GAP_FRAMES
        changes = np.flatnonzero(np.diff(np.r_[True, obs, True].astype(np.int8)))
        for a, b in zip(changes[::2], changes[1::2]):
            g_len = b - a
            # Verifica se esta entre observacoes validas (nao no inicio nem no fim do tempo)
            if g_len <= MAX_GAP_FRAMES and a > 0 and b < half_len:
                filled[a:b] = 1

        ball_filled[sl] = filled

        # Pequena fracao de jump outliers removidos em lacunas nao preenchidas
        unfilled_gaps = np.flatnonzero(~obs & (filled == 0))
        if len(unfilled_gaps) > 0:
            jump_cnt = max(1, int(len(unfilled_gaps) * 0.003))
            jump_idx = rng.choice(unfilled_gaps, size=jump_cnt, replace=False)
            jump_arr = np.zeros(half_len, dtype=np.int8)
            jump_arr[jump_idx] = 1
            ball_jump_removed[sl] = jump_arr

        # Trajetoria da bola
        has_ball = obs | (filled == 1)
        raw_bx, raw_by = _smooth_trajectory(half_len, rng, (3.0, pitch_l - 3.0), (3.0, pitch_w - 3.0))

        cur_bx = np.where(has_ball, raw_bx, np.nan).astype(np.float32)
        cur_by = np.where(has_ball, raw_by, np.nan).astype(np.float32)
        ball_x[sl] = cur_bx
        ball_y[sl] = cur_by

        # Derivadas: estritamente sobre segmentos observados (ball_filled == 0 e obs == 1)
        cur_bvx = np.full(half_len, np.nan, dtype=np.float32)
        cur_bvy = np.full(half_len, np.nan, dtype=np.float32)
        cur_bspeed = np.full(half_len, np.nan, dtype=np.float32)
        cur_bacc = np.full(half_len, np.nan, dtype=np.float32)
        cur_bdir = np.full(half_len, np.nan, dtype=np.float32)

        obs_runs = np.flatnonzero(np.diff(np.r_[False, obs, False].astype(np.int8)))
        for a, b in zip(obs_runs[::2], obs_runs[1::2]):
            if b - a >= 2:
                seg_x = cur_bx[a:b]
                seg_y = cur_by[a:b]
                vx_seg = np.gradient(seg_x) * fps
                vy_seg = np.gradient(seg_y) * fps
                spd_seg = np.hypot(vx_seg, vy_seg)
                ax_seg = np.gradient(vx_seg) * fps
                ay_seg = np.gradient(vy_seg) * fps
                acc_seg = np.hypot(ax_seg, ay_seg)

                cur_bvx[a:b] = vx_seg
                cur_bvy[a:b] = vy_seg
                cur_bspeed[a:b] = spd_seg
                cur_bacc[a:b] = acc_seg

                if b - a >= 3:
                    fast = (spd_seg[:-1] >= MIN_BALL_SPEED_DIR) & (spd_seg[1:] >= MIN_BALL_SPEED_DIR)
                    cross = np.abs(vx_seg[:-1] * vy_seg[1:] - vy_seg[:-1] * vx_seg[1:])
                    dot = vx_seg[:-1] * vx_seg[1:] + vy_seg[:-1] * vy_seg[1:]
                    dc = np.full(len(spd_seg), np.nan, dtype=np.float32)
                    dc[1:] = np.where(fast, np.arctan2(cross, dot), np.nan)
                    cur_bdir[a:b] = dc

        ball_vx[sl] = cur_bvx
        ball_vy[sl] = cur_bvy
        ball_speed[sl] = cur_bspeed
        ball_acc[sl] = cur_bacc
        ball_dir_change[sl] = cur_bdir

        # 2. Times (centroides e dispersoes)
        c_hx, c_hy = _smooth_trajectory(half_len, rng, (25.0, 65.0), (25.0, 43.0))
        c_ax, c_ay = _smooth_trajectory(half_len, rng, (40.0, 80.0), (25.0, 43.0))

        centroid_home_x[sl] = c_hx
        centroid_home_y[sl] = c_hy
        centroid_away_x[sl] = c_ax
        centroid_away_y[sl] = c_ay

        spread_home_x[sl] = rng.uniform(12.0, 22.0, size=half_len).astype(np.float32)
        spread_home_y[sl] = rng.uniform(10.0, 18.0, size=half_len).astype(np.float32)
        spread_away_x[sl] = rng.uniform(12.0, 22.0, size=half_len).astype(np.float32)
        spread_away_y[sl] = rng.uniform(10.0, 18.0, size=half_len).astype(np.float32)

        mean_speed_home[sl] = rng.uniform(1.2, 2.8, size=half_len).astype(np.float32)
        mean_speed_away[sl] = rng.uniform(1.2, 2.8, size=half_len).astype(np.float32)
        n_still_home[sl] = rng.integers(1, 6, size=half_len).astype(np.float32)
        n_still_away[sl] = rng.integers(1, 6, size=half_len).astype(np.float32)
        n_tracked_home[sl] = rng.integers(10, 12, size=half_len).astype(np.float32)
        n_tracked_away[sl] = rng.integers(10, 12, size=half_len).astype(np.float32)

        # 3. Proximidade (definidas apenas onde has_ball e True)
        d_h = rng.uniform(0.5, 7.5, size=half_len).astype(np.float32)
        d_a = rng.uniform(0.5, 7.5, size=half_len).astype(np.float32)
        d_near_home[sl] = np.where(has_ball, d_h, np.nan)
        d_near_away[sl] = np.where(has_ball, d_a, np.nan)
        d_diff[sl] = np.where(has_ball, d_h - d_a, np.nan)

        # Contagens de jogadores ao redor da bola (nunca zero sem bola, sempre NaN)
        n_home_r2[sl] = np.where(has_ball, (d_h < 2.0).astype(np.float32), np.nan)
        n_home_r3[sl] = np.where(has_ball, (d_h < 3.0).astype(np.float32) + rng.integers(0, 2, half_len), np.nan)
        n_home_r5[sl] = np.where(has_ball, (d_h < 5.0).astype(np.float32) + rng.integers(1, 3, half_len), np.nan)
        n_away_r2[sl] = np.where(has_ball, (d_a < 2.0).astype(np.float32), np.nan)
        n_away_r3[sl] = np.where(has_ball, (d_a < 3.0).astype(np.float32) + rng.integers(0, 2, half_len), np.nan)
        n_away_r5[sl] = np.where(has_ball, (d_a < 5.0).astype(np.float32) + rng.integers(1, 3, half_len), np.nan)

        press_h = np.exp(-d_h / PRESSURE_TAU) + rng.uniform(0.0, 0.5, size=half_len)
        press_a = np.exp(-d_a / PRESSURE_TAU) + rng.uniform(0.0, 0.5, size=half_len)
        pressure_home[sl] = np.where(has_ball, press_h, np.nan).astype(np.float32)
        pressure_away[sl] = np.where(has_ball, press_a, np.nan).astype(np.float32)

        # 4. Velocidade relativa (apenas onde velocidade observada da bola existe)
        valid_bvel = np.isfinite(cur_bvx)
        v_rel_h = rng.uniform(0.5, 12.0, size=half_len).astype(np.float32)
        v_rel_a = rng.uniform(0.5, 12.0, size=half_len).astype(np.float32)
        speed_rel_home[sl] = np.where(valid_bvel, v_rel_h, np.nan)
        speed_rel_away[sl] = np.where(valid_bvel, v_rel_a, np.nan)
        speed_radial_home[sl] = np.where(valid_bvel, rng.uniform(-10.0, 10.0, size=half_len), np.nan)
        speed_radial_away[sl] = np.where(valid_bvel, rng.uniform(-10.0, 10.0, size=half_len), np.nan)

        # 5. Campo
        oop = ((cur_bx < -PITCH_MARGIN) | (cur_bx > pitch_l + PITCH_MARGIN) |
               (cur_by < -PITCH_MARGIN) | (cur_by > pitch_w + PITCH_MARGIN)).astype(np.float32)
        out_of_pitch[sl] = np.where(has_ball, oop, np.nan)
        d_touch = np.minimum(cur_by, pitch_w - cur_by)
        d_goal = np.minimum(cur_bx, pitch_l - cur_bx)
        d_touchline[sl] = np.where(has_ball, d_touch, np.nan)
        d_goalline[sl] = np.where(has_ball, d_goal, np.nan)

        # 6. Set pieces (calculados estritamente por tempo)
        events_half: list[tuple[int, int, str, str]] = []
        cur_pos = f0 + int(rng.integers(20, 60))
        # Adiciona kickoff inicial do tempo
        kick_end = min(cur_pos + int(rng.integers(100, 200)), f1)
        events_half.append((cur_pos, kick_end, "kick off (start)", "H" if h == 1 else "A"))
        cur_pos = kick_end + int(rng.integers(500, 1500))

        while cur_pos < f1 - 300:
            sp_dur = int(rng.integers(80, 250))
            sp_t = str(rng.choice(sp_pool))
            sp_tm = str(rng.choice(sp_team_pool))
            sp_end = min(cur_pos + sp_dur, f1)
            events_half.append((cur_pos, sp_end, sp_t, sp_tm))
            cur_pos = sp_end + int(rng.integers(800, 2500))

        for ev_start, ev_end, t_name, tm_name in events_half:
            is_set_piece[ev_start:ev_end] = True
            sp_type[ev_start:ev_end] = t_name
            sp_team[ev_start:ev_end] = tm_name

        sp_starts = np.array([ev[0] for ev in events_half], dtype=int)
        sp_ends = np.array([ev[1] for ev in events_half], dtype=int)

        frames_half = np.arange(f0, f1 + 1)
        # to_next_sp
        idx_next = np.searchsorted(sp_starts, frames_half, side="left")
        ok_next = idx_next < len(sp_starts)
        frames_to_next_sp[f0 + np.flatnonzero(ok_next)] = (sp_starts[idx_next[ok_next]] - frames_half[ok_next]).astype(np.float32)
        for i_local, ok in enumerate(ok_next):
            if ok:
                next_sp_type[f0 + i_local] = events_half[idx_next[i_local]][2]

        # since_sp_end
        idx_prev = np.searchsorted(sp_ends, frames_half, side="right") - 1
        ok_prev = idx_prev >= 0
        frames_since_sp_end[f0 + np.flatnonzero(ok_prev)] = (frames_half[ok_prev] - sp_ends[idx_prev[ok_prev]]).astype(np.float32)
        for i_local, ok in enumerate(ok_prev):
            if ok:
                prev_sp_type[f0 + i_local] = events_half[idx_prev[i_local]][2]

        # 7. Fases (0..3 fases ativas)
        n_tags_favor_home[sl] = rng.integers(0, 3, size=half_len, dtype=np.int16)
        n_tags_favor_away[sl] = rng.integers(0, 3, size=half_len, dtype=np.int16)

        # 8. Homografia opcional
        reproj_error[sl] = rng.normal(1.2, 0.25, size=half_len).clip(0.3, 4.0).astype(np.float32)

    # 9. Pontape inicial: frames avaliados antes do kickoff do 2o tempo (2.934 frames)
    # No 2o tempo: f0 = 110808 ate kickoff 113742
    kickoff_2 = KICKOFF_FALLBACK.get(2, 113742)
    if len(blocks) >= 2:
        f0_2 = blocks[1][1]
        if kickoff_2 > f0_2:
            pre_kickoff[f0_2:kickoff_2] = True

    # Monta DataFrame na ordem exata de master_columns()
    cols_order = master_columns(include_optional=include_optional)
    data = {}

    local_vars = {
        "video_frame": video_frame,
        "half": half,
        "is_graded": is_graded,
        "frame_in_half": frame_in_half,
        "ball_x": ball_x,
        "ball_y": ball_y,
        "ball_vx": ball_vx,
        "ball_vy": ball_vy,
        "ball_speed": ball_speed,
        "ball_acc": ball_acc,
        "ball_dir_change": ball_dir_change,
        "ball_observed": ball_observed,
        "ball_jump_removed": ball_jump_removed,
        "ball_filled": ball_filled,
        "gap_length": gap_length,
        "since_last_obs": since_last_obs,
        "until_next_obs": until_next_obs,
        "centroid_home_x": centroid_home_x,
        "centroid_home_y": centroid_home_y,
        "centroid_away_x": centroid_away_x,
        "centroid_away_y": centroid_away_y,
        "spread_home_x": spread_home_x,
        "spread_home_y": spread_home_y,
        "spread_away_x": spread_away_x,
        "spread_away_y": spread_away_y,
        "mean_speed_home": mean_speed_home,
        "mean_speed_away": mean_speed_away,
        "n_still_home": n_still_home,
        "n_still_away": n_still_away,
        "n_tracked_home": n_tracked_home,
        "n_tracked_away": n_tracked_away,
        "d_near_home": d_near_home,
        "d_near_away": d_near_away,
        "d_diff": d_diff,
        "n_home_r2": n_home_r2,
        "n_home_r3": n_home_r3,
        "n_home_r5": n_home_r5,
        "n_away_r2": n_away_r2,
        "n_away_r3": n_away_r3,
        "n_away_r5": n_away_r5,
        "pressure_home": pressure_home,
        "pressure_away": pressure_away,
        "speed_rel_home": speed_rel_home,
        "speed_rel_away": speed_rel_away,
        "speed_radial_home": speed_radial_home,
        "speed_radial_away": speed_radial_away,
        "out_of_pitch": out_of_pitch,
        "d_touchline": d_touchline,
        "d_goalline": d_goalline,
        "is_set_piece": is_set_piece,
        "sp_type": sp_type,
        "sp_team": sp_team,
        "frames_to_next_sp": frames_to_next_sp,
        "frames_since_sp_end": frames_since_sp_end,
        "next_sp_type": next_sp_type,
        "prev_sp_type": prev_sp_type,
        "n_tags_favor_home": n_tags_favor_home,
        "n_tags_favor_away": n_tags_favor_away,
        "pre_kickoff": pre_kickoff,
        "reproj_error": reproj_error,
    }

    for col_name, dtype_name in cols_order:
        arr = local_vars[col_name]
        if dtype_name == "string":
            data[col_name] = pd.Series(arr, dtype="string")
        elif dtype_name == "bool":
            data[col_name] = pd.Series(arr, dtype=bool)
        else:
            data[col_name] = pd.Series(arr, dtype=dtype_name)

    return pd.DataFrame(data)


def save_mock_master(
    df: pd.DataFrame,
    output_path: str,
    schema_version: str = SCHEMA_VERSION,
) -> None:
    """Salva o DataFrame mock em formato Parquet com os metadados do schema."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    existing_meta = table.schema.metadata or {}
    new_meta = {**existing_meta, b"schema_version": schema_version.encode("utf-8")}
    table = table.replace_schema_metadata(new_meta)
    pq.write_table(table, output_path, compression="snappy")


def default_mock_path(data_dir: str | None = None) -> str:
    """Retorna o caminho padrao do mock: data/processed/master_features_mock.parquet."""
    root = data_dir or os.environ.get("MTAH_DATA_DIR") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )
    return os.path.join(root, "processed", "master_features_mock.parquet")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", default=GAME, help=f"identificador da partida (padrao: {GAME})")
    parser.add_argument("--seed", type=int, default=42, help="seed para gerador pseudoaleatorio (padrao: 42)")
    parser.add_argument("--output", help="caminho do arquivo parquet de saida")
    parser.add_argument("--data-dir", help="raiz dos dados (padrao: MTAH_DATA_DIR ou ./data)")
    parser.add_argument("--no-optional", action="store_true", help="ignorar colunas opcionais (reproj_error)")
    args = parser.parse_args(argv)

    output_path = args.output or default_mock_path(args.data_dir)
    print(f"Gerando master mock para o jogo {args.game} (seed={args.seed})...")
    df = generate_mock_master(
        game=args.game,
        seed=args.seed,
        include_optional=not args.no_optional,
    )

    save_mock_master(df, output_path)

    graded_mask = df["is_graded"]
    ball_missing_ratio = (df.loc[graded_mask, "ball_observed"] == 0).mean()
    print(f"Sucesso! Geradas {len(df):,} linhas e {len(df.columns)} colunas.")
    print(f"Bola ausente nos frames avaliados: {ball_missing_ratio:.1%}")
    print(f"Salvo em: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
