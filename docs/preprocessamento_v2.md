# Pré-processamento v2 — especificação

Versão em Markdown do `MTAH2027_preprocessamento_v2.pdf` (o PDF não vai ao repositório: `*.pdf` está no `.gitignore`).
Contrato de colunas e decisões fechadas: [`SCHEMA.md`](../SCHEMA.md). Em caso de divergência com os documentos oficiais do desafio (`DATA.md`, `RULES.md`, `EVALUATION.md`), valem os oficiais.

> Os trechos de código foram testados com dados sintéticos, não com os dados reais. São ponto de partida: nomes de colunas e formatos reais devem ser confirmados localmente.

## 1. Princípios (valem para todos os scripts)

- **Separar o observado do inferido.** Toda interpolação ou inferência é marcada; o dado original continua acessível.
- **Offline e bidirecional** (passado e futuro). Nada de filtro causal: o atraso nas trocas de posse custa caro no spellF1.
- **Sempre por tempo de jogo** (`half_video_blocks`). Nada de suavizar, interpolar ou derivar atravessando o intervalo.
- **Coletivo, não individual.** Os 46 slots trocam de pessoa; usar agregados por time.
- **Parâmetros justificados** por física, pela distribuição dos próprios dados de tracking (sem rótulos) ou por outros jogos. **Nunca ajustados nos 5 minutos públicos** e nunca atrelados a frames específicos.
- **Determinismo:** mesma entrada, mesma saída. Seeds fixas e versões pinadas (`RULES.md`: o repositório precisa regenerar o mesmo arquivo).
- **Dados do desafio não vão** para LLMs em nuvem (tabelas, valores impressos, trechos, capturas) **nem** para o repositório público. `data/processed/` está dentro de `data/`, já no `.gitignore`.
- **Nada de correção manual** do tracking do jogo fornecido (`RULES.md`: rótulos têm de vir de algoritmo).

## 2. Fluxo de dados e responsabilidades

```
ATD ----> prep_tracking.py -------> tracking_clean.parquet (+ slot_teams.parquet) --+--------------------+
          (Antônio)                                                                 |                    |
                                   prep_proximities.py <----------------------------+                    |
                                   (Hugo) --> proximities.parquet ---------------------------------------+
XML tag -> prep_smart_tagging.py --> tagging.parquet (+ kickoffs.json) -------------------------------------+
CSV hom -> prep_homography.py ----> homography.parquet (opcional) ----------------------------------------+
           (Fernando)
                                   build_master_table.py <------------------------------------------------+
                                   (Fernando) --> data/processed/master_features.parquet
```

| Script | Dono | Lê | Escreve |
|---|---|---|---|
| `preprocessing/prep_tracking.py` | Antônio | ATD (txt + xml), `games.py`, `kickoffs.json` | `tracking_clean.parquet`, `slot_teams.parquet` |
| `preprocessing/prep_proximities.py` | Hugo | `tracking_clean.parquet`, `slot_teams.parquet` | `proximities.parquet` |
| `preprocessing/prep_smart_tagging.py` / `prep_homography.py` | Fernando | XML do tagging; CSV da homografia | `tagging.parquet`, `kickoffs.json`, `homography.parquet` (opcional) |
| `preprocessing/build_master_table.py` | Fernando | os três acima | `master_features.parquet` |
| `preprocessing/make_mock_master.py`, `tests/test_master_schema.py` | Bernardo / Ahmed | `preprocessing/schema.py` | mock sintético e testes |

**Dependência A → B e V0.** O script de proximidades precisa das posições limpas. O Antônio libera a V0 do `tracking_clean.parquet` (metros, orientação por tempo, bola e slots sem filtros, mapeamento slot-time). Nomes e colunas ficam idênticos na V1 (com filtros); o Hugo programa contra a V0 e reexecuta sobre a V1.

## 3. Script A — `prep_tracking.py` (Antônio)

| Tarefa | Descrição |
|---|---|
| A1 Carga e alinhamento | Ler o ATD com `kloppy.metrica.load_tracking_epts` + `asset_paths()`, **ou** em streaming direto do `.txt`/`.xml` como em `starter/baseline.py` (muito mais rápido e leve em RAM; traz também o `s` escalar nativo). `video_frame = frame_id - 1`. Asserts: 199.319 linhas, faixa 0..199.318 em ordem. |
| A2 Tempos de jogo | `half_video_blocks('BAR-ATH')`: (1, 11185, 90651) e (2, 110808, 190197). Cada tempo é independente. |
| A3 Metros e orientação | Multiplicar por 105,3 × 68,0 (`games.py`). Descobrir por tempo se a casa defende x pequeno e refletir o campo para que cada time ataque sempre no mesmo sentido. Inverter também velocidade e aceleração. Assert: o flip do 2º tempo é o oposto do 1º. |
| A4 Bola | Filtro de saltos (cadeias mutuamente consistentes com reconexão sobre blocos fantasmas `> K` e retenção de cadeias longas disjuntas) → colunas de confiança depois do filtro → interpolação só de lacunas curtas → suavização zero-fase apenas para derivar nos trechos observados (`ball_observed == 1`), mascarando velocidades e aceleração como `NaN` onde `ball_filled == 1` → `ball_vx, ball_vy, ball_speed, ball_acc, ball_dir_change`. |
| A5 Jogadores | `slot_break` nos saltos de 1 frame; apagar a posição de pontos isolados. Velocidade vetorial por derivada da posição suavizada, por trechos entre quebras. Remover duplicatas (mesmo time a <1 m): zerar como `NaN` o slot duplicado no frame no `tracking_clean.parquet` (V1), para que A6 e B2–B4 já fiquem sem duplicatas. |
| A6 Agregados por time | `centroid_*`, `spread_*`, `mean_speed_*`, `n_still_*`, `n_tracked_*`, robustos à troca de slots. |
| A7 Time de cada slot | Verificar o mapeamento no XML (`starter/baseline.py: team_of_each_track`) e se há objetos que não são jogadores de linha. Exportar `slot_teams.parquet`. |
| A8 Saída | `tracking_clean.parquet` (V0, depois V1). |

### Filtro de saltos da bola (cadeias mutuamente consistentes)
```python
import numpy as np

def remove_ball_jump_outliers(x, y, vmax=38.0, fps=25.0, eps=1.0, k=30, min_chain=None):
    """x, y em metros (NaN onde falta). Mantem cadeias de observacoes mutuamente
    consistentes (|dr| <= vmax*dt + eps); reconecta sobre blocos fantasmas > k usando
    ancoras de fim de segmento e preserva cadeias longas disjuntas (>= min_chain)."""
    obs = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
    xo, yo = x.copy(), y.copy()
    m = len(obs)
    if m < 2:
        return xo, yo
    if min_chain is None:
        min_chain = k
    best = np.ones(m, dtype=np.int32)
    prev = np.full(m, -1, dtype=np.int32)
    run_start = np.zeros(m, dtype=np.int32)
    anchors = []
    for i in range(1, m):
        dt_prev = (obs[i] - obs[i - 1]) / fps
        step_dist = np.hypot(x[obs[i]] - x[obs[i - 1]], y[obs[i]] - y[obs[i - 1]])
        if step_dist <= vmax * dt_prev + eps:
            run_start[i] = run_start[i - 1]
            js = np.arange(max(run_start[i], i - k), i, dtype=np.int32)
        else:
            run_start[i] = i
            anchors.append(i - 1)
            start_j = max(0, i - k)
            js = np.arange(start_j, i, dtype=np.int32)
            if anchors:
                prior = np.asarray(anchors[-k:], dtype=np.int32)
                prior = prior[prior < start_j]
                if len(prior):
                    js = np.r_[prior, js]
        dt = (obs[i] - obs[js]) / fps
        dist = np.hypot(x[obs[i]] - x[obs[js]], y[obs[i]] - y[obs[js]])
        valid = js[dist <= vmax * dt + eps]
        if len(valid):
            j = valid[np.argmax(best[valid])]
            best[i], prev[i] = best[j] + 1, j
    keep = np.zeros(len(x), dtype=bool)
    blocked = np.zeros(m, dtype=bool)
    first_chain = True
    while not blocked.all():
        candidates = np.flatnonzero(~blocked)
        i = int(candidates[np.argmax(best[candidates])])
        chain = []
        while i >= 0 and not blocked[i]:
            chain.append(i)
            p = int(prev[i])
            if p >= 0 and blocked[p:i + 1].any():
                break
            i = p
        if not first_chain and len(chain) < min_chain:
            blocked[chain] = True
            continue
        first_chain = False
        for idx in chain:
            keep[obs[idx]] = True
        blocked[min(chain):max(chain) + 1] = True
    xo[~keep], yo[~keep] = np.nan, np.nan
    return xo, yo
```
Por que assim: um fantasma no meio de uma lacuna (mesmo que dure mais de `k` frames) não permite que seus frames internos roubem a cadeia real anterior (`run_start`), enquanto o retorno da bola real testa as âncoras anteriores ao salto (`anchors`) e fecha a ponte sobre o bloco fantasma. Além disso, preservar cadeias disjuntas com pelo menos `min_chain` observações garante que uma descontinuidade legítima (ex.: reposição rápida no sistema multi-bolas) nunca descarte minutos de bola real. Conferir a taxa de remoção ao longo do tempo (por janelas do jogo), sem criar regras para frames específicos.

### Colunas de confiança bidirecionais e interpolação curta
```python
import numpy as np
import pandas as pd

def gap_features(obs):
    """obs: bool (n,), True = bola observada (calcular DEPOIS do filtro de saltos)."""
    n = len(obs)
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(obs, idx, -1))
    nxt = np.minimum.accumulate(np.where(obs, idx, n)[::-1])[::-1]
    since_last = np.where(last >= 0, idx - last, np.nan)   # NaN antes da 1a observacao
    until_next = np.where(nxt < n, nxt - idx, np.nan)      # NaN depois da ultima
    edges = np.flatnonzero(np.diff(np.r_[True, obs, True].astype(int)))
    gap_len = np.zeros(n, int)
    for a, b in zip(edges[::2], edges[1::2]):               # cada buraco = [a, b)
        gap_len[a:b] = b - a
    return gap_len, since_last, until_next

def fill_short_gaps(x, gap_len, max_gap):
    filled = pd.Series(x).interpolate(method="linear", limit_area="inside").to_numpy()
    return np.where(gap_len > max_gap, np.nan, filled)      # buracos longos ficam NaN

# chamar SEMPRE por tempo de jogo, nunca atravessando o intervalo
```

### Suavização offline, derivadas observadas e mudança de direção da bola
```python
import numpy as np
from scipy.signal import savgol_filter

def smooth_ball_segment(seg):
    """Suavizacao offline de um trecho finito; segmentos curtos ficam inalterados."""
    seg = np.asarray(seg, dtype=np.float64)
    if len(seg) >= 7:
        return savgol_filter(seg, window_length=7, polyorder=2, mode="interp")
    return seg.copy()

def segment_gradient(values, valid, fps=25.0, smooth=True):
    """Deriva apenas dentro de segmentos contiguos validos (valid == True, comprimento >= 2).
    Para a bola, passar valid = (ball_observed == 1) para mascarar ball_filled == 1 como NaN."""
    out = np.full(len(values), np.nan, dtype=np.float32)
    edges = np.flatnonzero(np.diff(np.r_[False, valid, False].astype(np.int8)))
    for start, end in zip(edges[::2], edges[1::2]):
        if end - start >= 2:
            seg = smooth_ball_segment(values[start:end]) if smooth else values[start:end]
            out[start:end] = np.gradient(seg) * fps
    return out

def ball_direction_change(dvx, dvy, speed, min_speed=1.0):
    """Mudanca angular |dtheta| em [0, pi] entre t-1 e t; NaN se speed < min_speed."""
    dir_change = np.full(len(dvx), np.nan, dtype=np.float32)
    if len(dvx) >= 2:
        fast = (speed[:-1] >= min_speed) & (speed[1:] >= min_speed)
        cross = np.abs(dvx[:-1] * dvy[1:] - dvy[:-1] * dvx[1:])
        dot = dvx[:-1] * dvx[1:] + dvy[:-1] * dvx[1:]
        dir_change[1:] = np.where(fast, np.arctan2(cross, dot), np.nan).astype(np.float32)
    return dir_change
```

### Quebras de slot e pontos isolados
```python
import numpy as np

def slot_breaks(px, py, fps=25.0, vmax=12.0):
    """px, py: (T, N) em metros. Devolve (quebra, isolado), ambos (T, N) bool.
    quebra[t]   : salto entre t-1 e t -> proibido interpolar/derivar atravessando t
    isolado[t] : ponto que salta na entrada E na saida, com vizinhos consistentes
                   entre si -> erro de posicao: pode virar NaN"""
    dt = 1.0 / fps
    jump = np.hypot(np.diff(px, axis=0), np.diff(py, axis=0)) / dt > vmax   # (T-1, N)
    quebra = np.zeros(px.shape, bool)
    quebra[1:] = jump
    skip = np.hypot(px[2:] - px[:-2], py[2:] - py[:-2]) / (2 * dt)          # t-1 -> t+1
    isolado = np.zeros(px.shape, bool)
    isolado[1:-1] = jump[:-1] & jump[1:] & (skip <= vmax)
    return quebra, isolado
```
A suposição por trás de `slot_break` é que, numa troca de identidade, a posição antiga e a nova são de jogadores reais. Que o `s` nativo do ATD continue normal nesse frame **não foi verificado** (ver verificações, seção 9).

### Orientação por tempo de jogo
```python
import numpy as np

L, W = 105.3, 68.0               # usar as constantes do games.py

def home_defends_low_x(px, is_home, kick_frame, win=25):
    """px: (T, N) em metros, indexado pelo tempo de jogo. No pontape inicial cada
    time esta na propria metade. True = casa defende x=0 (ataca para x crescente)."""
    seg = px[kick_frame:kick_frame + win][:, is_home]
    return np.nanmean(seg) < L / 2

def orient(x, y, vx, vy, flip):
    """Rotacao de 180 graus: posicao E componentes de velocidade (e aceleracao)."""
    if not flip:
        return x, y, vx, vy
    return L - x, W - y, -vx, -vy

# teste: o 'flip' do 2o tempo deve ser o oposto do 1o (os times trocam de lado)
```

## 4. Script B — `prep_proximities.py` (Hugo)

| Tarefa | Descrição |
|---|---|
| B1 Entrada | Ler `tracking_clean.parquet` (V0 primeiro, V1 depois). Montar matrizes (T, N) de posição e velocidade por tempo de jogo. |
| B2 Proximidade por time | `d_near_home`, `d_near_away`, `d_diff`: menor distância de um jogador de cada time à bola. Slots NaN ignorados. |
| B3 Contagens por raio e pressão | `n_*_r2`, `n_*_r3`, `n_*_r5` (jogadores dentro de 2, 3 e 5 m) e `pressure_home`, `pressure_away` (`sum(exp(-d / 3.0))`, contínua). NaN quando não há bola, e não zero. |
| B4 Velocidade relativa e radial | `speed_rel_home`, `speed_rel_away` (norma da **diferença vetorial** bola − jogador mais próximo) e `speed_radial_home`, `speed_radial_away` (componente radial `(b - p) · (vb - vp) / max(d, 1e-3)`: negativa = aproximando, positiva = afastando). Como `ball_vx/vy` é `NaN` onde `ball_filled == 1`, ambas ficam `NaN` nesses frames. |
| B5 Campo | `out_of_pitch` (com margem), `d_touchline`, `d_goalline`. A orientação do campo não muda essas medidas. |
| B6 Saída | `proximities.parquet` com `video_frame` e `half`, seguindo o contrato. |

```python
import numpy as np

L, W = 105.3, 68.0

def masked_nearest(d, mask):
    """d: (T, N) distancias (NaN onde falta); mask: (N,) bool. -> (indice, distancia float32)."""
    dm = np.where(mask[None, :] & np.isfinite(d), d, np.inf)
    j = dm.argmin(axis=1)
    dist = dm[np.arange(len(d)), j]
    return j, np.where(np.isfinite(dist), dist, np.nan).astype(np.float32)

def proximity_features(bx, by, bvx, bvy, px, py, pvx, pvy, is_home, radii=(2.0, 3.0, 5.0), tau=3.0):
    dx = bx[:, None] - px
    dy = by[:, None] - py
    d = np.hypot(dx, dy)
    T = np.arange(len(bx))
    no_ball = np.isnan(bx) | np.isnan(by)
    out = {}
    for name, mask in (("home", is_home), ("away", ~is_home)):
        j, dist = masked_nearest(d, mask)
        rel_vx = bvx - pvx[T, j]
        rel_vy = bvy - pvy[T, j]
        rel = np.hypot(rel_vx, rel_vy)                       # diferenca VETORIAL
        safe_d = np.maximum(dist, 1e-3)
        radial = (dx[T, j] * rel_vx + dy[T, j] * rel_vy) / safe_d
        out[f"d_near_{name}"] = dist
        out[f"speed_rel_{name}"] = np.where(np.isnan(dist), np.nan, rel).astype(np.float32)
        out[f"speed_radial_{name}"] = np.where(np.isnan(dist), np.nan, radial).astype(np.float32)
        dm = np.where(mask[None, :] & np.isfinite(d), d, np.inf)
        for r in radii:
             cnt = (dm < r).sum(axis=1).astype(np.float32)
             cnt[no_ball] = np.nan                           # sem bola, contagem nao existe
             out[f"n_{name}_r{r:g}"] = cnt
        press = np.exp(-dm / tau).sum(axis=1).astype(np.float32)
        press[no_ball] = np.nan
        out[f"pressure_{name}"] = press
    out["d_diff"] = (out["d_near_home"] - out["d_near_away"]).astype(np.float32)
    return out

def pitch_features(x, y, margin=1.0):
    fora = ((x < -margin) | (x > L + margin) | (y < -margin) | (y > W + margin)).astype(np.float32)
    fora[np.isnan(x) | np.isnan(y)] = np.nan
    d_touch = np.minimum(y, W - y).astype(np.float32)
    d_goal = np.minimum(x, L - x).astype(np.float32)
    return fora, d_touch, d_goal                              # out_of_pitch, d_touchline, d_goalline
```

## 5. Script C — `prep_smart_tagging.py` e `prep_homography.py` (Fernando)

| Tarefa | Descrição |
|---|---|
| C1 Leitura | XML SportsCode do kloppy (ou `starter/load_smart_tagging.py`). Comparar texto sem diferenciar maiúsculas. `round(segundos * 25)`. Os 14 códigos estão no `DATA.md`. |
| C2 Set pieces | `is_set_piece`, `sp_type`, `sp_team`, separados das fases, mais `frames_to_next_sp`, `frames_since_sp_end`, `next_sp_type` e `prev_sp_type` (**calculados por tempo de jogo**, `NaN`/vazio quando `half == 0`). Os segundos antes de lateral, falta, tiro de meta ou saída pós-gol são candidatos a bola morta (com durações distintas conforme `next_sp_type`), e `sp_team` indica quem cobra o reinício quando disponível (`H`/`A`/`U`). **Atenção ao baixo recall do XML:** há apenas 5 `free kick` e 26 `throw-in` anotados no jogo inteiro, portanto reinícios sem tag ocorrerão e a detecção de bola morta (`D`) também deve usar os sinais cinemáticos do tracking (`n_still_*`, `mean_speed_*`, `gap_length`, `out_of_pitch`). |
| C3 Fases | `n_tags_favor_home` e `n_tags_favor_away`, sem perda por sobrescrita. Ofensiva favorece o time da tag; defensiva favorece o adversário. `Team = N/A` não conta (e não é falta de dado). |
| C4 Pontapé inicial | `pre_kickoff`: frames avaliados antes do pontapé inicial de cada tempo (2º tempo: `110808..113741`, 2.934 frames D). **Atenção:** a tag `Kick Off (Start)` no XML é uma janela de ~5 s que abre ~3,5 s antes do chute (`starter/load_smart_tagging.py` e `01_smart_tagging.ipynb`); checar **continência** (`start_frame <= kickoff <= end_frame` para `11185` e `113742`) e gravar `kickoffs.json`. |
| C5 Homografia (opcional) | `reprojection_error` (salvo como `reproj_error`) e `valid` do CSV (`NaN` onde `valid == 0` ou `half == 0`). Baixa prioridade, só para depuração. |
| C6 Checagens locais | (1) taxa de sobreposição de fases; (2) se ofensiva e defensiva do mesmo time coexistem (não deveriam); (3) onde o intervalo de set piece (e de `GOALS`) começa/termina em relação à bola. |
| C7 Saída | `tagging.parquet` (+ `homography.parquet`). |

```python
import numpy as np

OFF = {"BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION", "LONG BALL",
       "BALL IN FINAL THIRD", "BALL IN THE BOX", "PLAYERS IN THE BOX", "GOALS"}
DEF = {"DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
       "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION"}
TEAM = {"barcelona": "H", "atletic club": "A"}   # 'Atletic' (um t so), como no XML; N/A -> None

def build_tag_features(ev, blocks=None, n=199_319):
    """ev: DataFrame de eventos da C1 (load_tagging_events / events_from_frame), com colunas
    start_frame, end_frame (EXCLUSIVO: fatia semiaberta [a, b)), code, team ('H'/'A'/'U'), type."""
    favor = {"H": np.zeros(n, np.int16), "A": np.zeros(n, np.int16)}
    is_sp = np.zeros(n, bool)
    sp_type = np.full(n, "", dtype=object)
    sp_team = np.full(n, "", dtype=object)
    sp_events, kicks = [], []
    for _, r in ev.iterrows():
        code = str(r["code"]).strip().upper()
        a, b = int(r["start_frame"]), int(r["end_frame"])   # [a, b) semiaberto (sem + 1)
        team = str(r["team"]).strip().upper()
        if code == "SET PIECES":                            # set pieces: NUNCA misturar com fases
            typ = str(r["type"]).strip().lower()
            valid_team = team if team in ("H", "A", "U") else "U"
            if "kick off (start)" in typ:
                kicks.append((a, b))                        # janela bruta [a, b) da tag (~5 s ao redor da batida)
            slices = (
                [(max(a, f0), min(b, f1 + 1)) for _, f0, f1 in blocks if min(b, f1 + 1) > max(a, f0)]
                if blocks is not None else ([(a, b)] if b > a else [])
            )
            for ca, cb in slices:
                is_sp[ca:cb] = True
                sp_type[ca:cb] = typ
                sp_team[ca:cb] = valid_team
                sp_events.append((ca, cb - 1, typ))         # cb - 1 = ultimo frame ativo no tempo
            continue
        if team not in ("H", "A"):
            continue                                        # 'U' (N/A): nao diz a favor de quem
        other = "A" if team == "H" else "H"
        if code in OFF:
            favor[team][a:b] += 1                           # fase ofensiva favorece o time da tag
        elif code in DEF:
            favor[other][a:b] += 1                          # fase defensiva favorece o adversario
    if blocks is not None:
        in_half = np.zeros(n, bool)
        for _, f0, f1 in blocks:
            in_half[f0:f1 + 1] = True
        favor["H"][~in_half] = 0
        favor["A"][~in_half] = 0
    sp_events.sort(key=lambda item: (item[0], item[1], item[2]))
    starts = np.asarray([s for s, _, _ in sp_events], dtype=int)
    ends = np.asarray([e for _, e, _ in sp_events], dtype=int)
    sp_types = np.asarray([t for _, _, t in sp_events], dtype=object)
    return favor["H"], favor["A"], is_sp, sp_type, sp_team, starts, ends, sp_types, sorted(kicks)

def distance_to_set_pieces(starts, ends, sp_types, blocks, n=199_319):
    """Calcula distancias e tipos adjacentes apenas dentro de cada tempo (blocks = half_video_blocks).
    Fora dos tempos (half == 0) fica NaN / string vazia; nada cruza o intervalo."""
    starts = np.asarray(starts, dtype=int)
    ends = np.asarray(ends, dtype=int)
    sp_types = np.asarray(sp_types, dtype=object)
    to_next = np.full(n, np.nan, dtype=np.float32)
    since_end = np.full(n, np.nan, dtype=np.float32)
    next_type = np.full(n, "", dtype=object)
    prev_type = np.full(n, "", dtype=object)
    for _, f0, f1 in blocks:
        f = np.arange(f0, f1 + 1)
        m_half = (starts <= f1) & (ends >= f0)
        s_half = np.maximum(starts[m_half], f0)
        e_half = np.minimum(ends[m_half], f1)
        t_half = sp_types[m_half]
        i = np.searchsorted(s_half, f, side="left")         # 1o inicio >= f no mesmo tempo
        ok = i < len(s_half)
        to_next[f[ok]] = s_half[i[ok]] - f[ok]
        next_type[f[ok]] = t_half[i[ok]]
        j = np.searchsorted(e_half, f, side="right") - 1    # ultimo fim <= f no mesmo tempo
        ok = j >= 0
        since_end[f[ok]] = f[ok] - e_half[j[ok]]
        prev_type[f[ok]] = t_half[j[ok]]
    return to_next, since_end, next_type, prev_type

def pre_kickoff_mask(kick_frames, blocks, n=199_319):
    """kick_frames: dict {1: 11185, 2: 113742} (apos validar que caem dentro das janelas (a, b) de kicks)."""
    m = np.zeros(n, bool)
    for half, f0, f1 in blocks:                                     # blocks = half_video_blocks(...)
        k = kick_frames.get(half)
        if k is not None and f0 <= k <= f1:
             m[f0:k] = True                                         # antes do pontape inicial do tempo
    return m
```
> **Nota importante sobre `Kick Off (Start)` (`starter/01_smart_tagging.ipynb`, célula 11, e `starter/load_smart_tagging.py`):** no XML, cada tag `Kick Off (Start)` marca uma **janela `[a, b]` de ~5 s** ao redor do reinício, abrindo ~3,5 s **antes** do chute real (`a < 11185 <= b` no 1º tempo e `a < 113742 <= b` no 2º tempo). Se usássemos `a` como o frame do chute, perderíamos ~88 frames `D` antes do pontapé do 2º tempo e no 1º tempo `a < f0`. Por isso, `C4` verifica a **continência** `a <= kick_frame <= b` (como faz `load_smart_tagging.check_time_base()`) e usa os frames de pontapé `11185` e `113742` em `pre_kickoff_mask` e em `kickoffs.json`.

Hipóteses a confirmar, não fatos: (a) a classificação ofensiva/defensiva dos códigos segue as descrições do `OVERVIEW.md`; `DEFENSIVE TRANSITION` foi tratada como "o time da tag acabou de perder a bola"; (b) o valor do time vem escrito `Atletic Club`, com um t só, segundo o `DATA.md`. A tag é evidência, não resposta: é automática, tem falsos positivos e negativos, e os intervalos são fases, não posses.

## 5.1. Script D e Mock — `build_master_table.py`, `make_mock_master.py` e `tests/test_master_schema.py`

| Tarefa | Descrição |
|---|---|
| D1 Junção (`build_master_table.py`) | Ler `tracking_clean.parquet` (colunas de `indice`, `bola`, `confianca`, `times`), `proximities.parquet` (`proximidade`, `velocidade_relativa`, `campo`), `tagging.parquet` (`set_pieces`, `fases`, `pontape`) e opcionalmente `homography.parquet` (`reproj_error`). Validar que todos têm `N_FRAMES` linhas (`0..N_FRAMES-1`), mesmo `half`, sem colunas duplicadas, e que nomes e dtypes seguem `preprocessing.schema.master_columns(include_optional=...)`. Salvar `data/processed/master_features.parquet` com metadado `schema_version = SCHEMA_VERSION`. |
| M1 / D2 Mock (`make_mock_master.py`) | Gerar um DataFrame sintético determinístico (`np.random.default_rng(42)`) respeitando exatamente `preprocessing/schema.py` (60 colunas, tipos `int32`, `int8`, `int16`, `bool`, `float32`, `string`, e regras de `NaN` quando `half == 0` ou sem bola) para permitir desenvolvimento e teste de modelos sem depender dos arquivos locais de `data/`. |
| M2 / D3 Testes de contrato (`tests/test_master_schema.py`) | Validar que o mock e a saída de `build_master_table` (via `--master`) batem com `MASTER_GROUPS`, `INTERMEDIATE_FILES` e todas as invariantes de `NaN` e determinismo. |

## 5.2. Diretrizes para Generalização, Validação e Determinismo na Modelagem

- **Parametrização por partida (`games.py`):** Receber `--game` (default `games.GAME`) e buscar `n_video_frames`, `fps`, `pitch` e `half_video_blocks(game)` de `games.py` (usando `params.py` apenas como fallback), garantindo que o pipeline rode sem alterações nas 10 partidas da Fase 2.
- **Zero anotação manual da partida (`RULES.md`):** É estritamente proibido anotar manualmente qualquer frame do jogo fornecido (nem para treinar, nem para validar).
- **Como validar e escolher modelos sem overfit nos 5 min públicos:**
  1. **Métrica oficial de ranking (`evaluation/run_evaluation.py` e `possession_metrics.py`):** O ranking é decidido por **`spell_f1_mean` (`spellF1`, IoU temporal 0,3..0,7 sobre posses `H`/`A`, com `X` preenchido por *forward-fill* no tempo)**, tendo **`macro_f1` (`{H, A, D}`)** apenas como **1º critério de desempate** (seguido por `transF1` em $\pm 25$ frames e latência). Não existe fórmula ponderada combinando `spellF1` e `macroF1`.
  2. **Sanidade de sinal nos 5 min públicos (*smoke test*):** Checar automaticamente que nos frames públicos `H` tem `d_diff < 0` em média e `n_tags_favor_home > n_tags_favor_away` (e o oposto para `A`). Falhas indicam bug de orientação/alinhamento (nunca usar os 5 min públicos para escolher hiperparâmetros ou modelos).
  3. **Teste de robustez por auto-supervisão (*Masked Possession Reconstruction*) + Partidas externas (ex.: *Metrica Sample Games*):** Usar a ocultação artificial da bola no jogo como teste de regressão/robustez a lacunas (cuidando da circularidade nas transições) e, após confirmação via issue com a organização, validar escolhas de arquitetura e pós-processamento temporal em partidas abertas de outros jogos (permitido pelo `RULES.md` e `FAQ.md`).
- **Segurança de dados com agentes/LLMs (`FAQ.md` e Termos):** Manter os dados reais fora do workspace aberto por agentes de IA via variável de ambiente `MTAH_DATA_DIR` e rodar scripts sobre dados reais apenas no terminal local.
- **Determinismo bit-a-bit em GBDTs (`RULES.md`):** Em scripts de modelagem (LightGBM, CatBoost, XGBoost), fixar `PYTHONHASHSEED=0`, `seed=42`, `deterministic=True`, `force_col_wise=True` e `num_threads=1` (ou número fixo com redução determinística) para que duas execuções gerem exatamente o mesmo CSV.

## 6. Testes de sanidade automatizados

| Teste | Esperado | Origem |
|---|---|---|
| Linhas e índice | 199.319 linhas; `video_frame` de 0 a 199.318 em ordem | `DATA.md` |
| Bola presente (todos / avaliados) | 56,9% (113.488) / 59,3% (94.214), no dado cru | `DATA.md` |
| Objetos rastreados (avaliados) | média ≈ 20,5; pode passar de 22 | `DATA.md` |
| Limites dos tempos | 11.185–90.651 e 110.808–190.197 | `DATA.md` |
| Orientação | flip do 2º tempo oposto ao do 1º; velocidades invertidas quando há flip | projeto |
| Nada cruza o intervalo | nenhuma interpolação ou derivada atravessa a fronteira de um tempo | projeto |
| Confiança da bola | `ball_filled` só onde `gap_length` ≤ limite; NaN em `since_last_obs` só antes da 1ª observação | projeto |
| Filtro de saltos (sintético) | fantasma isolado e bloco longo de fantasmas (`> K`) removidos; cadeias longas disjuntas (multi-bolas) mantidas | unitário |
| Derivadas e direção da bola (sintético) | `ball_vx/vy/speed/acc/dir_change` são `NaN` onde `ball_filled == 1`; suavização reduz jitter; `ball_dir_change` = 0 em reta, > 0 em curva rápida e `NaN` se `< 1,0 m/s` | unitário |
| Velocidade relativa, radial e pressão (sintético) | condução: 0 m/s; sentidos opostos a 7 m/s: 14 m/s; `speed_radial < 0` aproximando e `> 0` afastando; `pressure_*` contínua e `NaN` sem bola | unitário |
| Quebras de slot (sintético) | salto permanente: só quebra; ponto isolado: isolado | unitário |
| Tagging e set pieces (sintético) | ofensiva de H + defensiva de A = 2 a favor de H; N/A não conta; `frames_to_next_sp`, `frames_since_sp_end`, `next_sp_type` e `prev_sp_type` restritos a cada tempo | unitário |
| Master | 60 colunas (59 obrigatórias + 1 opcional), sem colunas duplicadas; intermediários com 199.319 linhas | contrato |
| Determinismo | duas execuções geram hash idêntico | `RULES.md` |

Os testes unitários usam **só dados sintéticos** (não dependem dos dados do desafio, que não estão no repositório).

## 7. Parâmetros: valores de partida

Valor único e central em `preprocessing/params.py`. Confirmar cada um na distribuição dos próprios dados de tracking (sem rótulos) ou em outros jogos. **Não vale escolher pelo desempenho nos 5 minutos públicos.**

| Parâmetro | Valor | Justificativa | Como validar |
|---|---|---|---|
| vmax da bola (`BALL_VMAX`) | 38 m/s (faixa 35–38) | limite físico de um chute muito forte | histograma de velocidades; % descartado |
| eps do filtro (`BALL_EPS`) | 1 m | ruído de posição do tracker | sensibilidade 0,5 / 1 / 2 m |
| janela K (`BALL_K`) | 30 observações | cobre lacunas sem custo quadrático | estável para K de 20 a 60 |
| lacuna máxima para interpolar (`MAX_GAP_FRAMES`) | 12 frames (0,5 s) | trecho curto, quase contínuo | comparar 6 / 12 / 25 |
| vel. mín. para direção (`MIN_BALL_SPEED_DIR`) | 1,0 m/s | evita variação angular espúria em bola quase parada | distribuição de `ball_dir_change` por velocidade |
| vmax de jogador (`PLAYER_VMAX`) | 12 m/s | acima do sprint humano | histograma; saltos de 1 frame viram `slot_break` |
| duplicata de jogadores (`DEDUP_DIST`) | < 1 m, mesmo time | dois jogadores não ocupam o mesmo ponto | frames com >22 objetos antes/depois |
| jogador parado (`STILL_SPEED`) | < 0,5 m/s | marcha lenta | histograma de velocidades |
| janela de orientação (`ORIENT_WIN`) | 25 frames (1,0 s) | jogadores ainda na própria metade após o pontapé | conferência visual com o vídeo nos 2 tempos |
| raios de contagem (`RADII`) | 2, 3 e 5 m | escala de duelo e controle | o modelo escolhe depois |
| decaimento de pressão (`PRESSURE_TAU`) | 3,0 m | escala contínua de pressão imediata (`sum(exp(-d / 3))`) | consistência com `n_*_r2/r3/r5` |
| margem de `out_of_pitch` (`PITCH_MARGIN`) | 1 m | ruído perto das linhas | coordenadas da bola perto das linhas |

## 8. Verificações locais antes de congelar o schema
Rodar localmente e levar **conclusões (não tabelas)** ao grupo.
- Tamanho das lacunas da bola, % de frames em lacunas >1 s, % de pontos removidos pelo filtro ao longo do tempo, maior bloco removido e quantas cadeias disjuntas sobraram (verificar se não há fantasma estacionário $\ge K$ antes da 1ª ou após a última observação real do tempo).
- Fração de frames observados que ficam com velocidade válida (`ball_vx/vy` não-`NaN`) após o mascaramento de `ball_filled == 1`, e distribuição de `ball_acc` (tratá-la como feature ruidosa/secundária na modelagem).
- Correlação por slot entre o `s` nativo do ATD e a velocidade derivada $\sqrt{v_x^2 + v_y^2}$, e comportamento do `s` nativo nas trocas de slot (`slot_break`).
- O time de cada slot é constante? Há objetos que não são jogadores de linha? Com que frequência o jogador mais recuado (goleiro) some do tracking?
- Objetos por time por frame; frames com >22 objetos antes e depois da deduplicação.
- Sentido de ataque em cada tempo: a inferência bate com o vídeo?
- Smart Tagging: sobreposição de fases, coexistência de ofensiva/defensiva do mesmo time, valores exatos de `Team`.
- Set pieces: distribuição empírica das durações de bola parada por `sp_type` (não usar durações fixas sem medir nos dados); onde o intervalo começa/termina em relação à bola; o que fazem os jogadores no segmento pré-pontapé do 2º tempo.
- Frame do `Kick Off (Start)` de cada tempo e se a janela da tag contém 11.185 e 113.742.

### Registro da execução local (31/10/2026)
Com os assets táticos instalados localmente, o ATD completo produziu 199.319 linhas em
`data/processed/tracking_clean.parquet`; os arquivos processados são locais e ignorados pelo
Git. A contagem de frames com mais de 22 objetos, antes → depois da deduplicação, foi 695 →
586 no 1º tempo e 3.054 → 2.735 no 2º tempo. A tabela contém `flip=0` no 1º e `flip=1` no
2º tempo, conforme esperado.

O campo escalar `s` nativo do ATD foi comparado com a norma da velocidade derivada nos
2.975.960 pares finitos: correlação agregada 0,967 e mediana da diferença absoluta 0,109 m/s.
Nos 1.665 cortes de slot com `s` finito nos dois frames adjacentes, a mediana de `s` foi
1,25 m/s antes e no frame do corte; o percentil 95 no frame do corte foi 3,91 m/s e nenhum
valor excedeu 12 m/s. Isso indica que `s` pode permanecer normal durante os cortes espaciais,
portanto não deve ser usado sozinho para rejeitá-los.

Validação: `tests/test_prep_players.py` (8 testes) e suíte completa (75 testes) passaram em
31/10/2026. O teste sintético da contagem foi corrigido para refletir que quatro outros frames
continuam acima de 22 objetos; o frame com o ponto isolado cai para exatamente 22.

Pendente de confirmação externa: a dependência A3 (#10), o estado do PR e o compartilhamento
das conclusões com o grupo não puderam ser consultados nesta execução.

## 9. Cronograma
| Quando | Marco |
|---|---|
| até o fim de 27/10 | schema congelado (`SCHEMA.md`); V0 do `tracking_clean.parquet`; mock e testes de schema |
| 28/10 | versões completas dos três scripts; `build_master_table.py` rodando; master V0 real |
| 29/10 a 03/11 | refinos (Kalman/RTS, duplicatas, parâmetros), determinismo, comando único de pipeline, primeira submissão válida |
