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
| A4 Bola | Filtro de saltos (maior cadeia consistente) → colunas de confiança depois do filtro → interpolação só de lacunas curtas → suavização offline por trechos contínuos (Kalman + RTS ou Savitzky-Golay) → `ball_vx, ball_vy, ball_speed, ball_acc`. |
| A5 Jogadores | `slot_break` nos saltos de 1 frame; apagar a posição de pontos isolados. Velocidade vetorial por derivada da posição suavizada, por trechos entre quebras. Remover duplicatas (mesmo time a <1 m): zerar como `NaN` o slot duplicado no frame no `tracking_clean.parquet` (V1), para que A6 e B2–B4 já fiquem sem duplicatas. |
| A6 Agregados por time | `centroid_*`, `spread_*`, `mean_speed_*`, `n_still_*`, `n_tracked_*`, robustos à troca de slots. |
| A7 Time de cada slot | Verificar o mapeamento no XML (`starter/baseline.py: team_of_each_track`) e se há objetos que não são jogadores de linha. Exportar `slot_teams.parquet`. |
| A8 Saída | `tracking_clean.parquet` (V0, depois V1). |

### Filtro de saltos da bola (maior cadeia consistente)
```python
import numpy as np

def remove_ball_jump_outliers(x, y, vmax=38.0, fps=25.0, eps=1.0, K=30):
    """x, y em metros (NaN onde falta). Mantem a MAIOR cadeia de observacoes
    mutuamente consistentes (|dr| <= vmax*dt + eps); o resto vira NaN."""
    obs = np.flatnonzero(~np.isnan(x) & ~np.isnan(y))
    m = len(obs)
    xo, yo = x.copy(), y.copy()
    if m < 2:
        return xo, yo
    best = np.ones(m, int)               # tamanho da melhor cadeia terminando em i
    prev = np.full(m, -1)
    for i in range(1, m):
        js = np.arange(max(0, i - K), i) # so olha as K observacoes anteriores
        dt = (obs[i] - obs[js]) / fps
        d = np.hypot(x[obs[i]] - x[obs[js]], y[obs[i]] - y[obs[js]])
        ok = js[d <= vmax * dt + eps]
        if len(ok):
            j = ok[np.argmax(best[ok])]
            best[i], prev[i] = best[j] + 1, j
    keep = np.zeros(len(x), bool)
    i = int(np.argmax(best))
    while i != -1:                        # reconstroi a cadeia vencedora
        keep[obs[i]] = True
        i = prev[i]
    xo[~keep] = np.nan
    yo[~keep] = np.nan
    return xo, yo
```
Por que assim: um fantasma no meio de uma lacuna é compatível com o ponto anterior (há muito tempo para chegar lá), mas incompatível com a bola real que reaparece dois frames depois. A cadeia mais longa fica com a trajetória real. A janela de K observações mantém o custo em O(n·K). Conferir visualmente alguns trechos (permitido) e pelo percentual descartado, sem criar regras para frames específicos.

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
| B3 Contagens por raio | `n_*_r2`, `n_*_r3`, `n_*_r5`: jogadores de cada time dentro de 2, 3 e 5 m. NaN quando não há bola, e não zero. |
| B4 Velocidade relativa | `speed_rel_home`, `speed_rel_away`: norma da **diferença vetorial** entre a bola e o jogador mais próximo de cada time (a diferença de módulos dá zero quando cruzam em sentidos opostos). |
| B5 Campo | `out_of_pitch` (com margem), `d_touchline`, `d_goalline`. A orientação do campo não muda essas medidas. |
| B6 Saída | `proximities.parquet` com `video_frame` e `half`, seguindo o contrato. |

```python
import numpy as np

L, W = 105.3, 68.0

def masked_nearest(d, mask):
    """d: (T, N) distancias (NaN onde falta); mask: (N,) bool. -> (indice, distancia)."""
    dm = np.where(mask[None, :] & ~np.isnan(d), d, np.inf)
    j = dm.argmin(axis=1)
    dist = dm[np.arange(len(d)), j]
    return j, np.where(np.isfinite(dist), dist, np.nan)

def proximity_features(bx, by, bvx, bvy, px, py, pvx, pvy, is_home, radii=(2.0, 3.0, 5.0)):
    d = np.hypot(px - bx[:, None], py - by[:, None])
    T = np.arange(len(bx))
    no_ball = np.isnan(bx) | np.isnan(by)
    out = {}
    for name, mask in (("home", is_home), ("away", ~is_home)):
        j, dist = masked_nearest(d, mask)
        rel = np.hypot(bvx - pvx[T, j], bvy - pvy[T, j])     # diferenca VETORIAL
        out[f"d_near_{name}"] = dist
        out[f"speed_rel_{name}"] = np.where(np.isnan(dist), np.nan, rel)
        dm = np.where(mask[None, :], d, np.nan)
        for r in radii:
             cnt = (dm < r).sum(axis=1).astype(float)
             cnt[no_ball] = np.nan                           # sem bola, contagem nao existe
             out[f"n_{name}_r{r:g}"] = cnt
    out["d_diff"] = out["d_near_home"] - out["d_near_away"]
    return out

def pitch_features(x, y, margin=1.0):
    fora = ((x < -margin) | (x > L + margin) | (y < -margin) | (y > W + margin)).astype(float)
    fora[np.isnan(x) | np.isnan(y)] = np.nan
    return fora, np.minimum(y, W - y), np.minimum(x, L - x)   # fora, d_lateral, d_fundo
```

## 5. Script C — `prep_smart_tagging.py` e `prep_homography.py` (Fernando)

| Tarefa | Descrição |
|---|---|
| C1 Leitura | XML SportsCode do kloppy (ou `starter/load_smart_tagging.py`). Comparar texto sem diferenciar maiúsculas. `round(segundos * 25)`. Os 14 códigos estão no `DATA.md`. |
| C2 Set pieces | `is_set_piece`, `sp_type`, `sp_team`, separados das fases, mais `frames_to_next_sp` e `frames_since_sp_end` (**calculados por tempo de jogo**, `NaN` quando `half == 0`). Os segundos antes de lateral, falta ou tiro de meta são candidatos a bola morta, e `sp_team` indica quem cobra o reinício quando disponível (`H`/`A`/`U`). |
| C3 Fases | `n_tags_favor_home` e `n_tags_favor_away`, sem perda por sobrescrita. Ofensiva favorece o time da tag; defensiva favorece o adversário. `Team = N/A` não conta (e não é falta de dado). |
| C4 Pontapé inicial | `pre_kickoff`: frames avaliados antes do pontapé inicial de cada tempo (2º tempo: `110808..113741`, 2.934 frames D). **Atenção:** a tag `Kick Off (Start)` no XML é uma janela de ~5 s que abre ~3,5 s antes do chute (`starter/load_smart_tagging.py` e `01_smart_tagging.ipynb`); checar **continência** (`start_frame <= kickoff <= end_frame` para `11185` e `113742`) e gravar `kickoffs.json`. |
| C5 Homografia (opcional) | `reproj_error` e `valid` do CSV. Baixa prioridade, só para depuração. |
| C6 Checagens locais | (1) taxa de sobreposição de fases; (2) se ofensiva e defensiva do mesmo time coexistem (não deveriam); (3) onde o intervalo de set piece (e de `GOALS`) começa/termina em relação à bola. |
| C7 Saída | `tagging.parquet` (+ `homography.parquet`). |

```python
import numpy as np

OFF = {"BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION", "LONG BALL",
       "BALL IN FINAL THIRD", "BALL IN THE BOX", "PLAYERS IN THE BOX", "GOALS"}
DEF = {"DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
       "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION"}
TEAM = {"barcelona": "H", "atletic club": "A"}   # 'Atletic' (um t so), como no XML; N/A -> None

def build_tag_features(ev, n=199_319):
    """ev: DataFrame de eventos da C1 (load_tagging_events / events_from_frame), com colunas
    start_frame, end_frame (EXCLUSIVO: fatia semiaberta [a, b)), code, team ('H'/'A'/'U'), type."""
    favor = {"H": np.zeros(n, np.int16), "A": np.zeros(n, np.int16)}
    is_sp = np.zeros(n, bool)
    sp_type = np.full(n, "", dtype=object)
    sp_team = np.full(n, "", dtype=object)
    starts, ends, kicks = [], [], []
    for _, r in ev.iterrows():
        code = str(r["code"]).strip().upper()
        a, b = int(r["start_frame"]), int(r["end_frame"])   # [a, b) semiaberto (sem + 1)
        team = str(r["team"]).strip().upper()
        if code == "SET PIECES":                            # set pieces: NUNCA misturar com fases
            typ = str(r["type"]).strip().lower()
            is_sp[a:b] = True
            sp_type[a:b] = typ
            sp_team[a:b] = team if team in ("H", "A", "U") else "U"
            if b > a:
                starts.append(a); ends.append(b - 1)        # b - 1 = ultimo frame ativo do set piece
            if "kick off (start)" in typ:
                kicks.append((a, b))                        # janela [a, b) da tag (~5 s ao redor da batida)
            continue
        if team not in ("H", "A"):
            continue                                        # 'U' (N/A): nao diz a favor de quem
        other = "A" if team == "H" else "H"
        if code in OFF:
            favor[team][a:b] += 1                           # fase ofensiva favorece o time da tag
        elif code in DEF:
            favor[other][a:b] += 1                          # fase defensiva favorece o adversario
    return favor["H"], favor["A"], is_sp, sp_type, sp_team, np.sort(starts), np.sort(ends), sorted(kicks)

def distance_to_set_pieces(starts, ends, blocks, n=199_319):
    """Calcula distancias apenas dentro de cada tempo (blocks = half_video_blocks).
    Fora dos tempos (half == 0) fica NaN; nada cruza o intervalo."""
    starts = np.asarray(starts, dtype=int)
    ends = np.asarray(ends, dtype=int)
    to_next = np.full(n, np.nan, dtype=np.float32)
    since_end = np.full(n, np.nan, dtype=np.float32)
    for _, f0, f1 in blocks:
        f = np.arange(f0, f1 + 1)
        s_half = starts[(starts >= f0) & (starts <= f1)]
        e_half = ends[(ends >= f0) & (ends <= f1)]
        i = np.searchsorted(s_half, f, side="left")         # 1o inicio >= f no mesmo tempo
        ok = i < len(s_half)
        to_next[f[ok]] = s_half[i[ok]] - f[ok]
        j = np.searchsorted(e_half, f, side="right") - 1    # ultimo fim <= f no mesmo tempo
        ok = j >= 0
        since_end[f[ok]] = f[ok] - e_half[j[ok]]
    return to_next, since_end

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
| Filtro de saltos (sintético) | ponto fantasma no fim de uma lacuna removido, trajetória real mantida | unitário |
| Velocidade relativa (sintético) | condução: 0 m/s; sentidos opostos a 7 m/s: 14 m/s | unitário |
| Quebras de slot (sintético) | salto permanente: só quebra; ponto isolado: isolado | unitário |
| Tagging (sintético) | ofensiva de H + defensiva de A = 2 a favor de H; N/A não conta | unitário |
| Master | sem colunas duplicadas; intermediários com 199.319 linhas | contrato |
| Determinismo | duas execuções geram hash idêntico | `RULES.md` |

Os testes unitários usam **só dados sintéticos** (não dependem dos dados do desafio, que não estão no repositório).

## 7. Parâmetros: valores de partida

Valor único e central em `preprocessing/params.py`. Confirmar cada um na distribuição dos próprios dados de tracking (sem rótulos) ou em outros jogos. **Não vale escolher pelo desempenho nos 5 minutos públicos.**

| Parâmetro | Valor | Justificativa | Como validar |
|---|---|---|---|
| vmax da bola | 38 m/s (faixa 35–38) | limite físico de um chute muito forte | histograma de velocidades; % descartado |
| eps do filtro | 1 m | ruído de posição do tracker | sensibilidade 0,5 / 1 / 2 m |
| janela K | 30 observações | cobre lacunas sem custo quadrático | estável para K de 20 a 60 |
| vmax de jogador | 12 m/s | acima do sprint humano | histograma; saltos de 1 frame viram `slot_break` |
| lacuna máxima para interpolar | 12 frames (0,5 s) | trecho curto, quase contínuo | comparar 6 / 12 / 25 |
| margem de `out_of_pitch` | 1 m | ruído perto das linhas | coordenadas da bola perto das linhas |
| raios de contagem | 2, 3 e 5 m | escala de duelo e controle | o modelo escolhe depois |
| duplicata de jogadores | < 1 m, mesmo time | dois jogadores não ocupam o mesmo ponto | frames com >22 objetos antes/depois |
| jogador parado (`n_still`) | < 0,5 m/s | marcha lenta | histograma de velocidades |

## 8. Verificações locais antes de congelar o schema
Rodar localmente e levar **conclusões (não tabelas)** ao grupo.
- Tamanho das lacunas da bola, % de frames em lacunas >1 s, % de pontos removidos pelo filtro.
- Nas trocas de slot, o `s` nativo do ATD fica normal ou também dispara?
- O time de cada slot é constante? Há objetos que não são jogadores de linha?
- Objetos por time por frame; frames com >22 objetos.
- Sentido de ataque em cada tempo: a inferência bate com o vídeo?
- Smart Tagging: sobreposição de fases, coexistência de ofensiva/defensiva do mesmo time, valores exatos de `Team`.
- Set pieces: o intervalo começa quando a bola sai ou na cobrança? O que fazem os jogadores no segmento pré-pontapé do 2º tempo?
- Frame do `Kick Off (Start)` de cada tempo e se bate com 11.185 e 113.742.

## 9. Cronograma
| Quando | Marco |
|---|---|
| até o fim de 27/10 | schema congelado (`SCHEMA.md`); V0 do `tracking_clean.parquet`; mock e testes de schema |
| 28/10 | versões completas dos três scripts; `build_master_table.py` rodando; master V0 real |
| 29/10 a 03/11 | refinos (Kalman/RTS, duplicatas, parâmetros), determinismo, comando único de pipeline, primeira submissão válida |
