# SCHEMA — contrato dos arquivos de pré-processamento

**Versão do schema: `1.0` (congelado).** Mudanças só por issue aprovada pelo grupo, com atualização deste arquivo, de `preprocessing/schema.py` e do mock no mesmo PR.

Especificação completa: [`docs/preprocessamento_v2.md`](docs/preprocessamento_v2.md).

## Convenções (valem para todos os arquivos)

- **Formato:** Parquet (`pyarrow`), em `data/processed/` (ignorado pelo git; os dados do desafio **nunca** vão ao repositório).
- **Uma linha por frame de vídeo:** 199.319 linhas, `video_frame` = 0..199.318 em ordem (`video_frame = frame_id - 1` no ATD).
- **Índice:** todo arquivo intermediário tem `video_frame` (`int32`) e `half` (`int8`). No master, `half` aparece uma única vez.
- **`half`:** `1` ou `2` dentro dos blocos de `half_video_blocks('BAR-ATH')`; **`0` fora deles** (antes do jogo, intervalo, depois do apito). Fora dos blocos, **todas as colunas derivadas do tracking ficam NaN** (flags `0`), para nada cruzar o intervalo.
- **Tipos:** `float32` para medidas; `int8`/`int16` para flags e contagens sem NaN; `bool` para booleanos puros. Coluna que pode ter NaN é `float32` (inclusive flags 0/1/NaN e contagens).
- **NaN = "sem evidência"**, nunca zero. O significado do NaN de cada coluna está nas tabelas abaixo.
- **Metadados do Parquet:** `schema_version = "1.0"`. O `tracking_clean.parquet` também grava `tracking_stage = "V0"` ou `"V1"`.
- **Nomes:** `snake_case`; `_home`/`_away` para times (home = Barcelona, away = Athletic); `_r2/_r3/_r5` para raios; `_x/_y` para coordenadas.
- **Unidades:** metros, m/s, m/s², frames (25 fps), pixels (só `reproj_error`).
- **Orientação:** após `flip` por tempo, o campo vai de `x = 0` a `x = 105,3` e `y = 0` a `68,0`. **A casa sempre ataca para x crescente; o visitante, para x decrescente.** Velocidades e acelerações são invertidas junto com as posições.
- **Parâmetros:** todos em `preprocessing/params.py`. Nenhum é ajustado nos 5 minutos públicos.
- **Determinismo:** mesma entrada, mesma saída (seeds fixas, versões pinadas).

## Arquivos

| Arquivo | Dono | Conteúdo |
|---|---|---|
| `tracking_clean.parquet` | A (Antônio) | índice, `flip`, bola, confiança, agregados e colunas por slot (V0, depois V1) |
| `slot_teams.parquet` | A (Antônio) | uma linha por slot: time e se é jogador de linha |
| `proximities.parquet` | B (Hugo) | proximidade, velocidade relativa e campo |
| `tagging.parquet` | C (Fernando) | set pieces, fases, pré-pontapé |
| `kickoffs.json` | C (Fernando) | frame do `Kick Off (Start)` de cada tempo |
| `homography.parquet` | C (Fernando, opcional) | `reproj_error` |
| `master_features.parquet` | D (Fernando) | junção de tudo, só as colunas da seção "Master" |

## Master — colunas

### Índice (D, `build_master_table.py`)
| Coluna | Tipo | Significado |
|---|---|---|
| `video_frame` | int32 | 0..199.318 |
| `half` | int8 | 1, 2 ou 0 (fora dos tempos) |
| `is_graded` | bool | frame dentro dos blocos de `half_video_blocks` |
| `frame_in_half` | float32 | frames desde o 1º frame do bloco do tempo; NaN se `half == 0` |

### Bola (A)
| Coluna | Tipo | Unidade | NaN significa |
|---|---|---|---|
| `ball_x`, `ball_y` | float32 | m, orientados | bola sem evidência (não observada e não preenchida) |
| `ball_vx`, `ball_vy` | float32 | m/s, orientados | sem observação real vizinha suficiente (ou `ball_filled == 1`) |
| `ball_speed` | float32 | m/s | idem |
| `ball_acc` | float32 | m/s² (norma do vetor) | idem |
| `ball_dir_change` | float32 | rad (`0..π`) | variação angular entre `v[t-1]` e `v[t]`; NaN se `ball_speed < MIN_BALL_SPEED_DIR` (1,0 m/s) em `t-1` ou `t` |

`ball_x/ball_y` incluem as lacunas curtas interpoladas (NaN **sse** `ball_observed == 0` e `ball_filled == 0`), sem suavização da posição armazenada. As derivadas (`ball_vx`, `ball_vy`, `ball_speed`, `ball_acc`) são calculadas com suavização zero-fase apenas nos trechos observados (`ball_observed == 1`) e ficam `NaN` onde `ball_filled == 1`.

### Confiança da bola (A) — calculada **depois** do filtro de saltos
| Coluna | Tipo | Significado |
|---|---|---|
| `ball_observed` | int8 | 1 = posição do ATD mantida após o filtro de saltos |
| `ball_jump_removed` | int8 | 1 = o ATD trouxe posição, mas o filtro a removeu |
| `ball_filled` | int8 | 1 = posição interpolada (lacuna ≤ `MAX_GAP_FRAMES`) |
| `gap_length` | float32 | frames do buraco de bola que contém este frame; 0 se observada; NaN se `half == 0` |
| `since_last_obs` | float32 | frames desde a última observação (0 se observada); NaN **só** antes da 1ª observação do tempo |
| `until_next_obs` | float32 | frames até a próxima observação (0 se observada); NaN **só** depois da última do tempo |

### Times (A) — só jogadores de linha, com duplicatas removidas
| Coluna | Tipo | Unidade | Observação |
|---|---|---|---|
| `centroid_home_x`, `centroid_home_y`, `centroid_away_x`, `centroid_away_y` | float32 | m | NaN sem jogadores rastreados |
| `spread_home_x`, `spread_home_y`, `spread_away_x`, `spread_away_y` | float32 | m | desvio-padrão das posições do time |
| `mean_speed_home`, `mean_speed_away` | float32 | m/s | média das velocidades vetoriais |
| `n_still_home`, `n_still_away` | float32 | jogadores | com velocidade < `STILL_SPEED` |
| `n_tracked_home`, `n_tracked_away` | float32 | jogadores | slots válidos e únicos no frame |

### Proximidade (B)
Usa `ball_x/ball_y` (observadas + preenchidas). Slots NaN são ignorados.
| Coluna | Tipo | Unidade | NaN significa |
|---|---|---|---|
| `d_near_home`, `d_near_away` | float32 | m | sem bola ou sem jogadores do time |
| `d_diff` | float32 | m | `d_near_home - d_near_away` (negativo = casa mais perto) |
| `n_home_r2`, `n_home_r3`, `n_home_r5`, `n_away_r2`, `n_away_r3`, `n_away_r5` | float32 | jogadores | **sem bola** (nunca zero) |
| `pressure_home`, `pressure_away` | float32 | adimensional | `sum(exp(-d / PRESSURE_TAU))` com `PRESSURE_TAU = 3,0 m`; NaN **sem bola** |

### Velocidade relativa (B)
| Coluna | Tipo | Unidade | Observação |
|---|---|---|---|
| `speed_rel_home`, `speed_rel_away` | float32 | m/s | norma da **diferença vetorial** bola − jogador mais próximo do time; NaN sem velocidade observada da bola (ex.: sem bola ou `ball_filled == 1`) ou sem velocidade do jogador |
| `speed_radial_home`, `speed_radial_away` | float32 | m/s | componente radial de afastamento/aproximação `(b - p) · (vb - vp) / max(d, 1e-3)` (negativo = aproximando, positivo = afastando); NaN nas mesmas condições de `speed_rel_*` |

### Campo (B)
| Coluna | Tipo | Unidade | Observação |
|---|---|---|---|
| `out_of_pitch` | float32 | 0 / 1 / NaN | 1 = bola fora do campo + `PITCH_MARGIN`; NaN sem bola |
| `d_touchline` | float32 | m | `min(y, 68 − y)` (negativo = fora) |
| `d_goalline` | float32 | m | `min(x, 105,3 − x)` (negativo = fora) |

### Set pieces (C)
| Coluna | Tipo | Unidade | NaN significa |
|---|---|---|---|
| `is_set_piece` | bool | | — |
| `sp_type` | string | minúsculas, vazio se não há set piece | tipos do XML: `throw-in`, `corner kick`, `goal kick`, `free kick`, `kick off (after goal)`, `kick off (start)` |
| `sp_team` | string | `'H'`, `'A'`, `'U'` (`N/A`) ou vazio se não há set piece | time da bola parada no XML (`r["team"]` da `C1`) |
| `frames_to_next_sp` | float32 | frames | não há set piece depois **neste tempo** (ou `half == 0`) |
| `frames_since_sp_end` | float32 | frames | não houve set piece antes **neste tempo** (ou `half == 0`) |
| `next_sp_type` | string | minúsculas, vazio se `frames_to_next_sp` é NaN | tipo do próximo set piece neste tempo (distingue duração de parada antes de gol, tiro de meta, lateral etc.) |
| `prev_sp_type` | string | minúsculas, vazio se `frames_since_sp_end` é NaN | tipo do último set piece encerrado neste tempo |

### Fases (C)
| Coluna | Tipo | Significado |
|---|---|---|
| `n_tags_favor_home`, `n_tags_favor_away` | int16 | nº de fases ativas a favor do time (ofensiva do time da tag + defensiva do adversário). `Team = N/A` não conta |

### Pontapé (C)
| Coluna | Tipo | Significado |
|---|---|---|
| `pre_kickoff` | bool | frame avaliado antes do `Kick Off (Start)` do tempo (no 2º tempo: 2.934 frames, todos `D`) |

### Homografia (C, opcional)
| Coluna | Tipo | Unidade | NaN significa |
|---|---|---|---|
| `reproj_error` | float32 | px | `valid == 0` |

## Arquivos auxiliares

### `slot_teams.parquet` (A7)
| Coluna | Tipo | Significado |
|---|---|---|
| `slot` | int8 | 1..46 |
| `is_home` | bool | time do slot (Barcelona = True) |
| `is_field_player` | bool | True para jogadores dos dois clubes (inclui goleiros; o XML atribui todos os 46 slots a Barcelona ou Athletic); False apenas se um slot for identificado como não-jogador (ex.: árbitro) |

Se A7 mostrar que o time de um slot **não é constante**, a A7 avisa o grupo antes de seguir; o plano alternativo é uma coluna `s{NN}_is_home` por frame no `tracking_clean.parquet`, com `slot_teams.parquet` só para os slots constantes.

### `kickoffs.json` (C4)
```json
{"1": 11185, "2": 113742, "tag_windows": {"1": [start_1, end_1], "2": [start_2, end_2]}}
```
> **Atenção (`starter/01_smart_tagging.ipynb`, célula 11, e `starter/load_smart_tagging.py`):** no XML do Smart Tagging, a tag `Kick Off (Start)` marca uma **janela de ~5 s ao redor do pontapé** (abre ~3,5 s antes da batida), e **não** o frame exato do chute. Portanto, o teste correto sobre o XML é de **continência** (`start_frame <= kickoff <= end_frame`, como em `load_smart_tagging.check_time_base()`), e não de igualdade (`start_frame == kickoff`). Os frames exatos do pontapé inicial são `11185` (1º tempo) e `113742` (2º tempo, após os 2.934 frames `D` de `110808..113741`), publicados em `DATA.md`, `FAQ.md` e `starter/load_smart_tagging.py` (ou refinados dentro da janela da tag pelo início do movimento da bola no tracking).

### Colunas por slot em `tracking_clean.parquet`
`s01_x`, `s01_y`, `s01_vx`, `s01_vy`, `s01_break` … `s46_*` (`NN` com dois dígitos; metros e m/s, orientados). Mais: `flip` (int8: 1 se o campo foi refletido naquele tempo), `ball_x_raw`, `ball_y_raw` (posição original em metros, antes de qualquer filtro). Essas colunas **não** entram no master.
- **Pontos isolados e duplicatas (A5):** quando um slot tem ponto isolado ou é duplicata de outro slot do mesmo time no mesmo frame (distância < `DEDUP_DIST` = 1 m), sua posição `(sNN_x, sNN_y, sNN_vx, sNN_vy)` naquele frame já vira `NaN` no `tracking_clean.parquet` (V1). Assim, tanto os agregados de A6 quanto as proximidades e contagens de B2–B4 ficam livres de duplicatas automaticamente.

## Decisões fechadas (reunião de 27/10)

1. **Nomes das colunas:** os deste arquivo.
2. **`build_master_table.py`:** Fernando (issue [D1]).
3. **Raios e vmax de partida:** 2/3/5 m; bola 38 m/s; jogador 12 m/s (`params.py`; validar na distribuição dos dados, não nos 5 min públicos).
4. **`is_home`:** em `slot_teams.parquet`, fora do master.
5. **Formato:** Parquet; `schema_version = "1.0"`.
6. **Frame do pontapé:** `kickoffs.json` (C → A). A V0 usa 11.185 e 113.742 (valores publicados) com aviso; a V1 usa o `kickoffs.json` e confere com esses valores.
