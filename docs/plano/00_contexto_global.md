# Contexto global do plano de implementação

Extraído do [plano completo](plano_5_frentes.md): Seções 1, 2, 3, 6 e 7 e Apêndices A–E. As tarefas estão em [`docs/tasks/`](../tasks/), o mapeamento das issues em [`issues.md`](issues.md) e os contratos em [`docs/contracts.md`](../contracts.md). Em caso de divergência, vale o plano completo.

**Responsáveis** (definidos em 06/11/2026):

| Frente | Responsável | GitHub |
| :--- | :--- | :--- |
| **F1** Evidência de posse e avaliação | Fernando | `fernando-horita-siratuti` |
| **F2** Decodificação temporal e seleção | Hugo | `hugnarok` |
| **F3** Bola morta e documentação | Bernardo | `BernardoSilvaAndrade` |
| **F4** Dados externos (IDSSE) e trilha da transmissão | Antônio | `AntonioEscobarTorres` |
| **F5** Infraestrutura, integração e entrega (responsável pelo envio) | Ahmed | `HachemAhmed` |

O gate G1 (code freeze) é da F5.

---

## 1. Contexto fixo (toda tarefa herda isto)

### 1.1 O problema e a métrica
- Partida: **FC Barcelona (H, casa) × Athletic Club (A, visitante)**, LaLiga 2024-25, rodada 2.
- Saída: CSV `frame,label` com **199.319 linhas** (frames 0…199318 do vídeo tático, 25 fps). Rótulos: `H`, `A`, `D` (bola morta) e `X` (abstenção).
- **Frames avaliados:** 1º tempo `11185…90651` e 2º tempo `110808…190197`. Ambos vêm de `games.half_video_blocks("BAR-ATH")`, com **limites inclusivos**. Fora deles o rótulo é `X` e nunca é pontuado.
- **Dentro dos blocos avaliados, nunca usar `X`.** O spellF1 trata `X` como "repetir o rótulo anterior" e o macroF1 conta `X` como erro, então não há ganho em usá-lo.
- **2º tempo:** os frames `110808…113741` (2.934) são todos `D` (pré-pontapé, publicado no FAQ). O 1º tempo começa no próprio pontapé, com posse imediata.
- **Ranking:** `spellF1`. Posses (sequências máximas de `H` ou de `A`) são pareadas 1-para-1 com a verdade por IoU temporal, e o F1 é a média nos limiares 0,3…0,7. Desempates, nesta ordem: `macroF1`, F1 de transições (±25 frames) e latência média.
- Um `D` divide posses: `H H D D H H` são **duas** posses. Um `D` falso no meio de uma posse custa duas vezes: divide a posse real e cria uma falsa.
- **Sensibilidade a atraso** (EVALUATION.md): 1 frame não custa nada; 3 frames → 0,98; 8 frames → 0,89; 1 s → 0,65. Nada de filtros causais: tudo offline e bidirecional.
- **Regras de posse** (CHALLENGE.md): troca de posse é o outro time ganhar posse *controlada*. Toque sem controle, desvio e duelo aéreo sem controle posterior não trocam a posse. Em bola solta, a posse fica com o último time que a teve.
  - **Cuidado:** o GT é derivado mecanicamente do tracking corrigido. Na janela pública ele contém posses de 0,76 s e 1,32 s; a issue #17 questiona uma delas e não tem resposta pública. Não presuma que o GT ignora toques curtos.
- Diferenças de spellF1 menores que ~0,04 são ruído no jogo inteiro.

### 1.2 Regras e decisões da organização (vinculantes)
- **Rótulos têm de vir de algoritmo.** É proibido anotar o BAR-ATH à mão para qualquer fim: submissão, treino, validação ou escolha de parâmetros. Na issue #10 os organizadores vetaram escolher parâmetros com rótulos do BAR-ATH, mesmo que os rótulos nunca entrem no pipeline.
- **Janela pública** (`11185…18684`, 17 posses): serve só para checar se o pipeline está correto (alinhamento, formato, sentido dos rótulos). É **proibido** treinar, ajustar ou escolher entre variantes com ela.
- **Dados externos:** só de **outras partidas**, acessíveis e documentados.
  - O **IDSSE** foi aprovado explicitamente na issue #16 (licença CC BY 4.0). É obrigatório citá-lo no README e no relatório.
  - Recado dos organizadores: "todo hiperparâmetro e limiar escolhido no IDSSE ou em outros jogos".
- **Treinar no próprio BAR-ATH** usando o tracking fornecido como rótulo foi vetado na issue #2, embora o RULES.md (atualizado em 23/10) permita pseudo-rótulos algorítmicos. **Decisão deste plano: nada é treinado no BAR-ATH. Toda calibração usa o IDSSE.**
  - **Confirmado em 13/11** por e-mail dos organizadores: treino e ajuste de parâmetros só em dados externos; nem ajuste não supervisionado no BAR-ATH. Conferir a saída contra o vídeo também é proibido (D-012).
- **Reprodutibilidade** (issue #11): o CSV tem de sair **byte a byte igual** ao rodar o repositório, com os pesos commitados, sobre os dados fornecidos.
  - Inferência na CPU, versões fixadas, sementes fixas.
  - O treino não precisa ser bit-exato, mas o código de treino tem de estar publicado.
- **Permitido:** uso offline/bidirecional (issue #15). Transformar observações gerais do vídeo em regras também é permitido; regras ou constantes atreladas a frames específicos, não (issue #14).
- **Nenhum dado da partida pode ir para LLM em nuvem** (Termos, FAQ.md). Isso inclui trechos, tabelas, valores impressos e capturas de tela.
- O **repositório será público** na entrega e não pode conter dados da partida nem derivados quadro a quadro. Issues e PRs também ficam públicos.
- **Entrega** (Google Form, prazo 17/11/2026): `<team_name>.csv`, URL do repositório público e `<team_name>.pdf` (ou `.md`).
  - Uma resposta por equipe; vale a última enviada antes do prazo. Não há leaderboard nem feedback.
  - Validar o arquivo exato com `python evaluation/validate_submission.py <arquivo>`.
- Desde 03/11 os organizadores só respondem por e-mail: hackathon.talentarena@mobileworldcapital.com.

### 1.3 Estado do repositório (pré-processamento concluído e revisado)
- `python -m preprocessing.run_all` leva ≈25 s e ≈1,3 GB de RAM. Ele grava em `<data dir>/processed/` os arquivos `tagging.parquet`, `kickoffs.json`, `homography.parquet`, `tracking_clean.parquet`, `slot_teams.parquet`, `proximities.parquet` e `master_features.parquet`.
- **Determinismo verificado:** regenerados do zero num venv limpo, os 7 arquivos saem idênticos (SHA-256).
- `<data dir>` = `$MTAH_DATA_DIR` se a variável estiver definida; senão, `./data`. Os dados da equipe ficam **fora do repositório** (`~/mtah-data`).
- **Revisão de 04/11:** 152 testes passam, e o master real passa nos 17 testes de contrato. Mapeamento H/A, orientação nos dois tempos e sinais de bola morta foram conferidos no dado real.
  - Pontos conhecidos, deliberadamente não alterados: dois casos de borda do filtro de saltos da bola (a F3-07 mede a frequência) e a deduplicação a 1 m entre companheiros (efeito desprezível na posse).
- **Tabela mestra:** 199.319 linhas × 60 colunas (`SCHEMA.md` v1.0, `preprocessing/schema.py`). Colunas e significado no Apêndice C.
- **Convenções da tabela mestra:**
  - H ataca para x crescente nos dois tempos e A para x decrescente. Campo de 105,3 × 68,0 m.
  - `d_diff = d_near_home − d_near_away`: negativo significa casa mais perto da bola.
  - `sp_type`, `next_sp_type` e `prev_sp_type` vêm em **minúsculas**: `throw-in`, `corner kick`, `goal kick`, `free kick`, `kick off (after goal)`, `kick off (start)`. String vazia = nenhum.
  - `sp_team` ∈ {`H`, `A`, `U`, `""`}.
- **`NaN` = "sem evidência", nunca zero.** Fora dos blocos (`half == 0`), tudo que deriva de tracking é `NaN`; `n_tags_favor_*` vale 0 e as flags valem `False`.
- **Bola e jogadores** (números públicos, do DATA.md): a bola aparece em 59,3% dos frames avaliados e some de forma não aleatória (bola alta, aglomeração, chutes). Há ≈20,5 jogadores rastreados em média.
- **Smart Tagging tem recall baixo:** 59 bolas paradas no jogo inteiro, e todo lateral tem `Team = N/A`. Use como evidência fraca.

### 1.4 Regras globais de trabalho (colar no topo de cada `docs/tasks/<ID>.md`)
1. **Agentes nunca leem, abrem, imprimem nem executam código sobre os dados da partida** (`$MTAH_DATA_DIR`, `./data` ou qualquer parquet/CSV derivado deles).
   - Agentes trabalham com dados sintéticos e com o IDSSE (público, de outras partidas).
   - Testes usam só fixtures sintéticas em `tmp_path`. Um teste que leia o diretório de dados é um bug: a CI não tem dados e falharia.
2. **"Execução local (humano)"**: passos assim marcados são rodados por uma pessoa, no próprio terminal.
   - As ferramentas em `tools/` imprimem, por padrão, só `PASS`/`FAIL`. Números aparecem apenas com `--verbose`, para a pessoa ver.
   - Podem ir para issues, PRs e agentes: `PASS`/`FAIL`, hashes e estatísticas **dos rótulos previstos** (nº de posses, proporções de H/A/D).
   - Nunca vão: valores de features, quantis do tracking, linhas, arrays ou gráficos de trajetória da partida.
3. **Código em inglês**: identificadores, comentários, docstrings, mensagens de log e de erro, nomes de testes e chaves de config. Commits em inglês (Conventional Commits). Documentos em `docs/` podem ser em português; o relatório técnico final é em **inglês**.
4. **`NaN` mantém o significado "sem evidência".**
   - Proibido `fillna(0)` em tabelas de features.
   - A contribuição de um termo cuja feature é `NaN` vale 0 de forma explícita: `np.where(np.isfinite(x), termo, 0.0)`.
   - Imputação só é permitida dentro da matriz de desenho dos ajustes de calibração, e sempre com um indicador de ausência.
5. **Zero números mágicos.** Todo parâmetro fica em `configs/*.yaml`, lido por `modeling.config.load_config` (chave desconhecida ou ausente = erro). Mudar um parâmetro = criar um config novo, nunca editar o código.
6. **Determinismo:**
   - `np.random.default_rng(seed)` com semente explícita; `float64` no decodificador.
   - Ordenações estáveis (`kind="mergesort"`) e desempate sempre pelo menor índice.
   - Produtos internos de inferência escritos como `(X * w).sum(axis=1)` em `float64`, nunca `@`/BLAS. Sem paralelismo.
7. **Contratos congelados:** não alterar `SCHEMA.md` v1.0, `preprocessing/schema.py` nem as 60 colunas. Features novas são calculadas em memória pelos módulos de `modeling/`. Só as tarefas F4-02, F4-05 e F4-06 alteram `preprocessing/`, sempre com teste de regressão bit a bit (a saída padrão não pode mudar).
8. **Não editar nada em `evaluation/`** (código oficial). Importar como `from evaluation import possession_metrics as pm`.
9. **A janela pública nunca serve para ajustar ou escolher nada.** Nunca relatar score dela em issue, PR, relatório ou decisão.
10. **Nada atravessa o limite de um tempo:** janelas temporais, interpolações e buscas operam sempre por bloco de `half`.
11. Se a especificação for ambígua, comente na issue e pergunte. Não decida em silêncio.
12. **CI e higiene do repositório (desde 11/11, F5-03):**
   - Todo push e todo PR rodam `pytest -q` (a suíte inteira) no GitHub Actions, sem os dados da partida. Merge só com a CI verde.
   - `tests/test_repo_hygiene.py` falha se o commit tiver:
     - arquivo de dados (`.parquet .npy .npz .pkl .mp4 .xml`) fora de `tests/fixtures/`;
     - qualquer arquivo em `data/` além de `data/README.md`;
     - notebook com saída;
     - arquivo acima de 1 MB fora da allowlist (`models/**/*.joblib` até 5 MB).
   - Fixtures sintéticas `.npy`, `.npz`, `.pkl` ou `.parquet` em `tests/fixtures/` entram só com `git add -f`.
   - Preencha o template de PR, inclusive a pergunta obrigatória "Este PR contém algum valor derivado dos dados da partida?".
   - Regras completas e checklist de publicação: [`docs/data_safety.md`](../data_safety.md).

### 1.5 Fluxo Git
- Branch `f<n>/<ID>-slug` a partir da `main`. Uma tarefa por PR, PR pequeno, com `Closes #<issue>`.
- **Revisão cruzada obrigatória:** F1↔F3, F2↔F4; F5 revisa qualquer um. CI verde e squash merge.
- **Definition of Done**, tudo junto:
  - código e testes sintéticos;
  - `docs/tasks/<ID>.md` atualizado com o que foi feito e as decisões tomadas;
  - parâmetros em config;
  - entrada em `docs/decisions.md` se houve decisão;
  - resultado `PASS`/`FAIL` da execução local colado na issue, quando houver execução local;
  - revisão aprovada.
- Uma issue bloqueada **não começa** antes de a bloqueadora fechar. Se uma bloqueadora atrasar, avise no grupo no mesmo dia.
- Uma tarefa P1 ou P2 fechada como "não será feita" deixa de bloquear as outras.
- Antes de abrir o PR, rodar `pytest -q` (a suíte inteira).

---

## 2. Arquitetura-alvo (decisão tomada)

### 2.1 Visão geral

```
dados da partida ── preprocessing.run_all ──► master_features.parquet (60 colunas)
                                                   │
          ┌────────────────────────────────────────┼──────────────────────────────┐
          ▼                                        ▼                              ▼
  F1: team_logit (razão H×A)           F3: dead_logit (razão D×vivo)      F3: force_D (pré-pontapé)
          └─────────────────────┬──────────────────┴──────────────────────────────┘
                                ▼
       F2: emissão (C3) → Viterbi por tempo com duração mínima → pós-processamento
                                ▼
                rótulos H/A/D/X → CSV (F5) → validate_submission

IDSSE (7 jogos, rótulos por frame) ── F4: arrays → degradação → MESMO pré-processamento ──► mestres externos
     ├─► F1/F3: calibram os logits (regressão logística, validação deixando uma partida de fora)
     ├─► F2: estatísticas de transição + varredura do decodificador
     └─► F2-06: seleção pelo spellF1 externo (protocolo da Seção 6)
```

A mesma função, `modeling.pipeline.predict_labels(master, cfg)`, roda nos dois domínios (BAR-ATH e IDSSE).

### 2.2 Por que esta arquitetura
- **HMM de 3 estados (H, A, D) com evidências na forma de razões de verossimilhança.**
  - Evidência ausente vale 0, e o estado persiste. Isso implementa exatamente a regra "bola solta fica com o último time".
  - O Viterbi entrega a sequência globalmente ótima, sem oscilação (que destrói o spellF1) e sem atraso de fase.
- **Poucos parâmetros interpretáveis, calibrados no IDSSE.** O IDSSE tem rótulos por frame de bola viva/morta e de posse, da mesma natureza do GT (derivado do tracking). Calibrar fora do BAR-ATH é a única forma permitida de escolher parâmetros.
- **Pipeline idêntico nos dois domínios.** O IDSSE passa pelo **mesmo** pré-processamento (via F4-02), depois de degradado para parecer o ATD: bola some em ≈41% dos frames, trilhas quebram, aparecem fantasmas.
- **Descartados:**
  - GBDT ou redes por frame: alto risco de mudança de domínio IDSSE→ATD, pouco tempo e determinismo mais difícil.
  - Autotreino com pseudo-rótulos ou ajuste não supervisionado (EM) no BAR-ATH: vetado (issue #2; confirmado pelos organizadores em 13/11, D-012).
  - Harness de mascaramento no BAR-ATH: não tem verdade, e premia suavização excessiva (uma saída constante seria "perfeitamente robusta").
  - Amostras abertas da Metrica com rótulos derivados de eventos: os rótulos são aproximados, enquanto o IDSSE tem rótulos nativos por frame.
- **Ativos fora do modelo** (documentar no relatório): vídeo tático, vídeo da transmissão e homografia (só diagnóstico; `reproj_error` não entra no modelo). O **tracking da transmissão** é uma trilha exploratória (F4-05/F4-06): só entra se passar no portão de viabilidade e no protocolo da Seção 6.

### 2.3 Layout do código novo

```
modeling/
  __init__.py
  paths.py            data_dir, processed_dir, ext_dir, outputs_dir            (F5-01)
  contracts.py        C0, C2, C4, validações                                    (F5-01)
  io.py               C1, C6, persistência de logits                            (F5-01)
  config.py           C7: leitor estrito de YAML + hash                         (F5-01)
  variants.py         registros TEAM_KINDS / DEAD_KINDS                         (F5-01)
  pipeline.py         C8: predict_labels                                        (F5-02, F5-04)
  baseline.py         baseline_labels                                           (F5-02)
  make_submission.py  comando único                                             (F5-02)
  evidence/           control.py (F1-01), team.py (F1-02, F1-04, F1-05), collective.py (F1-04)
  deadball/           features.py (F3-01), dead.py (F3-02, F3-04), force.py (F3-03)
  decode/             emission.py, params.py (F2-01), viterbi.py (F2-02), postprocess.py (F2-04)
  calibration/        common.py, team.py (F1-05), dead.py (F3-04), boundary.py (F2-05), team_gbdt.py (F1-07), dead_gbdt.py (F3-08)
  external/           files.py (F5-01), idsse.py (F4-01), degrade.py (F4-03), build.py (F4-04)
  scoring/            metrics.py, external_eval.py (F1-03), label_stats.py (F2-03), sweep.py (F2-05), selection.py (F2-06)
  broadcast/          align.py (F4-05), fuse.py (F4-06)          — trilha exploratória
preprocessing/pipeline_api.py                                                   (F4-02)
models/               team_gbdt/all.joblib (F1-07), dead_gbdt/all.joblib (F3-08) — ≤ 5 MB cada
configs/              baseline.yaml, v1.yaml, v1s.yaml, v2.yaml, final.yaml, degrade_default.yaml,
                      decoder_idsse.yaml, plausibility.yaml, calib/team.yaml, calib/dead.yaml
tools/                scripts de execução local que leem dados da partida e imprimem PASS/FAIL
reports/              resultados agregados SÓ de dados externos (podem ser commitados)
outputs/              CSVs gerados (no .gitignore)
tests/                test_modeling_*.py (nomes únicos; não criar __init__.py em subpastas de teste)
```

Não criar pasta ou pacote chamado `eval` (sombreia a função embutida `eval` do Python). Não pôr código em pasta ignorada pelo git.

### 2.4 Contratos (C0–C9) — implementados em F5-01, F5-02 e F2-01; usados por todos

- **C0 Códigos de rótulo:** `H=0, A=1, D=2, X=3`, na mesma ordem de `evaluation.possession_metrics.LABELS`. O decodificador usa a ordem de estados `(H, A, D)`.
- **C1 Master:** `modeling.io.load_master(path=None, n_frames=None) -> pd.DataFrame`.
  - Valida colunas e dtypes contra `preprocessing.schema.master_columns(include_optional=False)`. `reproj_error` é aceito, mas não exigido.
  - Exige `video_frame == arange(len(df))` e, se `n_frames` for dado, `len(df) == n_frames`.
  - Com `path=None`, carrega o master do BAR-ATH (`processed_dir()/"master_features.parquet"`) e, se `n_frames` também for `None`, usa `games.GAMES[games.GAME]["n_video_frames"]`.
  - Com `path` explícito e `n_frames=None`, o comprimento não é conferido. É o caso dos mestres externos (IDSSE), que têm outro comprimento e 59 colunas (sem `reproj_error`).
- **C2 Evidências por frame:** `team_logit` e `dead_logit`, ambos `float32[n]`, finitos, sem `NaN`.
  - `team_logit` = razão de log-verossimilhança `log p(obs|H) − log p(obs|A)`: `> 0` favorece H, `< 0` favorece A, `0` = sem evidência. Limite `|t| ≤ 8`.
  - `dead_logit` = razão `log p(obs|D) − log p(obs|vivo)`: `> 0` favorece D, `0` = sem evidência. Limite `|d| ≤ 12`.
  - Ambos valem 0 onde `half == 0`.
  - Persistência opcional: `processed_dir()/f"logits_{cfg_name}.parquet"`, com colunas `video_frame, team_logit, dead_logit`.
- **C3 Emissão:** `modeling.decode.emission.logits_to_logscores(t, d, lam_team, lam_dead, force_d, neg=-1e6) -> float64[n, 3]`, na ordem `(H, A, D)`:
  ```
  logσ(x) = −np.logaddexp(0, −x)
  S_H = lam_team · (log 2 + logσ(t))
  S_A = lam_team · (log 2 + logσ(−t))
  S_D = lam_dead · d
  ```
  - Propriedades obrigatórias:
    - com `t = d = 0`, as três pontuações valem exatamente 0;
    - trocar `t` por `−t` troca `S_H` com `S_A`;
    - `S_H − S_A = lam_team · t`;
    - evidência forte de time empurra "vivo" no máximo `lam_team · log 2` por frame.
  - **Não usar a fórmula da v1** (`S_D = logσ(d)`, `S_H = logσ(−d) + logσ(t)`): com `d = 0` ela dá `S_H = S_D + logσ(t) < S_D` para qualquer `t`, e D vence sempre.
- **C4 Máscara** `force_d: bool[n]` (frames que obrigatoriamente são `D`). Na emissão: `S_D = 0` e `S_H = S_A = neg`.
- **C5 Decodificador:** `modeling.decode.viterbi.decode_match(scores, half, force_d, params, *, exit_logit=None) -> np.ndarray[dtype "<U1"]`.
  - Decodifica cada tempo (`half` = 1 e 2) separadamente.
  - `half == 0` → `X`; dentro dos blocos, nunca `X`.
  - `exit_logit` (F2-07, D-024): com `None`, as transições são constantes e o código é o de antes, sem nenhuma alteração. Senão é um `float64[n]`, finito dentro dos tempos. No frame `t`, só as saídas do subestado terminal de H e de A mudam: `q_o' = q_o·e^{x_t}/Z_t`, `p' = p/Z_t`, `q_d' = q_d/Z_t`, com `Z_t = p + q_d + q_o·e^{x_t}` (em log, com `logaddexp`). D e a cadeia do `min_dwell` não mudam. A transição que **chega** ao frame `t` usa `x_t`, então o valor do primeiro frame de cada tempo e os de `half == 0` não são usados.
  - O C8 passa `exit_logit = beta · h`, com `h` o hazard da F1-08 (seção `hazard` do config), ou `None` quando `hazard.kind` é `none` ou `beta` é 0.
- **C6 Submissão:** `modeling.io.write_submission(labels, path, n_expected=None)`.
  - Cabeçalho `frame,label`, UTF-8, `\n`, uma linha por frame `0..n−1`.
  - Depois de gravar, relê com `pm.read_label_csv` e chama `pm.validate_submission(df, n_expected)`; levanta erro se houver mensagem.
  - `n_expected` padrão = `games.GAMES[games.GAME]["n_video_frames"]`. Nunca escrever 199319 literal no código.
- **C7 Config:** `modeling.config.load_config(path) -> dict` faz validação estrita do esquema do Apêndice A.7 (chave desconhecida ou ausente = erro).
  - `config_hash(cfg)` = SHA-256 de `json.dumps(cfg, sort_keys=True, separators=(",", ":"))`.
  - Toda execução registra o nome e o hash do config.
- **C8 Pipeline:** `modeling.pipeline.predict_labels(master, cfg, *, holdout=None) -> np.ndarray["<U1"]`.
  - É a **única** função que transforma um master em rótulos, tanto no BAR-ATH quanto no IDSSE.
  - `holdout`: id da partida externa sendo avaliada. Componentes calibrados (logits logísticos, modelos GBDT e correção de fronteira) usam então os parâmetros ajustados **sem** essa partida.
- **C9 Formato externo** (`modeling/external/files.py`):
  - `ExternalMatch`, dataclass salva em `ext_dir()/"arrays"/f"{match_id}.npz"`, com os campos:
    ```
    match_id: str
    players: float32 (T, 46, 3)   # x_n, y_n em [0,1] (x ao longo do comprimento), velocidade em m/s
    ball: float32 (T, 3)          # x_n, y_n, velocidade (m/s ou NaN)
    ball_z: float32 (T,)          # altura em metros (usada só pela degradação)
    labels: int8 (T,)             # C0
    half: int8 (T,)               # 1, 2; 0 no intervalo artificial
    slot_team_ids: int8 (46,)     # 0 = casa (slots 0–23), 1 = visitante (slots 24–45)
    pitch: float64 (2,)           # (L, W) em metros, do estádio
    ```
  - Layout temporal: frames do 1º tempo, depois `GAP_FRAMES = 250` frames vazios (`NaN`, `half = 0`, rótulo `X`), depois o 2º tempo. `ExternalMatch.blocks()` → `[(1, f0, f1), (2, f0, f1)]`, com limites inclusivos.
  - Arquivos derivados (F4-04): `ext_dir()/"processed"/f"{match}__{degrade}.parquet"` (master de 59 colunas) e `f"{match}__{degrade}_labels.npy"` (int8, C0). `degrade` é `"none"` ou o nome de um arquivo `configs/degrade_<nome>.yaml`: `default`, `harder` e `easier` (F4-04), e `fused` (F4-06, se houver).
  - Funções de acesso: `IDSSE_MATCHES`, `GAP_FRAMES`, `arrays_path(match)`, `master_path(match, degrade)`, `labels_path(match, degrade)`, `load_external(match, degrade) -> (master, labels)`.

---

## 3. Calendário e marcos

| Marco | Data | Critério |
| :--- | :--- | :--- |
| **M0** Base pronta | dom 08/11 | contratos mergeados; `outputs/baseline.csv` passa em `validate_submission.py`; IDSSE baixado e convertido; features `ev_*`/`db_*` prontas |
| **M1** Motor v1 + dados externos | ter 10/11 | `make_submission --config configs/v1.yaml` gera CSV válido (Viterbi + logits v1); API em memória do pré-processamento; degradação; harness de avaliação |
| (envio de segurança) | qua 11/11 | formulário enviado com o CSV v1 (F5-05) |
| **M2** Calibração e varredura | qui 12/11 | mestres externos degradados prontos (qua 12h); logits calibrados; decodificador varrido; viés de fronteira calibrado |
| (portão da transmissão) | qui 12/11 12h | GO/NO-GO da trilha exploratória (F4-05) |
| (candidatas P2) | sex 13/11 12h | variante GBDT do dead (F3-08) e fusão da transmissão (F4-06) prontas, ou fechadas como "não será feita" (a F1-07 é P1 desde 11/11) |
| **M3** Método final | sex 13/11 18h | `reports/selection.md` + `configs/final.yaml`; **code freeze às 20h** (G1) |
| **M4** Reprodução limpa | sáb 14/11 | hash idêntico em duas máquinas, clonando do zero |
| Relatório | dom 15/11 | relatório técnico (inglês) e PDF finais |
| **Entrega** | seg 16/11 | formulário enviado; ter 17/11 é só folga |

O calendário assume trabalho no fim de semana de 07–08/11. Se o grupo não puder, empurre o M0 para seg 09/11 e aplique a lista de corte (Seção 7). O freeze **não se move**. Check-in assíncrono diário no grupo: "fiz / vou fazer / o que me bloqueia".

---

---

## 6. Protocolo de seleção (pré-registrado: valem estas regras, não a intuição)

> **Atualização de 11/11:** o teste de estresse passa a incluir os modos `hidelo` e `hidehi` (F4-04), e há passos novos na escada e uma checagem de estabilidade. Ver a seção de atualizações consolidadas em `docs/tasks/F2-06.md`, que prevalece sobre esta seção onde houver conflito.

Objetivo: escolher a variante final **sem usar a janela pública e sem nenhum rótulo do BAR-ATH**.

1. **Candidatas, da mais simples para a mais complexa:**
   - `baseline`;
   - `v1` (priors do Apêndice A);
   - `v1s` (v1 + decodificador varrido na F2-05);
   - `v2` (team e dead `calibrated`, partindo do decodificador do `v1s`; varrer de novo só a etapa 1, λ, e o pós-processamento);
   - `v2` com o modelo "gap" ligado (evidência coletiva);
   - mais o pós-processamento: cada regra que teve ganho isolado na F2-05, uma de cada vez;
   - mais a correção de viés de fronteira (`boundary_shift_file`, F2-05);
   - troca de `calibrated` por `gbdt` no team (F1-07) e, num passo separado, no dead (F3-08);
   - `ball_source: fused` (F4-06). Este passo é aceito pelos critérios próprios da F4-06, porque o ganho no IDSSE vem do proxy de visibilidade.
2. **Critério principal:** spellF1 médio nos 7 mestres degradados (`default`). Componentes calibrados são sempre avaliados com a partida avaliada fora do ajuste (`holdout = partida`).
3. **Aceitação sequencial:** cada passo da lista só é aceito se as duas condições valerem:
   - (a) o ganho pareado médio no spellF1 em `default` é maior que 1 EP desse ganho (`paired_delta`);
   - (b) **teste de estresse:** nos mestres `harder` e `easier`, o passo não piora o spellF1 em mais de 1 EP. Uma melhora que só aparece numa degradação não é confiável, porque a degradação é uma aproximação do ATD.
   
   Se não for aceito, fica a candidata anterior, e o passo seguinte parte dela.
4. **Filtros eliminatórios** (execução local no BAR-ATH, PASS/FAIL):
   - CSV válido;
   - `tools/check_public_window.py` PASS;
   - `tools/deadball_diagnostics.py` dentro das faixas de `configs/plausibility.yaml` com folga de ±40% (posses por minuto, duração média de posse, fração de D).
   
   Se a vencedora falhar, fica a melhor anterior que passa.
5. **Desempate** quando os ganhos ficam dentro de 1 EP: macroF1 externo; persistindo o empate, a candidata mais simples.
6. **Proibido:** qualquer uso do score da janela pública e qualquer ajuste fora do IDSSE.
7. **Registro:**
   - `reports/selection.md`: tabelas por partida, médias, EPs, deltas pareados e a decisão de cada passo;
   - `configs/final.yaml`: cópia do vencedor com `name: final`;
   - `docs/decisions.md`.
8. **Caminho sem IDSSE:** se a F4 não entregar os mestres até qua 11/11 às 18h, ver a Seção 7. No pior caso, o final é `v1` com os priors do Apêndice A, sem varredura, e a limitação vai para o relatório.

---

## 7. Riscos, plano B e lista de corte

| Risco | Plano B |
| :--- | :--- |
| Atraso no caminho crítico | lista de corte abaixo; envio de segurança na qua 11/11 garante um CSV válido |
| Mudança de domínio (IDSSE limpo → ATD automático) | degradação + checagem de realismo (F4-04), que inclui velocidades e bola parada; mestres `__none` só para diagnóstico, nunca para calibrar ou selecionar (D-009); poucos parâmetros calibrados; aceitar só com ganho > 1 EP |
| Convenção de posse do DFL diferente da do GT Metrica | preferir parâmetros estruturais robustos; documentar como limitação |
| Download, RAM ou disco do IDSSE | baixar do espelho no Hugging Face com revisão fixada (o figshare bloqueia scripts); uma partida por vez; `--delete-raw`; checar espaço livre antes |
| Não determinismo | threads = 1; somas explícitas em float64; F5-06 em duas máquinas |
| Vazamento de dados para o repositório ou para LLM em nuvem | regras 1.4.1 e 1.4.2; F5-03; auditoria da F5-09 |
| Formulário ou conta Google | F5-08 na sex 06/11; envio de segurança; envio final na seg 16/11 |
| A trilha da transmissão consumir tempo do caminho crítico | é P2, começa só depois da F4-04, tem portão GO/NO-GO na qui 12/11 às 12h e nunca bloqueia tarefas P0/P1 |
| Variantes novas parecerem melhores só por acaso | aceitação com ganho > 1 EP e teste de estresse em `harder`/`easier` |

**Lista de corte:**
- **P0 (nunca cortar):** F5-01, F5-02, F5-04, F5-05, F5-06, F5-07, F5-08, F5-09, F1-01, F1-02, F2-01, F2-02, F2-06, F3-01, F3-02, F3-03, F3-05, G1.
- **P1 (cortar só pelos gatilhos abaixo):** F4-01, F4-02, F4-03, F4-04, F1-03, F1-04, F1-05, F2-03, F2-04, F2-05, F3-04, F3-06, F5-03; desde 11/11 também F1-06 e F1-07 (evidência de time é o gargalo de spellF1).
- **P2 (primeiras a cair):** F3-07, F3-08, F4-05, F4-06.

**Gatilhos** (decididos pela F5 no check-in do dia):
- **F4-04 não está pronta na qua 11/11 às 18h:** as calibrações (F1-05, F3-04) e a varredura (F2-05) esperam pelos mestres degradados. **Nunca** usam os mestres `__none` (limpos): no IDSSE limpo a bola fica congelada durante a bola morta, o que vaza o rótulo para as features (D-009).
- **Também não está pronta na qui 12/11 às 12h:** Seção 6, passo 8.
- **Calibrações não estão prontas na qui 12/11 às 20h:** a seleção fica só entre `baseline`, `v1` e `v1s`.
- **F4-05 sem GO até qui 12/11 às 12h:** a F4-06 é fechada como "não será feita".
- **Candidatas P2 (F3-08, F4-06) não prontas até sex 13/11 às 12h:** fechadas como "não será feita"; deixam de bloquear a F2-06. A F1-07 é P1 desde 11/11 e não cai por este gatilho; o prazo dela é combinado com a F5.
- **O freeze não se move.**

**Fora do escopo desta entrega:** detecção automática do pontapé inicial a partir dos dados; detecção própria da bola no vídeo.

---

## Apêndice A — Especificação de referência (valores iniciais e esquema de config)

Todos os valores são **priors físicos ou de domínio, não calibrados**. Eles moram em `configs/*.yaml` e é **proibido otimizá-los contra a janela pública**. Só mudam pelas tarefas F1-05, F3-04, F2-03 e F2-05, sempre no IDSSE.

### A.1 `team_logit` `rules_v1`
`kappa=3.0` · `sigma_d=2.0` m · `r_ctrl=2.0` m · `l_prox=3.0` m · `v_rel=6.0` m/s · `ball_conf_filled=0.6` · `contest_factor=0.5` · `rel_w_when_nan=0.5`.

Evidência coletiva: `win_frames=25`, `w_adv=0.0`, `w_spread=0.0`, `w_speed=0.0` (desligada).
```
t = −kappa · tanh(d_diff / sigma_d) · ev_prox_w · ev_rel_w · ev_ball_conf · contest
t = clip(t, −kappa, +kappa);  t = 0.0 onde a bola não é válida, d_diff é NaN ou half == 0
```
Com `d_diff = d_near_home − d_near_away`: casa mais perto → `d_diff < 0` → `t > 0`.

### A.2 Parâmetros das features de bola morta (`dead.params`)
`long_gap_frames=12` · `exit_line_m=3.0` · `exit_speed_mps=0.5` · `restart_line_m=1.5` · `goal_area_depth_m=6.0` · `goal_area_half_width_m=9.2` · `pitch_w_m=68.0` · `air_speed_mps=8.0` · `air_max_frames=100` · `smooth_frames=25` · `rest_speed_mps=0.3` · `rest_min_frames=25` · `fast_speed_mps=6.0` · `sp_prox_tau_frames=250` · `sp_after_tau_frames=75` · `goal_window_frames=1500`.

### A.3 `dead_logit` `rules_v1` (pesos em `dead.params.weights`)
É a soma dos termos abaixo, recortada em ±12. Cada termo vale 0 quando a feature é NaN.

| Termo | Contribuição |
| :--- | :---: |
| `out_of_pitch == 1` | +4,0 |
| `db_gap_kind == 2` (saída) | +3,0 |
| `db_gap_kind == 3` (aérea) | −1,0 |
| `db_gap_kind == 4` (longa indeterminada) | +0,5 |
| `db_still_frac` | +3,0 · clip((x − 0,3)/0,4, 0, 1) |
| `db_mean_speed` alta | −2,0 · clip((v − 2,5)/2,0, 0, 1) |
| `db_mean_speed` baixa | +1,5 · clip((1,5 − v)/1,0, 0, 1) |
| `db_ball_rest == 1` | +2,0 |
| `db_fast_in_pitch == 1` | −3,0 |
| [tag] `is_set_piece` | +1,0 |
| [tag] `db_sp_prox` | +1,0 · valor |
| [tag] `db_sp_after` | −1,0 · valor |
| [tag] `db_goal_window == 1` | +2,0 |

O pré-pontapé não entra no logit; ele é a máscara `force_d` (C4).

**Por que `is_set_piece` pesa só +1,0** (era +2,5 na v1): a janela da tag cobre também o depois do reinício. A tag `Kick Off (Start)` abre ~3,5 s antes do chute e dura ~5 s (starter, `01_smart_tagging.ipynb`). Um peso alto atrasaria o fim do D, e portanto o início da posse, depois de cada reinício marcado.

### A.4 Decodificador (25 fps)
- `lam_team=0.2`, `lam_dead=0.2`. Os frames a 25 fps são fortemente autocorrelacionados, e λ < 1 evita contar a mesma evidência 25 vezes por segundo.
- Duração média: `H`/`A` = 15 s, `D` = 20 s. Saindo de H: 50% → A, 50% → D; saindo de A: 50% → H, 50% → D; saindo de D: 50% → H, 50% → A.
- `min_dwell`: `H`/`A` = 12 frames, `D` = 25 frames. `neg=-1e6`. Distribuição inicial uniforme.
- `configs/decoder_idsse.yaml` (F2-03) substitui as durações e o `exit_split` pelos valores do IDSSE.

### A.5 Pós-processamento
Padrão: tudo desligado. `midpoint_neutral_switch: false`, `d_bridge_frames: 0`, `min_spell_frames: 0`. Cada regra só é ligada pela F2-06.

### A.6 Degradação (`configs/degrade_default.yaml`)
- **Bola:**
  - `z_air_m=1.5`, `air_margin_frames=3`, `p_air=0.85`;
  - `crowd_n=3`, `crowd_r_m=2.0`, `crowd_hazard=0.02`;
  - `target_ball_visible=0.593`;
  - perfil de lacunas `{mix_geometric: 0.6, geometric_mean_frames: 8, lognormal_median_frames: 40, lognormal_sigma: 1.0, max_frames: 1500}`;
  - `ball_noise_m=0.25`;
  - `phantom_len_frames=[5, 60]`, `phantom_frac=0.01`.
- **Jogadores:**
  - `player_visible_mean_s=30.0` e `player_missing_mean_s=3.0` (este é só o ponto de partida da bisseção);
  - `target_tracked_mean=20.5`;
  - `player_noise_m=0.15`;
  - `dup_offset_m=0.3`, `dup_len_s=[1.0, 5.0]`, `dup_frac=0.02`.
- **Variantes de estresse** (mesmas chaves da `default`; só mudam estes valores):
  - `configs/degrade_harder.yaml`: `target_ball_visible=0.50`, `phantom_frac=0.02`, `player_noise_m=0.25`, `target_tracked_mean=19.5`;
  - `configs/degrade_easier.yaml`: `target_ball_visible=0.68`, `phantom_frac=0.005`, `player_noise_m=0.10`, `target_tracked_mean=21.5`.

### A.7 Esquema dos configs (validado por `load_config`)
```yaml
name: v1                       # str; igual ao nome do arquivo sem .yaml
pipeline: hmm                  # baseline | hmm
ball_source: tactical          # tactical | fused (F4-06); obrigatório em todo config
baseline:                      # obrigatório se e somente se pipeline == baseline
  d_gap_frames: 100
  slow_speed_mps: 1.5
  min_spell_frames: 25
team:                          # obrigatório se e somente se pipeline == hmm (vale para dead, decoder e postprocess)
  kind: rules_v1               # none | rules_v1 | calibrated | gbdt
  params:
    pitch_l_m: 105.3           # usados pela troca de times da F1-07
    pitch_w_m: 68.0
    kappa: 3.0
    sigma_d: 2.0
    r_ctrl: 2.0
    l_prox: 3.0
    v_rel: 6.0
    ball_conf_filled: 0.6
    contest_factor: 0.5
    rel_w_when_nan: 0.5
    collective: {win_frames: 25, w_adv: 0.0, w_spread: 0.0, w_speed: 0.0}
  calibrated_file: null        # ex.: configs/calib/team.yaml (calibrated) ou configs/calib/team_gbdt.yaml (gbdt)
dead:
  kind: rules_v1               # none | rules_v1 | calibrated | gbdt
  params:                      # todas as chaves de A.2, mais:
    weights: {out_of_pitch: 4.0, gap_exit: 3.0, gap_air: -1.0, gap_long: 0.5,
              still: 3.0, speed_high: -2.0, speed_low: 1.5, ball_rest: 2.0,
              fast_in_pitch: -3.0, tag_set_piece: 1.0, tag_sp_prox: 1.0,
              tag_sp_after: -1.0, tag_goal_window: 2.0}
  calibrated_file: null
decoder:
  lam_team: 0.2
  lam_dead: 0.2
  mean_duration_s: {H: 15.0, A: 15.0, D: 20.0}
  exit_split: {H: {A: 0.5, D: 0.5}, A: {H: 0.5, D: 0.5}, D: {H: 0.5, A: 0.5}}
  min_dwell_frames: {H: 12, A: 12, D: 25}
postprocess:
  midpoint_neutral_switch: false
  boundary_shift_file: null    # ex.: configs/calib/boundary.yaml (F2-05)
  d_bridge_frames: 0
  min_spell_frames: 0
```
`configs/baseline.yaml` usa só `name`, `pipeline: baseline`, `ball_source` e `baseline`.

---

## Apêndice B — Armadilhas verificadas (ler antes de implementar)

1. `video_frame = frame_id − 1`. Os índices do CSV são os de `video_frame` (0…199318).
2. `games.half_video_blocks` devolve limites **inclusivos**. A fatia Python correspondente é `[first, last + 1)`.
3. Nada atravessa o limite de um tempo: janelas, buscas e interpolações rodam por bloco.
4. `NaN` ≠ 0. Sem bola, a evidência vale 0 em **logit** (neutra), mas a feature continua `NaN`.
5. `d_diff = home − away`. Trocar a convenção inverte o sinal do modelo inteiro.
6. Um `D` no meio de uma posse custa duas vezes. Nada de `X` dentro dos tempos.
7. O pré-pontapé do 2º tempo (2.934 frames) é `D`. O 1º tempo começa direto no pontapé.
8. Tags: recall baixo; `sp_type` em minúsculas; a janela de uma bola parada vai além do reinício.
9. **Emissão:** usar o C3 desta versão. Com `t = d = 0`, as três pontuações têm de dar 0.
10. **IDSSE:**
    - baixar do espelho no Hugging Face com revisão fixada; o figshare bloqueia scripts;
    - usar `only_alive=False`;
    - o kloppy transforma `BallPossession` ausente em "visitante";
    - a origem das coordenadas é o centro do campo;
    - as dimensões do campo variam por estádio.
11. Vetores externos precisam de X no intervalo entre os tempos, ou as posses atravessam o intervalo na métrica.
12. Modelo de time calibrado: antissimétrico, sem intercepto e sem centralizar as features.
13. Modelo de bola morta calibrado: subtrair `logit(π_D)`, senão a proporção de D é contada duas vezes.
14. Nunca usar `reproj_error`, `frame_in_half` ou `video_frame` como feature; eles deixam o modelo decorar o momento do jogo. O BAR-ATH tem 60 colunas e os mestres externos têm 59.
15. Inferência: `(X * w).sum(axis=1)` em float64, nunca `@`; threads = 1 em `make_submission`.
16. Não criar pacote `eval` e não colocar código em pasta ignorada pelo git.
17. Agentes não rodam nada sobre os dados da partida. Testes não leem `MTAH_DATA_DIR`.
18. Novos testes ficam em `tests/` com prefixo `test_modeling_`, com nomes únicos e sem `__init__.py` em subpastas.

---

## Apêndice C — Mapa do código existente (o que usar e como)

| Onde | O quê |
| :--- | :--- |
| `games.py` | `GAME = "BAR-ATH"`; `GAMES[GAME]["n_video_frames"]` (199319), `["fps"]` (25), `["pitch"]` (105.3, 68.0); `half_video_blocks(GAME)` → `[(1, 11185, 90651), (2, 110808, 190197)]`; `DATA_DIR` (respeita `MTAH_DATA_DIR`) |
| `preprocessing/run_all.py` | `run_pipeline(data_dir=None, processed_dir=None, skip_homography=False, check_determinism=False) -> int` (0 = ok) |
| `preprocessing/schema.py` | `master_columns(include_optional=True) -> [(nome, dtype)]`; `MASTER_GROUPS`; `OPTIONAL_COLUMNS = ("reproj_error",)` |
| `preprocessing/params.py` | `N_FRAMES`, `FPS`, `PITCH_L`, `PITCH_W`, `MAX_GAP_FRAMES=12`, `STILL_SPEED=0.5`, `RADII`, `PRESSURE_TAU`, `PITCH_MARGIN=1.0` |
| `preprocessing/prep_tracking.py` | `prepare_tracking(players, ball, slot_team_ids, blocks=None, pitch=(PITCH_L, PITCH_W), n_frames=N_FRAMES, kickoffs=None, stage="V1", data_dir=None, slot_is_field_player=None) -> (frame_df, slot_teams_df)`; `players` (T, N, 3) normalizado (x, y, velocidade); `ball` (T, 3) |
| `preprocessing/prep_proximities.py` | `assemble_inputs(frame, slot_teams, tracking_stage, n_frames, n_slots)`; `build_proximities_table(inputs)` (ganha `pitch=` na F4-02); `pitch_features(ball_x, ball_y, length, width, margin)` |
| `preprocessing/build_master_table.py` | `build_master(processed_dir, blocks=None, n_frames=N_FRAMES)`; `index_columns(blocks, n_frames)`; `write_master(master, path)`; `file_sha256(path)` |
| `preprocessing/export_tagging.py` | `build_tagging_dataframe(ev, blocks, exact_kickoff_frames, n_frames)` |
| `preprocessing/make_mock_master.py` | `generate_mock_master(game="BAR-ATH", seed=42, include_optional=True) -> DataFrame` (199.319 × 60, ~5 s) |
| `evaluation/possession_metrics.py` | `LABELS = ("H","A","D","X")`; `label_codes(labels)`; `read_label_csv(path)`; `validate_submission(df, n_expected) -> None` ou uma mensagem; `evaluate(gt_codes, pred_codes, taus=(12, 25), bootstrap_n=0) -> dict` (chaves `spell_f1_mean`, `macro_f1`, `alive_dead_f1`, `transitions[25]["f1"]`, `transitions[25]["mean_abs_latency"]`, `n_gt_spells`, `n_pred_spells`, …) |
| `evaluation/validate_submission.py` | CLI oficial: `python evaluation/validate_submission.py <csv>` |

**Colunas do master** (`SCHEMA.md` v1.0):
- **Índice:** `video_frame`, `half`, `is_graded`, `frame_in_half`.
- **Bola:** `ball_x`, `ball_y`, `ball_vx`, `ball_vy`, `ball_speed`, `ball_acc`, `ball_dir_change`.
- **Confiança:** `ball_observed`, `ball_jump_removed`, `ball_filled`, `gap_length`, `since_last_obs`, `until_next_obs`.
- **Times:** `centroid_{home,away}_{x,y}`, `spread_{home,away}_{x,y}`, `mean_speed_*`, `n_still_*`, `n_tracked_*`.
- **Proximidade:** `d_near_home`, `d_near_away`, `d_diff`, `n_{home,away}_r{2,3,5}`, `pressure_*`.
- **Velocidade relativa:** `speed_rel_*`, `speed_radial_*`.
- **Campo:** `out_of_pitch`, `d_touchline`, `d_goalline`.
- **Bolas paradas:** `is_set_piece`, `sp_type`, `sp_team`, `frames_to_next_sp`, `frames_since_sp_end`, `next_sp_type`, `prev_sp_type`.
- **Fases:** `n_tags_favor_home`, `n_tags_favor_away`.
- **Pontapé:** `pre_kickoff`.
- **Opcional:** `reproj_error`.

---

## Apêndice D — IDSSE (dados externos)

- **O que é:** 7 partidas completas da Bundesliga e da 2. Bundesliga, temporada 2022/23, com dados de posição (DFL) a 25 fps. A bola tem, em cada frame, `BallStatus` (1 = viva, 0 = morta) e `BallPossession` (1 = casa, 2 = visitante).
- **Ids:** `J03WMX`, `J03WN1`, `J03WOH`, `J03WOY`, `J03WPY`, `J03WQQ`, `J03WR9`.
- **Acesso:** baixar do espelho no Hugging Face (`pysport/idsse-data`, revisão `a715a38dfbaf5f58e431727c2b78d174101a703c`), que é público e sem login. A hospedagem original (figshare/Springer Nature, usada por `sportec.load_open_tracking_data` do kloppy) bloqueia downloads por script com uma proteção anti-robô (HTTP 202 + página HTML, verificado em 05/11). O kloppy continua sendo usado para **ler** os arquivos já baixados (`sportec.load_tracking` com caminhos locais).
- **Licença:** CC BY 4.0, uso aprovado pelos organizadores na issue #16. A citação é obrigatória no README e no relatório (texto exato na F4-01).
- **Diferenças para o ATD:**
  - o tracking é corrigido, completo e com a bola em quase todo frame; a degradação (F4-03) cuida disso;
  - a convenção de posse do DFL pode diferir da do GT Metrica nos toques curtos e duelos (limitação a documentar).
- **Recursos:** os XMLs de posição somam ≈2,5 GB (de 333 a 399 MB por partida). Processar uma partida por vez e checar o espaço livre em disco antes. Os arquivos ficam em `$MTAH_EXT_DIR` (padrão `~/mtah-external`), fora do repositório.
- **Quem precisa dos arquivos no próprio computador:**
  - F4: gera tudo (F4-01, F4-03, F4-04);
  - F2: F2-03 (os `.npz`, a partir de ter 10/11), F2-05 e F2-06 (mestres);
  - F1: F1-05 e F1-06 (mestres);
  - F3: F3-04 (mestres);
  - F5: não precisa (a submissão final usa só os pesos e configs commitados).
  
  Como obter: regenerar com os comandos da F4-04 (resultado idêntico, sementes fixas) ou, como plano B, pegar o pacote compartilhado pela F4. O IDSSE é público e de outros jogos: os agentes podem ler esses arquivos à vontade.
- **Na avaliação final:** os organizadores não precisam do IDSSE para reproduzir o CSV, porque o comando único usa só os dados do BAR-ATH e os pesos commitados. O IDSSE só é necessário para refazer as calibrações, cujo código fica publicado (issue #11).

---

## Apêndice E — O que mudou em relação à v1 do plano, e por quê

| Item | v1 | v2 | Motivo |
| :--- | :--- | :--- | :--- |
| Emissão C3 | `S_D=logσ(d)`, `S_H=logσ(−d)+logσ(t)` | `S_H=λ(log2+logσ(t))`, `S_D=λ·d` | com `d = 0`, D vencia H/A para qualquer `t`; em simulação, uma posse clara virava D inteira |
| Dados externos | amostras da Metrica + rótulos derivados de eventos, com gate próprio (G1 da v1) | IDSSE, com rótulos nativos por frame; sem gate (G1 agora é o code freeze, que era o G3) | aprovado na issue #16; mesma natureza do GT; menos trabalho e menos erro |
| Parâmetros de transição | hard-EM no BAR-ATH (F2-04 da v1) | estatísticas dos rótulos do IDSSE (F2-03) + varredura (F2-05) | autoaprendizado no jogo da competição é arriscado e as regras são ambíguas; o IDSSE tem verdade |
| Robustez | mascaramento no BAR-ATH (F4-04 da v1) | removido; avaliação no IDSSE degradado | sem verdade, premia suavização excessiva; exigia mexer no `prep_tracking` |
| Forward-backward | F2-03 da v1 | removido | não era usado na decisão |
| Cache por hash | F5-04 da v1 | removido | o pré-processamento roda em ~25 s |
| Calendário | 30/10 → 16/11 | 05/11 → 16/11, com lista de corte e envio de segurança | data real de início |
| `is_set_piece` no dead_logit | +2,5 | +1,0 | a janela da tag vai além do reinício |
| `db_goal_window` | `"Kick Off (After Goal)"` | `"kick off (after goal)"` | no master os tipos estão em minúsculas; a v1 nunca dispararia |
| Pastas | `eval/`, `external/` ignorada no git, `pipeline/` | tudo em `modeling/`; dados externos em `$MTAH_EXT_DIR` | `eval` sombreia a função embutida; pasta de código ignorada não seria commitada |
| `prep_proximities.pitch_features` | dimensões fixas do BAR-ATH | `pitch` por partida (F4-02) | estádios do IDSSE têm outras dimensões |
| Modelos calibrados | sem restrições | time antissimétrico, sem intercepto; dead com correção de prior | evita aprender viés de mandante e contar a proporção de D duas vezes |
| `min_dwell` inicial | H/A 25, D 50 | H/A 12, D 25 (e varridos) | o GT tem posses de 19 e 33 frames |
| `lam` inicial | 1,0 | 0,2 (varrido) | a autocorrelação a 25 fps conta a mesma evidência muitas vezes |
| Smoke test público | imprimia o spellF1 "uma vez" | só PASS/FAIL, sem score | evita a tentação de escolher pela janela pública |
| Agentes e dados | "humano devolve agregados" | só PASS/FAIL, hashes e estatísticas dos rótulos previstos | Termos proíbem extratos dos dados em LLM em nuvem |
| Envio | só no fim | envio de segurança na qua 11/11 + final na seg 16/11 | o formulário aceita edição; vale a última resposta |
| Relatório | sem idioma definido | inglês | organizadores internacionais |
| Fronteiras (v2.1) | só Viterbi + ponto médio | + correção do atraso sistemático por tipo de transição, calibrada no IDSSE | o spellF1 pune atraso: 8 frames → 0,89; 1 s → 0,65 |
| Robustez de domínio (v2.1) | só a degradação padrão | + degradações `harder`/`easier` na regra de aceitação | a degradação é uma aproximação do ATD; ganho que só aparece nela não é confiável |
| Modelos das evidências (v2.1) | só regressão logística | + variantes GBDT, opcionais (F1-07, F3-08) | interações não lineares, sobretudo na bola morta |
| Tracking da transmissão (v2.1) | fora do escopo | trilha exploratória com portão GO/NO-GO (F4-05, F4-06) | a bola tática some em 41% dos frames; a TV pode ver parte deles |
