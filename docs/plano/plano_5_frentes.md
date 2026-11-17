# Plano de Execução em 5 Frentes — More than a Hack 2027 (Ball Possession) — v2

**Período:** de 05/11/2026 (pré-processamento concluído e revisado) até a entrega em **17/11/2026** (meta interna: enviar na **seg 16/11**).
Este arquivo é a **fonte única** para criar as issues do GitHub. Ele é autocontido: contexto, arquitetura, contratos, tarefas, dependências, calendário e critérios de aceite.

> **v2 — revisão de 05/11/2026.** A v1 foi escrita sem acesso ao código. Esta versão foi conferida contra o código real do repositório, as regras e as respostas dos organizadores nas issues oficiais. Mudanças principais (lista completa no Apêndice E):
> 1. **Bug corrigido na emissão (C3).** Na fórmula da v1, com `dead_logit = 0` vale `S_H = S_D + λ·logσ(t) < S_D` para **qualquer** `team_logit`: `D` vence H e A mesmo com evidência clara de posse, e só sai "vivo" o frame com evidência ativa contra bola morta.
>    - Numa simulação de 10 s de posse da casa com 3 s sem bola, a v1 decodifica tudo como `D`; a fórmula corrigida decodifica tudo como `H`.
>    - Era o pior erro possível para o spellF1.
> 2. **Dados externos = IDSSE** (7 jogos da Bundesliga com bola viva/morta e time com a posse **em cada frame**, aprovado pelos organizadores na issue #16). Ele substitui as amostras da Metrica com rótulos derivados de eventos. O gate de dados externos da v1 e a consulta à organização deixam de existir. Nesta versão, **G1 é o code freeze** (era o G3 na v1).
> 3. **Removidos:** o harness de mascaramento no BAR-ATH (sem verdade, premia suavização excessiva), o hard-EM das transições, o forward-backward e o cache por hash (o pré-processamento roda em ~25 s). A avaliação passa a ser feita direto no IDSSE degradado, onde existe verdade.
> 4. **Calendário refeito para 12 dias**, com lista de corte e envio de segurança.
> 5. **Especificações amarradas às assinaturas reais do código** e às armadilhas verificadas. Por exemplo: o loader do IDSSE no kloppy descarta todo `D` por padrão (`only_alive=True`).
>
> **v2.1 (05/11, sem restrição de tempo):** foram adicionadas quatro estratégias com ganho esperado. Todas são **candidatas**: só entram na versão final se passarem no protocolo da Seção 6.
> - Correção do atraso sistemático das fronteiras por tipo de transição, calibrada no IDSSE (F2-04, F2-05).
> - Teste de estresse com degradações mais difícil e mais fácil, contra a incerteza de domínio (F4-04, Seção 6).
> - Variantes com árvores de decisão (GBDT) para as evidências de posse e de bola morta (F1-07, F3-08).
> - Trilha exploratória com a câmera de transmissão para recuperar bola perdida, com portão de decisão (F4-05, F4-06).

---

## 0. Instruções para o agente que vai criar as issues

> **Executado em 06/11/2026:** labels, 6 milestones, 37 issues (#56 a #92) com responsáveis e as 58 dependências nativas foram criados. O mapeamento está em [`issues.md`](issues.md). O project board está em https://github.com/users/HachemAhmed/projects/2, com as visões "Tabela" e "Quadro" e os quatro colegas como colaboradores. O GitHub não permite ligá-lo à aba "Projects" do repositório, porque o projeto é de outra conta.

Você **não implementa nada**. Você só organiza o trabalho no GitHub a partir deste arquivo.

1. **Arquivos de contexto no repositório**, num único PR:
   - `docs/plano/00_contexto_global.md` = Seções 1, 2, 3, 6 e 7 e Apêndices A a E deste arquivo, com texto integral.
   - `docs/contracts.md` = Seção 2.4, com texto integral.
   - `docs/tasks/<ID>.md` = um por tarefa: as "Regras globais" (Seção 1.4) coladas no topo, depois o texto integral da tarefa (Seção 4) e por fim links para `docs/plano/00_contexto_global.md` e `docs/contracts.md`.
2. **Labels:** `front:F1` … `front:F5`, `type:impl`, `type:analysis`, `type:gate`, `type:doc`, `priority:P0`, `priority:P1`, `priority:P2`, `needs-local-run` (exige execução humana com os dados reais da partida), `status:blocked`.
3. **Milestones (com data de vencimento):** M0 dom 08/11 · M1 ter 10/11 · M2 qui 12/11 · M3 sex 13/11 · M4 sáb 14/11 · Entrega seg 16/11.
4. **Uma issue por tarefa: 37 no total** (36 tarefas + o gate G1). Título `[ID] título`. Corpo: resumo de até 5 linhas; link para `docs/tasks/<ID>.md`; campos front, prazo, esforço, prioridade, marco, "Bloqueada por" e "Bloqueia". As issues são escritas em português; o código é sempre em inglês.
5. **Dependências:** usar o recurso nativo de issue dependencies do GitHub ("Relationships → Blocked by"; REST `POST /repos/{owner}/{repo}/issues/{issue_number}/dependencies/blocked_by` com corpo `{"issue_id": <id numérico da bloqueadora>}`). Se não estiver disponível, usar o fallback: label `status:blocked` e a primeira linha do corpo `Bloqueada por #N, #M`. A Seção 5 lista **todas** as 58 arestas. Crie todas e verifique que não há ciclo.
6. **Project board** com colunas `Bloqueada · Pronta · Em andamento · Em revisão · Concluída` e campos front, prazo, esforço e prioridade.
7. **Ao terminar**, devolva uma tabela `ID → número da issue` e a lista de arestas criadas.
8. Se algo neste arquivo estiver ambíguo, **pergunte em vez de inventar**.

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
  - Autotreino com pseudo-rótulos no BAR-ATH: zona cinzenta nas regras (issue #2).
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
  - `exit_logit` (F2-07, D-025): com `None`, as transições são constantes e o código é o de antes, sem nenhuma alteração. Senão é um `float64[n]`, finito dentro dos tempos. No frame `t`, só as saídas do subestado terminal de H e de A mudam: `q_o' = q_o·e^{x_t}/Z_t`, `p' = p/Z_t`, `q_d' = q_d/Z_t`, com `Z_t = p + q_d + q_o·e^{x_t}` (em log, com `logaddexp`). D e a cadeia do `min_dwell` não mudam. A transição que **chega** ao frame `t` usa `x_t`, então o valor do primeiro frame de cada tempo e os de `half == 0` não são usados.
  - O C8 passa `exit_logit = beta · h`, com `h` o hazard da F1-08 (seção `hazard` do config), ou `None` quando `hazard.kind` é `none` ou `beta` é 0.
  - Durações (F2-08, D-027): `params.duration_stages` = `r_s` (config `decoder.duration_stages`). Para o estado `s`, `m_s − 1` subestados da cadeia do `min_dwell` (sem laço) e depois `r_s` estágios, cada um com laço `p_s` e avanço `1 − p_s`; só o último estágio (o subestado terminal) sai para outro estado, com `(1 − p_s)·exit_split[s, ·]`, e só ele recebe a distribuição inicial. `p_s = 1 − r_s/(mean_s − m_s + 1)` mantém a média; erro se `mean_s − m_s + 1 <= r_s`. Com `r_s = 1` a topologia, a ordem dos subestados e os valores são os de antes, byte a byte.
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

## 4. Frentes e tarefas

**Responsáveis** (definidos em 06/11/2026):

| Frente | Responsável | GitHub |
| :--- | :--- | :--- |
| **F1** Evidência de posse e avaliação | Fernando | `fernando-horita-siratuti` |
| **F2** Decodificação temporal e seleção | Hugo | `hugnarok` |
| **F3** Bola morta e documentação | Bernardo | `BernardoSilvaAndrade` |
| **F4** Dados externos (IDSSE) e trilha da transmissão | Antônio | `AntonioEscobarTorres` |
| **F5** Infraestrutura, integração e entrega (responsável pelo envio) | Ahmed | `HachemAhmed` |

O gate G1 (code freeze) é da F5.

Esforço: **P** ≤ 0,5 dia · **M** 1–2 dias. Prioridade: **P0** nunca cortar · **P1** cortar só pelos gatilhos da Seção 7 · **P2** primeira a cair.

---

### FRENTE F5 — Infraestrutura, integração e entrega

#### F5-01 — Esqueleto `modeling/`, contratos e configuração
**Prazo:** sex 06/11 · **Esforço:** M · **Prioridade:** P0 · **Marco:** M0 · **Bloqueada por:** — · **Bloqueia:** F5-02, F5-03, F1-01, F2-01, F3-01, F3-03, F3-05

**Objetivo:** criar o pacote `modeling/` com os contratos C0–C2, C4, C6, C7 e C9 e a infraestrutura de configuração, para as outras frentes programarem em paralelo contra interfaces fixas.

**Arquivos:**
- `modeling/__init__.py`, `paths.py`, `contracts.py`, `io.py`, `config.py`, `variants.py`;
- subpacotes vazios com `__init__.py`: `evidence/`, `deadball/`, `decode/`, `calibration/`, `scoring/`;
- `modeling/external/__init__.py` e `modeling/external/files.py`;
- `configs/baseline.yaml`, `requirements-modeling.txt`, `.python-version`, `docs/contracts.md`;
- `tests/test_modeling_contracts.py`.

**Especificação:**
1. **`paths.py`:**
   - `repo_root() -> Path`;
   - `data_dir()` = `Path(games.DATA_DIR)`;
   - `processed_dir()` = `data_dir()/"processed"`;
   - `ext_dir()` = `Path(os.environ.get("MTAH_EXT_DIR", "~/mtah-external")).expanduser().resolve()`, com `ValueError` se ficar dentro de `repo_root()`;
   - `outputs_dir()` = `repo_root()/"outputs"`.
2. **`contracts.py`:**
   - `STATES = ("H", "A", "D")` e `LABEL_CODES = {"H": 0, "A": 1, "D": 2, "X": 3}`, com um assert de que a ordem coincide com `pm.LABELS`;
   - `TEAM_LOGIT_MAX = 8.0` e `DEAD_LOGIT_MAX = 12.0`;
   - `validate_logits(team, dead, n)`: shape `(n,)`, `float32`, finitos, dentro dos limites; `ValueError` com mensagem clara;
   - `validate_force_d(mask, half)`: bool, shape `(n,)`, nenhum `True` onde `half == 0`;
   - `validate_labels(labels, half)`: dtype `<U1`, valores ⊂ {H, A, D, X}, `X` **se e somente se** `half == 0`.
3. **`io.py`:** `validate_master`, `load_master` (C1), `write_submission` (C6), `save_logits(team, dead, name)` e `load_logits(name)`.
4. **`config.py`:** `load_config(path)` e `config_hash(cfg)` (C7, esquema do Apêndice A.7). Validar tipos, chaves obrigatórias, chaves desconhecidas e faixas:
   - durações > `min_dwell`;
   - linhas de `exit_split` somando 1 (tolerância 1e-9);
   - `lam_* > 0`.
5. **`variants.py`:**
   - registros `TEAM_KINDS` e `DEAD_KINDS`, com decoradores `register_team(name)` e `register_dead(name)`;
   - assinatura comum `fn(master: pd.DataFrame, params: dict, *, holdout: str | None) -> np.ndarray` (float32);
   - registrar o tipo `"none"` (zeros) nos dois registros;
   - `load_kinds()` importa `modeling.evidence.team` e `modeling.deadball.dead`, se existirem, para popular os registros.
6. **`external/files.py`:** tudo do C9, inclusive `ExternalMatch.save(path)` / `ExternalMatch.load(path)` (`np.savez_compressed`, validando chaves, shapes e dtypes) e `load_external`.
7. **Dependências e arquivos de apoio:**
   - `requirements-modeling.txt` = `-r requirements-preprocessing.txt` + `pyyaml==6.0.2` + `scikit-learn==1.5.2` (combinação testada com Python 3.12 e numpy 2.0.2);
   - `.python-version` com `3.12`;
   - adicionar `outputs/` ao `.gitignore`;
   - `docs/contracts.md` = Seção 2.4 deste plano, literal.

**Testes** (`tests/test_modeling_contracts.py`):
- `validate_logits` rejeita NaN, inf, dtype errado, shape errado e valores fora dos limites;
- `validate_labels` rejeita X dentro dos blocos e não-X fora deles;
- round-trip de logits em `tmp_path` (com `monkeypatch.setenv("MTAH_DATA_DIR", ...)` **e** `monkeypatch.setattr(games, "DATA_DIR", ...)`);
- `write_submission` gera arquivo que passa em `pm.validate_submission` e falha com `n` errado;
- `load_config` rejeita chave desconhecida e chave ausente;
- `config_hash` é igual para dicts iguais com ordem diferente;
- `ext_dir()` dentro do repositório levanta erro;
- `ExternalMatch` faz round-trip.

Para master, usar `preprocessing.make_mock_master.generate_mock_master(seed=42)` (199.319 linhas, ~5 s), numa fixture com `scope="session"`.

**Aceite:** pytest verde; PR mergeado até sex 06/11.

**Armadilhas:** não fixar 199319; `video_frame` é 0-based; nenhuma função faz `fillna`.

#### F5-02 — Baseline, comando único e checagens da janela pública
**Prazo:** sáb 07/11 · **Esforço:** M · **Prioridade:** P0 · **Marco:** M0 · **Bloqueada por:** F5-01 · **Bloqueia:** F5-04, F1-03

**Objetivo:** ter cedo um CSV válido gerado pelo nosso pipeline e o comando único que todas as variantes vão usar.

**Arquivos:** `modeling/baseline.py`, `modeling/pipeline.py`, `modeling/make_submission.py`, `tools/check_public_window.py`, `tests/test_modeling_baseline.py`, `tests/test_modeling_pipeline.py`.

**Especificação:**
1. **`baseline.py::baseline_labels(master, params) -> np.ndarray["<U1"]`** (`params` = seção `baseline` do config), por bloco de `half`:
   - (a) `H` onde `d_diff < 0` e `A` onde `d_diff > 0`, só onde `d_diff` é finito.
   - (b) Os demais frames herdam o último rótulo H/A anterior no mesmo tempo. Os frames iniciais herdam o primeiro H/A do tempo. Se o tempo não tiver nenhum, `H`.
   - (c) `D` onde `out_of_pitch == 1`, ou `pre_kickoff`, ou (`gap_length >= d_gap_frames` e `mean_speed_home < slow_speed_mps` e `mean_speed_away < slow_speed_mps`). Comparações com NaN são falsas.
   - (d) Segmentos H/A com menos de `min_spell_frames` frames: varrer da esquerda para a direita e trocar pelo rótulo do segmento H/A adjacente anterior. Se o anterior for `D` ou não existir, usar o do segmento H/A adjacente posterior. Repetir até não mudar nada (no máximo 20 passadas). `D` nunca é alterado.
   - (e) `half == 0` → `X`.
2. **`pipeline.py::predict_labels(master, cfg, *, holdout=None)`** (C8):
   - `cfg["pipeline"] == "baseline"` → `baseline_labels`;
   - `"hmm"` → `NotImplementedError` até a F5-04;
   - sempre termina com `validate_labels`.
3. **`make_submission.py`** (CLI `python -m modeling.make_submission --config <yaml> [--out <csv>] [--skip-preprocessing] [--data-dir <dir>]`):
   - (0) No topo do módulo, **antes de importar numpy**, fazer `os.environ.setdefault(v, "1")` para `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` e `MKL_NUM_THREADS`.
   - (1) Sem `--skip-preprocessing`, chamar `preprocessing.run_all.run_pipeline(data_dir=...)` e abortar se o retorno for ≠ 0.
   - (2) `load_master()`.
   - (3) `predict_labels`.
   - (4) `write_submission`; caminho padrão `outputs/<cfg["name"]>.csv`.
   - (5) Imprimir nome e hash do config, SHA-256 do CSV e um resumo **dos rótulos previstos**: proporção de H/A/D nos frames avaliados e nº de posses por tempo.
4. **`tools/check_public_window.py --pred <csv> [--verbose]`** (execução local): lê `evaluation/gt/BAR-ATH_possession_gt_public.csv` e o master, e imprime só `PASS`/`FAIL` para:
   - (i) CSV válido (`pm.validate_submission`);
   - (ii) nenhum X nos frames avaliados, e só X fora deles;
   - (iii) média de `d_diff` menor nos frames GT-H que nos GT-A;
   - (iv) saldo `n_tags_favor_home − n_tags_favor_away` maior nos GT-H que nos GT-A;
   - (v) `centroid_home_x < centroid_away_x` na média dos 25 frames após cada pontapé (frames de `preprocessing.params.KICKOFF_FALLBACK`, os valores publicados);
   - (vi) nos frames vivos do GT público, acerto do time previsto > 0,5 (detecta H/A trocados);
   - (vii) fração prevista de `D` maior nos frames GT-D do que nos GT-vivos.
   
   **Não** calcula nem imprime spellF1 ou macroF1.

**Testes:**
- baseline sobre o mock: 199.319 rótulos, só H/A/D/X, X se e somente se `half == 0`;
- nenhum segmento H/A menor que `min_spell_frames` dentro de um tempo, salvo se o tempo for todo menor;
- `pre_kickoff` → D;
- duas execuções iguais;
- CLI com `--skip-preprocessing` e master mock salvo num `tmp_path` apontado por `MTAH_DATA_DIR` gera CSV válido;
- `check_public_window` com GT e master sintéticos, em casos PASS e FAIL.

**Execução local (humano):** `python -m modeling.make_submission --config configs/baseline.yaml`, depois `python tools/check_public_window.py --pred outputs/baseline.csv`. Colar na issue só os PASS/FAIL e o resumo de rótulos.

**Aceite:** CSV real passa em `evaluation/validate_submission.py`; checagens PASS.

**Armadilhas:** `pre_kickoff` é bool; o 1º tempo começa com posse, não com D.

#### F5-03 — Higiene do repositório e CI
**Prazo:** dom 08/11 · **Esforço:** P · **Prioridade:** P1 · **Marco:** M0 · **Bloqueada por:** F5-01 · **Bloqueia:** —

**Arquivos:** `tests/test_repo_hygiene.py`, `.github/workflows/tests.yml`, `.github/pull_request_template.md`, `docs/data_safety.md`, `.gitignore`.

**Especificação:**
1. **`.gitignore`:** garantir `data/*` (exceto `data/README.md`), `outputs/`, `*.parquet`, `*.npy`, `*.npz`, `*.pkl` e `*.mp4`. **Não** ignorar `configs/` nem `reports/`.
2. **`test_repo_hygiene.py`** (via `git ls-files`; pular se `git` estiver indisponível) falha se houver:
   - arquivo com extensão de dados (`.parquet .npy .npz .pkl .mp4 .xml`) fora de `tests/fixtures/`;
   - qualquer arquivo sob `data/` exceto `data/README.md`;
   - notebook `.ipynb` com outputs;
   - arquivo > 1 MB fora da allowlist (`evaluation/gt/*`, `evaluation/results/baseline_*.csv` e `models/**/*.joblib` até 5 MB).
3. **GitHub Actions:** em push e PR, Ubuntu, Python 3.12, `pip install -r requirements-modeling.txt` e `pytest -q`.
4. **Template de PR:** checklist da Definition of Done (1.5) mais a pergunta obrigatória "Este PR contém algum valor derivado dos dados da partida?".
5. **`docs/data_safety.md`:**
   - os dados ficam fora do repositório, via `MTAH_DATA_DIR`;
   - `.geminiignore` e configurações de IDE são camada extra, nunca a proteção principal;
   - agentes não rodam nada sobre os dados da partida;
   - checklist antes de tornar o repositório público.

**Aceite:** CI verde no PR. O teste de higiene falha se um `.parquet` for adicionado de propósito (verificar localmente e reverter).

#### F5-04 — Integração do motor HMM (v1) no comando único
**Prazo:** ter 10/11 · **Esforço:** M · **Prioridade:** P0 · **Marco:** M1 · **Bloqueada por:** F5-02, F1-02, F2-02, F3-02, F3-03 · **Bloqueia:** F5-05, F2-05, F3-06, F4-06

**Objetivo:** `predict_labels` com `pipeline: hmm` ponta a ponta, idêntico no BAR-ATH e no IDSSE.

**Especificação** — em `pipeline.py`, para `pipeline == "hmm"`:
1. `variants.load_kinds()`.
2. `t = TEAM_KINDS[cfg["team"]["kind"]](master, cfg["team"], holdout=holdout)`.
3. `d = DEAD_KINDS[cfg["dead"]["kind"]](master, cfg["dead"], holdout=holdout)`.
4. `validate_logits(t, d, len(master))`.
5. `force_d = build_force_d(master)` (F3-03).
6. `params = DecoderParams.from_config(cfg["decoder"])` (F2-01).
7. `scores = logits_to_logscores(t, d, params.lam_team, params.lam_dead, force_d)`.
8. `labels = decode_match(scores, half, force_d, params)` (F2-02).
9. `labels, post_stats = postprocess(labels, master, t, d, force_d, cfg["postprocess"], holdout=holdout)` (F2-04; até a F2-04 existir, a identidade com a imposição final). Registrar `post_stats` no log.
10. `validate_labels`.

Criar também `configs/v1.yaml` (Apêndice A).

**Testes:** ponta a ponta no mock com `configs/v1.yaml`:
- rótulos válidos;
- SHA-256 do CSV igual em duas execuções;
- `pre_kickoff` → D;
- nenhum X dentro dos blocos.

**Execução local (humano):** gerar `outputs/v1.csv`, validá-lo e rodar `tools/check_public_window.py`. Colar na issue os PASS/FAIL, o hash e o resumo dos rótulos.

**Aceite:** CSV v1 válido e checagens PASS.

#### F5-05 — Envio de segurança
**Prazo:** qua 11/11 · **Esforço:** P · **Prioridade:** P0 · **Marco:** M1 · **Bloqueada por:** F5-04, F5-08, F3-05 · **Bloqueia:** —

**Especificação:**
- Enviar o formulário com o melhor CSV válido disponível (v1), o rascunho do relatório exportado em PDF (`<team_name>.pdf`) e a URL do repositório.
- Vale a última resposta antes do prazo: este envio só garante que existe algo válido caso dê problema no fim.
- Registrar em `docs/decisions.md` a data e o SHA-256 do CSV enviado.
- **Lembrete obrigatório no registro:** o repositório precisa estar **público** antes de 17/11 em qualquer cenário (ver F5-09).

**Aceite:** comprovante de envio salvo; o CSV enviado é o arquivo validado (mesmo SHA-256).

#### F5-06 — Reprodução limpa e determinismo entre máquinas
**Prazo:** sáb 14/11 · **Esforço:** M · **Prioridade:** P0 · **Marco:** M4 · **Bloqueada por:** G1 · **Bloqueia:** F5-09

**Especificação:**
1. **`docs/reproducibility.md` e seção "How to run" do README:**
   - requisitos: Python 3.12 e `pip install -r requirements-modeling.txt`;
   - onde pôr os dados: `./data`, como no DATA.md oficial, ou `MTAH_DATA_DIR`;
   - comando único: `python -m modeling.make_submission --config configs/final.yaml --out outputs/<team_name>.csv`;
   - SHA-256 esperado do CSV.
2. **Primeira máquina:** um membro que **não** implementou o motor clona o repositório do zero numa pasta nova e cria um venv novo. Põe os dados em `./data` dessa pasta (o caminho que os organizadores vão usar), roda o comando e compara o SHA-256.
3. **Segunda máquina:** outro membro repete o procedimento num computador diferente.

Se der diferença: corrigir (exceção ao freeze aprovada por 2 pessoas) e repetir.

**Aceite:** SHA-256 idêntico nas duas máquinas e igual ao do candidato; o comando roda sem nenhum ajuste manual.

#### F5-07 — Relatório técnico final (inglês)
**Prazo:** dom 15/11 · **Esforço:** M · **Prioridade:** P0 · **Marco:** Entrega · **Bloqueada por:** G1, F3-05 · **Bloqueia:** F5-09

**Especificação:** completar `docs/technical_report/report.md`, com 4–6 páginas, em inglês:
1. **Approach and why.**
2. **Assets used and dropped:**
   - ATD tático: base de tudo;
   - Smart Tagging: só como evidência fraca de bola parada;
   - homografia e vídeos: não usados no modelo, e por quê;
   - tracking da transmissão: usado ou "testado e descartado", conforme a trilha F4-05/F4-06, com o motivo.
3. **External data:**
   - IDSSE, com citação completa e licença CC BY 4.0;
   - o que foi usado e como foi degradado;
   - nenhum rótulo é do BAR-ATH.
4. **Possession rules implemented:** bola solta = persistência; duelos; reinícios; pré-pontapé; nenhum X dentro dos tempos.
5. **Validation protocol:** IDSSE com uma partida de fora de cada vez, regra de parcimônia e teste de estresse; a janela pública nunca foi usada para escolher. Listar também as variantes testadas e descartadas pelo protocolo, com o motivo.
6. **Limitations:**
   - mudança de domínio DFL→Metrica;
   - convenção de posse possivelmente diferente;
   - recall das tags;
   - casos de borda do filtro da bola.
7. **Reproduction:** comando, versões e SHA-256.
8. **How the provided game was used (compliance):**
   - nenhum rótulo humano e nenhum treino no BAR-ATH;
   - estatísticas **não rotuladas** do tracking usadas só para tornar a degradação do IDSSE realista (perfil de lacunas opcional, checagem de realismo) — uso não supervisionado, permitido pelo RULES.md;
   - constantes publicadas pelos organizadores (blocos dos tempos e frames dos pontapés, DATA.md/FAQ.md);
   - a janela pública usada só para checagens PASS/FAIL de correção, nunca para escolher nada;
   - nenhum dado da partida enviado a LLMs em nuvem.

Figuras só agregadas e só de dados externos. Exportar `<team_name>.pdf` (pandoc ou similar) e conferir a renderização. Todos os membros revisam.

**Aceite:** cobre todos os itens exigidos pelo SUBMISSION.md; nenhum dado da partida; nomes de arquivo corretos.

#### F5-08 — Formulário e nome da equipe
**Prazo:** sex 06/11 · **Esforço:** P · **Prioridade:** P0 · **Marco:** M0 · **Bloqueada por:** — · **Bloqueia:** F5-05, F5-09

**Especificação:**
- Confirmar que o e-mail com a URL do formulário chegou a todos.
- Testar o acesso com a conta Google do responsável pelo envio.
- Fixar o nome exato da equipe (igual ao do cadastro) e o `<team_name>` dos arquivos.
- Registrar tudo em `docs/decisions.md`.
- Se o e-mail não chegou: escrever para hackathon.talentarena@mobileworldcapital.com.

**Aceite:** nome da equipe, responsável pelo envio e acesso ao formulário confirmados.

#### F5-09 — Pacote final, repositório público e envio
**Prazo:** seg 16/11 · **Esforço:** P · **Prioridade:** P0 · **Marco:** Entrega · **Bloqueada por:** F5-06, F5-07, F5-08 · **Bloqueia:** —

**Especificação:**
1. **Gerar e validar o CSV:** gerar `<team_name>.csv` a partir da tag `submission-v1` com o comando único. Validar **o arquivo exato** com `evaluation/validate_submission.py`. Conferir que o SHA-256 é o mesmo da F5-06.
2. **Auditoria antes de tornar público:**
   - `git log --all --stat`: nenhum arquivo de dados em todo o histórico;
   - varrer issues, PRs e comentários atrás de valores derivados dos dados da partida, e editar/apagar se houver;
   - README com instruções de execução e citação do IDSSE.
3. **Publicar:** tornar o repositório público e abrir a URL numa janela anônima.
4. **Enviar** o formulário (CSV, URL e PDF).
5. **Registrar:** guardar o comprovante e anotar em `docs/decisions.md`.

**Aceite:** comprovante registrado; repositório público; o arquivo enviado é o arquivo validado.

---

### FRENTE F4 — Dados externos (IDSSE)

#### F4-01 — IDSSE: download, conversão e rótulos por frame
**Prazo:** dom 08/11 · **Esforço:** M · **Prioridade:** P1 · **Marco:** M0 · **Bloqueada por:** — · **Bloqueia:** F4-03, F4-04, F2-03

**Objetivo:** transformar as 7 partidas do IDSSE em `ExternalMatch` (C9), com rótulos H/A/D por frame.

**Arquivos:** `modeling/external/idsse.py`, `tests/test_modeling_idsse.py`, `tests/fixtures/idsse_mini_tracking.xml`, `docs/external_data.md`, `reports/idsse_overview.md`. Até a F5-01 ser mergeada, criar só os `__init__.py` necessários; o esqueleto completo vem da F5-01.

**Especificação:**
1. **`download(match_id) -> tuple[Path, Path]`:**
   - **Fonte: o espelho no Hugging Face** `pysport/idsse-data`, com a revisão fixada `a715a38dfbaf5f58e431727c2b78d174101a703c` (público, sem login, CC BY 4.0). URL: `https://huggingface.co/datasets/pysport/idsse-data/resolve/<revisão>/<arquivo>`.
   - **Não usar o figshare**, nem `kloppy._providers.sportec.get_IDSSE_url` ou `sportec.load_open_tracking_data`. Ele tem proteção anti-robô (AWS WAF) e responde HTTP 202 com uma página HTML no lugar do arquivo (verificado em 05/11).
   - Arquivos de cada partida `m`:
     - meta: `DFL_02_01_matchinformation_DFL-COM-<com>_DFL-MAT-<m>.xml`;
     - posições: `DFL_04_03_positions_raw_observed_DFL-COM-<com>_DFL-MAT-<m>.xml`;
     - `<com>` = `000001` para J03WMX e J03WN1, e `000002` para as outras cinco;
     - os eventos (`DFL_03_02_events_raw_...`) não são necessários.
   - Baixar com `urllib.request.urlretrieve` para `ext_dir()/"raw"/`, mantendo o nome original. Pular se o arquivo já existe.
   - Conferir que o arquivo começa com `<?xml`; se não começar, apagar e falhar com erro claro.
   - Tamanho: ≈2,5 GB no total (de 333 a 399 MB por arquivo de posições); os de meta são pequenos.
   - Registrar tamanho e SHA-256 de cada arquivo em `docs/external_data.md`, junto com a fonte e a revisão.
2. **`load_match(match_id) -> ExternalMatch`:**
   - `ds = kloppy.sportec.load_tracking(meta_data=meta_path, raw_data=tracking_path, only_alive=False)`. **`only_alive=False` é obrigatório**: o padrão `True` descarta todos os frames de bola morta.
   - Coordenadas no sistema padrão do provedor (metros, origem no **centro** do campo).
   - Campo `L, W = ds.metadata.pitch_dimensions.pitch_length, ds.metadata.pitch_dimensions.pitch_width` (atributos `PitchX`/`PitchY`, próprios de cada estádio).
   - Normalizar: `x_n = x / L + 0.5` e `y_n = 0.5 − y / W`.
3. **Slots:**
   - jogadores da casa (`ds.metadata.teams[0]`) nos slots 0–23 e visitantes (`teams[1]`) nos slots 24–45, na ordem da primeira aparição no tracking;
   - `slot_team_ids` = 0 para casa e 1 para visitante;
   - erro se uma equipe tiver mais jogadores do que slots;
   - velocidade `speed` (m/s) no 3º canal.
4. **Layout temporal** conforme o C9. Os frames de cada tempo têm de ser contíguos: frames ausentes viram linhas `NaN` com rótulo X, e são contados e reportados.
5. **Rótulos (C0):**
   - `D` onde `frame.ball_state == BallState.DEAD`;
   - senão, `H` se `frame.ball_owning_team` é a equipe da casa e `A` se é a visitante.
   - **Validação obrigatória.** O kloppy mapeia `BallPossession` ausente ou 0 para "visitante" sem avisar. Por isso:
     - escrever `parse_ball_labels(tracking_path)`, que lê o XML com `lxml.etree.iterparse` e, no `FrameSet` com `TeamId == "BALL"`, coleta de cada `Frame` os atributos `N`, `BallStatus` e `BallPossession`;
     - todo frame vivo com `BallPossession ∉ {1, 2}` recebe rótulo X, e esses frames são contados;
     - em todos os outros frames, os rótulos do kloppy e os do XML têm de coincidir (assert).
6. **Bola:** `ball[:, 0:2]` normalizada e `ball[:, 2]` = velocidade; `ball_z` (metros) vai separado.
7. **Gravação:** salvar em `arrays_path(match_id)`.
   - Processar **uma partida por vez**: a máquina de referência tem ≈15 GB de RAM.
   - Checar o espaço livre em disco antes.
   - Opção `--delete-raw` apaga os XMLs depois de converter.
8. **CLI:** `python -m modeling.external.idsse --matches all|<ids separados por vírgula> [--delete-raw]`.
9. **`reports/idsse_overview.md`** (dados públicos, pode ser commitado), por partida:
   - frames por tempo;
   - % de frames vivos;
   - % de H, A e D;
   - frames X por validação;
   - dimensões do campo.

**Testes** (sem rede):
- separar a conversão em `frames_to_arrays(records, ...)`, que recebe registros simples, e testá-la com registros sintéticos: normalização, layout com intervalo, slots, rótulos;
- `parse_ball_labels` com o XML mínimo em `tests/fixtures/` (2 tempos × 10 frames, incluindo um frame vivo com `BallPossession="0"`), verificando X nesse frame;
- `ExternalMatch` com todas as chaves e dtypes;
- um smoke test com rede (`@pytest.mark.network`, pulado por padrão), que carrega uma partida real com `limit=500`.

**Execução:** o IDSSE é público; o próprio agente pode baixar e converter as 7 partidas.

**Aceite:**
- 7 `.npz` gerados;
- `reports/idsse_overview.md` escrito;
- atribuição CC BY 4.0 em `docs/external_data.md` e no README (seção "External data"). Citação: Bassek, M., Rein, R., Weber, H., Memmert, D. (2025). *An integrated dataset of spatiotemporal and event data in elite soccer.* Scientific Data 12, 195. https://doi.org/10.1038/s41597-025-04505-y

**Armadilhas:** `only_alive`; `BallPossession` ausente vira "visitante" no kloppy; origem das coordenadas no centro; dimensões do campo por estádio.

#### F4-02 — API em memória do pré-processamento (sem mudar a saída do BAR-ATH)
**Prazo:** seg 09/11 · **Esforço:** M · **Prioridade:** P1 · **Marco:** M1 · **Bloqueada por:** — · **Bloqueia:** F4-04

**Objetivo:** permitir rodar **exatamente** o mesmo pré-processamento (A-V1, B, D) sobre arrays de outra partida, com outro tamanho de campo e sem tags.

**Arquivos:** `preprocessing/prep_proximities.py`, `preprocessing/build_master_table.py`, `preprocessing/export_tagging.py`, o novo `preprocessing/pipeline_api.py` e `tests/test_pipeline_api.py`.

**Especificação:**
1. **`prep_proximities.build_proximities_table(inputs, pitch=(PITCH_L, PITCH_W))`:** repassar `length=pitch[0], width=pitch[1]` para `pitch_features`. Hoje ela usa sempre as dimensões do BAR-ATH; o comportamento padrão continua igual.
2. **`export_tagging.empty_tagging_dataframe(blocks, n_frames) -> pd.DataFrame`:** mesmas colunas, ordem e dtypes de `build_tagging_dataframe`:
   - `video_frame` int32 e `half` int8;
   - `is_set_piece` = False;
   - strings vazias com dtype `string`;
   - distâncias float32 NaN;
   - `n_tags_favor_*` int16 zero;
   - `pre_kickoff` = False.
3. **`build_master_table.assemble_master(tracking, proximities, tagging, homography, blocks, n_frames) -> pd.DataFrame`:**
   - recebe DataFrames em memória; `homography=None` gera master sem `reproj_error`;
   - faz as mesmas checagens de `build_master`: chaves, cruzamento de `is_graded`/`frame_in_half`, casts e nenhum valor fora dos tempos;
   - `build_master(processed_dir, ...)` passa a apenas ler os arquivos (com `check_metadata`) e delegar a `assemble_master`.
4. **`pipeline_api.build_master_from_arrays(players, ball, slot_team_ids, blocks, pitch) -> pd.DataFrame`**, encadeando:
   1. `prepare_tracking(players, ball, slot_team_ids, blocks=blocks, pitch=pitch, kickoffs={h: f0 for h, f0, _ in blocks}, stage="V1")`;
   2. `assemble_inputs(frame, slot_teams, "V1", n_frames=T, n_slots=players.shape[1])`;
   3. `build_proximities_table(inputs, pitch=pitch)`;
   4. `empty_tagging_dataframe(blocks, T)`;
   5. `assemble_master(..., homography=None, blocks=blocks, n_frames=T)`.

**Testes:**
- (a) **Teste "golden", escrito antes de mudar qualquer código:**
  - gerar, com os helpers sintéticos de `tests/test_build_master_table.py`, um diretório processado em `tmp_path`;
  - gravar o master com `write_master` e fixar no teste o SHA-256 obtido;
  - depois do refactor, o hash tem de ser idêntico.
- (b) **`build_master_from_arrays` com arrays sintéticos:** T = 600 (tempo 1 = 0–249, intervalo 250–349, tempo 2 = 350–599), 3+3 jogadores, bola em movimento.
  - Saem 59 colunas na ordem do schema (sem `reproj_error`), `video_frame` 0..T−1, `half` correto e NaN fora dos tempos.
  - `out_of_pitch` respeita o `pitch` passado: com a bola em x = 106,2 m e margem de 1 m, o resultado é "fora" com `pitch=(105, 68)` e "dentro" com `pitch=(105.3, 68)`.
- (c) A suíte inteira continua verde.

**Execução local (humano):** `python -m preprocessing.run_all --processed-dir <pasta temporária>` e comparar o SHA-256 dos 7 arquivos com os atuais. Colar na issue só `IDENTICAL`/`DIFFERENT`.

**Aceite:** 7 × IDENTICAL; testes verdes.

**Armadilhas:** não mudar a ordem nem os dtypes de nenhuma coluna; `assemble_inputs` só confere os blocos publicados quando `n_frames == 199319`.

#### F4-03 — Degradação realista do IDSSE
**Prazo:** ter 10/11 · **Esforço:** M · **Prioridade:** P1 · **Marco:** M1 · **Bloqueada por:** F4-01 · **Bloqueia:** F4-04

**Objetivo:** dar ao tracking limpo do IDSSE o ruído do ATD, para que a calibração e a seleção meçam o que importa.

**Arquivos:** `modeling/external/degrade.py`, `configs/degrade_default.yaml` (Apêndice A.6), `tools/ball_gap_profile.py` (execução local), `tests/test_modeling_degrade.py`.

**Especificação:** `degrade(match: ExternalMatch, cfg: dict, seed: int) -> ExternalMatch`.
- Nunca altera a entrada e não mexe nos rótulos.
- Trabalha em metros (`x_m = x_n · L`, `y_m = y_n · W`) e devolve normalizado.
- Usa um único `rng = np.random.default_rng(seed)` e aplica as operações sempre nesta ordem:

**Bola:**
1. **Bola alta:** em cada trecho com `ball_z > z_air_m`, apaga o trecho inteiro (mais `air_margin_frames` de cada lado) com probabilidade `p_air`.
2. **Aglomeração:** em frames com pelo menos `crowd_n` jogadores a menos de `crowd_r_m` da bola, inicia uma lacuna com taxa `crowd_hazard` por frame.
3. **Lacunas de fundo:** com taxa `bg_hazard`, até a fração de frames com bola visível dentro dos tempos atingir `target_ball_visible` ± 0,01 (0,593, o valor público do DATA.md). Ajustar `bg_hazard` por bisseção determinística (no máximo 30 iterações).
   - Os comprimentos de lacuna das etapas 2 e 3 são sorteados do perfil de lacunas (abaixo).
4. **Ruído gaussiano** na posição (`ball_noise_m`).
5. **Fantasmas:** rajadas com duração inteira uniforme em `phantom_len_frames`, numa posição fixa sorteada em [−5, L+5] × [−5, W+5] m, até cobrir `phantom_frac` dos frames dos tempos.
6. Velocidade da bola = NaN.

**Jogadores:**
7. **Apagões por jogador**, num processo alternado: tempo visível ~ exponencial de média `player_visible_mean_s`; tempo ausente ~ exponencial de média `player_missing_mean_s`. Ajustar `player_missing_mean_s` por bisseção até a média de jogadores rastreados nos tempos ser `target_tracked_mean` ± 0,3 (20,5, valor público).
8. **Fragmentação:** ao reaparecer, o jogador vai para o slot livre de menor índice da mesma equipe; o slot antigo fica NaN.
9. **Ruído gaussiano** (`player_noise_m`).
10. **Duplicatas:** cópia de um jogador, deslocada por N(0, `dup_offset_m`), num slot livre da mesma equipe, por `dup_len_s` (uniforme), até `dup_frac` dos frames dos tempos terem mais de 22 objetos.

**Perfil de lacunas:**
- Padrão: a mistura paramétrica documentada em `configs/degrade_default.yaml`.
- Opcional: `tools/ball_gap_profile.py` (execução local, humano) grava `<data dir>/processed/ball_gap_profile.json` com o histograma de comprimentos de lacuna do BAR-ATH dentro dos tempos. Se o arquivo existir, `degrade` usa ele.
- O script imprime só `OK`. O JSON nunca é commitado, colado ou mostrado a agentes.

**Testes:**
- mesma semente → arrays idênticos; semente diferente → arrays diferentes;
- rótulos inalterados e entrada não modificada;
- numa partida sintética, fração de bola visível e média de rastreados dentro das tolerâncias;
- fantasmas e duplicatas na fração esperada.

**Aceite:** testes verdes; config padrão documentado.

#### F4-04 — Mestres externos (limpos e degradados) e checagem de realismo
**Prazo:** qua 11/11 às 12h · **Esforço:** M · **Prioridade:** P1 · **Marco:** M2 · **Bloqueada por:** F4-01, F4-02, F4-03 · **Bloqueia:** F1-05, F3-04, F2-05

**Arquivos:** `modeling/external/build.py`, `tools/compare_external_realism.py`, `reports/external_masters.md`, `tests/test_modeling_external_build.py`.

**Especificação:**
1. **CLI `python -m modeling.external.build --matches all --degrade none|default|harder|easier --seed 0`.** Os modos `harder` e `easier` (Apêndice A.6) servem ao teste de estresse da Seção 6. Para cada partida:
   - carrega o `.npz`;
   - aplica `degrade` com a semente `seed + IDSSE_MATCHES.index(match)`, se pedido;
   - chama `preprocessing.pipeline_api.build_master_from_arrays`;
   - grava o master e os rótulos nos caminhos do C9.
2. **`reports/external_masters.md`** (dados públicos), por partida e por modo:
   - % de bola observada, % `ball_filled` e % `ball_jump_removed`;
   - média de `n_tracked_home + n_tracked_away`;
   - % de D.
3. **`tools/compare_external_realism.py [--verbose]`** (execução local, humano): compara as mesmas estatísticas do master do BAR-ATH (frames avaliados) com a média dos mestres degradados, e imprime `PASS`/`FAIL` por estatística, com tolerância relativa de ±20%.
   - Se houver FAIL, a pessoa ajusta `configs/degrade_default.yaml` e reconstrói.
   - Divergências que restarem vão para `docs/decisions.md`.
4. **Distribuição para o resto da equipe.** F1, F2 e F3 precisam dos arquivos externos a partir de ter 10/11 (a F2-03 usa os `.npz`) e qua 11/11 (mestres). Em `docs/external_data.md`, documentar a forma padrão de obtê-los, que é regenerar localmente:
   ```
   export MTAH_EXT_DIR=~/mtah-external
   python -m modeling.external.idsse --matches all --delete-raw
   python -m modeling.external.build --matches all --degrade none --seed 0
   python -m modeling.external.build --matches all --degrade default --seed 0
   python -m modeling.external.build --matches all --degrade harder --seed 0
   python -m modeling.external.build --matches all --degrade easier --seed 0
   ```
   - Publicar no mesmo documento o SHA-256 esperado de cada `.npz`, master e rótulo.
   - Implementar `python -m modeling.external.build --verify`, que compara os arquivos locais com esses hashes e imprime `IDENTICAL`/`DIFFERENT`. Diferenças de último bit entre CPUs são possíveis e irrelevantes para a calibração, mas devem ser avisadas no grupo.
   - **Plano B**, se o download atrasar alguém: a pessoa da F4 compacta `ext_dir()/arrays` e `ext_dir()/processed` com o README de atribuição dentro e compartilha numa pasta da equipe. A licença CC BY 4.0 permite.
   - Esses arquivos **nunca** entram no repositório.

**Testes:** com uma partida sintética pequena no formato C9, o build gera um master que passa em `validate_master` e rótulos com X exatamente onde `half == 0`.

**Aceite:** 28 mestres (7 partidas × 4 modos) gerados; realismo PASS ou divergências documentadas; comandos de regeneração e hashes publicados em `docs/external_data.md`.

#### F4-05 — Trilha exploratória: viabilidade da câmera de transmissão (portão GO/NO-GO)
**Prazo:** qui 12/11 às 12h · **Esforço:** M · **Prioridade:** P2 · **Marco:** M2 · **Bloqueada por:** — · **Bloqueia:** F4-06

**Começar só depois de entregar a F4-04:** o caminho crítico tem prioridade.

**Contexto:**
- A bola do tracking tático some em 41% dos frames avaliados.
- O tracking da transmissão (TV) tem sua própria bola, presente em 50,5% dos seus frames. Mas está em outra linha do tempo, com cortes e replays, e não há tabela de correspondência (DATA.md).
- Mapear as duas linhas do tempo por algoritmo é permitido ("Everything is fair game, in any combination, as input to an algorithm").
- Se a TV enxergar a bola em parte dos frames em que a câmera tática a perde, o modelo ganha evidência justamente nos lances disputados.

**Execução:** são dados da partida. O agente escreve código e testes sintéticos; a pessoa roda no próprio terminal (regras 1.4.1 e 1.4.2).

**Arquivos:** `modeling/broadcast/__init__.py`, `modeling/broadcast/align.py`, `tools/broadcast_feasibility.py`, `tests/test_modeling_broadcast_align.py`. Em `preprocessing/prep_tracking.py`, só um parâmetro novo, `n_slots=N_SLOTS`, em `read_atd` (a transmissão tem 51 slots), com o padrão inalterado.

**Especificação:**
1. **Leitura:**
   - transmissão: `paths = games.asset_paths(source="broadcast")`; `read_atd(paths["atd_raw"], paths["atd_meta"], n_frames=games.source_frames("broadcast"), n_slots=51)`; mais o `valid` da homografia da transmissão;
   - tático: `read_atd` padrão, com coordenadas cruas, **não** orientadas.
2. **`align_broadcast(tactical, broadcast) -> Mapping`**, em que `Mapping` = vetor `int32[T_transmissão]` com o frame tático correspondente (−1 se não houver), mais a orientação escolhida:
   - **orientação:** testar identidade e rotação de 180° (`x → 1 − x`, `y → 1 − y`) e ficar com a de menor custo global;
   - **alinhamento grosso:** correlação cruzada das posições da bola (x e y em metros) a 5 fps, só onde as duas câmeras têm bola, sobre todos os deslocamentos possíveis;
   - **refino por blocos de 2 s** (50 frames da transmissão, com homografia válida e pelo menos 6 jogadores visíveis):
     - escolher o deslocamento inteiro, em ±25 frames em volta do grosso (ou do bloco anterior), que minimiza a mediana da distância ao jogador tático mais próximo **da mesma equipe**;
     - aceitar o bloco se o custo for ≤ `align_max_cost_m` (1,5 m) e o segundo melhor deslocamento (a mais de 10 frames) custar pelo menos 1,5 × o melhor;
   - **segmentos ao vivo:** blocos consecutivos com deslocamentos que diferem no máximo 2 frames. Descartar segmentos com menos de 5 blocos (replays e cortes produzem saltos).
3. **`tools/broadcast_feasibility.py [--verbose]`**, dentro dos tempos avaliados, imprime `PASS`/`FAIL` para:
   - (i) o mapeamento cobre pelo menos 50% dos frames;
   - (ii) custo mediano de pareamento de jogadores ≤ 1,0 m;
   - (iii) a bola da transmissão existe em pelo menos 15% dos frames táticos **sem** bola (`ball_observed == 0`): é o ganho potencial;
   - (iv) onde as duas câmeras veem a bola, distância mediana ≤ 2,0 m.
   
   **GO** se as quatro passarem. Números só com `--verbose`, para a pessoa.
4. **Registro:** anotar a decisão (GO/NO-GO, com os PASS/FAIL) em `docs/decisions.md`. Se for NO-GO, fechar a F4-06 como "não será feita" e citar no relatório como estratégia testada e descartada, com o motivo.

**Testes (sintéticos):**
- gerar tracking tático com 22 jogadores e bola;
- criar uma "transmissão" que mostra só os jogadores perto da bola, com dois segmentos ao vivo de deslocamentos conhecidos, um bloco de replay (frames anteriores repetidos) e ruído;
- `align_broadcast` deve recuperar os deslocamentos com erro de no máximo 1 frame, descartar o replay e acertar a orientação, inclusive no caso girado.

**Aceite:** decisão registrada até qui 12/11 às 12h.

#### F4-06 — Trilha exploratória: fusão da bola da transmissão (se GO)
**Prazo:** sex 13/11 às 12h · **Esforço:** M · **Prioridade:** P2 · **Marco:** M3 · **Bloqueada por:** F4-05, F5-04 · **Bloqueia:** F2-06

**Arquivos:** `modeling/broadcast/fuse.py`; alterações em `preprocessing/prep_tracking.py`, `preprocessing/run_all.py` e `modeling/make_submission.py`; `configs/degrade_fused.yaml`; `tests/test_modeling_broadcast_fuse.py`.

**Especificação:**
1. **`fused_ball(tactical_ball, broadcast_ball, mapping) -> ball (T, 3)`:**
   - nos frames táticos **sem** bola cujo frame de transmissão mapeado tem bola, preencher com a posição da transmissão, na orientação escolhida;
   - **nunca** sobrescrever uma observação tática;
   - a velocidade (3º canal) fica NaN.
2. **Pré-processamento com bola fundida:** `python -m preprocessing.run_all --ball-source fused --processed-dir <data dir>/processed_fused`.
   - A bola fundida entra **antes** do filtro de saltos (A4), que continua valendo para ela.
   - Com `--ball-source tactical` (o padrão), a saída tem de continuar bit a bit idêntica: teste de regressão e checagem local de hashes, como na F4-02.
3. **Comando único:** chave `ball_source: tactical | fused` no config (Apêndice A.7). Com `fused`, `make_submission` roda o pré-processamento com a bola fundida e carrega o master de `processed_fused/`.
4. **Estimativa do ganho no IDSSE (proxy):**
   - a pessoa mede localmente a fração de bola visível nos frames avaliados do master fundido e grava esse número em `configs/degrade_fused.yaml`, uma cópia da `default` só com `target_ball_visible` trocado;
   - a F4 constrói os mestres `fused` e avalia a melhor candidata corrente em `default` × `fused`;
   - o ganho pareado estima quanto vale ver mais bola.
5. **Critérios para a Seção 6** (todos obrigatórios):
   - o master fundido passa nos testes de contrato (`--master`);
   - `tools/check_public_window.py` dá PASS com ele;
   - as checagens da F4-05 dão PASS;
   - o ganho do proxy é maior que 1 EP.

**Testes:**
- a fusão nunca sobrescreve frames táticos observados;
- só preenche frames mapeados;
- com `ball_source: tactical`, o resultado é idêntico ao atual.

**Aceite:** variante `fused` disponível até sex 13/11 às 12h, com os critérios avaliados; senão, fechar como "não será feita".

---

### FRENTE F1 — Evidência de posse (`team_logit`) e avaliação

#### F1-01 — Features de controle
**Prazo:** dom 08/11 · **Esforço:** M · **Prioridade:** P0 · **Marco:** M0 · **Bloqueada por:** F5-01 · **Bloqueia:** F1-02

**Arquivos:** `modeling/evidence/control.py`, `tests/test_modeling_control.py`.

**Especificação:** `build_control_features(master, params) -> pd.DataFrame`, com o mesmo índice do master. Colunas float32, exceto onde indicado; NaN = sem evidência:
- `ev_ball_valid` (bool): `ball_x` e `ball_y` finitos.
- `ev_ball_conf`: 1,0 se `ball_observed == 1`; `ball_conf_filled` se `ball_filled == 1`; 0,0 nos demais casos.
- `ev_dmin = np.fmin(d_near_home, d_near_away)`; NaN se ambos forem NaN.
- `ev_prox_w = exp(−max(0, ev_dmin − r_ctrl) / l_prox)`; NaN se `ev_dmin` for NaN.
- `ev_rel_near`: `speed_rel_home` se `d_near_home <= d_near_away` (empate → casa), senão `speed_rel_away`; NaN se indisponível.
- `ev_rel_w = 1 / (1 + (ev_rel_near / v_rel)²)`. Vale `rel_w_when_nan` quando `ev_rel_near` é NaN com bola válida, e NaN quando a bola não é válida.
- `ev_contest` (0/1 em float32; NaN sem bola): 1 se `n_home_r2 >= 1` e `n_away_r2 >= 1`.

**Testes:**
- pesos dentro de [0, 1];
- `ev_prox_w` decrescente em `ev_dmin` e `ev_rel_w` decrescente em `ev_rel_near`;
- NaN quando a bola é NaN;
- trocar as colunas home/away mantém `ev_dmin`, `ev_prox_w` e `ev_contest`;
- nenhum `fillna`.

#### F1-02 — `team_logit` v1 (prior)
**Prazo:** seg 09/11 · **Esforço:** P · **Prioridade:** P0 · **Marco:** M1 · **Bloqueada por:** F1-01 · **Bloqueia:** F1-04, F5-04

**Arquivos:** `modeling/evidence/team.py`, `tests/test_modeling_team.py`.

**Especificação:** `@register_team("rules_v1") team_logit_rules_v1(master, params, *, holdout=None)`, com os parâmetros do Apêndice A.1:
```
t = −kappa · tanh(d_diff / sigma_d) · ev_prox_w · ev_rel_w · ev_ball_conf · (contest_factor se ev_contest == 1, senão 1)
```
- `t = 0` onde a bola não é válida, onde `d_diff` é NaN e onde `half == 0`.
- Recorte em ±`kappa`; saída float32.
- Nenhuma dependência de frames vizinhos.

**Testes:**
- casa mais perto → `t > 0`;
- trocar home↔away inverte o sinal;
- NaN → 0;
- `|t|` cresce com `|d_diff|`, com o resto fixo;
- limitado por `kappa`;
- uma mudança de `d_diff` no frame t0 muda `t` exatamente em t0.

#### F1-03 — Harness de avaliação (métrica oficial, partidas externas)
**Prazo:** ter 10/11 · **Esforço:** M · **Prioridade:** P1 · **Marco:** M1 · **Bloqueada por:** F5-02 · **Bloqueia:** F1-05, F3-04, F2-05

**Arquivos:** `modeling/scoring/metrics.py`, `modeling/scoring/external_eval.py`, `tests/test_modeling_scoring.py`.

**Dados externos:** não são necessários para esta tarefa, cujos testes são sintéticos. Quando os mestres existirem (F4-04, qua 11/11), quem tiver os arquivos roda o CLI com `configs/baseline.yaml` e `configs/v1.yaml` e commita os relatórios em `reports/external/`. É o primeiro número de eficácia da equipe.

**Especificação:**
1. **`metrics.score(gt_codes, pred_codes) -> dict`:** chama `pm.evaluate(gt_codes, pred_codes)` (não reimplementar nada) e devolve:
   - `spell_f1_mean`, `macro_f1`, `alive_dead_f1`;
   - `trans_f1_25 = r["transitions"][25]["f1"]` e `mean_abs_latency_25`;
   - `n_gt_spells` e `n_pred_spells`.
2. **`external_eval.evaluate_config(cfg, degrade="default", matches=IDSSE_MATCHES) -> pd.DataFrame`.** Para cada partida:
   - `master, gt = load_external(match, degrade)`;
   - `pred = predict_labels(master, cfg, holdout=match)`;
   - `score(gt, pm.label_codes(pred))`.
   
   O resultado tem uma linha por partida, mais uma linha `mean` e uma linha `se` (desvio-padrão amostral / √n).
3. **`external_eval.paired_delta(df_a, df_b, metric="spell_f1_mean") -> (mean_delta, se_delta)`**, pareado por partida.
4. **CLI `python -m modeling.scoring.external_eval --config <yaml> --degrade default|none`:** imprime a tabela e grava `reports/external/<cfg name>__<degrade>.md` (dados externos; pode ser commitado).

**Testes:**
- previsão perfeita → spellF1 = 1,0;
- deslocar a previsão em 0, 3, 8 e 25 frames → spellF1 não crescente;
- `score` coincide com `pm.evaluate`, campo a campo;
- `evaluate_config` com 2 partidas sintéticas no formato C9 (monkeypatch de `load_external`) e config baseline.

**Armadilhas:** o vetor de verdade tem X no intervalo entre os tempos; sem isso, as posses atravessariam o intervalo.

#### F1-04 — Evidência coletiva para lacunas da bola (desligada por padrão)
**Prazo:** qua 11/11 · **Esforço:** M · **Prioridade:** P1 · **Marco:** M2 · **Bloqueada por:** F1-02 · **Bloqueia:** F1-05

**Arquivos:** `modeling/evidence/collective.py`, alteração em `modeling/evidence/team.py`, `tests/test_modeling_collective.py`.

**Especificação:** `build_collective_features(master, params) -> pd.DataFrame`. Calculada por tempo, sem atravessar o limite, com janela centrada de ±`win_frames` (25):
- `ev_adv_home` = derivada centrada de `centroid_home_x` (m/s); `ev_adv_away` = −(derivada centrada de `centroid_away_x`), porque A ataca para x decrescente.
- `ev_coll_adv = ev_adv_home − ev_adv_away`.
- `ev_coll_spread = spread_home_x · spread_home_y − spread_away_x · spread_away_y`.
- `ev_coll_speed = mean_speed_home − mean_speed_away`.

As três são antissimétricas: trocar os times inverte o sinal. Valem NaN se faltar dado ou se a janela sair do tempo.

Em `team_logit_rules_v1`, somar `w_adv·ev_coll_adv + w_spread·ev_coll_spread + w_speed·ev_coll_speed` **apenas** onde a bola não é válida ou `gap_length > 12`. Pesos padrão 0 (desligado); termo NaN → 0.

**Testes:**
- com os pesos em 0, a saída é idêntica bit a bit à da F1-02;
- antissimetria;
- nada atravessa o limite do tempo;
- derivada correta numa rampa conhecida.

#### F1-05 — Calibração do `team_logit` no IDSSE
**Prazo:** qui 12/11 · **Esforço:** M · **Prioridade:** P1 · **Marco:** M2 · **Bloqueada por:** F1-03, F1-04, F4-04 · **Bloqueia:** F1-06, F1-07, F2-06

**Arquivos:** `modeling/calibration/common.py`, `modeling/calibration/team.py`, `configs/calib/team.yaml`, `@register_team("calibrated")` em `modeling/evidence/team.py`, `reports/calibration_team.md`, `tests/test_modeling_calibration_team.py`.

**Pré-requisito local (dados externos):** os mestres do IDSSE em `$MTAH_EXT_DIR`, obtidos com os comandos de `docs/external_data.md` (F4-04) e conferidos com `python -m modeling.external.build --verify`, ou pelo pacote compartilhado pela F4 (plano B). São dados públicos: o agente pode rodar esses comandos e ler os arquivos.

**Especificação:**
- **Dados:** mestres `__default` (degradados) e seus rótulos; só frames vivos (H ou A); subamostragem de 1 a cada 5 frames.
- **Dois modelos**, ambos `sklearn.linear_model.LogisticRegression(solver="lbfgs", fit_intercept=False, max_iter=1000)` e ambos **antissimétricos**: features ímpares sob troca de times e sem intercepto. Assim, trocar os times inverte o logit e nenhum viés de mandante é aprendido do IDSSE.
  - **"ball"** (bola válida): `f1 = −tanh(d_diff/2)`, `f2 = −tanh(d_diff/6)`, `f3 = f1·ev_prox_w`, `f4 = f1·ev_rel_w`, `f5 = f1·ev_ball_conf`, `f6 = f1·(1 − ev_contest)`, `f7 = pressure_home − pressure_away`, `f8 = n_home_r3 − n_away_r3`.
  - **"gap"** (bola não válida): `ev_coll_adv`, `ev_coll_spread`, `ev_coll_speed`. NaN → 0 só na matriz de desenho; o frame é mantido.
- **Escala:** dividir cada feature pelo desvio-padrão do treino, **sem centralizar** (centralizar quebraria a antissimetria). As escalas vão para o YAML.
- **Regularização:** `C ∈ {0.01, 0.1, 1.0}`, escolhido pela log-loss média com uma partida de fora de cada vez (7 dobras).
- **`configs/calib/team.yaml`:** `C`, escalas e pesos de `all` (ajuste com as 7 partidas) e de `folds[<match>]` (ajuste com as outras 6).
- **Tipo `calibrated`:**
  - `t = clip(Σ w·f/scale, ±8)`, com soma explícita em float64;
  - usa `folds[holdout]` quando `holdout` é dado e `all` caso contrário;
  - modelo "ball" onde a bola é válida e "gap" onde não é;
  - 0 onde `half == 0`.
- **Não aplicar correção de prior:** sem intercepto, o modelo já assume 0,5 e devolve uma razão de verossimilhança.
- **`reports/calibration_team.md`:** log-loss e AUC por dobra e sinais dos coeficientes. f1–f6 devem sair positivos; se algum sair negativo, registrar.

**Testes:**
- em dados sintéticos gerados com pesos conhecidos, recupera os sinais;
- trocar as colunas home/away inverte `t`;
- ajuste determinístico com dados fixos;
- `holdout` usa a dobra certa.

**Aceite:** YAML e relatório gerados. A adoção é decidida só pela F2-06.

#### F1-06 — Análise de erros agregada (IDSSE)
**Prazo:** sex 13/11 · **Esforço:** P · **Prioridade:** P2 · **Marco:** M3 · **Bloqueada por:** F1-05 · **Bloqueia:** —

**Pré-requisito local (dados externos):** os mestres do IDSSE em `$MTAH_EXT_DIR`, obtidos com os comandos de `docs/external_data.md` (F4-04) e conferidos com `python -m modeling.external.build --verify`, ou pelo pacote compartilhado pela F4 (plano B). São dados públicos: o agente pode rodar esses comandos e ler os arquivos.

**Especificação:** com a melhor variante corrente, sobre os mestres degradados, gerar `reports/error_analysis.md` com:
- **taxa de erro por frame, por categoria:**
  - bola visível, lacuna curta, lacuna longa;
  - frames a até 2 s de uma troca de posse verdadeira;
  - frames a até 3 s de um reinício (fim de um D);
- **posses verdadeiras perdidas, por faixa de duração:** < 2 s, 2–10 s, > 10 s.

O relatório alimenta a seção de limitações do relatório técnico.

#### F1-07 — Variante GBDT do `team_logit` (opcional)
**Prazo:** sex 13/11 às 12h · **Esforço:** M · **Prioridade:** P2 · **Marco:** M3 · **Bloqueada por:** F1-05 · **Bloqueia:** F2-06

**Objetivo:** testar se um modelo não linear (árvores com gradient boosting) extrai mais da evidência de posse do que a regressão logística da F1-05. Entra na seleção como candidata e só é adotado se ganhar com folga.

**Arquivos:** `modeling/calibration/team_gbdt.py`, `@register_team("gbdt")` em `modeling/evidence/team.py`, `models/team_gbdt/all.joblib`, `configs/calib/team_gbdt.yaml`, `reports/calibration_team_gbdt.md`, `tests/test_modeling_team_gbdt.py`.

**Pré-requisito local (dados externos):** os mestres do IDSSE em `$MTAH_EXT_DIR`, obtidos com os comandos de `docs/external_data.md` (F4-04) e conferidos com `python -m modeling.external.build --verify`, ou pelo pacote compartilhado pela F4 (plano B). São dados públicos: o agente pode rodar esses comandos e ler os arquivos.

**Especificação:**
- **Modelo:** `sklearn.ensemble.HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=500, l2_regularization=1.0, early_stopping=False, random_state=0)`. Já vem no scikit-learn fixado e aceita NaN nativamente, sem imputação.
- **Dados:** os mesmos da F1-05 (mestres degradados, frames vivos, 1 a cada 5), só frames com bola válida. Onde a bola não é válida, continua valendo o modelo "gap" logístico da F1-05.
- **Features e regra de troca de times** (`swap`), obrigatória para a antissimetria:
  - pares que trocam entre si: (`d_near_home`, `d_near_away`), (`n_home_r2`, `n_away_r2`), (`n_home_r3`, `n_away_r3`), (`n_home_r5`, `n_away_r5`), (`pressure_home`, `pressure_away`), (`speed_rel_home`, `speed_rel_away`), (`speed_radial_home`, `speed_radial_away`), (`mean_speed_home`, `mean_speed_away`), (`n_still_home`, `n_still_away`), (`n_tracked_home`, `n_tracked_away`);
  - bola, que gira 180°: `ball_x → pitch_l_m − ball_x`, `ball_y → pitch_w_m − ball_y`, `ball_vx → −ball_vx`, `ball_vy → −ball_vy` (`pitch_l_m` e `pitch_w_m` vêm do config);
  - simétricas, que não mudam: `ball_speed`, `ev_ball_conf`, `d_touchline`, `d_goalline`, `gap_length`.
- **Treino com aumento:** cada amostra entra também na versão trocada, com o rótulo trocado (H↔A).
- **Logit antissimétrico por construção:** `t = clip((g(x) − g(swap(x))) / 2, ±8)`, em que `g(·)` é `decision_function` (log-odds de H). Assim `t(swap(x)) = −t(x)` exatamente, e não há prior a corrigir.
- **Dobras:** um modelo por dobra (sem a partida) e um com as 7 partidas.
  - Só `models/team_gbdt/all.joblib` é commitado, porque é o único usado no CSV final. Os modelos de dobra ficam em `ext_dir()/"models"/"team_gbdt"/`.
  - Salvar com `joblib.dump(..., compress=3)`. Registrar a versão do scikit-learn no YAML e recusar carregar com versão diferente.
  - O arquivo commitado tem no máximo 5 MB.
- **`reports/calibration_team_gbdt.md`:** log-loss e AUC por dobra, lado a lado com a F1-05.

**Testes:**
- antissimetria exata (`t(swap(x)) == −t(x)`) em dados sintéticos;
- treino determinístico (duas execuções dão as mesmas previsões);
- `holdout` usa o modelo da dobra certa;
- salvar e carregar preserva as previsões bit a bit.

**Aceite:** modelos, YAML e relatório gerados até sex 13/11 às 12h. A adoção é decidida só pela F2-06. Se não ficar pronto até lá, fechar como "não será feita".

---

### FRENTE F3 — Bola morta (`dead_logit`) e documentação

#### F3-01 — Features de bola morta
**Prazo:** dom 08/11 · **Esforço:** M · **Prioridade:** P0 · **Marco:** M0 · **Bloqueada por:** F5-01 · **Bloqueia:** F3-02

**Arquivos:** `modeling/deadball/features.py`, `tests/test_modeling_deadball_features.py`.

**Especificação:** `build_deadball_features(master, params) -> pd.DataFrame`, calculada por bloco de `half`; nenhuma janela ou busca atravessa o limite. Parâmetros no Apêndice A.3.

**`db_gap_kind`** (int8): vale 0 fora de lacuna longa. Para cada trecho máximo com `ball_observed == 0` e `gap_length > long_gap_frames` (12), classificar assim:
- **2 = saída**, se ocorrer (a) ou (b):
  - (a) o último frame observado antes do trecho tem `out_of_pitch == 1`, ou está a menos de `exit_line_m` de uma linha (`d_touchline` ou `d_goalline`) com a velocidade apontando para essa linha (componente normal > `exit_speed_mps`);
  - (b) o primeiro frame observado depois do trecho está em posição de reinício:
    - `d_touchline < restart_line_m` (lateral ou escanteio), ou
    - área de meta: `d_goalline < goal_area_depth_m` e `|ball_y − pitch_w_m/2| < goal_area_half_width_m`.
- **3 = aérea**, se não for saída, a velocidade da bola no último frame observado for > `air_speed_mps` e o trecho tiver no máximo `air_max_frames`.
- **4 = longa indeterminada**, nos demais casos.

Sem frame observado de um dos lados, usar só o outro; sem nenhum dos dois, classificar como 4.

**Demais features:**
- `db_still_frac = (n_still_home + n_still_away) / (n_tracked_home + n_tracked_away)`, NaN se o denominador for 0 ou NaN; depois média móvel centrada de `smooth_frames` (nanmean; NaN se a janela for toda NaN).
- `db_mean_speed` = nanmean(`mean_speed_home`, `mean_speed_away`), com a mesma média móvel.
- `db_ball_rest` (0/1): o frame pertence a um trecho de pelo menos `rest_min_frames` frames consecutivos com `ball_observed == 1` e `ball_speed < rest_speed_mps` (NaN não conta).
- `db_fast_in_pitch` (0/1): `ball_observed == 1`, `ball_speed > fast_speed_mps` e `out_of_pitch == 0`.
- `db_sp_prox = exp(−frames_to_next_sp / sp_prox_tau_frames)` e `db_sp_after = exp(−frames_since_sp_end / sp_after_tau_frames)`; NaN se a distância for NaN.
- `db_goal_window` (0/1): `next_sp_type == "kick off (after goal)"` (em **minúsculas**, como no master) e `frames_to_next_sp < goal_window_frames`.

**Testes:**
- trajetórias sintéticas para cada tipo de lacuna: saída pela lateral, saída pela linha de fundo com reinício na área de meta, bola aérea, lacuna longa;
- bordas dos tempos;
- NaN;
- `db_goal_window` com o valor exato `"kick off (after goal)"`.

**Armadilhas:** não confundir bola alta (a posse continua) com bola fora (D); é a distinção mais importante desta frente. `pitch_w_m` vem do config (68,0): o master não carrega as dimensões do campo.

#### F3-02 — `dead_logit` v1 (prior)
**Prazo:** seg 09/11 · **Esforço:** P · **Prioridade:** P0 · **Marco:** M1 · **Bloqueada por:** F3-01 · **Bloqueia:** F3-04, F5-04

**Arquivos:** `modeling/deadball/dead.py`, `tests/test_modeling_dead.py`.

**Especificação:** `@register_dead("rules_v1")`:
- soma dos termos do Apêndice A.3; termo com feature NaN vale 0;
- 0 onde `half == 0`; recorte em ±12; saída float32;
- separar a parte das tags (`is_set_piece`, `db_sp_prox`, `db_sp_after`, `db_goal_window`) numa função `dead_tag_prior(master, params) -> float64[n]`, que a F3-04 reutiliza.

**Testes:**
- monotonicidade termo a termo;
- NaN → 0;
- `out_of_pitch == 1` com o resto neutro → positivo forte;
- bola rápida em campo → negativo;
- limites respeitados.

#### F3-03 — `force_D` (pré-pontapé)
**Prazo:** sáb 07/11 · **Esforço:** P · **Prioridade:** P0 · **Marco:** M0 · **Bloqueada por:** F5-01 · **Bloqueia:** F5-04

**Especificação:** `modeling/deadball/force.py::build_force_d(master) -> np.ndarray[bool]` = `master["pre_kickoff"]` convertido para bool, validado com `validate_force_d`. No IDSSE fica todo False. O docstring deve dizer que o 1º tempo começa no pontapé (não há pré-pontapé) e que, no 2º tempo, a máscara cobre 2.934 frames.

**Testes:** no mock, a contagem de `True` é igual à de `pre_kickoff`; nada fora dos tempos; nada no 1º tempo.

#### F3-04 — Calibração do `dead_logit` no IDSSE
**Prazo:** qui 12/11 · **Esforço:** M · **Prioridade:** P1 · **Marco:** M2 · **Bloqueada por:** F3-02, F1-03, F4-04 · **Bloqueia:** F2-06, F3-08

**Arquivos:** `modeling/calibration/dead.py`, `configs/calib/dead.yaml`, `@register_dead("calibrated")` em `modeling/deadball/dead.py`, `reports/calibration_dead.md`, `tests/test_modeling_calibration_dead.py`.

**Pré-requisito local (dados externos):** os mestres do IDSSE em `$MTAH_EXT_DIR`, obtidos com os comandos de `docs/external_data.md` (F4-04) e conferidos com `python -m modeling.external.build --verify`, ou pelo pacote compartilhado pela F4 (plano B). São dados públicos: o agente pode rodar esses comandos e ler os arquivos.

**Especificação:**
- **Dados:** mestres degradados e rótulos; frames dos tempos com rótulo ≠ X; subamostragem de 1 a cada 5; alvo = D × vivo.
- **Features** (sem tags, que não existem no IDSSE):
  - `has_ball` (indicador);
  - `out_of_pitch` (NaN → 0, junto com `has_ball`);
  - one-hot de `db_gap_kind` ∈ {2, 3, 4};
  - `db_still_frac`, `db_mean_speed`, `db_ball_rest`, `db_fast_in_pitch`;
  - `log1p(min(gap_length, 1500))` (NaN → 0, com indicador);
  - `ball_speed` recortada em 30 (NaN → 0, junto com `has_ball`).
- **Ajuste:** padronizar com média e desvio do treino, com intercepto. `C ∈ {0.01, 0.1, 1.0}`, escolhido pela log-loss com uma partida de fora de cada vez.
- **Conversão para razão de verossimilhança:** `d_fit = logit_modelo − logit(π_D)`, onde π_D é a fração de D no treino da dobra. Sem essa correção, a proporção de D seria contada duas vezes: no logit e nas transições do HMM.
- **Tipo `calibrated`:** `d = clip(d_fit + dead_tag_prior(master, params), ±12)`.
  - Os pesos das tags ficam fixos no prior do Apêndice A.3 (documentar).
  - Soma explícita em float64; `holdout` seleciona a dobra.
- **`configs/calib/dead.yaml`:** médias, desvios, pesos, intercepto, π_D e `C`, para `all` e para cada dobra.
- **`reports/calibration_dead.md`:** log-loss e AUC por dobra e sinais. Sinais esperados: `out_of_pitch` positivo, `db_fast_in_pitch` negativo, `db_mean_speed` negativo.

**Testes:**
- recuperação de sinais em dados sintéticos;
- correção de prior: com features neutras, `d_fit ≈ 0`;
- `holdout` usa a dobra certa.

#### F3-05 — Registro de decisões e esqueleto do relatório técnico (inglês)
**Prazo:** seg 09/11 · **Esforço:** M · **Prioridade:** P0 · **Marco:** M1 · **Bloqueada por:** F5-01 · **Bloqueia:** F5-05, F5-07

**Especificação:**
- **`docs/decisions.md`:** entradas com data, decisão, opções consideradas, evidência e autor. Registrar já as decisões deste plano:
  - IDSSE como dado externo;
  - nada é treinado no BAR-ATH;
  - emissão corrigida;
  - ativos descartados.
- **`docs/technical_report/report.md`**, em inglês, com as seções da F5-07. Já preencher:
  - o pré-processamento (a partir de `docs/preprocessamento_v2.md`);
  - os ativos usados e descartados;
  - a citação do IDSSE.
- Cada frente completa a sua seção do relatório como parte da Definition of Done das próprias tarefas.

#### F3-06 — Diagnósticos de bola morta no BAR-ATH (execução local)
**Prazo:** sex 13/11 · **Esforço:** P · **Prioridade:** P1 · **Marco:** M3 · **Bloqueada por:** F5-04, F2-03 · **Bloqueia:** —

**Especificação:** `tools/deadball_diagnostics.py --pred <csv> [--verbose]`, sobre o master e os rótulos previstos. Imprime PASS/FAIL:
- **contra as faixas de `configs/plausibility.yaml` (F2-03), com folga de ±40%:**
  - fração de D por tempo;
  - posses por minuto por tempo;
  - duração média das posses;
- **contra limiares fixos no próprio script (só informativo):**
  - fração de frames com `out_of_pitch == 1` previstos como não-D < 0,2;
  - fração de segmentos D sem nenhuma evidência (sem `out_of_pitch`, sem lacuna de saída, sem jogadores lentos e sem tag) < 0,3.

Números só aparecem com `--verbose`. O resultado alimenta os filtros eliminatórios da F2-06.

#### F3-07 — Frequência dos casos de borda do filtro da bola (execução local)
**Prazo:** sex 13/11 · **Esforço:** P · **Prioridade:** P2 · **Marco:** M3 · **Bloqueada por:** — · **Bloqueia:** —

**Contexto:** a revisão de 04/11 mostrou, com dados sintéticos, dois casos em que `remove_ball_jump_outliers` erra:
- (a) um fantasma parado por 3 s ou mais vence a bola real quando ela é detectada de forma esparsa;
- (b) numa reposição com várias bolas, se a bola nova aparece em até ~0,4 s, os últimos frames da bola antiga (o cruzamento da linha) são apagados.

No jogo corrido sintético, o filtro mantém 100% dos pontos reais.

**Especificação:** `tools/jump_filter_edge_cases.py [--verbose]`, que lê `tracking_clean.parquet` e o master e conta, por tempo:
- (a) trechos de pelo menos 75 frames em que a bola mantida está parada (velocidade < 0,5 m/s) e fora do campo (`out_of_pitch == 1`) enquanto a velocidade média dos jogadores é > 3 m/s;
- (b) observações removidas nos 25 frames anteriores a um salto > 20 m seguido de bola parada.

PASS se as duas contagens forem < 10 por tempo. Se der FAIL, registrar em `docs/decisions.md`. Só corrigir se houver tempo antes do freeze: readmitir as observações removidas que continuam suavemente um trecho mantido, com testes sintéticos dos dois cenários.

#### F3-08 — Variante GBDT do `dead_logit` (opcional)
**Prazo:** sex 13/11 às 12h · **Esforço:** M · **Prioridade:** P2 · **Marco:** M3 · **Bloqueada por:** F3-04 · **Bloqueia:** F2-06

**Objetivo:** o mesmo da F1-07, agora para bola morta, onde as interações entre sinais (bola perto da linha, jogadores desacelerando, lacuna longa) tendem a pesar mais.

**Arquivos:** `modeling/calibration/dead_gbdt.py`, `@register_dead("gbdt")` em `modeling/deadball/dead.py`, `models/dead_gbdt/all.joblib`, `configs/calib/dead_gbdt.yaml`, `reports/calibration_dead_gbdt.md`, `tests/test_modeling_dead_gbdt.py`.

**Pré-requisito local (dados externos):** os mestres do IDSSE em `$MTAH_EXT_DIR`, obtidos com os comandos de `docs/external_data.md` (F4-04) e conferidos com `python -m modeling.external.build --verify`, ou pelo pacote compartilhado pela F4 (plano B). São dados públicos: o agente pode rodar esses comandos e ler os arquivos.

**Especificação:**
- Mesmo modelo, hiperparâmetros, dobras, persistência e regra de versão da F1-07.
- Dados e alvo da F3-04: D × vivo, frames dos tempos com rótulo ≠ X, 1 a cada 5.
- Features da F3-04, **sem** imputação nem indicadores (o modelo aceita NaN): `out_of_pitch`, `db_gap_kind`, `db_still_frac`, `db_mean_speed`, `db_ball_rest`, `db_fast_in_pitch`, `gap_length`, `ball_speed`, mais `d_touchline` e `d_goalline`.
- Razão de verossimilhança: `d_fit = decision_function(x) − logit(π_D)`, com π_D da dobra.
- `d = clip(d_fit + dead_tag_prior(master, params), ±12)`, com as tags fixas no prior, como na F3-04.
- Relatório lado a lado com a F3-04.

**Testes:** correção de prior (com π_D conhecido); treino determinístico; dobra certa; salvar e carregar preserva as previsões bit a bit.

**Aceite:** como na F1-07.

---

### FRENTE F2 — Decodificação temporal e seleção

#### F2-01 — Emissão e parâmetros do decodificador
**Prazo:** sáb 07/11 · **Esforço:** P · **Prioridade:** P0 · **Marco:** M0 · **Bloqueada por:** F5-01 · **Bloqueia:** F2-02, F2-03

**Arquivos:** `modeling/decode/emission.py`, `modeling/decode/params.py`, `tests/test_modeling_emission.py`.

**Especificação:**
- **`logits_to_logscores(team_logit, dead_logit, lam_team, lam_dead, force_d, neg=-1e6) -> float64[n, 3]`:** exatamente como o C3.
- **`DecoderParams`**, dataclass imutável em float64, com os campos:
  - `lam_team` e `lam_dead`;
  - `mean_duration_frames` para H, A e D;
  - `exit_split`: matriz 3×3 com diagonal 0 e cada linha somando 1;
  - `min_dwell` para H, A e D (inteiros ≥ 1);
  - `neg`.
- **`DecoderParams.from_config(cfg_decoder, fps=25)`:**
  - valida `mean_duration_frames[s] > min_dwell[s]`;
  - calcula `p_stay[s] = 1 − 1/(mean_duration_frames[s] − min_dwell[s] + 1)`, de modo que a duração média esperada de um segmento é exatamente `mean_duration_frames[s]`;
  - guarda as log-transições em float64.

**Testes:**
- `t = d = 0` → as três pontuações valem exatamente 0;
- trocar `t` por `−t` troca `S_H` com `S_A`, e `S_H − S_A = lam_team·t`;
- estabilidade com `|logit| = 1e4` (sem inf ou NaN);
- `force_d` aplicado corretamente;
- `p_stay` reproduz a duração média: simulação de Monte Carlo com semente fixa, tolerância de 2%.

**Armadilha (corrigida nesta versão):** na fórmula da v1 do plano, com `d = 0` valia `S_H = S_D + λ·logσ(t) < S_D` para qualquer `t`. D vencia todo frame sem evidência ativa contra bola morta, mesmo com posse clara. Numa simulação de 10 s de posse com 3 s sem bola, a v1 decodificava tudo como D.

#### F2-02 — Viterbi por tempo com duração mínima
**Prazo:** seg 09/11 · **Esforço:** M · **Prioridade:** P0 · **Marco:** M1 · **Bloqueada por:** F2-01 · **Bloqueia:** F2-04, F5-04

**Arquivos:** `modeling/decode/viterbi.py`, `tests/test_modeling_viterbi.py`.

**Especificação:** `decode_match(scores, half, force_d, params)`:
- **Por tempo:** cada bloco contíguo com `half == h` (h = 1, 2) é decodificado separadamente; `half == 0` → `X`.
- **Cadeia expandida:** um estado `s` com duração mínima `m_s` vira os subestados `(s,1)…(s,m_s)`, com índice global na ordem H, A, D. Transições:
  - `(s,k) → (s,k+1)` com log-prob 0, para k < m_s;
  - `(s,m_s) → (s,m_s)` com `log p_stay[s]`;
  - `(s,m_s) → (s',1)` com `log(1 − p_stay[s]) + log exit_split[s,s']`, para s' ≠ s;
  - se `m_s = 1`, `(s,1)` é ao mesmo tempo o primeiro e o último subestado.
  
  A emissão de `(s,k)` é `scores[t, s]`.
- **Início:** só os subestados `(s, m_s)` recebem `log(1/3) + scores[0, s]`; os demais começam em −inf. Assim, o primeiro segmento do tempo não precisa cumprir a duração mínima.
- **Fim:** argmax sobre todos os subestados no último frame; o último segmento também não precisa cumprir a duração mínima.
- **Recursão:** em log, float64, com laço em t vetorizado nos subestados; ponteiros `int16` (K = Σm_s ≤ 32767); desempate pelo menor índice (`np.argmax`).
- **Saída:** backtracking e colapso de `(s,k)` para `s`.

**Testes (sintéticos):**
- (a) igualdade com a enumeração exaustiva de todas as sequências válidas, com n ≤ 10, para combinações de `m_s` ∈ {1, 2, 3} e scores aleatórios com semente;
- (b) duração mínima respeitada em todos os segmentos, exceto o primeiro e o último de cada tempo;
- (c) um trecho com scores 0 entre evidências do mesmo time mantém o rótulo;
- (d) uma mudança brusca de evidência em t0, forte e longa, põe a fronteira em t0 ± 3;
- (e) determinismo;
- (f) `force_d` respeitado;
- (g) X exatamente onde `half == 0` e nunca dentro dos blocos.

**Aceite:** menos de 10 s por tempo com 80.000 frames e K ≈ 50, medido no mock.

#### F2-03 — Estatísticas dos rótulos IDSSE (transições, durações, faixas de plausibilidade)
**Prazo:** ter 10/11 · **Esforço:** P · **Prioridade:** P1 · **Marco:** M1 · **Bloqueada por:** F2-01, F4-01 · **Bloqueia:** F2-05, F3-06

**Arquivos:** `modeling/scoring/label_stats.py`, `configs/decoder_idsse.yaml`, `configs/plausibility.yaml`, `reports/idsse_label_stats.md`, `tests/test_modeling_label_stats.py`.

**Pré-requisito local (dados externos):** os 7 `.npz` do IDSSE em `$MTAH_EXT_DIR/arrays`, obtidos com `python -m modeling.external.idsse --matches all --delete-raw` (F4-01) ou pelo pacote compartilhado pela F4 (plano B). São dados públicos: o agente pode rodar esse comando e ler os arquivos.

**Especificação:** a partir dos rótulos (C0) dos 7 `.npz`, por tempo, calcular:
- os segmentos;
- a duração média, em segundos, de H, A e D (por estado e com H e A também juntos);
- as contagens de transição entre estados, e `exit_split` = contagens normalizadas por linha;
- posses por minuto, duração média de posse e fração de D.

Gerar:
- **`configs/decoder_idsse.yaml`:** seção `decoder` completa, com as durações e o `exit_split` do IDSSE e `lam_*` e `min_dwell` do Apêndice A.4;
- **`configs/plausibility.yaml`:** mínimo e máximo, entre os 14 tempos, de posses por minuto, duração média de posse e fração de D.

Tudo vem de dados externos e pode ser commitado.

**Testes:** com uma sequência sintética conhecida, recupera as durações e o `exit_split` exatos; frames X são ignorados e nunca contados como segmento.

#### F2-04 — Pós-processamento por posse
**Prazo:** qua 11/11 · **Esforço:** M · **Prioridade:** P1 · **Marco:** M2 · **Bloqueada por:** F2-02 · **Bloqueia:** F2-05

**Arquivos:** `modeling/decode/postprocess.py`, `tests/test_modeling_postprocess.py`.

**Especificação:** `postprocess(labels, master, team_logit, dead_logit, force_d, cfg_post, *, holdout=None) -> (labels, stats)`. As regras são independentes, ligadas por config e aplicadas por tempo, nesta ordem:
1. **`midpoint_neutral_switch`:** para cada fronteira direta H↔A (sem D entre elas) que esteja dentro ou na borda de um trecho máximo com `team_logit == 0` (sem evidência de time), mover a fronteira para o ponto médio do trecho (`início + comprimento // 2`).
   - Motivo: com evidência neutra, o Viterbi empata e põe a troca numa das bordas, por desempate. O ponto médio minimiza o erro esperado de IoU.
2. **`boundary_shift_file`** (ligada se não for `null`): deslocar cada fronteira pelo atraso sistemático calibrado no IDSSE para o seu tipo (`H>A`, `A>H`, `H>D`, `A>D`, `D>H`, `D>A`).
   - Os valores vêm do arquivo da F2-05: `folds[holdout]` quando `holdout` é dado; senão, `all`.
   - Deslocamento = −(mediana da latência), em frames inteiros.
   - Nunca mover uma fronteira para além da vizinha (cada segmento mantém pelo menos 1 frame), nem mexer em fronteiras vizinhas de `force_d` ou das bordas do tempo.
   - Motivo: o spellF1 pune atraso (8 frames → 0,89; 1 s → 0,65), e um atraso sistemático (por exemplo, o D começar sempre tarde) é corrigível.
3. **`d_bridge_frames`** (ligada se > 0): um segmento D mais curto que isso, entre dois segmentos do **mesmo** time, vira esse time. Nunca vale em frames de `force_d`.
4. **`min_spell_frames`** (ligada se > 0): um segmento H/A mais curto que isso, cujos dois vizinhos têm o mesmo rótulo, vira esse rótulo. Varrer da esquerda para a direita até não mudar nada (no máximo 20 passadas).
5. **Imposição final (sempre):** `force_d` → D; `half == 0` → X; validar que não há X dentro dos tempos.

A função é idempotente e determinística, e devolve a tupla `(labels, stats)`, em que `stats` é um dict com quantas mudanças cada regra fez.

**Testes:**
- casos curtos `H D H`, `A D A` e `H A H`;
- ponto médio num trecho neutro, e nenhuma mudança quando há evidência;
- deslocamento de fronteira aplicado por tipo, limitado pelas fronteiras vizinhas, e com a dobra certa;
- idempotência das regras 1, 3 e 4 (a regra 2 desloca uma vez por chamada: testar que a imposição final continua válida depois dela);
- com todas as regras desligadas, o resultado é a identidade (mais a imposição final).

#### F2-05 — Varredura de sensibilidade do decodificador (IDSSE)
**Prazo:** qui 12/11 · **Esforço:** M · **Prioridade:** P1 · **Marco:** M2 · **Bloqueada por:** F1-03, F2-03, F2-04, F4-04, F5-04 · **Bloqueia:** F2-06

**Arquivos:** `modeling/scoring/sweep.py`, `modeling/calibration/boundary.py`, `configs/calib/boundary.yaml`, `reports/sensitivity_decoder.md`, `configs/v1s.yaml`.

**Pré-requisito local (dados externos):** os mestres do IDSSE em `$MTAH_EXT_DIR`, obtidos com os comandos de `docs/external_data.md` (F4-04) e conferidos com `python -m modeling.external.build --verify`, ou pelo pacote compartilhado pela F4 (plano B). São dados públicos: o agente pode rodar esses comandos e ler os arquivos.

**Especificação:** partindo de `configs/decoder_idsse.yaml` com logits `rules_v1`, sobre os mestres degradados, fazer uma varredura por etapas (uma coordenada de cada vez). Registrar média e erro-padrão (EP) do spellF1 em cada ponto.
1. `lam_team × lam_dead` ∈ {0,05; 0,1; 0,2; 0,4; 0,8}² (25 pontos).
2. No melhor λ: `min_dwell` H/A ∈ {1, 12, 25} × D ∈ {1, 25, 50} (9 pontos).
3. Fator de escala das durações médias ∈ {0,5; 1; 2} (3 pontos).
4. Pós-processamento: cada regra ligada isoladamente (midpoint; `d_bridge` 25 e 50; `min_spell` 13 e 25).
5. **Viés de fronteira** (`modeling/calibration/boundary.py`), com a config escolhida nas etapas 1–4:
   - em cada partida, emparelhar as transições previstas com as verdadeiras do mesmo tipo usando as funções oficiais `pm.valid_runs`, `pm.transitions` e `pm.match_transitions(gt, pred, tol=50, typed="pair")`, e medir a latência (frame previsto − frame verdadeiro);
   - gravar em `configs/calib/boundary.yaml` a mediana por tipo para `all` e para cada dobra (calculada sem a partida); tipos com menos de 20 pares ficam com 0;
   - avaliar a config com e sem `boundary_shift_file`; o deslocamento entra como um passo da Seção 6.

**Regra de escolha (pré-registrada):**
- o platô é o conjunto de pontos cuja média fica a menos de 1 EP da melhor;
- dentro do platô, escolher o ponto cujos vizinhos imediatos na grade também estão no platô;
- se nenhum ponto satisfizer, ficar com o melhor;
- empate: menor λ, depois menor `min_dwell`.

Gerar `configs/v1s.yaml` (v1 com o decodificador escolhido) e o relatório (tabelas de dados externos).

**Aceite:** relatório com recomendação e justificativa. O script roda com qualquer config e é usado de novo na F2-06.

#### F2-06 — Protocolo de seleção e `configs/final.yaml`
**Prazo:** sex 13/11 às 18h · **Esforço:** M · **Prioridade:** P0 · **Marco:** M3 · **Bloqueada por:** F2-05, F1-05, F3-04, F1-07, F3-08, F4-06 · **Bloqueia:** G1

**Pré-requisito local (dados externos):** os mestres do IDSSE em `$MTAH_EXT_DIR`, obtidos com os comandos de `docs/external_data.md` (F4-04) e conferidos com `python -m modeling.external.build --verify`, ou pelo pacote compartilhado pela F4 (plano B). São dados públicos: o agente pode rodar esses comandos e ler os arquivos.

**Especificação:** aplicar **exatamente** a Seção 6 e produzir:
- `reports/selection.md`;
- `configs/v2.yaml`;
- `configs/final.yaml`;
- uma entrada em `docs/decisions.md`.

Se alguma das tarefas F1-05, F3-04, F1-07, F3-08 ou F4-06 for fechada como "não será feita", a seleção usa só as variantes disponíveis. As candidatas P2 que não estiverem prontas na sex 13/11 às 12h ficam de fora. O passo de plausibilidade no BAR-ATH é execução local (humano), com `tools/deadball_diagnostics.py` → PASS/FAIL.

---

### Gate de congelamento

#### G1 — Code freeze
**Prazo:** sex 13/11 às 20h · **Tipo:** gate · **Dono:** F5 · **Bloqueada por:** F2-06 · **Bloqueia:** F5-06, F5-07

**Especificação:** proteger a `main`: só entram correções de bug e documentação, com 2 aprovações. Criar a tag `submission-v1` no commit que contém `configs/final.yaml`. Depois do congelamento não entra nenhuma mudança de método.

---

#### F1-08, F2-07, F2-08 — Candidatas novas de 13/11 (Ahmed)
Acrescentadas em 13/11 (D-012) para atacar a contagem baixa de posses e o atraso nas fronteiras. Especificações completas em `docs/tasks/`:
- **F1-08 — Detector de troca de posse (hazard) no IDSSE** ([#124](https://github.com/AntonioEscobarTorres/more-than-a-hack-2027-brasileirao/issues/124)): evidência simétrica, por frame, de que a posse está trocando. Bloqueada por F1-03 e F4-04.
- **F2-07 — Transições guiadas pela evidência de troca (IO-HMM)** ([#125](https://github.com/AntonioEscobarTorres/more-than-a-hack-2027-brasileirao/issues/125)): o hazard modula a troca H↔A no Viterbi. Bloqueada por F1-08 e F2-05.
- **F2-08 — Durações semi-Markov (binomial negativa)** ([#126](https://github.com/AntonioEscobarTorres/more-than-a-hack-2027-brasileirao/issues/126)): `r` estágios depois do `min_dwell`. Bloqueada por F2-03 e F2-05.

As três bloqueiam a F2-06, deixam o v1 byte a byte igual quando desligadas e só entram pela regra da Seção 6.

---

## 5. Arestas de dependência (todas as relações "bloqueada por") — 67 arestas (58 + 9 de 13/11)

| Tarefa | Bloqueada por |
| :--- | :--- |
| F5-01 | — |
| F5-02 | F5-01 |
| F5-03 | F5-01 |
| F5-04 | F5-02, F1-02, F2-02, F3-02, F3-03 |
| F5-05 | F5-04, F5-08, F3-05 |
| F5-06 | G1 |
| F5-07 | G1, F3-05 |
| F5-08 | — |
| F5-09 | F5-06, F5-07, F5-08 |
| F4-01 | — |
| F4-02 | — |
| F4-03 | F4-01 |
| F4-04 | F4-01, F4-02, F4-03 |
| F4-05 | — |
| F4-06 | F4-05, F5-04 |
| F1-01 | F5-01 |
| F1-02 | F1-01 |
| F1-03 | F5-02 |
| F1-04 | F1-02 |
| F1-05 | F1-03, F1-04, F4-04 |
| F1-06 | F1-05 |
| F1-07 | F1-05 |
| F3-01 | F5-01 |
| F3-02 | F3-01 |
| F3-03 | F5-01 |
| F3-04 | F3-02, F1-03, F4-04 |
| F3-05 | F5-01 |
| F3-06 | F5-04, F2-03 |
| F3-07 | — |
| F3-08 | F3-04 |
| F2-01 | F5-01 |
| F2-02 | F2-01 |
| F2-03 | F2-01, F4-01 |
| F2-04 | F2-02 |
| F2-05 | F1-03, F2-03, F2-04, F4-04, F5-04 |
| F1-08 | F1-03, F4-04 |
| F2-07 | F1-08, F2-05 |
| F2-08 | F2-03, F2-05 |
| F2-06 | F2-05, F1-05, F3-04, F1-07, F3-08, F4-06, F1-08, F2-07, F2-08 |
| G1 | F2-06 |

**Caminho crítico:** F4-01 → F4-03 → F4-04 → (F1-05, F3-04, F2-05) → F2-06 → G1 → F5-06 → F5-09. Uma tarefa P1/P2 fechada como "não será feita" deixa de bloquear (ver Seção 7).

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
- `duration_stages` (F2-08, D-027): `H`/`A`/`D` = 1, a duração geométrica depois do `min_dwell`. `r > 1` estágios dão uma binomial negativa deslocada com a mesma média.
- `configs/decoder_idsse.yaml` (F2-03) substitui as durações e o `exit_split` pelos valores do IDSSE.

### A.5 Pós-processamento
Padrão: tudo desligado. `midpoint_neutral_switch: false`, `d_bridge_frames: 0`, `min_spell_frames: 0`. Cada regra só é ligada pela F2-06.

### A.6 Degradação (`configs/degrade_default.yaml`)
- `name=default`; tolerâncias dos alvos: `target_ball_visible_tolerance=0.01` e
  `target_tracked_mean_tolerance=0.3`.
- **Bola:**
  - `z_air_m=1.5`, `air_margin_frames=3`, `p_air=0.85`;
  - `crowd_n=3`, `crowd_r_m=2.0`, `crowd_hazard=0.02`;
  - `target_ball_visible=0.593`;
  - perfil de lacunas `{mix_geometric: 0.6, geometric_mean_frames: 8, lognormal_median_frames: 40, lognormal_sigma: 1.0, max_frames: 1500}`;
  - `bg_hazard_init=0.01`, `hazard_max=1.0`, `bisection_iterations=30`;
  - `ball_noise_m=0.25`;
  - `phantom_len_frames=[5, 60]`, `phantom_frac=0.01`, `phantom_margin_m=5.0`.
- **Jogadores:**
  - `player_visible_mean_s=30.0` e `player_missing_mean_s_init=3.0` (ponto de partida),
    `player_missing_mean_s_max=120.0`, `player_bisection_iterations=30`, `fps=25`;
  - `target_tracked_mean=20.5`;
  - `player_noise_m=0.15`;
  - `dup_offset_m=0.3`, `dup_len_s=[1.0, 5.0]`, `dup_frac=0.02`,
    `dup_max_attempts=10000`, `overtracked_threshold=22`.
- `modeling.config.load_config` valida todas as chaves, exige `name` igual ao sufixo do arquivo
  e rejeita valores ausentes ou desconhecidos.
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
  duration_stages: {H: 1, A: 1, D: 1}   # F2-08 (D-027): estágios depois do min_dwell; 1 = geométrica
postprocess:
  midpoint_neutral_switch: false
  boundary_shift_file: null    # ex.: configs/calib/boundary.yaml (F2-05)
  d_bridge_frames: 0
  min_spell_frames: 0
hazard:                        # F2-07 (D-025): hazard da F1-08 nas saídas H↔A do decodificador
  kind: none                   # none | calibrated
  calibrated_file: null        # obrigatório se kind != none, ex.: configs/calib/hazard.yaml
  beta: 0.0                    # >= 0; exit_logit = beta · h; kind none ou beta 0 = decodificador de antes
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
