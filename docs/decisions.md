# Registro de decisões

Registro único das decisões do projeto. Toda tarefa que toma uma decisão acrescenta uma entrada aqui
(faz parte da Definition of Done, Seção 1.5 do [contexto global](plano/00_contexto_global.md)).
O relatório técnico ([`technical_report/report.md`](technical_report/report.md)) é escrito a partir
deste arquivo.

## Como registrar

- Uma entrada por decisão, com id sequencial `D-NNN`. Entradas novas vão no **fim** da seção "Decisões".
- Uma entrada nunca é apagada nem reescrita. Se a decisão mudar, crie uma entrada nova e marque a antiga
  como `Substituída por D-NNN`.
- Campos obrigatórios: data, autor, status, decisão, opções consideradas e evidência.
- **Evidência:** só o que pode ser público (regra global 2). Valem `PASS`/`FAIL`, hashes, estatísticas dos
  rótulos previstos, resultados no IDSSE e links para issues, PRs e arquivos de `reports/`. Nunca valores
  de features, quantis do tracking, linhas ou gráficos da partida, e nunca o score da janela pública
  (regra global 9).
- "Issue oficial #N" é uma issue do repositório dos organizadores
  (`metrica-sports/more-than-a-hack-2027`). "#N" sem qualificação é deste repositório.

Modelo:

```markdown
### D-NNN — Título curto

- **Data:** AAAA-MM-DD
- **Autor:** nome (frente, tarefa)
- **Status:** Vigente | Substituída por D-NNN
- **Decisão:** o que foi decidido, em uma ou duas frases.
- **Opções consideradas:**
  - opção escolhida;
  - opção descartada, e por quê.
- **Evidência:** links, PASS/FAIL, hashes, resultados no IDSSE.
```

## Índice

| Id | Data | Decisão | Status |
| :--- | :--- | :--- | :--- |
| [D-001](#d-001--idsse-como-único-dado-externo) | 2026-11-05 | IDSSE como único dado externo | Vigente |
| [D-002](#d-002--nada-é-treinado-nem-ajustado-no-bar-ath) | 2026-11-05 | Nada é treinado nem ajustado no BAR-ATH | Vigente |
| [D-003](#d-003--emissão-corrigida-contrato-c3) | 2026-11-05 | Emissão corrigida (contrato C3) | Vigente |
| [D-004](#d-004--ativos-usados-e-descartados) | 2026-11-05 | Ativos usados e descartados | Vigente |
| [D-005](#d-005--arquitetura-hmm-de-3-estados-com-viterbi-por-tempo) | 2026-11-05 | Arquitetura: HMM de 3 estados com Viterbi por tempo | Vigente |
| [D-006](#d-006--seleção-pré-registrada-sem-a-janela-pública) | 2026-11-05 | Seleção pré-registrada, sem a janela pública | Vigente |
| [D-007](#d-007--features-de-bola-morta-interpretações-da-especificação) | 2026-11-10 | Features de bola morta: interpretações da especificação | Vigente |
| [D-008](#d-008--dead_logit-rules_v1-rampas-no-código-e-convenção-de-params) | 2026-11-10 | `dead_logit` `rules_v1`: rampas no código e convenção de `params` | Vigente |
| [D-009](#d-009--mestres-__none-só-para-diagnóstico) | 2026-11-11 | Mestres `__none` só para diagnóstico | Vigente |
| [D-010](#d-010--integração-do-hmm-no-predict_labels) | 2026-11-11 | Integração do HMM no `predict_labels` | Vigente |
| [D-011](#d-011--calibrated_file-chega-às-variantes-dentro-de-params) | 2026-11-11 | `calibrated_file` chega às variantes dentro de `params` | Vigente |
| [D-012](#d-012--resposta-dos-organizadores-de-1311-e-candidatas-novas) | 2026-11-13 | Resposta dos organizadores de 13/11 e candidatas novas | Vigente |
| [D-013](#d-013--nome-da-equipe-formulário-e-envio-único) | 2026-11-13 | Nome da equipe, formulário e envio único | Vigente |
| [D-014](#d-014--casos-de-borda-do-filtro-da-bola-como-são-contados) | 2026-11-13 | Casos de borda do filtro da bola: como são contados | Vigente |
| [D-015](#d-015--diagnósticos-de-bola-morta-como-são-calculados) | 2026-11-13 | Diagnósticos de bola morta: como são calculados | Vigente |
| [D-016](#d-016--degradação-do-idsse-ajustada-pelo-realismo-f4-04) | 2026-11-15 | Degradação do IDSSE ajustada pelo realismo (F4-04) | Vigente |

## Decisões

### D-001 — IDSSE como único dado externo

- **Data:** 2026-11-05
- **Autor:** equipe (plano v2); registrada por Bernardo (F3-05)
- **Status:** Vigente
- **Decisão:** o único dado externo do projeto é o IDSSE: 7 partidas completas da Bundesliga e da
  2. Bundesliga (2022/23), com dados de posição a 25 fps e, em cada frame, bola viva/morta (`BallStatus`)
  e time com a posse (`BallPossession`). Ele é usado para calibrar as evidências, estimar os parâmetros de
  transição, varrer o decodificador e escolher a variante final. Antes disso, cada partida é degradada
  para parecer o tracking automático (F4-03) e passa pelo **mesmo** pré-processamento do BAR-ATH (F4-02).
- **Opções consideradas:**
  - **IDSSE (escolhida):** rótulos nativos por frame, da mesma natureza do GT (derivados do tracking);
    licença CC BY 4.0; uso aprovado pelos organizadores.
  - Amostras abertas da Metrica com rótulos derivados de eventos (plano v1): descartada. Os rótulos seriam
    aproximados, e a construção deles daria mais trabalho e mais erro.
  - Não usar dado externo: descartada. Sem verdade fora do BAR-ATH não há forma permitida de escolher
    parâmetros (ver D-002).
- **Evidência:**
  - aprovação explícita na issue oficial #16; recado dos organizadores: "todo hiperparâmetro e limiar
    escolhido no IDSSE ou em outros jogos";
  - fonte, revisão fixada, tamanhos e SHA-256 dos arquivos em [`external_data.md`](external_data.md);
  - estatísticas por partida em [`reports/idsse_overview.md`](../reports/idsse_overview.md): 1.002.644
    frames no total, nenhum frame vivo com posse inválida;
  - citação obrigatória (README e relatório): Bassek, M., Rein, R., Weber, H., Memmert, D. (2025).
    *An integrated dataset of spatiotemporal and event data in elite soccer.* Scientific Data 12, 195.
    https://doi.org/10.1038/s41597-025-04505-y
- **Limitações conhecidas:** mudança de domínio (tracking corrigido do DFL → tracking automático) e
  convenção de posse do DFL possivelmente diferente da do GT nos toques curtos e duelos. As duas vão para a
  seção de limitações do relatório.

### D-002 — Nada é treinado nem ajustado no BAR-ATH

- **Data:** 2026-11-05
- **Autor:** equipe (plano v2); registrada por Bernardo (F3-05)
- **Status:** Vigente
- **Decisão:** nenhum parâmetro, peso ou limiar é treinado, calibrado ou escolhido com rótulos do BAR-ATH,
  sejam humanos, do GT público ou pseudo-rótulos algorítmicos. Toda calibração e toda escolha usam o IDSSE
  (D-001). Do BAR-ATH só se usam:
  - as constantes publicadas pelos organizadores (blocos dos tempos e frames dos pontapés, DATA.md e FAQ.md);
  - estatísticas **não rotuladas** do tracking, e apenas para tornar a degradação do IDSSE realista
    (alvos de visibilidade da bola e de jogadores rastreados, perfil de lacunas);
  - a janela pública, só para checagens `PASS`/`FAIL` de correção (D-006).
- **Opções consideradas:**
  - **Calibrar tudo fora do BAR-ATH (escolhida).**
  - Autotreino com pseudo-rótulos algorítmicos no BAR-ATH (hard-EM das transições, na v1): descartada.
    O RULES.md (atualizado em 23/10) permite pseudo-rótulos, mas a issue oficial #2 vetou treinar no próprio
    jogo com o tracking fornecido como rótulo. É uma zona cinzenta, e o IDSSE tem verdade.
  - Harness de mascaramento da bola no BAR-ATH para medir robustez (v1): descartada. Não tem verdade e
    premia suavização excessiva (uma saída constante seria "perfeitamente robusta").
  - Escolher parâmetros com rótulos do BAR-ATH que não entram no pipeline: vetada pelos organizadores na
    issue oficial #10.
- **Evidência:** issues oficiais #2 e #10; RULES.md ("Training and validation data"); Seção 1.2 e Apêndice E
  do [contexto global](plano/00_contexto_global.md).
- **Confirmação (13/11):** os organizadores responderam por e-mail que todo treino e todo ajuste de
  parâmetros devem usar só dados externos, e que o que não é explicitamente permitido não é permitido.
  Isso fecha a zona cinzenta acima: nem ajuste não supervisionado (EM) no BAR-ATH. Ver D-012.

### D-003 — Emissão corrigida (contrato C3)

- **Data:** 2026-11-05 (implementada em 2026-11-09, F2-01, PR #102)
- **Autor:** equipe (plano v2); implementação de Hugo (F2); registrada por Bernardo (F3-05)
- **Status:** Vigente
- **Decisão:** a emissão do decodificador é a do contrato C3 ([`contracts.md`](contracts.md)), com
  `logσ(x) = −logaddexp(0, −x)`:

  ```
  S_H = lam_team · (log 2 + logσ(t))
  S_A = lam_team · (log 2 + logσ(−t))
  S_D = lam_dead · d
  ```

  Com `t = d = 0` as três pontuações valem exatamente 0: sem evidência, nenhum estado é favorecido e o
  estado anterior persiste, que é a regra "bola solta fica com o último time".
- **Opções consideradas:**
  - **Fórmula C3 da v2 (escolhida).** Propriedades: neutra sem evidência; trocar `t` por `−t` troca `S_H`
    com `S_A`; `S_H − S_A = lam_team · t`; evidência forte de time empurra "vivo" no máximo
    `lam_team · log 2` por frame.
  - Fórmula da v1 (`S_D = logσ(d)`, `S_H = logσ(−d) + logσ(t)`): descartada por erro. Com `d = 0` vale
    `S_H = S_D + logσ(t) < S_D` para **qualquer** `t`: D vence H e A mesmo com evidência clara de posse.
- **Evidência:**
  - demonstração algébrica acima;
  - simulação sintética do plano: 10 s de posse da casa com 3 s sem bola. A v1 decodifica tudo como `D`;
    a fórmula corrigida decodifica tudo como `H`;
  - as propriedades são testadas em `tests/test_modeling_emission.py` (dados sintéticos).

### D-004 — Ativos usados e descartados

- **Data:** 2026-11-05
- **Autor:** equipe (plano v2); registrada por Bernardo (F3-05)
- **Status:** Vigente. A linha do tracking da transmissão será atualizada por uma entrada nova depois do
  portão GO/NO-GO da F4-05 (qui 12/11).
- **Decisão:**

  | Ativo | Situação | Como é usado, ou por que não |
  | :--- | :--- | :--- |
  | ATD tático (tracking automático) | **Usado** | Base de tudo: bola, jogadores e todas as features do master. |
  | Smart Tagging | **Usado, como evidência fraca** | Só nos termos de bola parada do `dead_logit` (pesos baixos) e na conferência dos frames de pontapé. As contagens de fase (`n_tags_favor_*`) ficam no master, mas não entram no modelo. |
  | Homografia tática | **Fora do modelo** | Só diagnóstico. `reproj_error` é coluna opcional do master e é proibido usá-lo como feature (Apêndice B, item 14). |
  | Vídeo tático | **Fora do modelo** | Só conferência visual humana (por exemplo, o sentido de ataque). Detecção própria da bola no vídeo está fora do escopo. |
  | Vídeo da transmissão | **Fora do modelo** | Não é alinhado ao vídeo tático, tem cortes, replays e zoom. |
  | Tracking da transmissão | **Trilha exploratória** | Só entra se passar no portão de viabilidade (F4-05) e no protocolo de seleção (F4-06, D-006). Caso contrário, é relatado como "testado e descartado". |

- **Opções consideradas:**
  - **Modelo só com o ATD tático, com as tags como evidência fraca (escolhida).**
  - Tags com peso alto: descartada. São automáticas, têm recall baixo, todo lateral vem com `Team = N/A`
    (DATA.md) e a janela de uma bola parada vai além do reinício, o que atrasaria o início da posse. Por
    isso `is_set_piece` pesa +1,0 (era +2,5 na v1).
  - Modelo de visão sobre os vídeos: descartada. Pouco tempo, inferência determinística em CPU mais difícil,
    e o ganho esperado é menor que o da calibração no IDSSE.
  - `reproj_error` como feature: descartada. Não existe nos mestres externos (59 colunas) e deixa o modelo
    decorar o momento do jogo.
- **Evidência:** Seção 2.2 e Apêndices A.3 e B do [contexto global](plano/00_contexto_global.md); DATA.md
  (seções "Smart Tagging", "Homography" e "The broadcast recording").

### D-005 — Arquitetura: HMM de 3 estados com Viterbi por tempo

- **Data:** 2026-11-05
- **Autor:** equipe (plano v2); registrada por Bernardo (F3-05)
- **Status:** Vigente
- **Decisão:** o método é um HMM de 3 estados (H, A, D). As evidências por frame são duas razões de
  log-verossimilhança (`team_logit` e `dead_logit`, contrato C2). A decodificação é um Viterbi offline e
  bidirecional, separado por tempo de jogo, com duração mínima por estado. A mesma função
  (`modeling.pipeline.predict_labels`) roda no BAR-ATH e no IDSSE.
- **Opções consideradas:**
  - **HMM com poucos parâmetros interpretáveis (escolhida):** evidência ausente vale 0 e o estado persiste;
    a sequência é globalmente ótima, sem oscilação (que destrói o spellF1) e sem atraso de fase.
  - Classificador por frame (GBDT ou rede) como método principal: descartada. Alto risco de mudança de
    domínio IDSSE→ATD, pouco tempo e determinismo mais difícil. Variantes GBDT só das evidências continuam
    como candidatas opcionais (F1-07, F3-08), sujeitas à D-006.
  - Filtros causais: descartada. O spellF1 pune atraso (8 frames → 0,89; 1 s → 0,65, EVALUATION.md), e o uso
    offline é permitido (issue oficial #15).
- **Evidência:** Seção 2.2 do [contexto global](plano/00_contexto_global.md); EVALUATION.md.

### D-006 — Seleção pré-registrada, sem a janela pública

- **Data:** 2026-11-05
- **Autor:** equipe (plano v2); registrada por Bernardo (F3-05)
- **Status:** Vigente
- **Decisão:** a variante final é escolhida pelo protocolo da Seção 6 do
  [contexto global](plano/00_contexto_global.md), fixado antes de existir qualquer resultado:
  - critério: spellF1 médio nos 7 mestres degradados (`default`), com a partida avaliada fora do ajuste;
  - cada passo, do mais simples para o mais complexo, só é aceito se o ganho pareado médio for maior que
    1 erro-padrão **e** se o passo não piorar mais que 1 erro-padrão nas degradações `harder` e `easier`;
  - a janela pública (frames 11185…18684) serve só para checagens `PASS`/`FAIL` de correção (alinhamento,
    formato, sentido dos rótulos). O score dela nunca é usado para escolher nem relatado.
- **Opções consideradas:**
  - **Protocolo pré-registrado no IDSSE (escolhida).**
  - Escolher pelo score da janela pública: proibida pelas regras (5 minutos, 17 posses) e vetada na issue
    oficial #10.
  - Aceitar qualquer ganho médio no IDSSE: descartada. Diferenças de spellF1 menores que ~0,04 são ruído no
    jogo inteiro, e a degradação é só uma aproximação do ATD.
- **Evidência:** issue oficial #10; EVALUATION.md; o resultado de cada passo ficará em `reports/selection.md`
  (F2-06).
- **Esclarecimento (13/11):** os filtros eliminatórios no BAR-ATH (CSV válido, checagens da janela pública,
  faixas de plausibilidade tiradas do IDSSE) e a checagem de estabilidade são checagens automáticas, que a
  resposta dos organizadores permite. Eles só eliminam candidatas; nunca as ordenam nem ajustam parâmetros.
  Ver D-012.

### D-007 — Features de bola morta: interpretações da especificação

- **Data:** 2026-11-10
- **Autor:** Bernardo (F3, F3-01)
- **Status:** Vigente
- **Decisão:** pontos que a especificação da F3-01 deixava em aberto, em `modeling/deadball/features.py`:
  1. **Saída tem precedência sobre bola aérea.** Uma bola rápida que sai pela linha é `saída` (2), nunca
     `aérea` (3).
  2. **Lado da linha mais próxima sem o comprimento do campo.** `dead.params` só tem `pitch_w_m`. O lado é
     lido das próprias distâncias: no lado baixo `coord − d = 0`; no lado alto `coord − d = tamanho − 2d`,
     que perto de uma linha é maior que meia largura do campo nos dois eixos. Teste: `2·(coord − d) < pitch_w_m`.
  3. **Os frames observados de uma lacuna são os adjacentes a ela, no mesmo tempo.** Uma lacuna encostada no
     limite do tempo usa só o outro lado.
  4. **Média móvel centrada:** a janela do frame `i` é `[i − w//2, i + (w−1)//2]`, cortada nas pontas do
     bloco (média dos frames disponíveis).
  5. **Indicadores 0/1 em `int8`**, valendo 0 fora dos tempos; as features reais valem NaN fora dos tempos.
- **Opções consideradas:**
  - (2) acrescentar `pitch_l_m` a `dead.params`: descartada, porque muda o esquema de config (A.7), que é
    contrato;
  - (2) comparar `coord == d` com tolerância: descartada, porque exige uma tolerância sem justificativa;
  - (4) janela cheia obrigatória (NaN nas pontas): descartada, porque apagaria a evidência justamente no
    início e no fim de cada tempo;
  - (5) indicadores em `float32` com NaN: descartada, porque os indicadores são sempre definidos (a ausência
    de bola já vale 0).
- **Evidência:** `tests/test_modeling_deadball_features.py` (81 testes sintéticos), com um teste por ponto
  acima.

### D-008 — `dead_logit` `rules_v1`: rampas no código e convenção de `params`

- **Data:** 2026-11-10
- **Autor:** Bernardo (F3, F3-02)
- **Status:** Vigente
- **Decisão:** em `modeling/deadball/dead.py`:
  1. **Os pontos das três rampas do Apêndice A.3 são constantes nomeadas no código** (`STILL_RAMP`,
     `SPEED_HIGH_RAMP`, `SPEED_LOW_RAMP`), e não chaves de config. Os 13 pesos vêm de `dead.params.weights`.
  2. **A variante recebe `cfg["dead"]["params"]`**, a mesma convenção de `team_logit_rules_v1` (já no
     `main`) e de `build_deadball_features`. O texto da F5-04 fala em passar `cfg["dead"]`.
  3. **`dead_tag_prior` devolve a soma dos quatro termos de tag sem recorte**, valendo 0 onde `half == 0`.
     O recorte em ±12 é feito uma vez, no fim, por quem soma (`rules_v1` agora, `calibrated` na F3-04).
- **Opções consideradas:**
  - (1) acrescentar as rampas a `dead.params`: descartada por enquanto. O esquema do Apêndice A.7 é contrato
    (C7, validação estrita) e só tem os pesos; mudá-lo exige issue aprovada pela F5. As rampas definem a
    forma da variante `rules_v1`, e a variante `calibrated` (F3-04) não as usa;
  - (2) aceitar tanto `cfg["dead"]` quanto `cfg["dead"]["params"]`: descartada, porque esconderia um erro de
    integração. Com a seção inteira, a função falha com `KeyError` em vez de devolver um resultado errado;
  - (3) recortar o prior das tags: descartada, porque recortar duas vezes mudaria a soma perto do limite.
- **Evidência:** `tests/test_modeling_dead.py` (51 testes sintéticos); contrato C2 conferido no mock com
  `validate_logits`; teste de mutação manual (8 alterações no código, todas detectadas pelos testes).

### D-009 — Mestres `__none` só para diagnóstico

- **Data:** 2026-11-11
- **Autor:** Ahmed (F5, revisão de 11/11)
- **Status:** Vigente
- **Decisão:** os mestres externos `__none` (IDSSE sem degradação) servem só para diagnóstico. As
  calibrações (F1-05, F3-04), a varredura (F2-05) e a seleção (F2-06) usam apenas os mestres degradados. Se a
  F4-04 atrasar, essas tarefas esperam; no pior caso vale o passo 8 da Seção 6. Esta decisão substitui o
  gatilho da Seção 7 que mandava calibrar nos `__none`.
- **Opções consideradas:**
  - **Esperar pelos mestres degradados (escolhida).**
  - Calibrar nos `__none` se a F4-04 atrasar (gatilho anterior da Seção 7): descartada. No IDSSE limpo, a
    posição da bola fica congelada durante a bola morta. Um modelo de bola morta ajustado nesses mestres
    aprenderia "bola parada ⇒ D", o que não vale no tracking automático.
- **Evidência** (só dados públicos do IDSSE, medidos com um script exploratório não commitado):
  - nas partidas J03WMX, J03WN1, J03WOH e J03WOY, a velocidade da bola fica abaixo de 0,3 m/s em
    99,8%–99,9% dos frames de D, contra 1%–2% dos frames de bola viva, e a bola é visível em 100% dos
    frames de D;
  - o artefato sobrevive à degradação atual (`configs/degrade_default.yaml`): no J03WOY degradado, a bola
    fica parada por 1 s (deslocamento menor que 0,5 m) em 61% dos frames de D com bola visível, contra 3%
    com a bola viva;
  - a estatística "fração de bola parada por 1 s" passa a fazer parte da checagem de realismo da F4-04
    (atualização de 11/11 na #74); ver também as atualizações de 11/11 nas #79, #80 e #81.

### D-010 — Integração do HMM no `predict_labels`

- **Data:** 2026-11-11
- **Autor:** Ahmed (F5, F5-04)
- **Status:** Vigente
- **Decisão:** em `modeling/pipeline.py`:
  1. **As variantes recebem `cfg["team"]["params"]` e `cfg["dead"]["params"]`**, e não a seção inteira, como
     fixou a D-008. O texto da F5-04 mandava passar `cfg["dead"]`.
  2. **Enquanto `modeling/decode/postprocess.py` (F2-04) não existir, o pós-processamento é a identidade
     mais a imposição final** (`force_d` → D, `half == 0` → X). O módulo da F2-04 é usado automaticamente
     quando existir. Um config que ligue alguma regra antes disso é erro (`NotImplementedError`), e um erro
     de import dentro do módulo não é escondido. Os `post_stats` vão para o log.
  3. **`validate_logits` recebe `half`**, para conferir a regra do C2 de que os dois logits valem 0 onde
     `half == 0`.
  4. **`DecoderParams.from_config` recebe os FPS de `games.py`**, a mesma fonte que `load_config` usa para
     validar `mean_duration_s` contra `min_dwell_frames`.
- **Opções consideradas:**
  - (2) levantar erro sempre que a F2-04 faltar: descartada, porque bloquearia a F5-05 por uma tarefa P1
    cujas regras ficam desligadas no v1;
  - (2) ignorar em silêncio as regras ligadas: descartada, porque o config diria uma coisa e o CSV faria
    outra;
  - (4) o padrão `fps=25` de `from_config`: descartado, porque é um número fixo no código, enquanto
    `games.py` já tem a taxa da partida.
- **Evidência:** `tests/test_modeling_pipeline_hmm.py` (29 testes sintéticos) e PR #117. Execução local
  (humano) no BAR-ATH: CSV válido e as 7 checagens de `tools/check_public_window.py` em `PASS`; config `v1`
  com sha256 `cef43128…`, CSV com sha256 `783dff2d…` (valores completos na #117).

### D-011 — `calibrated_file` chega às variantes dentro de `params`

- **Data:** 2026-11-11
- **Autor:** Ahmed (F5, revisão de 11/11)
- **Status:** Vigente
- **Decisão:** `modeling.pipeline.variant_params(section)` monta o `params` que cada variante recebe:
  1. sem `calibrated_file` na seção (`null`), a variante recebe `params` sem mudança, como na D-010;
  2. com `calibrated_file`, ela recebe uma **cópia** de `params` com a chave a mais
     `params["calibrated_file"]`, um caminho absoluto. Um caminho relativo é lido a partir da raiz do
     repositório, para o resultado não depender do diretório de trabalho;
  3. o config nunca é modificado, e `params` com a chave `calibrated_file` é erro (ela pertence à seção).
- **Motivo:** a D-010 passava só `params`, e o esquema do C7 deixa `calibrated_file` no nível da seção, onde
  nenhuma variante o via. Os tipos `calibrated` (F1-05, F3-04) e `gbdt` (F1-07, F3-08) não teriam como achar
  os pesos ajustados. As atualizações de 11/11 das #79 e #81 já anunciavam esta interface.
- **Opções consideradas:**
  - passar a seção inteira às variantes: descartada, porque mudaria a assinatura dos `rules_v1` (D-008);
  - pôr `calibrated_file` dentro de `params` no YAML: descartada, porque muda o esquema do C7 e o
    `load_config` já valida a chave no nível da seção.
- **Evidência:** `tests/test_modeling_pipeline_hmm.py` (testes de `variant_params` e de um tipo
  `calibrated` registrado no teste). O v1 não muda: as duas seções têm `calibrated_file: null`.

### D-012 — Resposta dos organizadores de 13/11 e candidatas novas

- **Data:** 2026-11-13
- **Autor:** Ahmed (F5)
- **Status:** Vigente
- **Contexto:** a equipe perguntou por e-mail se um ajuste não supervisionado (EM/Baum-Welch) do HMM sobre o
  tracking do BAR-ATH, sem nenhum rótulo, era permitido. A resposta (e-mail de 13/11) repetiu o e-mail de
  esclarecimento:
  - **não permitido:** conferir a saída contra o vídeo; rótulos à mão ou qualquer dado produzido por pessoas
    sobre o jogo; enviar os dados fornecidos a serviços de IA; treinar, ajustar ou calibrar no GT de posse ou
    no vídeo fornecido;
  - **permitido:** checagens automáticas usando só os dados fornecidos; inspecionar os dados e o vídeo para
    entender o que conta como posse; usar como entrada o vídeo, o tracking, o Smart Tagging ou outros dados
    gerados, desde que o algoritmo não tenha sido treinado no GT nem no vídeo; entregar arquivos de etapas
    intermediárias; usar qualquer dataset com direito de uso;
  - "If you need to train (or retrain), you MUST use only external data. If your algorithm has parameters
    that need to be tuned, you MUST tune them on external data", e "anything that is not explicitly allowed
    is not permitted".
- **Decisão:**
  1. **Nenhum ajuste no BAR-ATH, nem não supervisionado.** O EM proposto em 11/11 não será feito. A D-002
     continua valendo, agora confirmada.
  2. **Os filtros eliminatórios e a checagem de estabilidade da F2-06 ficam como estão:** são checagens
     automáticas, com faixas externas, que só eliminam. Ficam documentados no relatório técnico com essa
     natureza.
  3. **Ninguém confere a saída contra o vídeo**, nem para depurar (por exemplo, rótulos previstos sobrepostos
     ao vídeo). Olhar o vídeo para entender a definição de posse continua permitido.
  4. **Três candidatas novas para a F2-06**, todas ajustadas só no IDSSE e com a saída do v1 byte a byte
     igual quando desligadas: F1-08 (hazard de troca, #124), F2-07 (transições guiadas, #125) e F2-08
     (durações semi-Markov, #126). Entram na escada da Seção 6 depois de `d_cross_frames`, com a mesma regra
     de aceitação.
  5. **Hash do config:** F2-07 e F2-08 acrescentam chaves obrigatórias a todos os configs. O hash do
     `configs/v1.yaml` muda, mas os rótulos não; o envio de segurança registra o commit e o hash da época.
- **Opções consideradas:**
  - EM não supervisionado no BAR-ATH: descartado pela resposta acima.
  - Trocar o fallback dos filtros eliminatórios por "investigar bug": descartado. As checagens são
    permitidas, e sem o fallback uma vencedora degenerada no BAR-ATH poderia ser enviada.
  - Pesos das tags do Smart Tagging ajustados no BAR-ATH: proibido; os pesos fixos do Apêndice A.3 continuam.
- **Evidência:** e-mail dos organizadores de 13/11/2026 (hackathon.talentarena@mobileworldcapital.com);
  RULES.md; issues #124, #125 e #126.

### D-013 — Nome da equipe, formulário e envio único

- **Data:** 2026-11-13
- **Autor:** Ahmed (F5, F5-08)
- **Status:** Vigente
- **Decisão:**
  1. **Nome da equipe:** `brasileirao`, igual ao do cadastro. É o valor do campo "Team name" e o
     `<team_name>` dos arquivos: `brasileirao.csv` e `brasileirao.pdf`.
  2. **Responsável pelo envio:** Ahmed (`HachemAhmed`). O acesso ao formulário foi testado com a conta Google
     dele em 13/11. Só ele envia, porque o formulário aceita uma resposta por equipe.
  3. **Envio único na seg 16/11 (F5-09).** Não haverá envio de segurança separado: a F5-05 (#77) foi
     cancelada. O prazo oficial (17/11) não informa horário nem fuso, então
     17/11 fica só como folga.
  4. **Campos do formulário:** confirmação de e-mail (obrigatória); Team name; **Team members names**, campo
     que não aparece no SUBMISSION.md (preencher com os nomes reais completos dos 5 membros, como no
     cadastro); Possession file `.csv` (até 10 MB); Public code repository URL; Short technical document
     (até 10 MB).
- **Opções consideradas:**
  - Envio único em 16/11: escolhido pela equipe.
  - Envio de segurança antes, editado em 16/11 (plano original da F5-05): descartado pela equipe. Sem ele,
    o envio de 16/11 precisa sair certo de primeira; a print do formulário não mostra se a resposta pode
    ser editada depois.
- **Evidência:** e-mail dos organizadores com a URL do formulário (13/11); SUBMISSION.md, seção "How to
  submit"; issue #57.

### D-014 — Casos de borda do filtro da bola: como são contados

- **Data:** 2026-11-13
- **Autor:** Bernardo (F3, F3-07)
- **Status:** Vigente
- **Decisão:** `tools/jump_filter_edge_cases.py` conta, por tempo, a marca dos dois casos de borda de
  `remove_ball_jump_outliers`, com os limiares fixados pela F3-07. Pontos que a especificação deixava em
  aberto:
  1. **(a) A "bola mantida" inclui os frames interpolados.** Um trecho de bola parada e fora do campo pode
     atravessar lacunas curtas preenchidas pelo pré-processamento. Nesses frames a velocidade é a da própria
     interpolação.
  2. **(a) "Enquanto os jogadores correm"** vale se o trecho contém 75 frames consecutivos em que a média da
     velocidade dos jogadores é maior que 3 m/s. Um trecho em que todo frame passa de 3 m/s também é contado.
  3. **(b) O salto é medido entre duas observações mantidas consecutivas**, no mesmo tempo.
  4. **(b) "Seguido de bola parada"** vale se a bola mantida fica parada em algum dos 25 frames a partir do
     pouso.
  5. **(b) Contam todas as observações removidas nos 25 frames antes do pouso**, cada uma uma única vez. Com
     `--verbose`, a ferramenta diz também quantas delas continuam a trajetória mantida dentro do alcance
     físico do filtro; para isso usa as posições originais de `tracking_clean.parquet`.
  6. **Entrada inutilizável não é FAIL:** arquivo ausente, ou master e `tracking_clean` de execuções
     diferentes, dão `ERROR` e código de saída 2.
  7. **O filtro não foi alterado.** A correção prevista na tarefa só entra se a execução local der FAIL,
     houver tempo antes do freeze e a F5 aprovar: ela muda a saída de `preprocessing/`, que a regra global 7
     congela.
- **Opções consideradas:**
  - (1) trechos só de frames observados: descartada. No cenário sintético, o fantasma que vence a bola real
    vira uma cadeia de trechos observados de ~8 frames ligados por frames interpolados (as observações reais
    removidas). Nenhum trecho observado chegaria a 75 frames, e a checagem passaria sempre;
  - (2) exigir mais de 3 m/s em todos os frames: descartada, porque uma oscilação de um frame quebraria o
    trecho. A leitura escolhida nunca conta menos que essa;
  - (4) exigir bola parada no próprio frame do pouso: descartada. A velocidade do master é suavizada dentro
    do trecho observado e fica contaminada pelo salto nos primeiros frames;
  - (7) corrigir o filtro já: descartada, pelos motivos do item 7.
- **Evidência:** `tests/test_modeling_jump_filter_edge_cases.py` (52 testes sintéticos). Os dois cenários
  foram reproduzidos passando arrays sintéticos pelo pré-processamento real: a ferramenta conta o fantasma
  parado e as observações removidas antes da reposição, e não conta nada em jogo corrido. Teste de mutação
  manual: 14 alterações no código, todas detectadas.

### D-015 — Diagnósticos de bola morta: como são calculados

- **Data:** 2026-11-13
- **Autor:** Bernardo (F3, F3-06)
- **Status:** Vigente
- **Nota:** o id D-014 é da F3-07 (PR #130).
- **Decisão:** em `tools/deadball_diagnostics.py`:
  1. **As três estatísticas são calculadas com a mesma função que gerou as faixas** (`label_stats.half_stats`,
     F2-03). Assim, posse, D e frames X são definidos exatamente como no IDSSE.
  2. **As três checagens de faixa valem por tempo**, inclusive a duração média de posse. As faixas são o
     mínimo e o máximo dos 14 tempos do IDSSE.
  3. **Folga de ±40% multiplicativa:** a faixa aceita é `[min × 0,6, max × 1,4]`.
  4. **Só as três checagens de faixa são eliminatórias** e decidem o resultado final e o código de saída
     (o filtro do passo 4 da F2-06). As duas de limiar fixo aparecem como `PASS`/`FAIL` com a marca
     `[informative]`, como pede a tarefa ("só informativo").
  5. **Evidência de bola morta num segmento D:** basta um frame do segmento com bola fora do campo, lacuna
     classificada como saída (F3-01), jogadores lentos, tag de bola parada ativa ou pré-pontapé. "Jogadores
     lentos" é o critério em que os termos `still` ou `speed_low` do `dead_logit` `rules_v1` passam a
     contribuir: fração de parados > 0,3 ou velocidade média < 1,5 m/s (as rampas da D-008, sem limiar novo).
     O pré-pontapé conta como evidência porque é D por definição (C4).
  6. **As features de bola morta usam `dead.params` de `configs/v1.yaml`** (Apêndice A), trocável por
     `--config`.
- **Opções consideradas:**
  - (2) duração média de posse no jogo inteiro: descartada. As faixas são por tempo, e misturar os dois
    tempos compararia grandezas diferentes;
  - (4) as cinco checagens eliminatórias: descartada. A tarefa diz que as duas últimas são só informativas,
    e os limiares delas não vêm de dados externos;
  - (5) definir "jogadores lentos" com um limiar novo: descartada, porque seria um número sem origem. Os
    limiares das rampas já são priors do Apêndice A.3.
- **Evidência:** `tests/test_modeling_deadball_diagnostics.py` (43 testes sintéticos); teste de mutação
  manual (12 alterações; 11 detectadas, e a restante é equivalente, porque o caso "nenhum tempo" é barrado
  antes, na validação da entrada).

### D-016 — Degradação do IDSSE ajustada pelo realismo (F4-04)

- **Data:** 2026-11-15
- **Autor:** Ahmed (revisão e fechamento da F4-04, #74, sobre a implementação do Antônio)
- **Status:** Vigente
- **Decisão:** os mestres externos usam `configs/degrade_default.yaml` com os valores abaixo. As outras
  chaves não mudaram.

  | Chave | Antes (A.6 / 11/11) | Agora | Motivo (estatística da F4-04) |
  | --- | ---: | ---: | --- |
  | `player_noise_m` | 0,15 | 0,01 | (a) velocidade dos jogadores (feito pelo Antônio) |
  | `unfreeze_drift_mps` | — | 1,5 | (b) velocidade da bola e (c) bola parada |
  | `unfreeze_drift_tau_s` (nova) | — | 1,0 | idem: velocidade persistente, ver abaixo |
  | `p_unfreeze_hide` | — | 0,5 | grade da etapa 4 (escolha abaixo) |
  | `p_air` | 0,85 | 0,0 | (f) visibilidade com jogadores rápidos |
  | `bg_slow_speed_mps` (nova) | — | 1,0 | (f) visibilidade com jogadores lentos |
  | `bg_slow_factor` (nova) | — | 40 | idem |
  | `gap_profile.lognormal_median_frames` | 40 | 150 | (e) lacunas longas |
  | `gap_profile.mix_geometric` | 0,6 | 0,3 | (e) lacunas longas |

  Duas mudanças de código em `modeling/external/degrade.py`:
  1. **Deriva da bola morta com velocidade persistente.** Na versão anterior, cada frame recebia um passo
     independente, então a 0,5 m/s a bola andava só ~0,08 m por segundo e continuava "parada" para (c). A
     velocidade agora segue um AR(1) por eixo (tempo de correlação `unfreeze_drift_tau_s`), e a trajetória é
     uma ponte entre a posição congelada e a do reinício (sem salto nas pontas).
  2. **Lacunas de fundo mais prováveis com jogo lento.** A chance de iniciar uma lacuna de fundo é
     multiplicada por `bg_slow_factor` nos frames em que a média das velocidades médias dos dois times é menor
     que `bg_slow_speed_mps` (a mesma borda da faixa lenta de (f)). O passo lê só a velocidade dos jogadores,
     nunca os rótulos, e não muda a sequência de sorteios. `bg_slow_factor = 1` desliga.

  **Grade de `p_unfreeze_hide`** (J03WMX e J03WN1, outras chaves fixas no `default` acima):

  | p | (a) | (b) | (c) | (d) | (e) | (f) |
  | ---: | --- | --- | --- | --- | --- | --- |
  | 0,00 | PASS | PASS | PASS | FAIL | PASS | FAIL |
  | 0,25 | PASS | PASS | PASS | FAIL | PASS | FAIL |
  | 0,50 | PASS | PASS | PASS | FAIL | PASS | PASS |
  | 0,75 | PASS | PASS | FAIL | FAIL | PASS | PASS |
  | 1,00 | inviável: esconder toda a bola morta passa da cota de 40,7% de bola invisível | | | | | |

  Conjunto plausível ((b), (c) e (f) em PASS): {0,5}, então `default` = 0,5. A faixa tem largura < 0,5, então
  `hidelo` = 0,25 e `hidehi` = 0,75 (`default` ± 0,25). `harder` e `easier` = `default` com `ball_noise_m` e
  `player_noise_m` × 1,5 e × 0,5 (0,375/0,015 e 0,125/0,005).

  **Divergência que fica:** (d), posses por minuto do baseline, falha em todos os candidatos. O baseline faz
  mais posses por minuto no IDSSE degradado do que no BAR-ATH. Fica registrada aqui e vai para as limitações
  do relatório. (d) não é eliminatória pelo aceite da F4-04.
- **Opções consideradas:**
  - manter a degradação de 11/11 e só varrer `p_unfreeze_hide`: descartada. Nenhum valor passou em (b), (c)
    e (f) (p = 0; 0,25; 0,5 com (b), (c), (d), (e) e (f) em FAIL; 0,75 e 1,0 inviáveis);
  - baixar `ball_noise_m`: descartada. A velocidade da bola do BAR-ATH fica acima da do IDSSE degradado, e
    a causa era a bola parada na bola morta;
  - esconder mais bola morta (p alto) para acertar (f): limitada pela cota de visibilidade (p = 1,0 é inviável
    em todas as partidas, e 0,75 era inviável no J03WN1 com `p_air` = 0,85). A visibilidade por velocidade só
    passou com o fator de jogo lento;
  - `bg_slow_factor` = 100 (também passou em (b), (c) e (f) com p = 0,6): descartado em favor de 40, o menos
    extremo que passou.
- **Evidência:** só `PASS`/`FAIL` e, para orientar o ajuste, o lado (acima/abaixo) do BAR-ATH em relação à
  média do IDSSE degradado em cada estatística. Nenhum valor do BAR-ATH saiu da execução local. Foram 6 rodadas
  de candidatos, cada uma com 2 partidas. Os números do lado do IDSSE estão em
  [`reports/external_masters.md`](../reports/external_masters.md). Testes sintéticos novos em
  `tests/test_modeling_external_degrade.py` (deriva: deslocamento em 1 s; jogo lento: máscara, fator neutro,
  concentração das lacunas). Hashes e comandos de regeneração em [`docs/external_data.md`](external_data.md).
- **Checagem final (7 partidas, `tools/compare_external_realism.py`):** `default`: (a), (b), (c), (e) e (f)
  PASS, (d) FAIL. `hidelo`: (f) e (d) FAIL. `hidehi`: (c) e (d) FAIL. Os modos de sensibilidade cercam o
  `default` por definição e não precisam passar.
- **Item 7 da F4-04 (`bg_hazard`):** no `default`, o `bg_hazard` final fica em 61–72% do valor com
  p = 0 (sem esconder bola morta) nas duas partidas da grade (J03WMX 0,000315 contra 0,000440; J03WN1 0,000211
  contra 0,000346), acima do limite de 10% que exigiria comentar na issue antes de gerar os mestres.
- **Risco conhecido:** com esses valores, a bola do IDSSE degradado fica visível em ~30% dos frames de D e em
  ~80% dos de jogo. É o padrão que (f) mede no BAR-ATH (bola some com jogadores lentos), mas é uma pista forte
  para "bola invisível ⇒ D". A F3-04 e a F2-05 devem conferir a estabilidade em `hidelo` e `hidehi`.

## Registro de envios

Preenchido pela F5 (F5-09; envio único, ver D-013). O CSV enviado tem de ser o arquivo validado, com o mesmo SHA-256.

> **Lembrete obrigatório:** o repositório precisa estar **público** antes de 17/11/2026 em qualquer
> cenário (ver F5-09).

| Data | Envio | Config (nome e hash) | Commit ou tag | SHA-256 do CSV | `validate_submission` | Autor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| _pendente_ | Envio final (F5-09) | | | | | |
