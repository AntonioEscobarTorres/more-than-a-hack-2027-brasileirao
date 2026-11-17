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
| [D-016](#d-016--degradação-do-idsse-ajustada-pelo-realismo-f4-04) | 2026-11-15 | Degradação do IDSSE ajustada pelo realismo (F4-04) | Vigente (`harder`/`easier` substituídos pela D-017) |
| [D-017](#d-017--harder-e-easier-pelo-apêndice-a6-e-estatísticas-originais-do-realismo-f4-04) | 2026-11-15 | `harder`/`easier` pelo Apêndice A.6 e estatísticas originais do realismo (F4-04) | Vigente |
| [D-018](#d-018--hazard-de-troca-de-posse-ajuste-e-checagens-f1-08) | 2026-11-15 | Hazard de troca de posse: ajuste e checagens (F1-08) | Vigente |
| [D-019](#d-019--calibração-do-dead_logit-no-idsse-features-e-o-r2) | 2026-11-15 | Calibração do `dead_logit` no IDSSE: features e o R2 | Vigente (R2 revisado; NaN na inferência, 16/11) |
| [D-020](#d-020--trilha-da-câmera-de-transmissão-descartada-no-go-f4-05) | 2026-11-16 | Trilha da câmera de transmissão descartada: NO-GO (F4-05) | Revista para a bola pela D-024 |
| [D-021](#d-021--team_logit-calibrado-com-a-variante-touch-e-o-config-v2-revisão-da-f1-05f1-06) | 2026-11-16 | `team_logit` calibrado com a variante `touch` e o config `v2` (F1-05/F1-06) | Vigente (toque substituído pela D-022) |
| [D-022](#d-022--toque-só-por-persistência-e-o-config-v2_base-checagem-do-toque-no-bar-ath) | 2026-11-16 | Toque só por persistência e o config `v2_base` (checagem do toque no BAR-ATH) | Vigente |
| [D-023](#d-023--team_logit-gbdt-e-o-config-v2_gbdt-revisão-da-f1-07) | 2026-11-16 | `team_logit` GBDT e o config `v2_gbdt` (revisão da F1-07) | Vigente |
| [D-024](#d-024--bola-fundida-da-transmissão-f4-06-revisão-da-d-020) | 2026-11-16 | Bola fundida da transmissão (F4-06): revisão da D-020 | Vigente (opção B) |
| [D-025](#d-025--seleção-f2-06-vencedora-v2_gbdt_fused) | 2026-11-16 | Seleção F2-06: vencedora `v2_gbdt_fused` | Vigente |

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
- **Status:** Vigente. A definição de `harder` e `easier` foi substituída pela D-017.
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
- **Termômetro independente (feito depois da escolha, nunca usado para ajustar):** o starter dos organizadores
  (`baseline_smooth50`: jogador mais próximo da bola, votação em ±50 frames) rodado no IDSSE, 7 partidas,
  seed 0 + índice, comparado com os números publicados por eles no BAR-ATH (`EVALUATION.md`):

  | Dados | macroF1 | spellF1 | posses previstas / reais |
  | --- | ---: | ---: | ---: |
  | BAR-ATH (publicado) | 0,672 | 0,243 | não publicado |
  | IDSSE limpo | 0,392 | 0,308 | 0,87 |
  | IDSSE degradação anterior (`main` 0b4e9d8) | 0,416 | 0,219 | 1,37 |
  | IDSSE degradação desta decisão | 0,629 | 0,254 | 1,13 |

  A distância em macroF1 cai de 0,256 para 0,043, e o spellF1 fica a 0,011 do publicado (o ruído do spellF1 é
  ~0,04). O macroF1 do starter depende quase todo de "bola invisível ⇒ D". Como ele também é alto no BAR-ATH, a
  bola do ATD de fato some mais na bola morta, o que reduz o risco abaixo, sem eliminá-lo.
- **Risco conhecido:** com esses valores, a bola do IDSSE degradado fica visível em ~30% dos frames de D e em
  ~80% dos de jogo. É o padrão que (f) mede no BAR-ATH (bola some com jogadores lentos), mas é uma pista forte
  para "bola invisível ⇒ D". A F3-04 e a F2-05 devem conferir a estabilidade em `hidelo` e `hidehi`.

### D-017 — `harder` e `easier` pelo Apêndice A.6 e estatísticas originais do realismo (F4-04)

- **Data:** 2026-11-15
- **Autor:** Ahmed (revisão da F4-04 depois do merge do #132)
- **Status:** Vigente
- **Decisão:**
  1. **`harder` e `easier` seguem o Apêndice A.6** sobre o `default` da D-016. A D-016 tinha só multiplicado o
     ruído da bola e dos jogadores, e os dois modos ficaram quase iguais ao `default`, o que esvaziava o teste de
     estresse da F2-06. Agora:

     | Chave | `default` | `harder` | `easier` | Origem |
     | --- | ---: | ---: | ---: | --- |
     | `target_ball_visible` | 0,593 | 0,50 | 0,68 | A.6 |
     | `phantom_frac` | 0,01 | 0,02 | 0,005 | A.6 |
     | `player_noise_m` | 0,01 | 0,0167 | 0,0067 | A.6 (0,25 e 0,10 sobre 0,15), reescalado para o `default` novo (item 6 de 11/11) |
     | `target_tracked_mean` | 20,5 | **20,0** | **21,3** | A.6 pede 19,5 e 21,5; ver abaixo |
     | `ball_noise_m` | 0,25 | 0,25 | 0,25 | o A.6 não muda o ruído da bola |

     As outras chaves são as do `default`, inclusive `p_unfreeze_hide` = 0,5.
  2. **Metas de jogadores viáveis nas 7 partidas.** O J03WN1 rastreia em média no máximo 21,05 jogadores, mesmo
     sem esconder nenhum, então 21,5 ± 0,3 é inviável no `easier`. 21,3 é o maior valor viável. No `harder`, com
     19,5 quase não sobram frames com 22 jogadores no J03WN1, e a fração de duplicatas (`dup_frac` 0,02) não é
     alcançada nem com 200 mil tentativas. 20,0 é viável e continua abaixo do `default`.
  3. **Estatísticas originais no script de realismo.** O item 3 original pedia comparar com o BAR-ATH as
     estatísticas do relatório. A atualização de 11/11 acrescentou (a)–(f) sem tirar essas.
     `tools/compare_external_realism.py` agora também tem (g) bola observada, (h) bola preenchida, (i) saltos da
     bola removidos e (j) média de jogadores rastreados, todas com ±20%. O % de D não é comparado, porque o
     BAR-ATH não tem rótulos fora da janela pública, que não pode ser usada.
  4. **Divergências que ficam (sem novo ajuste):** (h) e (i) dão `FAIL` no `default`. São frações pequenas (bola
     preenchida ~0,3% e saltos removidos 1–3% dos frames do IDSSE degradado) e vêm do pré-processamento, não de
     um alvo da degradação. Não houve rodada de ajuste nova contra o BAR-ATH.
- **Opções consideradas:**
  - `harder` com `dup_frac` 0,01 e 19,5 jogadores: também viável no J03WN1. Descartada para manter as
    duplicatas do `default` e fazer nos dois modos o mesmo tipo de ajuste (só a meta de jogadores);
  - gerar `harder`/`easier` sem o J03WN1: descartada, porque o aceite pede os 42 mestres;
  - ajustar a degradação para (h) e (i): descartada, pelo mesmo motivo da D-016 (regra dos organizadores e prazo).
- **Evidência:** `tools/compare_external_realism.py --degrade default` (7 partidas, sem `--verbose`): (a) PASS,
  (b) PASS, (c) PASS, (d) FAIL, (e) PASS, (f) PASS, (g) PASS, (h) FAIL, (i) FAIL, (j) PASS. Estatísticas por
  partida dos modos novos em [`reports/external_masters.md`](../reports/external_masters.md); hashes em
  [`docs/external_data.md`](external_data.md). Testes: `test_harder_and_easier_follow_appendix_a6_relative_to_the_new_default`
  e `test_build_runs_the_real_degradation_on_the_synthetic_match` (o teste do build antes caía num fallback sem
  degradar).

### D-018 — Hazard de troca de posse: ajuste e checagens (F1-08)

- **Data:** 2026-11-15
- **Autor:** Ahmed (F1-08, #124)
- **Status:** Vigente
- **Decisão:**
  1. **`bridge_d_frames = 0`.** Nos rótulos do IDSSE nenhum D entre dois times dura menos de 41 frames, e o D
     entre times diferentes dura o mesmo que o D dentro de um time (medianas 476 e 551 frames). É bola parada,
     não passagem de posse, então o alvo conta só trocas diretas H↔A.
  2. **`W` escolhido junto com `C`** em {12, 25, 50} × {0,01; 0,1; 1}, pela log-loss deixando uma partida de
     fora, aninhada em cada dobra. `W` não muda o alvo. Os 8 modelos escolheram 50, a borda da grade, com ganho
     pequeno por degrau. A grade não foi estendida depois de ver o resultado.
  3. **Checagem nova da "bola invisível" (risco da D-016):** se o AUC isolado de uma feature de lacuna mudar mais
     que 0,10 entre `hidelo` e `hidehi`, as três saem do ajuste. A margem também vale para a checagem de bola
     parada do item 7. Nenhuma disparou: as maiores diferenças foram 0,005 (lacunas) e −0,38 (bola parada, no
     sentido oposto ao artefato).
- **Opções consideradas:** `bridge_d_frames = 25` (o valor inicial) não juntava nada e foi trocado por 0 para
  deixar isso explícito; escolher `target_w_frames` pela log-loss foi descartado, porque ele muda o alvo e as
  log-losses deixam de ser comparáveis (fica para a F2-07).
- **Evidência:** [`reports/calibration_hazard.md`](../reports/calibration_hazard.md): AUC fora da dobra 0,71
  no `default` (0,68–0,74 nos outros modos). Nas trocas que o v1 perde, o pico do hazard tem AUC 0,72 contra
  janelas sem troca. Só IDSSE.

### D-019 — Calibração do `dead_logit` no IDSSE: features e o R2

- **Data:** 2026-11-15 (revisada no mesmo dia: R2)
- **Autor:** Bernardo (F3, F3-04)
- **Status:** Vigente
- **Decisão:**
  1. **`db_ball_rest` e `ball_speed` ficam fora do ajuste, sempre** (atualização de 11/11, item 3). No IDSSE a
     bola congela durante a bola morta, e as duas vazariam o rótulo. `db_fast_in_pitch` continua.
  2. **Imputação só na matriz de desenho, sempre com indicador** (regra 4): `out_of_pitch` usa `has_ball`;
     `db_still_frac`, `db_mean_speed` e `log1p(gap_length)` ganham uma coluna `*_missing`. No IDSSE elas nunca
     faltam, então os indicadores ficam constantes e com peso 0.
  3. **`C` escolhido pela log-loss média com uma partida de fora**, empate para o menor `C`. A log-loss
     de validação usa os frames subamostrados da partida de fora.
  4. **O arquivo guarda os `dead.params` que constroem as features**, e o tipo `calibrated` recusa um config
     com valores diferentes (os pesos das tags podem mudar).
  5. **Interações entre os três fatores de uma lacuna** (conjunto `interactions`): os produtos dois a dois do
     tipo de lacuna (`gap_exit`, `gap_long`), do comprimento (`log_gap`) e da velocidade média dos jogadores
     (`mean_speed`): `exit_x_log_gap`, `long_x_log_gap`, `exit_x_speed`, `long_x_speed` e `speed_x_log_gap`. Os
     produtos usam os valores já imputados, e o indicador de cada fator continua marcando a falta (regra 4).
     A lacuna aérea fica sem interação: é rara e nunca passa de `air_max_frames`.
  6. **O conjunto é escolhido pela log-loss, nunca pelo R2.** O ajuste roda o `full` e o `interactions` e
     fica com o de menor log-loss média com uma partida de fora entre os que passam no R1 em todos os
     modelos; empate fica com o `full`, o mais simples. Se nenhum passar no R1, entra o `reduced` (item 4 da
     atualização de 11/11). O R2 só é relatado. O relatório mostra os dois conjuntos lado a lado.
  7. **Sinais esperados pela inclinação efetiva.** Com interações, o peso de `mean_speed` sozinho não é o
     efeito da velocidade. A checagem usa a inclinação do `d_fit` em cada feature, na média do treino e em
     cada cenário R1–R3. Sem interações ela coincide com a checagem antiga (sinal do peso).
  8. **Resultado:** os 4 arquivos usam o `interactions`, com os três sinais esperados. No `dead.yaml`
     (`C = 1,0`), a log-loss com uma partida de fora cai de 0,2517 (`full`) para 0,2476 e melhora em 6 das 7
     partidas. **R1 e R2 passam nos 8 modelos**: R1 = −2,44 no `all` (dobras de −2,68 a −2,28), R2 = +1,03
     (dobras de +0,81 a +1,21). O R3 vai de +0,35 para −0,56, ou seja, a posse cadenciada com lacuna agora
     puxa para jogo vivo. `dead_pooled` e `dead_hidehi` também passam no R2 (+0,89 e +1,49). **O
     `dead_hidelo` falha no R2** (−0,44; ver abaixo). O `calibrated` segue para a F2-06.
- **Por que o R2 falhava no conjunto `full` (e por que a versão anterior desta decisão estava errada):** a
  versão anterior dizia que os dados do IDSSE contradizem o R2, porque só 26% dos frames de lacuna de saída
  com os jogadores entre 2,3 e 2,7 m/s são bola morta. Esse número junta lacunas de todos os comprimentos, e
  o R2 fixa `gap_length = 50`. Só com as lacunas de até 100 frames, a mesma faixa de velocidade tem 0,56 de
  bola morta (296 frames); com todas, 0,26, porque as lacunas de centenas de frames dominam a contagem. Uma
  lacuna de saída longa com os jogadores correndo é quase sempre jogo vivo depois do reinício, com a bola
  ainda invisível; uma curta é a bola morta que acabou de começar. Nos frames parecidos com cada cenário
  (mesmo tipo de lacuna, lacuna ≤ 100 frames, jogadores a ±0,4 m/s), o valor fiel aos dados é
  **R1 = −2,27 [−3,38; −1,59]** e **R2 = +0,70 [+0,07; +1,27]** (intervalo de 90%, bootstrap por lacuna,
  39 lacunas no R2). O `full` acerta o R1 (−2,02),
  mas dá −1,15 no R2, fora do intervalo. Com um único coeficiente para `log_gap`, ele aprende "lacuna longa ⇒
  bola morta", que vale com os jogadores lentos e se inverte com eles correndo. O `interactions` fica dentro
  dos dois intervalos (−2,44 e +1,03). O pedido de revisão do R2 (jogadores a ~1,5 m/s) feito na issue #81
  fica sem efeito.
- **Limite (`hidelo`):** no `hidelo` a bola fica escondida em menos bolas mortas (`p_unfreeze_hide` 0,25).
  Duas coisas pioram ali. O dado do cenário fica fraco: R2 = +0,57 [−0,24; +1,27] (32 lacunas), com o
  intervalo cruzando o zero. E o `dead_hidelo` fica abaixo dos dados nos dois cenários: R1 = −3,74 (dados
  −2,24 [−3,61; −1,51]) e R2 = −0,44, negativo com qualquer `C` da grade (−0,59 a −0,41, exploração desta
  revisão). Ou seja, o R2 depende de quanto a bola some na bola morta, a principal incerteza da D-016. O
  `dead_hidelo` não entra na seleção (item 7 da atualização de 11/11), mas a checagem de estabilidade da
  F2-06 deve olhar esse ponto. O `dead_pooled`, que vê os três modos, passa no R2 (+0,89).
- **Avaliação cruzada (item 5):** log-loss média do `dead` com cada dobra na partida de fora: `default`
  0,2479, `hidelo` 0,2733 e `hidehi` 0,2097, menores que as do `full` nos três modos (0,2519, 0,2750 e
  0,2142). Em relação ao `default`, o `hidelo` fica +10,3% (+10,8% no `dead_pooled`), acima do limite de
  10% do item 5: fica registrado aqui. A razão sobe porque o `default` melhora mais que o `hidelo`, não
  porque o `hidelo` piora.
- **Opções consideradas** (os números das variantes intermediárias vêm da exploração desta revisão, com o
  mesmo ajuste e as mesmas dobras, e não estão no relatório):
  - manter o `full` com o R2 falhando (versão anterior desta decisão): descartada, porque os dados apoiam o
    R2 no comprimento de lacuna do cenário e o `interactions` tem log-loss menor;
  - forçar o R2 a passar (peso mínimo para `gap_exit`, ou +3 fixo como no `rules_v1`): descartada. Seria um
    número escolhido por nós, e o R2 nunca entra na escolha;
  - só `gap_exit` × velocidade: log-loss quase igual à do `full` e R2 ainda negativo (−0,91);
  - só os produtos com `log_gap`: R2 positivo, mas com margem de +0,01 a +0,02 na pior dobra e log-loss pior
    que a do conjunto completo de interações;
  - uma feature de posição dentro da lacuna (frames desde o início): é a descrição mais fiel (a evidência da
    saída está no começo da lacuna), mas os cenários da tarefa não dizem a posição, então pediria uma
    mudança da especificação (regra 11). Fica como melhoria futura;
  - melhorar o detector de saída da F3-01: não ataca a causa principal (lacunas que atravessam o reinício) e
    muda features compartilhadas com o `rules_v1`.
- **Evidência:** `reports/calibration_dead.md`: escolha do conjunto (log-loss por partida e R1–R3 de cada
  conjunto), cenários contra os dados com intervalo, tabela de P(D) com todas as lacunas e só com as de até
  100 frames, inclinações efetivas, avaliação cruzada e AUC de cada feature sozinha. O ajuste é
  reprodutível: rodando de novo, os 4 arquivos e o relatório saem idênticos byte a byte. Testes:
  `tests/test_modeling_calibration_dead.py` (sintéticos).
- **Revisão da F5 (16/11):**
  - **Falta na inferência (corrige o item 2).** Como `db_still_frac` e `db_mean_speed` nunca faltam no
    IDSSE, imputar 0 na inferência lia uma velocidade NaN como "todos parados": `d_fit` de cerca de +4,2
    no `dead.yaml` (−3 com a fração de parados NaN), contra a regra 4 (NaN é "sem evidência"). Agora, só na
    inferência, a entrada cujo indicador nunca disparou no ajuste recebe a média do treino do próprio
    modelo (`inference_fill`): o valor padronizado fica 0 e as interações usam esse mesmo valor. Uma
    entrada que faltou no ajuste mantém a convenção ajustada (0 mais o indicador). O ajuste não muda: os 4
    arquivos e o relatório saem idênticos. No BAR-ATH as duas features são finitas em todos os frames dos
    tempos (`PASS`), então a submissão não muda; é uma proteção.
  - **Deslocamento das entradas.** O modelo depende muito de `db_mean_speed` (inclinação de cerca de
    −2,8 por m/s na média do treino), e a checagem (a) da F4-04 só compara a mediana bruta com ±20%. Nova
    ferramenta `tools/dead_feature_shift.py` (execução local, só `PASS`/`FAIL`): finitude das duas
    features, quantis 10/50/90 de `db_mean_speed` a até 0,25 m/s da referência IDSSE e, informativa, a
    fração de `d_fit > 0`. Resultado no BAR-ATH: `PASS` em todas, com `default` e com os três modos, para
    `dead.yaml` e `dead_pooled.yaml`.
  - **Efeito no score (spellF1 no IDSSE, uma partida de fora por vez, decodificador de
    `decoder_idsse.yaml`, melhor `lam_dead` de cada variante):** com o `team` `rules_v1`, o `calibrated`
    piora (−0,007 `default`, −0,013 `hidehi`, ≈ 0 `hidelo`), apesar da AUC por frame maior (0,963 contra
    0,937). Com o `team` calibrado (F1-05), ganha: `lam_dead` 0,2 contra `rules_v1` com 0,35 dá +0,011
    no `default` (6/7 partidas), +0,029 no `hidelo` (7/7) e −0,008 no `hidehi` (3/7); média de 0,362 para
    0,373. O `dead_pooled` fica igual. **Na F2-06, testar o `dead` calibrado junto com o `team`
    calibrado**, nunca contra o `team` `rules_v1`. Ressalvas: 7 partidas, `lam_dead` escolhido nelas, e a
    F2-06 decide.

### D-020 — Trilha da câmera de transmissão descartada: NO-GO (F4-05)

- **Data:** 2026-11-16
- **Autor:** Antônio (F4, F4-05)
- **Status:** Vigente
- **Decisão:** a trilha exploratória do tracking da câmera de transmissão (TV) recebeu veredito **NO-GO** no portão de viabilidade da F4-05 e está descartada do pipeline do modelo. A tarefa dependente F4-06 (#84) não será feita, e o tracking da transmissão entra no relatório técnico ([`technical_report/report.md`](technical_report/report.md)) como estratégia testada e descartada.
- **Opções consideradas:**
  - **Descartar o tracking da transmissão e fechar a F4-06 (escolhida):** embora o alinhamento fino por blocos de 2 s e segmentos ao vivo atinja alta precisão espacial nos trechos mapeados (critérios ii e iv em `PASS`), a cobertura total dos frames avaliados ficou abaixo do mínimo exigido (critério i em `FAIL`) e a recuperação da bola nos frames em que o tracking tático a perde também ficou abaixo do limiar (critério iii em `FAIL`). O ganho potencial sobre os lances oclusos não compensa a complexidade adicional.
  - Integrar a bola da transmissão ao pipeline mesmo com cobertura parcial (F4-06): descartada pelo critério explícito de aceite da F4-05 (exige `PASS` nos quatro critérios para `GO`).
- **Evidência:** execução local (humano) de `tools/broadcast_feasibility.py` com `configs/broadcast.yaml`:
  - `(i)` cobertura de mapeamento dos frames avaliados: `FAIL`;
  - `(ii)` custo mediano de pareamento de jogadores nos frames mapeados: `PASS`;
  - `(iii)` bola da transmissão observada nos frames táticos sem bola: `FAIL`;
  - `(iv)` concordância da posição da bola quando ambas as câmeras a observam: `PASS`;
  - veredito final: `NO-GO`.
  - Código e testes sintéticos em `modeling/broadcast/align.py`, `tools/broadcast_feasibility.py` e `tests/test_modeling_broadcast_align.py` (issues #78 e #84).
- **Revisão da F5 (16/11):**
  - **Falha no alinhamento.** O refino só procurava ±25 frames em volta de **um** deslocamento grosso por
    tempo (ou do bloco anterior). Se o deslocamento muda dentro do tempo (uma edição ou deriva da
    transmissão), o trecho ao vivo com o outro deslocamento nunca era procurado e ficava sem mapeamento:
    num teste sintético com dois trechos no mesmo tempo, um deles ficou com 0% mapeado, mesmo com um salto de
    só 1,6 s. Os testes da F4-05 não pegavam isso porque os dois trechos ficavam em tempos diferentes.
    Corrigido com deslocamentos grossos locais em janelas deslizantes de 60 s (`local_coarse_seconds`,
    passo de 20 s; 0 desliga), usados como centros extras da busca; valor fixado no teste sintético, antes de
    rodar no BAR-ATH. Testes novos em `tests/test_modeling_broadcast_align.py` (seção 6).
  - **O veredito não muda.** Execução local de `tools/broadcast_feasibility.py`: com o algoritmo original
    (`local_coarse_seconds: 0`) reproduz exatamente o resultado acima, e com a correção também dá (i) `FAIL`,
    (ii) `PASS`, (iii) `FAIL`, (iv) `PASS`, **NO-GO**.
  - **Não é limite dos dados.** Tetos com alinhamento perfeito (todo frame da transmissão com homografia
    válida, ou com bola, mapeado): (i) e (iii) dariam `PASS`. A ordem dos times (casa primeiro) é a mesma
    nos metadados das duas câmeras (`PASS`). A cobertura fica abaixo do limiar por causa do próprio
    alinhamento: os critérios de aceite dos blocos (custo, segundo melhor, 6 jogadores) foram fixados antes e
    **não podem ser ajustados no BAR-ATH** (D-012: parâmetros só em dados externos). O NO-GO fica.
  - O merge do #141 tinha apagado o bloco "Revisão da F5 (16/11)" da D-019; ele foi restaurado.


### D-021 — `team_logit` calibrado com a variante `touch` e o config `v2` (revisão da F1-05/F1-06)

- **Data:** 2026-11-16 (renumerada de D-020, que colidiu com a D-020 da F4-05)
- **Autor:** Ahmed (revisão da F5 das tarefas F1-05, #79, e F1-06, #87)
- **Status:** Vigente, exceto o toque e o item 3, substituídos pela D-022
- **Decisão:**
  1. **`configs/calib/team.yaml` passa a usar a variante `touch`** (f1–f8 + f9), não a `bridge_touch`, que era o
     padrão do comando. A ponte de passe (f10/f11) sobe muito a AUC por frame (0,92 contra 0,85), mas junta
     posses vizinhas: a razão posses previstas/verdadeiras cai e o spellF1 cai nos três modos (`default`,
     `hidelo`, `hidehi`), com qualquer `lam_team` de 0,2 a 1,0. O padrão de `--variant` agora é `touch`.
  2. **Limiares do toque escolhidos no IDSSE degradado** (atualização 4 da F1-05): grade de 81 pontos
     (`r_touch_m` {1,5; 2; 3}, `touch_min_frames` {2; 3; 5}, `touch_acc_mps2` {5; 8; 12},
     `touch_dir_change_rad` {0,3; 0,5; 0,8}), pela log-loss média do modelo "ball" com uma partida de fora,
     minimizada sobre `C`; empate fica com o primeiro ponto da grade. Escolhido: 1,5 m, 5 frames, 5 m/s²,
     0,3 rad. `r_touch_m` e `touch_min_frames` saíram na borda da grade; a grade não foi estendida depois de
     ver o resultado.
  3. **A condição de "mudança brusca" do toque é inerte no IDSSE degradado:** o ruído da posição da bola faz
     `ball_acc`/`ball_dir_change` passarem do limiar em ~99% dos frames com bola válida, com qualquer ponto da
     grade (por isso os limiares de aceleração e de direção empatam). Na prática, o f9 mede "o mesmo time a
     até 1,5 m da bola em 5 dos 9 frames da janela". Se a bola do BAR-ATH for mais lisa, o f9 enfraquece
     fora da amostra. Checagem local nova: `python tools/touch_feature_shift.py` (só `PASS`/`FAIL`). A
     eliminatória (a) exige a fração de mudança brusca a até 0,10 da referência IDSSE. Se der `FAIL`, ajustar
     de novo sem o toque (`--variant base`) ou sem a condição brusca, sempre no IDSSE.
  4. **Config `configs/v2.yaml`:** o `v1` com `team` `calibrated` (`configs/calib/team.yaml`), `dead`
     `calibrated` (`configs/calib/dead.yaml`, D-019), o decodificador de `configs/decoder_idsse.yaml` e
     `lam_team = 0,35` (`lam_dead` 0,2). O `lam_team` foi escolhido antes da última varredura (com os limiares
     antigos do toque); a varredura com o arquivo final deu um platô entre 0,3 e 0,4 (diferenças de até
     0,0015 de spellF1, abaixo do ruído), e 0,35 é o centro dele. Pós-processamento desligado.
- **Opções consideradas:** manter a `bridge_touch` (pior nos três modos); `lam_team` 0,4, o melhor ponto nos
  três modos por menos de 0,002 (descartado: escolher de novo nas mesmas 7 partidas só acrescenta ruído);
  estender a grade do toque (descartado pela regra da D-018).
- **Evidência (spellF1 médio no IDSSE degradado, 7 partidas, pesos de cada partida ajustados nas outras 6):**

  | Modo | `v1` + `decoder_idsse` | melhor `rules_v1` (dead calibrado, `lam_team` 0,35) | **`v2`** | `v2` − `v1`: partidas que ganham, pior caso |
  | :--- | ---: | ---: | ---: | :--- |
  | `default` | 0,343 | 0,359 | **0,400** | +0,057; 7/7; +0,035 |
  | `hidelo` | 0,309 | 0,332 | **0,383** | +0,074; 7/7; +0,054 |
  | `hidehi` | 0,383 | 0,392 | **0,425** | +0,042; 7/7; +0,013 |

  Contra o melhor `rules_v1`, o `v2` ganha nas 7 partidas nos três modos (+0,041, +0,051, +0,033).
  Relatórios: [`reports/calibration_team.md`](../reports/calibration_team.md) e
  [`reports/error_analysis.md`](../reports/error_analysis.md). **Ressalvas:** `lam_team` e o conjunto de
  variantes foram escolhidos nas mesmas 7 partidas (os pesos não: cada partida usa a dobra sem ela), o IDSSE
  degradado não é o BAR-ATH (D-016), e nada foi medido no BAR-ATH.

### D-022 — Toque só por persistência e o config `v2_base` (checagem do toque no BAR-ATH)

- **Data:** 2026-11-16
- **Autor:** Ahmed (revisão da F5 da F1-05, #79)
- **Status:** Vigente
- **Contexto:** a checagem local `python tools/touch_feature_shift.py` (D-021, item 3) deu `FAIL` na
  eliminatória (a) no BAR-ATH. A fração de frames com "mudança brusca" da bola fica a mais de 0,10 da
  referência do IDSSE degradado (~0,99). A informativa (b) também deu `FAIL`; a (c), `PASS`. Só esses
  `PASS`/`FAIL` saíram da máquina local.
- **Decisão:**
  1. **O toque passa a ser só persistência** (`touch_require_abrupt: false` em `configs/v2.yaml`, gravado em
     `feature_params` de `configs/calib/team.yaml`): "o mesmo time a até `r_touch_m` da bola em pelo menos
     `touch_min_frames` frames da janela". No IDSSE a condição brusca já era verdadeira em ~99% dos frames,
     então o modelo ajustado praticamente não muda, e a feature deixa de depender do ruído da bola, que é
     diferente no BAR-ATH. O padrão do código continua `true` (a especificação da atualização 4).
  2. **Limiares escolhidos de novo** pela mesma regra da D-021 (log-loss com uma partida de fora), agora só em
     `r_touch_m` × `touch_min_frames` (9 pontos; os de aceleração e direção ficam no valor inicial, sem uso):
     1,5 m e 5 frames, os mesmos de antes.
  3. **A checagem muda com o arquivo:** sem a condição brusca, a (a) sai como `PASS` "not used" e a (b) — a
     fração de frames com toque, tolerância relativa de ±30%, fixada na D-021 antes deste resultado — passa a
     ser **eliminatória**, porque o f9 passa a depender só dela.
  4. **Alternativa pronta, `configs/v2_base.yaml`:** o `v2` sem o toque (variante `base`,
     `configs/calib/team_base.yaml`), com `lam_team = 0,3` (o melhor ponto da `base` nos três modos, entre
     0,3, 0,35 e 0,4). Vale **se a (b) der `FAIL`** com o `v2`. Sem toque, a checagem dá `PASS` (nada a
     verificar).
- **Evidência (spellF1 médio no IDSSE degradado, 7 partidas, `dead` calibrado):**

  | Config | `default` | `hidelo` | `hidehi` |
  | :--- | ---: | ---: | ---: |
  | `v2` com o toque antigo (D-021, `lam_team` 0,35) | 0,400 | 0,383 | 0,425 |
  | **`v2`, toque só por persistência (`lam_team` 0,35)** | **0,402** | **0,385** | **0,426** |
  | `v2_base`, sem toque (`lam_team` 0,3) | 0,389 | 0,375 | 0,412 |
  | `v1` + `decoder_idsse` | 0,343 | 0,309 | 0,383 |

  O `lam_team` do `v2` continua 0,35: a varredura deu um platô (0,4015–0,4019 no `default` entre 0,3 e 0,4).
- **Ordem de execução local antes do envio:** `python tools/touch_feature_shift.py`. Com `PASS`, enviar o
  `v2`; com `FAIL`, enviar o `v2_base`.

### D-023 — `team_logit` GBDT e o config `v2_gbdt` (revisão da F1-07)

- **Data:** 2026-11-16
- **Autor:** Ahmed (revisão da F5 da F1-07, #82 / PR #144)
- **Status:** Vigente
- **Contexto:** o PR #144 entregou só a inferência e o módulo de janelas. A inferência chamava uma função
  inexistente (`swap_teams_master`), as janelas tinham um frame a mais, e faltavam o treino, os modelos, o YAML,
  o relatório e os testes. A revisão completou a tarefa pela especificação de `docs/tasks/F1-07.md`.
- **Decisão:**
  1. **Troca de times na matriz de features.** Toda feature do GBDT é par casa/fora, ímpar (muda de sinal) ou
     par (não muda). A troca `S` é uma permutação com sinais, então `S S X == X` bit a bit e
     `t = clip((g(X) − g(S X)) / 2, ±8)` é exatamente antissimétrico. `S X(m)` é igual a `X(swap(m))` (a
     definição da especificação) a menos de arredondamento; o teste confere isso e `t(swap(m)) == −t(m)` num
     master sintético. Assim não é preciso recalcular as features num master trocado, e a antissimetria vale
     exatamente também nos dados reais (o espelhamento `pitch_l_m − x` não é uma involução exata em float).
  2. **Features:** f1–f9 da F1-05 (com os limiares do toque de `configs/calib/team.yaml`, persistência só), a
     ponte de passe (τ 15 e 35), `ev_coll_adv`, `ev_coll_speed`, `d_diff`, os pares brutos de proximidade,
     pressão e velocidade, `ev_ball_conf`, `d_touchline`, `d_goalline`, `gap_length` e as janelas de
     `modeling/evidence/windows.py` (W 12, 25, 50). Posição da bola fica de fora (o sentido de ataque muda no
     intervalo).
  3. **Fora do modelo** (`feature_params.exclude_features`): `n_tracked_*` e `ev_coll_spread`, que deram `FAIL`
     na checagem (a) de `tools/gbdt_feature_shift.py` no BAR-ATH; e `n_still_*`, `ball_speed`,
     `since_last_obs`, `until_next_obs`, que dependem da cobertura da câmera ou do ruído da bola (a lição da
     D-022) e acrescentam no máximo 0,003 de spellF1 no IDSSE (ablação no relatório).
  4. **Frames sem bola válida continuam com o modelo "gap" logístico da F1-05** (`gap_model: logistic`): o GBDT
     de lacuna perdeu no log-loss fora da dobra e no spellF1 (relatório lado a lado, atualização 4).
  5. **`configs/v2_gbdt.yaml`** = `v2` com `team.kind: gbdt` e `lam_team = 0,3`, escolhido pela regra da F2-05
     (platô a 1 EP do melhor ponto no `default`, depois o melhor pior caso nos três modos) na grade 0,3–0,6; a
     primeira rodada mostrou queda abaixo de 0,3 e acima de 0,5.
  6. **Modelos:** um por dobra em `ext_dir()/models/team_gbdt/` (não commitados) e `models/team_gbdt/all.joblib`
     (0,92 MB, o único usado no CSV). O YAML registra o scikit-learn 1.5.2, e a inferência recusa outra versão
     ou outra lista de features. Treino e inferência em uma thread; o CSV sai idêntico em duas execuções.
- **Evidência (spellF1 médio no IDSSE degradado, 7 partidas, cada uma com os modelos treinados sem ela):**

  | Config | `default` | `hidelo` | `hidehi` | posses previstas / verdadeiras (`default`) |
  | :--- | ---: | ---: | ---: | ---: |
  | `v2` (enviado) | 0,402 | 0,385 | 0,426 | 0,70 |
  | **`v2_gbdt`** | **0,489** | **0,466** | **0,524** | 0,66 |

  O ganho vem dos frames com bola: AUC fora da dobra 0,969 contra 0,850 da logística, nas 7 partidas.
- **Checagens locais no BAR-ATH** (só `PASS`/`FAIL` e estatísticas dos rótulos previstos saíram da máquina):
  `python tools/gbdt_feature_shift.py --degrade default hidelo hidehi` `PASS` em (a), (b) e (c);
  `tools/check_public_window.py` `PASS` nas 7. CSV do `v2_gbdt`: SHA-256
  `3d9570c55a398c8153fa90289567d8a7e33c3cecac363afefaccee7d7fbed2d0`, H 0,3203, A 0,2065, D 0,4732, 148 posses
  no 1º tempo e 166 no 2º (o `v2` tem 164 e 174).
- **Envio:** o envio único (D-013) já foi feito com o `v2`. O `v2_gbdt` só substitui o `v2` se a equipe decidir
  reenviar e o formulário permitir; nesse caso, antes do envio, rodar `tools/gbdt_feature_shift.py` e
  `tools/check_public_window.py` e registrar o novo CSV no registro de envios.

### D-024 — Bola fundida da transmissão (F4-06): revisão da D-020

- **Data:** 2026-11-16
- **Autor:** Ahmed (revisão da F5 do PR #148, F4-06, #84)
- **Status:** Vigente. **Opção B aprovada pela F5 (Ahmed) em 16/11.** A D-020 continua valendo para o resto da trilha da transmissão (jogadores).
- **Contexto:** a F4-06 foi entregue no PR #148 depois do NO-GO da F4-05 (D-020). Três dos quatro critérios da própria F4-06 passam. O que falha é "as checagens da F4-05 dão `PASS`". Detalhes e números em [`tasks/F4-06.md`](tasks/F4-06.md), na seção "Execução e revisão".
- **O fato novo:** os dois critérios da F4-05 que falham medem **quanto** a transmissão acrescenta: (i) a cobertura do mapeamento e (iii) a bola onde a tática falta. Os que medem **se ela está certa** passam: (ii) o pareamento dos jogadores e (iv) a concordância da bola. A D-020 descartou a trilha porque o ganho não compensaria. Agora o ganho foi medido no proxy da F4-06 com a melhor candidata, o `v2_gbdt`: **+0,0064 ± 0,0020** de spellF1 no IDSSE (3,2 EP, 6 de 7 partidas). Os limiares da F4-05 continuam como estão: nada foi ajustado no BAR-ATH.
- **Opções:**
  - **A. Manter a D-020:** a bola fundida fica fora. É a regra pré-registrada ao pé da letra, e o ganho é pequeno.
  - **B. Rever a D-020 só para a bola** (recomendada pela revisão): `ball_source: fused` entra na escada da F2-06 como último passo de dados, com a regra de sempre (ganho pareado acima de 1 EP no `default`). Sustentam essa opção:
    - os critérios de precisão passam;
    - o ganho no proxy está bem acima de 1 EP;
    - o master fundido passa nos testes de contrato, na janela pública e na checagem de transferência do GBDT, e muda só 0,69% dos rótulos previstos no BAR-ATH;
    - o caminho tático continua bit a bit idêntico.

    Riscos:
    - é uma revisão feita depois de conhecido o resultado do portão;
    - o proxy é otimista (posição exata no IDSSE) e não tem versões `fused` dos modos de estresse.
- **Consequências da opção B:**
  - a #84 foi reaberta e fechada como feita;
  - a candidata é `configs/v2_gbdt_fused.yaml` (o `v2_gbdt` com `ball_source: fused`). Ela entra na escada da F2-06 como último passo de dados. Se a F2-06 não terminar a tempo, é ela que vai para o envio, no lugar do `v2_gbdt`;
  - antes de qualquer envio com ela: rodar `make_submission` (que gera `processed_fused`), `tools/gbdt_feature_shift.py` e `tools/check_public_window.py` sobre o master fundido, e registrar o CSV.

### D-025 — Seleção F2-06: vencedora `v2_gbdt_fused`

- **Data:** 2026-11-16
- **Autor:** Hugo (F2, F2-06)
- **Status:** Vigente
- **Decisão:** a escada da Seção 6 (implementada em `modeling/scoring/selection.py`) escolhe
  `configs/final.yaml` = `v2_gbdt` com `ball_source: fused` (equivalente a `configs/v2_gbdt_fused.yaml`).
  Passos aceitos (ganho pareado > 1 EP no `default` e estresse ok, quando aplicável):
  `baseline` → `v1` → `v1s` → `v2_base` → `v2` → `team_gbdt` → `fused`.
  Rejeitados: `dead_pooled`, todas as regras de pós-processamento isoladas e `boundary_shift_file`.
  F3-08 / F2-07 / F2-08 ficaram de fora (ainda abertas ou opcionais).
- **Opções consideradas:**
  - parar em `v2` ou `v2_gbdt` tático: descartadas — o GBDT e o `fused` passaram a regra de ganho;
  - aceitar pós-processamento / boundary da F2-05: descartado — ganho nulo ou negativo no `default`.
- **Evidência:** [`reports/selection.md`](../reports/selection.md); spellF1 IDSSE
  `v2_gbdt` 0,4895±0,0083 → `fused` 0,4958±0,0083 (Δ +0,0064±0,0020, alinhado à D-024).
  Checagens BAR-ATH da vencedora: **pending** (execução local humana).

## Registro de envios

Preenchido pela F5 (F5-09; envio único, ver D-013). O CSV enviado tem de ser o arquivo validado, com o mesmo SHA-256.

> **Lembrete obrigatório:** o repositório precisa estar **público** antes de 17/11/2026 em qualquer
> cenário (ver F5-09).

| Data | Envio | Config (nome e hash) | Commit ou tag | SHA-256 do CSV | `validate_submission` | Autor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-11-16 | Envio final (F5-09) | `v2` (`6a992f351783382f2532c34d5299307ce61ea20326ecf1e002aee5b95d2b241b`) | `ffe5eae` (merge do #142) | `50b12bb537f1c62b6dd033da1efc6e03ab243cf2f77dfd44709c90201f0ae66a` | `PASS` (`tools/check_public_window.py`, check (i)) | Ahmed |

Checagens locais do CSV acima: `tools/touch_feature_shift.py` `PASS` (a informativa (c) deu `FAIL`, sem efeito
no resultado, D-022), `tools/dead_feature_shift.py --config configs/v2.yaml` `PASS` em todas e as 7 de
`tools/check_public_window.py --pred outputs/v2.csv` `PASS`. Rótulos previstos: 158.857
frames avaliados, H 0,3216, A 0,2051, D 0,4732; 164 posses no 1º tempo e 174 no 2º. Comando:
`python -m modeling.make_submission --config configs/v2.yaml` (o CSV foi gerado no branch do #142, com a mesma
árvore de arquivos do `ffe5eae`).
