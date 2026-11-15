# Issues do plano de implementação

37 issues (36 tarefas + o gate G1) e 58 dependências nativas do GitHub ("blocked by"), criadas em 06/11/2026 a partir do [plano completo](plano_5_frentes.md). Em 13/11 entraram mais 3 tarefas (#124–#126, D-012) e 9 dependências.

Quadro do projeto (colunas Bloqueada → Pronta → Em andamento → Em revisão → Concluída): https://github.com/users/HachemAhmed/projects/2

| Issue | ID | Título | Responsável | Marco | Prazo | Prioridade | Bloqueada por |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| [#56](https://github.com/AntonioEscobarTorres/more-than-a-hack-2027-brasileirao/issues/56) | [F5-01](../tasks/F5-01.md) | Esqueleto `modeling/`, contratos e configuração | Ahmed | M0 | sex 06/11 | P0 | — |
| [#57](https://github.com/AntonioEscobarTorres/more-than-a-hack-2027-brasileirao/issues/57) | [F5-08](../tasks/F5-08.md) | Formulário e nome da equipe | Ahmed | M0 | sex 06/11 | P0 | — |
| [#58](https://github.com/AntonioEscobarTorres/more-than-a-hack-2027-brasileirao/issues/58) | [F2-01](../tasks/F2-01.md) | Emissão e parâmetros do decodificador | Hugo | M0 | sáb 07/11 | P0 | #56 |
| [#59](https://github.com/AntonioEscobarTorres/more-than-a-hack-2027-brasileirao/issues/59) | [F3-03](../tasks/F3-03.md) | `force_D` (pré-pontapé) | Bernardo | M0 | sáb 07/11 | P0 | #56 |
| [#60](https://github.com/AntonioEscobarTorres/more-than-a-hack-2027-brasileirao/issues/60) | [F5-02](../tasks/F5-02.md) | Baseline, comando único e checagens da janela pública | Ahmed | M0 | sáb 07/11 | P0 | #56 |
| [#61](https://github.com/AntonioEscobarTorres/more-than-a-hack-2027-brasileirao/issues/61) | [F1-01](../tasks/F1-01.md) | Features de controle | Fernando | M0 | dom 08/11 | P0 | #56 |
| [#62](https://github.com/AntonioEscobarTorres/more-than-a-hack-2027-brasileirao/issues/62) | [F3-01](../tasks/F3-01.md) | Features de bola morta | Bernardo | M0 | dom 08/11 | P0 | #56 |
| [#63](https://github.com/AntonioEscobarTorres/more-than-a-hack-2027-brasileirao/issues/63) | [F4-01](../tasks/F4-01.md) | IDSSE: download, conversão e rótulos por frame | Antônio | M0 | dom 08/11 | P1 | — |
| [#64](https://github.com/AntonioEscobarTorres/more-than-a-hack-2027-brasileirao/issues/64) | [F5-03](../tasks/F5-03.md) | Higiene do repositório e CI | Ahmed | M0 | dom 08/11 | P1 | #56 |
| [#65](https://github.com/AntonioEscobarTorres/more-than-a-hack-2027-brasileirao/issues/65) | [F1-02](../tasks/F1-02.md) | `team_logit` v1 (prior) | Fernando | M1 | seg 09/11 | P0 | #61 |
| [#66](https://github.com/AntonioEscobarTorres/more-than-a-hack-2027-brasileirao/issues/66) | [F2-02](../tasks/F2-02.md) | Viterbi por tempo com duração mínima | Hugo | M1 | seg 09/11 | P0 | #58 |
| [#67](https://github.com/AntonioEscobarTorres/more-than-a-hack-2027-brasileirao/issues/67) | [F3-02](../tasks/F3-02.md) | `dead_logit` v1 (prior) | Bernardo | M1 | seg 09/11 | P0 | #62 |
| [#68](https://github.com/AntonioEscobarTorres/more-than-a-hack-2027-brasileirao/issues/68) | [F3-05](../tasks/F3-05.md) | Registro de decisões e esqueleto do relatório técnico (inglês) | Bernardo | M1 | seg 09/11 | P0 | #56 |
| [#69](https://github.com/AntonioEscobarTorres/more-than-a-hack-2027-brasileirao/issues/69) | [F4-02](../tasks/F4-02.md) | API em memória do pré-processamento (sem mudar a saída do BAR-ATH) | Antônio | M1 | seg 09/11 | P1 | — |
| [#70](https://github.com/AntonioEscobarTorres/more-than-a-hack-2027-brasileirao/issues/70) | [F1-03](../tasks/F1-03.md) | Harness de avaliação (métrica oficial, partidas externas) | Fernando | M1 | ter 10/11 | P1 | #60 |
| [#71](https://github.com/AntonioEscobarTorres/more-than-a-hack-2027-brasileirao/issues/71) | [F2-03](../tasks/F2-03.md) | Estatísticas dos rótulos IDSSE (transições, durações, faixas de plausibilidade) | Hugo | M1 | ter 10/11 | P1 | #58, #63 |
| [#72](https://github.com/AntonioEscobarTorres/more-than-a-hack-2027-brasileirao/issues/72) | [F4-03](../tasks/F4-03.md) | Degradação realista do IDSSE | Antônio | M1 | ter 10/11 | P1 | #63 |
| [#73](https://github.com/AntonioEscobarTorres/more-than-a-hack-2027-brasileirao/issues/73) | [F5-04](../tasks/F5-04.md) | Integração do motor HMM (v1) no comando único | Ahmed | M1 | ter 10/11 | P0 | #60, #65, #66, #67, #59 |
| [#74](https://github.com/AntonioEscobarTorres/more-than-a-hack-2027-brasileirao/issues/74) | [F4-04](../tasks/F4-04.md) | Mestres externos (limpos e degradados) e checagem de realismo | Antônio | M2 | qua 11/11 às 12h | P1 | #63, #69, #72 |
| [#75](https://github.com/AntonioEscobarTorres/more-than-a-hack-2027-brasileirao/issues/75) | [F1-04](../tasks/F1-04.md) | Evidência coletiva para lacunas da bola (desligada por padrão) | Fernando | M2 | qua 11/11 | P1 | #65 |
| [#76](https://github.com/AntonioEscobarTorres/more-than-a-hack-2027-brasileirao/issues/76) | [F2-04](../tasks/F2-04.md) | Pós-processamento por posse | Hugo | M2 | qua 11/11 | P1 | #66 |
| [#77](https://github.com/AntonioEscobarTorres/more-than-a-hack-2027-brasileirao/issues/77) | [F5-05](../tasks/F5-05.md) | Envio de segurança | Ahmed | M1 | qua 11/11 | P0 | #73, #57, #68 |
| [#78](https://github.com/AntonioEscobarTorres/more-than-a-hack-2027-brasileirao/issues/78) | [F4-05](../tasks/F4-05.md) | Trilha exploratória: viabilidade da câmera de transmissão (portão GO/NO-GO) | Antônio | M2 | qui 12/11 às 12h | P2 | — |
| [#79](https://github.com/AntonioEscobarTorres/more-than-a-hack-2027-brasileirao/issues/79) | [F1-05](../tasks/F1-05.md) | Calibração do `team_logit` no IDSSE | Fernando | M2 | qui 12/11 | P1 | #70, #75, #74 |
| [#80](https://github.com/AntonioEscobarTorres/more-than-a-hack-2027-brasileirao/issues/80) | [F2-05](../tasks/F2-05.md) | Varredura de sensibilidade do decodificador (IDSSE) | Hugo | M2 | qui 12/11 | P1 | #70, #71, #76, #74, #73 |
| [#81](https://github.com/AntonioEscobarTorres/more-than-a-hack-2027-brasileirao/issues/81) | [F3-04](../tasks/F3-04.md) | Calibração do `dead_logit` no IDSSE | Bernardo | M2 | qui 12/11 | P1 | #67, #70, #74 |
| [#82](https://github.com/AntonioEscobarTorres/more-than-a-hack-2027-brasileirao/issues/82) | [F1-07](../tasks/F1-07.md) | Variante GBDT do `team_logit` (opcional) | Fernando | M3 | sex 13/11 às 12h | P2 | #79 |
| [#83](https://github.com/AntonioEscobarTorres/more-than-a-hack-2027-brasileirao/issues/83) | [F3-08](../tasks/F3-08.md) | Variante GBDT do `dead_logit` (opcional) | Bernardo | M3 | sex 13/11 às 12h | P2 | #81 |
| [#84](https://github.com/AntonioEscobarTorres/more-than-a-hack-2027-brasileirao/issues/84) | [F4-06](../tasks/F4-06.md) | Trilha exploratória: fusão da bola da transmissão (se GO) | Antônio | M3 | sex 13/11 às 12h | P2 | #78, #73 |
| [#85](https://github.com/AntonioEscobarTorres/more-than-a-hack-2027-brasileirao/issues/85) | [F2-06](../tasks/F2-06.md) | Protocolo de seleção e `configs/final.yaml` | Hugo | M3 | sex 13/11 às 18h | P0 | #80, #79, #81, #82, #83, #84, #124, #125, #126 |
| [#86](https://github.com/AntonioEscobarTorres/more-than-a-hack-2027-brasileirao/issues/86) | [G1](../tasks/G1.md) | Code freeze | Ahmed | M3 | sex 13/11 às 20h | P0 | #85 |
| [#87](https://github.com/AntonioEscobarTorres/more-than-a-hack-2027-brasileirao/issues/87) | [F1-06](../tasks/F1-06.md) | Análise de erros agregada (IDSSE) | Fernando | M3 | sex 13/11 | P2 | #79 |
| [#88](https://github.com/AntonioEscobarTorres/more-than-a-hack-2027-brasileirao/issues/88) | [F3-06](../tasks/F3-06.md) | Diagnósticos de bola morta no BAR-ATH (execução local) | Bernardo | M3 | sex 13/11 | P1 | #73, #71 |
| [#89](https://github.com/AntonioEscobarTorres/more-than-a-hack-2027-brasileirao/issues/89) | [F3-07](../tasks/F3-07.md) | Frequência dos casos de borda do filtro da bola (execução local) | Bernardo | M3 | sex 13/11 | P2 | — |
| [#90](https://github.com/AntonioEscobarTorres/more-than-a-hack-2027-brasileirao/issues/90) | [F5-06](../tasks/F5-06.md) | Reprodução limpa e determinismo entre máquinas | Ahmed | M4 | sáb 14/11 | P0 | #86 |
| [#91](https://github.com/AntonioEscobarTorres/more-than-a-hack-2027-brasileirao/issues/91) | [F5-07](../tasks/F5-07.md) | Relatório técnico final (inglês) | Ahmed | Entrega | dom 15/11 | P0 | #86, #68 |
| [#92](https://github.com/AntonioEscobarTorres/more-than-a-hack-2027-brasileirao/issues/92) | [F5-09](../tasks/F5-09.md) | Pacote final, repositório público e envio | Ahmed | Entrega | seg 16/11 | P0 | #90, #91, #57 |
| [#124](https://github.com/AntonioEscobarTorres/more-than-a-hack-2027-brasileirao/issues/124) | [F1-08](../tasks/F1-08.md) | Detector de troca de posse (hazard) no IDSSE | Ahmed | M3 | sáb 14/11 às 18h | P1 | #70, #74 |
| [#125](https://github.com/AntonioEscobarTorres/more-than-a-hack-2027-brasileirao/issues/125) | [F2-07](../tasks/F2-07.md) | Transições guiadas pela evidência de troca (IO-HMM) | Ahmed | M3 | sáb 14/11 às 18h | P1 | #124, #80 |
| [#126](https://github.com/AntonioEscobarTorres/more-than-a-hack-2027-brasileirao/issues/126) | [F2-08](../tasks/F2-08.md) | Durações semi-Markov (binomial negativa) no decodificador | Ahmed | M3 | sáb 14/11 às 18h | P1 | #71, #80 |
