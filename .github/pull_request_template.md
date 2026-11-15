Closes #

## O que muda
<!-- Resumo curto. Uma tarefa por PR (Seção 1.5 do plano). -->

## Dados da partida (obrigatório)
**Este PR contém algum valor derivado dos dados da partida?**
- [ ] Não.
- [ ] Sim → **não abra o PR.** Valores de features, quantis do tracking, linhas, arrays ou gráficos da partida nunca entram no repositório (regra 1.4.2; ver `docs/data_safety.md`). Só podem aparecer `PASS`/`FAIL`, hashes e estatísticas dos rótulos previstos.

## Definition of Done (Seção 1.5)
- [ ] Código e testes sintéticos (nenhum teste lê `MTAH_DATA_DIR` ou `./data`).
- [ ] `pytest -q` (a suíte inteira) passou localmente, e a CI está verde.
- [ ] `docs/tasks/<ID>.md` atualizado com o que foi feito e as decisões tomadas.
- [ ] Parâmetros em config (`configs/*.yaml`), sem números mágicos no código.
- [ ] Entrada em `docs/decisions.md`, se houve decisão.
- [ ] Resultado `PASS`/`FAIL` da execução local colado na issue, quando houver execução local.
- [ ] Revisão aprovada (cruzada: F1↔F3, F2↔F4; F5 revisa qualquer um).
