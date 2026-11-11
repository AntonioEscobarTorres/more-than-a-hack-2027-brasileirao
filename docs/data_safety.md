# Segurança dos dados da partida

Os dados do BAR-ATH são licenciados só para a competição. Os Termos proíbem redistribuí-los (inclusive num repositório público) e proíbem enviar os dados, ou qualquer extrato deles, a um LLM hospedado em nuvem ([RULES.md](../RULES.md#the-data), [FAQ.md](../FAQ.md)). O repositório **vai ficar público** antes de 17/11/2026 (F5-09), e tudo o que já foi commitado fica no histórico para sempre.

## 1. Onde os dados ficam

- **Fora do repositório**, num diretório apontado por `MTAH_DATA_DIR` (por exemplo, `export MTAH_DATA_DIR=~/mtah-data` no `~/.bashrc`). Todo script lê os dados de lá; sem a variável, usa `./data`.
- `./data` funciona, mas fica dentro da pasta que os agentes de IA abrem. Preferir `MTAH_DATA_DIR`.
- Saídas geradas ficam fora do git: `<data dir>/processed/` (master, logits) e `outputs/` (CSVs de rótulos).

## 2. As camadas de proteção

| Camada | O que faz | Protege de verdade? |
| :--- | :--- | :--- |
| `MTAH_DATA_DIR` fora do repositório | os dados nem estão na árvore de trabalho | **Sim, é a proteção principal** |
| `.gitignore` | ignora `data/*` (menos `data/README.md`), `outputs/`, `*.parquet`, `*.npy`, `*.npz`, `*.pkl`, `*.mp4` e o GT completo | sim, contra `git add` acidental (mas `git add -f` passa) |
| `tests/test_repo_hygiene.py` + CI | falha se o índice do git tiver arquivo de dados fora de `tests/fixtures/`, qualquer arquivo em `data/` além do README, notebook com saída ou arquivo > 1 MB fora da allowlist | sim, em todo push e PR |
| `.geminiignore`, configurações de IDE e de agentes | pedem ao agente que não leia certos caminhos | **não**: é só uma camada extra. O agente ainda pode ler o arquivo por outro caminho (terminal, outro nome) |

Fixtures sintéticas em `tests/fixtures/` são os únicos arquivos de dados permitidos no git. Como `*.npy`, `*.parquet` etc. estão no `.gitignore`, elas entram com `git add -f`, o que torna o commit sempre deliberado.

## 3. Agentes de IA (regras 1.4.1 e 1.4.2 do plano)

- **Agentes nunca leem, abrem, imprimem nem executam código sobre os dados da partida** (`$MTAH_DATA_DIR`, `./data` ou qualquer parquet/CSV derivado). Também não leem `outputs/*.csv`: são rótulos da partida inteira.
- Agentes trabalham com dados sintéticos (`preprocessing.make_mock_master`) e com o IDSSE (público, de outras partidas).
- Passos que tocam os dados são **execução local (humano)**: a pessoa roda no próprio terminal. As ferramentas em `tools/` imprimem só `PASS`/`FAIL`; números só com `--verbose`, para a pessoa ver.
- **Podem ir** para agentes, issues e PRs: `PASS`/`FAIL`, hashes e estatísticas **dos rótulos previstos** (nº de posses, proporções de H/A/D).
- **Nunca vão:** valores de features, quantis do tracking, linhas, arrays ou gráficos de trajetória da partida, nem score da janela pública (regra 1.4.9).
- Testes usam só fixtures sintéticas em `tmp_path`. Um teste que leia o diretório de dados é um bug (a CI não tem dados).

## 4. Checklist antes de tornar o repositório público (F5-09)

Rodar cada item no próprio terminal e marcar na issue da F5-09.

- [ ] **CI verde no `main`**, incluindo `tests/test_repo_hygiene.py`.
- [ ] **Nada de dados no histórico**, não só no `main` (o teste de higiene só olha o índice atual):
  ```bash
  git log --all --diff-filter=A --name-only --format= | sort -u \
    | grep -iE '\.(parquet|npy|npz|pkl|mp4|xml|csv|ipynb|txt|json)$|^data/'
  ```
  Esperado (estado de 11/11/2026): `data/README.md`, `evaluation/gt/BAR-ATH_possession_gt_public.csv` e `…_public_meta.json` (oficiais, publicados pela organização), `evaluation/results/baseline_*.csv`, `requirements*.txt`, `tests/fixtures/idsse_mini_tracking.xml` e os notebooks `starter/*.ipynb` e `tests/prep_tracking.ipynb`. Qualquer coisa nova nessa lista precisa de explicação.
- [ ] **Nenhum blob grande no histórico** fora da allowlist:
  ```bash
  git rev-list --objects --all \
    | git cat-file --batch-check='%(objecttype) %(objectsize) %(rest)' \
    | awk '$1=="blob" && $2>1000000 {print $2, $3}' | sort -u -k2
  ```
- [ ] **Nenhum notebook com saída em nenhuma versão do histórico.** Abrir pessoalmente as versões antigas que tiverem saída. Em 11/11/2026 havia uma: `tests/prep_tracking.ipynb` no commit `7fbb12e`, com uma saída do tipo `error` (3 linhas de traceback). Conferir com `git show 7fbb12e:tests/prep_tracking.ipynb` se ela mostra algum dado da partida.
- [ ] **Nenhum GT completo:** nenhum `*_possession_gt.csv`, `*_possession_gt_meta.json`, `*_possession_gt_reference.csv` ou `possession_by_frame.pkl` no histórico (mesmo comando do segundo item).
- [ ] **`reports/` só com resultados de dados externos (IDSSE)**, nunca números do BAR-ATH.
- [ ] **Issues, PRs e comentários** também ficam públicos: conferir que nenhum tem valores derivados da partida ou score da janela pública.
- [ ] **Nenhum segredo** (tokens, chaves, e-mails pessoais além dos de autoria dos commits).
- [ ] Se algo vazou no histórico: **não** tornar público. Reescrever o histórico (`git filter-repo`) e combinar com a equipe antes, porque todos precisam clonar de novo.
