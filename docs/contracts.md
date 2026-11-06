# Contratos de modelagem (C0–C9)

Fonte: Seção 2.4 do [plano de implementação](plano/plano_5_frentes.md). Mudar um contrato exige issue aprovada pela F5, atualizando este arquivo e o plano no mesmo PR.

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
- **C5 Decodificador:** `modeling.decode.viterbi.decode_match(scores, half, force_d, params) -> np.ndarray[dtype "<U1"]`.
  - Decodifica cada tempo (`half` = 1 e 2) separadamente.
  - `half == 0` → `X`; dentro dos blocos, nunca `X`.
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
