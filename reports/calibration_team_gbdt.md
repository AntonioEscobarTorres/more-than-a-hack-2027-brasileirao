# Team logit GBDT calibration (F1-07)

- degrade (training): `default`; samples: alive frames, 1 every 5
- scikit-learn `1.5.2`; model `models/team_gbdt/all.joblib` (0.92 MB); 63 features
- model params: `max_iter=300`, `learning_rate=0.05`, `max_leaf_nodes=31`, `min_samples_leaf=500`, `l2_regularization=1.0`, `early_stopping=False`, `random_state=0`
- logistic reference (F1-05): `configs/calib/team.yaml`
- decoder evaluation: every match predicted with the fold models trained without it

## Out-of-fold log-loss and AUC (GBDT vs F1-05 logistic, same rows)

| Match | ball LL | ball LL logistic | ball AUC | ball AUC logistic | gap LL | gap LL logistic | gap AUC | gap AUC logistic |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| J03WMX | 0.2034 | 0.4414 | 0.9744 | 0.8668 | 0.6179 | 0.5629 | 0.7426 | 0.7914 |
| J03WN1 | 0.2656 | 0.4812 | 0.9574 | 0.8435 | 0.6792 | 0.5669 | 0.7173 | 0.7851 |
| J03WOH | 0.2331 | 0.4857 | 0.9668 | 0.8348 | 0.5372 | 0.5042 | 0.8104 | 0.8387 |
| J03WOY | 0.2430 | 0.5020 | 0.9644 | 0.8221 | 0.5657 | 0.4946 | 0.7867 | 0.8476 |
| J03WPY | 0.2139 | 0.4546 | 0.9705 | 0.8451 | 0.5505 | 0.4824 | 0.7959 | 0.8462 |
| J03WQQ | 0.2171 | 0.4591 | 0.9707 | 0.8579 | 0.5679 | 0.4529 | 0.7935 | 0.8809 |
| J03WR9 | 0.1828 | 0.4098 | 0.9773 | 0.8822 | 0.5402 | 0.4923 | 0.8136 | 0.8467 |
| mean | 0.2227 | 0.4620 | 0.9688 | 0.8503 | 0.5798 | 0.5080 | 0.7800 | 0.8338 |

## spellF1 on degraded IDSSE by `lam_team` (mean ± SE over 7 matches)

Reference `v2` is the submitted config. Count ratio = predicted / true possessions.

### `degrade=default`

| config | lam_team | spellF1 | SE | count ratio |
|:---|---:|---:|---:|---:|
| gbdt_gap_gbdt | 0.3 | 0.4794 | 0.0103 | 0.667 |
| gbdt_gap_gbdt | 0.35 | 0.4789 | 0.0099 | 0.692 |
| gbdt_gap_gbdt | 0.4 | 0.4783 | 0.0100 | 0.708 |
| gbdt_gap_gbdt | 0.5 | 0.4761 | 0.0106 | 0.746 |
| gbdt_gap_gbdt | 0.6 | 0.4728 | 0.0101 | 0.784 |
| gbdt_gap_logistic | 0.3 | 0.4895 | 0.0083 | 0.657 |
| gbdt_gap_logistic | 0.35 | 0.4868 | 0.0085 | 0.682 |
| gbdt_gap_logistic | 0.4 | 0.4860 | 0.0091 | 0.696 |
| gbdt_gap_logistic | 0.5 | 0.4850 | 0.0088 | 0.729 |
| gbdt_gap_logistic | 0.6 | 0.4795 | 0.0088 | 0.755 |
| v2 | 0.35 | 0.4019 | 0.0084 | 0.703 |

### `degrade=hidelo`

| config | lam_team | spellF1 | SE | count ratio |
|:---|---:|---:|---:|---:|
| gbdt_gap_gbdt | 0.3 | 0.4566 | 0.0100 | 0.672 |
| gbdt_gap_gbdt | 0.35 | 0.4573 | 0.0096 | 0.699 |
| gbdt_gap_gbdt | 0.4 | 0.4563 | 0.0094 | 0.723 |
| gbdt_gap_gbdt | 0.5 | 0.4564 | 0.0106 | 0.763 |
| gbdt_gap_gbdt | 0.6 | 0.4518 | 0.0102 | 0.801 |
| gbdt_gap_logistic | 0.3 | 0.4659 | 0.0085 | 0.653 |
| gbdt_gap_logistic | 0.35 | 0.4636 | 0.0081 | 0.683 |
| gbdt_gap_logistic | 0.4 | 0.4643 | 0.0077 | 0.703 |
| gbdt_gap_logistic | 0.5 | 0.4642 | 0.0083 | 0.735 |
| gbdt_gap_logistic | 0.6 | 0.4591 | 0.0081 | 0.766 |
| v2 | 0.35 | 0.3848 | 0.0093 | 0.706 |

### `degrade=hidehi`

| config | lam_team | spellF1 | SE | count ratio |
|:---|---:|---:|---:|---:|
| gbdt_gap_gbdt | 0.3 | 0.5197 | 0.0117 | 0.664 |
| gbdt_gap_gbdt | 0.35 | 0.5196 | 0.0116 | 0.685 |
| gbdt_gap_gbdt | 0.4 | 0.5206 | 0.0124 | 0.703 |
| gbdt_gap_gbdt | 0.5 | 0.5182 | 0.0127 | 0.738 |
| gbdt_gap_gbdt | 0.6 | 0.5160 | 0.0122 | 0.768 |
| gbdt_gap_logistic | 0.3 | 0.5237 | 0.0094 | 0.658 |
| gbdt_gap_logistic | 0.35 | 0.5223 | 0.0100 | 0.682 |
| gbdt_gap_logistic | 0.4 | 0.5219 | 0.0105 | 0.700 |
| gbdt_gap_logistic | 0.5 | 0.5212 | 0.0105 | 0.734 |
| gbdt_gap_logistic | 0.6 | 0.5163 | 0.0101 | 0.758 |
| v2 | 0.35 | 0.4263 | 0.0101 | 0.704 |

## Choice

- gap model: `logistic`
- lam_team: `0.3`
- worst-case mean spellF1 over the modes: `0.4659`
## Feature ablation and transfer (decision D-023)

The first run used all 71 features. `tools/gbdt_feature_shift.py` (local, PASS/FAIL only) then
failed check (a) on the BAR-ATH for `n_tracked_home` and `ev_coll_spread`. Ablations on the same
protocol (fold models, `gap_model: logistic`, spellF1 mean over 7 matches):

| feature set | lam_team | default | hidelo | hidehi |
|:---|---:|---:|---:|---:|
| all 71 features | 0.5 | 0.4836 | 0.4650 | 0.5214 |
| without `ball_speed`, `n_tracked_*`, `n_still_*`, `since_last_obs`, `until_next_obs` | 0.35 | 0.4831 | 0.4618 | 0.5193 |
| F1-05 features + windows only (no raw pairs, no even columns but `ev_ball_conf`) | 0.5 | 0.4795 | 0.4589 | 0.5148 |

The raw camera/ball-noise columns add at most 0.003, so the final model excludes them together with
the two failing features (`feature_params.exclude_features`). The tables above this section are the
final model. With it, `tools/gbdt_feature_shift.py --degrade default hidelo hidehi` gives PASS on
(a), (b) and (c).

The `lam_team` grid of the first run (0.2 … 0.5, then 0.6 and 0.7) decreased on both sides of the
plateau; the final choice uses the F2-05 rule (plateau within 1 SE of the best `default` point, then
best worst case over `default`/`hidelo`/`hidehi`).
