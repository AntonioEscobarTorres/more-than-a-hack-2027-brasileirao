# Aggregated Error Analysis on External IDSSE Matches (F1-06)

- config: `v2` (`6a992f351783382f2532c34d5299307ce61ea20326ecf1e002aee5b95d2b241b`)
- team.kind: `calibrated` (calibrated_file: `configs/calib/team.yaml`)
- dead.kind: `calibrated` (calibrated_file: `configs/calib/dead.yaml`)
- decoder: `lam_team` = `0.35`, `lam_dead` = `0.2`
- degrade: `default`
- matches: `7`
- mean `spell_f1_mean`: `0.4019` | mean `macro_f1`: `0.8127`
- **possession count ratio (pred / true)**: `0.6968` (mean across matches: `0.7034`, total: `2133/3061`)

## 1. Per-Match Summary

| Match | spell_f1_mean | macro_f1 | alive_dead_f1 | n_gt_spells | n_pred_spells | count_ratio |
|:---|---:|---:|---:|---:|---:|---:|
| J03WMX | 0.3840 | 0.8232 | 0.9321 | 511 | 313 | 0.6125 |
| J03WN1 | 0.3776 | 0.8151 | 0.9281 | 500 | 304 | 0.6080 |
| J03WOH | 0.4023 | 0.8153 | 0.9252 | 402 | 295 | 0.7338 |
| J03WOY | 0.4113 | 0.7933 | 0.9118 | 456 | 325 | 0.7127 |
| J03WPY | 0.4000 | 0.8222 | 0.9351 | 421 | 299 | 0.7102 |
| J03WQQ | 0.4454 | 0.8050 | 0.9141 | 401 | 322 | 0.8030 |
| J03WR9 | 0.3926 | 0.8147 | 0.9215 | 370 | 275 | 0.7432 |

## 2. Per-Frame Error Rate by Situation Category

| Category | Graded Frames | Error Frames | Pooled Error Rate | Mean Match Error Rate |
|:---|---:|---:|---:|---:|
| `all_graded` | 1002644 | 170825 | 0.1704 | 0.1704 |
| `ball_visible` | 576821 | 108662 | 0.1884 | 0.1889 |
| `short_gap` | 2813 | 827 | 0.2940 | 0.2911 |
| `long_gap` | 423010 | 61336 | 0.1450 | 0.1449 |
| `near_possession_switch_2s` | 257736 | 94363 | 0.3661 | 0.3664 |
| `near_restart_3s` | 53124 | 17201 | 0.3238 | 0.3249 |

## 3. Missed Ground-Truth Possessions by Duration (IoU < 0.5)

| Duration Bin | GT Possessions | Missed Possessions | Miss Rate | Ball Visible Frac | Gap Frac | `|t| < 0.5` (Near 0) | Wrong Sign (`t` opp) | Right Sign (`|t| >= 0.5`) | Pred `D` Frac | Pred Opp Team Frac |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| < 2 s | 919 | 878 | 0.9554 | 0.7313 | 0.2687 | 0.4061 | 0.2924 | 0.3016 | 0.1337 | 0.5382 |
| 2-10 s | 1508 | 998 | 0.6618 | 0.7261 | 0.2739 | 0.3699 | 0.2260 | 0.4041 | 0.0928 | 0.3739 |
| > 10 s | 634 | 139 | 0.2192 | 0.6587 | 0.3413 | 0.2521 | 0.0927 | 0.6552 | 0.1275 | 0.1132 |
| All | 3061 | 2015 | 0.6583 | 0.7036 | 0.2964 | 0.3339 | 0.1883 | 0.4779 | 0.1096 | 0.3042 |

## 4. Comparison Across Variants

| Variant | spell_f1_mean | macro_f1 | Count Ratio (Pred/GT) | Miss Rate (< 2 s) | Miss Rate (2-10 s) | Miss Rate (> 10 s) | Switch <=2s Error | Restart <=3s Error |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| `v1` | 0.3429 | 0.7688 | 0.4534 | 0.9869 | 0.7745 | 0.3344 | 0.4300 | 0.5181 |
| `v2` | 0.4019 | 0.8127 | 0.6968 | 0.9554 | 0.6618 | 0.2192 | 0.3661 | 0.3238 |

## 5. Diagnostic Conclusions & Actionable Feedback (F1-05 / F2-06 / Report Limitations)

1. **Short and Medium Possessions Dominate Missed Spells:**
   - Possessions `< 2 s` have a miss rate of `95.5%` (`878/919`), and `2-10 s` possessions have a miss rate of `66.2%` (`998/1508`).
   - Overall predicted-to-true possession count ratio is `0.697` (`2133` predicted vs `3061` ground truth), confirming that brief turnovers are absorbed by the surrounding possession or merged across transitions.
2. **What Happened in `team_logit` During Missed Possessions:**
   - Across all missed possessions, the ball was visible on `70.4%` of frames and in a tracking gap on `29.6%` of frames.
   - On `33.4%` of missed-possession frames, `team_logit` was near zero (`|t| < 0.5`); on `18.8%` it had the wrong sign; and on `47.8%` it already had the **correct sign (`|t| >= 0.5`)** but was overridden by decoder transition costs (`lam_team`) or dead-ball (`D`) predictions (`11.0%` predicted as `D`, `30.4%` as opponent).
3. **Transition and Restart Vulnerability:**
   - Frame error rate jumps from `18.8%` on visible-ball frames to `36.6%` within `+-2 s` of a true possession switch and `32.4%` within `3 s` after a dead-ball restart (`D` end).
   - **Candidates for F1-05 / F2-06:** (a) sharper evidence for possessions shorter than 2 s; (b) tune `lam_team` jointly with the calibrated `team_logit`, since frames with the right sign are still overridden by the transition cost; (c) dead-ball restart exits.
