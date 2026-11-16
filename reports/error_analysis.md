# Aggregated Error Analysis on External IDSSE Matches (F1-06)

- config: `v2` (`3c8245346763c156edd9cb0c7b86633f82169d730eb580c27b19854f2c1ea5a5`)
- team.kind: `calibrated` (calibrated_file: `configs/calib/team.yaml`)
- dead.kind: `calibrated` (calibrated_file: `configs/calib/dead.yaml`)
- decoder: `lam_team` = `0.35`, `lam_dead` = `0.2`
- degrade: `default`
- matches: `7`
- mean `spell_f1_mean`: `0.4002` | mean `macro_f1`: `0.8124`
- **possession count ratio (pred / true)**: `0.6959` (mean across matches: `0.7025`, total: `2130/3061`)

## 1. Per-Match Summary

| Match | spell_f1_mean | macro_f1 | alive_dead_f1 | n_gt_spells | n_pred_spells | count_ratio |
|:---|---:|---:|---:|---:|---:|---:|
| J03WMX | 0.3800 | 0.8223 | 0.9319 | 511 | 310 | 0.6067 |
| J03WN1 | 0.3736 | 0.8144 | 0.9281 | 500 | 304 | 0.6080 |
| J03WOH | 0.4023 | 0.8152 | 0.9252 | 402 | 295 | 0.7338 |
| J03WOY | 0.4113 | 0.7934 | 0.9118 | 456 | 325 | 0.7127 |
| J03WPY | 0.4000 | 0.8222 | 0.9351 | 421 | 299 | 0.7102 |
| J03WQQ | 0.4415 | 0.8046 | 0.9141 | 401 | 322 | 0.8030 |
| J03WR9 | 0.3926 | 0.8147 | 0.9215 | 370 | 275 | 0.7432 |

## 2. Per-Frame Error Rate by Situation Category

| Category | Graded Frames | Error Frames | Pooled Error Rate | Mean Match Error Rate |
|:---|---:|---:|---:|---:|
| `all_graded` | 1002644 | 171094 | 0.1706 | 0.1707 |
| `ball_visible` | 576821 | 108922 | 0.1888 | 0.1893 |
| `short_gap` | 2813 | 836 | 0.2972 | 0.2943 |
| `long_gap` | 423010 | 61336 | 0.1450 | 0.1449 |
| `near_possession_switch_2s` | 257736 | 94580 | 0.3670 | 0.3671 |
| `near_restart_3s` | 53124 | 17212 | 0.3240 | 0.3251 |

## 3. Missed Ground-Truth Possessions by Duration (IoU < 0.5)

| Duration Bin | GT Possessions | Missed Possessions | Miss Rate | Ball Visible Frac | Gap Frac | `|t| < 0.5` (Near 0) | Wrong Sign (`t` opp) | Right Sign (`|t| >= 0.5`) | Pred `D` Frac | Pred Opp Team Frac |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| < 2 s | 919 | 880 | 0.9576 | 0.7308 | 0.2692 | 0.4053 | 0.2915 | 0.3033 | 0.1332 | 0.5366 |
| 2-10 s | 1508 | 1001 | 0.6638 | 0.7272 | 0.2728 | 0.3702 | 0.2254 | 0.4044 | 0.0925 | 0.3747 |
| > 10 s | 634 | 140 | 0.2208 | 0.6532 | 0.3468 | 0.2520 | 0.0922 | 0.6558 | 0.1267 | 0.1136 |
| All | 3061 | 2021 | 0.6602 | 0.7022 | 0.2978 | 0.3337 | 0.1875 | 0.4787 | 0.1091 | 0.3042 |

## 4. Comparison Across Variants

| Variant | spell_f1_mean | macro_f1 | Count Ratio (Pred/GT) | Miss Rate (< 2 s) | Miss Rate (2-10 s) | Miss Rate (> 10 s) | Switch <=2s Error | Restart <=3s Error |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| `v1` | 0.3429 | 0.7688 | 0.4534 | 0.9869 | 0.7745 | 0.3344 | 0.4300 | 0.5181 |
| `v2` | 0.4002 | 0.8124 | 0.6959 | 0.9576 | 0.6638 | 0.2208 | 0.3670 | 0.3240 |

## 5. Diagnostic Conclusions & Actionable Feedback (F1-05 / F2-06 / Report Limitations)

1. **Short and Medium Possessions Dominate Missed Spells:**
   - Possessions `< 2 s` have a miss rate of `95.8%` (`880/919`), and `2-10 s` possessions have a miss rate of `66.4%` (`1001/1508`).
   - Overall predicted-to-true possession count ratio is `0.696` (`2130` predicted vs `3061` ground truth), confirming that brief turnovers are absorbed by the surrounding possession or merged across transitions.
2. **What Happened in `team_logit` During Missed Possessions:**
   - Across all missed possessions, the ball was visible on `70.2%` of frames and in a tracking gap on `29.8%` of frames.
   - On `33.4%` of missed-possession frames, `team_logit` was near zero (`|t| < 0.5`); on `18.8%` it had the wrong sign; and on `47.9%` it already had the **correct sign (`|t| >= 0.5`)** but was overridden by decoder transition costs (`lam_team`) or dead-ball (`D`) predictions (`10.9%` predicted as `D`, `30.4%` as opponent).
3. **Transition and Restart Vulnerability:**
   - Frame error rate jumps from `18.9%` on visible-ball frames to `36.7%` within `+-2 s` of a true possession switch and `32.4%` within `3 s` after a dead-ball restart (`D` end).
   - **Candidates for F1-05 / F2-06:** (a) sharper evidence for possessions shorter than 2 s; (b) tune `lam_team` jointly with the calibrated `team_logit`, since frames with the right sign are still overridden by the transition cost; (c) dead-ball restart exits.
