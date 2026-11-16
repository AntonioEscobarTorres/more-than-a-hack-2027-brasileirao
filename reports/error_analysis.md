# Aggregated Error Analysis on External IDSSE Matches (F1-06)

- config: `v1_calibrated` (`b4499b1654130630d88b295a5ac074cc210a153554c9e22e24bb2039bb0616c8`)
- team.kind: `calibrated` (calibrated_file: `/home/fernando/more-than-a-hack-2027/configs/calib/team.yaml`)
- degrade: `default`
- matches: `7`
- mean `spell_f1_mean`: `0.3571` | mean `macro_f1`: `0.7912`
- **possession count ratio (pred / true)**: `0.4224` (mean across matches: `0.4254`, total: `1293/3061`)

## 1. Per-Match Summary

| Match | spell_f1_mean | macro_f1 | alive_dead_f1 | n_gt_spells | n_pred_spells | count_ratio |
|:---|---:|---:|---:|---:|---:|---:|
| J03WMX | 0.3624 | 0.7993 | 0.8855 | 511 | 201 | 0.3933 |
| J03WN1 | 0.3160 | 0.7952 | 0.9004 | 500 | 181 | 0.3620 |
| J03WOH | 0.3526 | 0.7999 | 0.8947 | 402 | 163 | 0.4055 |
| J03WOY | 0.3674 | 0.7931 | 0.8977 | 456 | 206 | 0.4518 |
| J03WPY | 0.3848 | 0.7932 | 0.8954 | 421 | 184 | 0.4371 |
| J03WQQ | 0.3565 | 0.7678 | 0.8642 | 401 | 188 | 0.4688 |
| J03WR9 | 0.3600 | 0.7900 | 0.8860 | 370 | 170 | 0.4595 |

## 2. Per-Frame Error Rate by Situation Category

| Category | Graded Frames | Error Frames | Pooled Error Rate | Mean Match Error Rate |
|:---|---:|---:|---:|---:|
| `all_graded` | 1002644 | 188766 | 0.1883 | 0.1883 |
| `ball_visible` | 579634 | 105704 | 0.1824 | 0.1828 |
| `short_gap` | 23 | 1 | 0.0435 | 0.2500 |
| `long_gap` | 422987 | 83061 | 0.1964 | 0.1963 |
| `near_possession_switch_2s` | 257736 | 105497 | 0.4093 | 0.4098 |
| `near_restart_3s` | 53124 | 24218 | 0.4559 | 0.4562 |

## 3. Missed Ground-Truth Possessions by Duration (IoU < 0.5)

| Duration Bin | GT Possessions | Missed Possessions | Miss Rate | Ball Visible Frac | Gap Frac | `|t| < 0.5` (Near 0) | Wrong Sign (`t` opp) | Right Sign (`|t| >= 0.5`) | Pred `D` Frac | Pred Opp Team Frac |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| < 2 s | 919 | 911 | 0.9913 | 0.7386 | 0.2614 | 0.2719 | 0.4368 | 0.2913 | 0.2515 | 0.4982 |
| 2-10 s | 1508 | 1160 | 0.7692 | 0.7023 | 0.2977 | 0.2370 | 0.2513 | 0.5117 | 0.2463 | 0.3201 |
| > 10 s | 634 | 204 | 0.3218 | 0.6137 | 0.3863 | 0.1217 | 0.0494 | 0.8288 | 0.3309 | 0.0393 |
| All | 3061 | 2275 | 0.7432 | 0.6727 | 0.3273 | 0.1973 | 0.1944 | 0.6084 | 0.2786 | 0.2327 |

## 4. Comparison Across Variants

| Variant | spell_f1_mean | macro_f1 | Count Ratio (Pred/GT) | Miss Rate (< 2 s) | Miss Rate (2-10 s) | Miss Rate (> 10 s) | Switch <=2s Error | Restart <=3s Error |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| `rules_v1` | 0.3429 | 0.7688 | 0.4534 | 0.9869 | 0.7745 | 0.3344 | 0.4300 | 0.5181 |
| `calibrated` | 0.3571 | 0.7912 | 0.4224 | 0.9913 | 0.7692 | 0.3218 | 0.4093 | 0.4559 |

## 5. Diagnostic Conclusions & Actionable Feedback (F1-05 / F2-06 / Report Limitations)

1. **Short and Medium Possessions Dominate Missed Spells:**
   - Possessions `< 2 s` have a miss rate of `99.1%` (`911/919`), and `2-10 s` possessions have a miss rate of `76.9%` (`1160/1508`).
   - Overall predicted-to-true possession count ratio is `0.422` (`1293` predicted vs `3061` ground truth), confirming that brief turnovers are absorbed by the surrounding possession or merged across transitions.
2. **What Happened in `team_logit` During Missed Possessions:**
   - Across all missed possessions, the ball was visible on `67.3%` of frames and in a tracking gap on `32.7%` of frames.
   - On `19.7%` of missed-possession frames, `team_logit` was near zero (`|t| < 0.5`); on `19.4%` it had the wrong sign; and on `60.8%` it already had the **correct sign (`|t| >= 0.5`)** but was overridden by decoder transition costs (`lam_team`) or dead-ball (`D`) predictions (`27.9%` predicted as `D`, `23.3%` as opponent).
3. **Transition and Restart Vulnerability:**
   - Frame error rate jumps from `18.2%` on visible-ball frames to `40.9%` within `+-2 s` of a true possession switch and `45.6%` within `3 s` after a dead-ball restart (`D` end).
   - **Recommendations for F1-05 / F2-06:** (a) Include the controlled-touch feature (`f9`) which sharpens short-spell evidence without over-smoothing switches; (b) pair calibrated `team_logit` with a higher `lam_team` in the F2-06 sweep so correct-sign short possessions overcome the ~10.8-nat H<->A transition barrier; (c) evaluate dead-ball restart exit transitions carefully.
