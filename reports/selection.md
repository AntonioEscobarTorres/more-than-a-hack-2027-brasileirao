# Selection protocol (F2-06, Section 6)

Primary metric: spellF1 on degraded IDSSE (`default`). Acceptance: paired gain > 1 SE on `default`, and no stress mode (`harder`/`easier`/`hidelo`/`hidehi`) worsens spellF1 by more than 1 SE. `fused` is gain-only on `default` (D-024). BAR-ATH filters are local PASS/FAIL only (never used to rank).

**Ladder winner:** `v2_gbdt_deadgbdt_nb_hzn2_fused` (spellF1 0.5399 on `fused`). **`configs/final.yaml`** is this winner **without the duration stages** (`duration_stages` D 2 → 1, same λ), after the control below (D-028, item 3).

## Step: baseline — baseline

- seeded ladder with `baseline`: spellF1 0.1729±0.0049, spell ratio 2.7711

| match | spell_f1_mean | macro_f1 | alive_dead_f1 | trans_f1_25 | mean_abs_latency_25 | n_gt_spells | n_pred_spells |
|:---|---:|---:|---:|---:|---:|---:|---:|
| J03WMX | 0.1594 | 0.7043 | 0.8739 | 0.1568 | 10.9954 | 511.0000 | 1261.0000 |
| J03WN1 | 0.1731 | 0.6709 | 0.8226 | 0.1471 | 10.5114 | 500.0000 | 1353.0000 |
| J03WOH | 0.1603 | 0.6778 | 0.8457 | 0.1353 | 12.1488 | 402.0000 | 1168.0000 |
| J03WOY | 0.1687 | 0.6775 | 0.8524 | 0.1525 | 10.7526 | 456.0000 | 1142.0000 |
| J03WPY | 0.1781 | 0.7001 | 0.8770 | 0.1531 | 11.1590 | 421.0000 | 1201.0000 |
| J03WQQ | 0.1977 | 0.6743 | 0.8607 | 0.1345 | 11.2903 | 401.0000 | 1090.0000 |
| J03WR9 | 0.1727 | 0.7056 | 0.8698 | 0.1398 | 10.7921 | 370.0000 | 1200.0000 |
| mean | 0.1729 | 0.6872 | 0.8574 | 0.1456 | 11.0928 | 437.2857 | 1202.1429 |
| se | 0.0049 | 0.0058 | 0.0072 | 0.0034 | 0.2019 | 20.1609 | 32.2146 |

## Step: v1 — v1 (Appendix A priors)

- candidate: `v1` vs current `baseline`
- default paired Δ spellF1: +0.1560±0.0105 (gain_ok=True, current 0.1729, candidate 0.3289, ratios 2.7711→0.4250)
- candidate decoder: lam_team 0.2, lam_dead 0.2, duration_stages {'H': 1, 'A': 1, 'D': 1}
- **accepted:** True
- stress `harder`: Δ +0.1512±0.0092 ok=True (curr 0.1322 / cand 0.2834, ratio 3.3403→0.3943)
- stress `easier`: Δ +0.1600±0.0100 ok=True (curr 0.2053 / cand 0.3653, ratio 2.5725→0.4428)
- stress `hidelo`: Δ +0.1341±0.0112 ok=True (curr 0.1620 / cand 0.2961, ratio 2.7668→0.4095)
- stress `hidehi`: Δ +0.1809±0.0114 ok=True (curr 0.1882 / cand 0.3691, ratio 2.7655→0.4503)

### candidate on default

| match | spell_f1_mean | macro_f1 | alive_dead_f1 | trans_f1_25 | mean_abs_latency_25 | n_gt_spells | n_pred_spells |
|:---|---:|---:|---:|---:|---:|---:|---:|
| J03WMX | 0.3246 | 0.7720 | 0.8738 | 0.3217 | 5.8041 | 511.0000 | 205.0000 |
| J03WN1 | 0.2799 | 0.7803 | 0.9005 | 0.3030 | 4.9248 | 500.0000 | 176.0000 |
| J03WOH | 0.3527 | 0.7851 | 0.8918 | 0.3178 | 6.6667 | 402.0000 | 182.0000 |
| J03WOY | 0.3028 | 0.7531 | 0.8785 | 0.3164 | 6.8955 | 456.0000 | 186.0000 |
| J03WPY | 0.3355 | 0.7663 | 0.8871 | 0.3338 | 5.6212 | 421.0000 | 181.0000 |
| J03WQQ | 0.3620 | 0.7468 | 0.8505 | 0.3124 | 7.6803 | 401.0000 | 188.0000 |
| J03WR9 | 0.3445 | 0.7683 | 0.8762 | 0.3255 | 5.7712 | 370.0000 | 171.0000 |
| mean | 0.3289 | 0.7674 | 0.8798 | 0.3187 | 6.1948 | 437.2857 | 184.1429 |
| se | 0.0110 | 0.0052 | 0.0060 | 0.0037 | 0.3517 | 20.1609 | 4.1024 |

## Step: v1s — v1s (F2-05 decoder)

- candidate: `v1s` vs current `v1`
- default paired Δ spellF1: +0.0313±0.0113 (gain_ok=True, current 0.3289, candidate 0.3601, ratios 0.4250→0.6240)
- candidate decoder: lam_team 0.4, lam_dead 0.4, duration_stages {'H': 1, 'A': 1, 'D': 1}
- **accepted:** True
- stress `harder`: Δ +0.0268±0.0085 ok=True (curr 0.2834 / cand 0.3102, ratio 0.3943→0.5928)
- stress `easier`: Δ +0.0432±0.0084 ok=True (curr 0.3653 / cand 0.4085, ratio 0.4428→0.6307)
- stress `hidelo`: Δ +0.0292±0.0108 ok=True (curr 0.2961 / cand 0.3253, ratio 0.4095→0.6076)
- stress `hidehi`: Δ +0.0335±0.0101 ok=True (curr 0.3691 / cand 0.4026, ratio 0.4503→0.6488)

### candidate on default

| match | spell_f1_mean | macro_f1 | alive_dead_f1 | trans_f1_25 | mean_abs_latency_25 | n_gt_spells | n_pred_spells |
|:---|---:|---:|---:|---:|---:|---:|---:|
| J03WMX | 0.3304 | 0.7629 | 0.8698 | 0.3210 | 6.0250 | 511.0000 | 271.0000 |
| J03WN1 | 0.3745 | 0.7845 | 0.8952 | 0.3523 | 6.6532 | 500.0000 | 268.0000 |
| J03WOH | 0.3792 | 0.7774 | 0.8835 | 0.3402 | 6.7248 | 402.0000 | 271.0000 |
| J03WOY | 0.3360 | 0.7469 | 0.8721 | 0.3418 | 7.5679 | 456.0000 | 275.0000 |
| J03WPY | 0.3682 | 0.7656 | 0.8837 | 0.3569 | 5.9937 | 421.0000 | 271.0000 |
| J03WQQ | 0.3742 | 0.7372 | 0.8474 | 0.3259 | 8.2517 | 401.0000 | 298.0000 |
| J03WR9 | 0.3584 | 0.7665 | 0.8731 | 0.3404 | 6.0511 | 370.0000 | 236.0000 |
| mean | 0.3601 | 0.7630 | 0.8750 | 0.3398 | 6.7525 | 437.2857 | 270.0000 |
| se | 0.0074 | 0.0062 | 0.0057 | 0.0049 | 0.3281 | 20.1609 | 6.8522 |

## Step: v2_base — v2_base (calibrated, no touch)

- candidate: `v2_base` vs current `v1s`
- default paired Δ spellF1: +0.0291±0.0100 (gain_ok=True, current 0.3601, candidate 0.3892, ratios 0.6240→0.7002)
- candidate decoder: lam_team 0.3, lam_dead 0.2, duration_stages {'H': 1, 'A': 1, 'D': 1}
- **accepted:** True
- stress `harder`: Δ +0.0474±0.0097 ok=True (curr 0.3102 / cand 0.3576, ratio 0.5928→0.7150)
- stress `easier`: Δ +0.0136±0.0075 ok=True (curr 0.4085 / cand 0.4221, ratio 0.6307→0.6728)
- stress `hidelo`: Δ +0.0492±0.0099 ok=True (curr 0.3253 / cand 0.3745, ratio 0.6076→0.6979)
- stress `hidehi`: Δ +0.0096±0.0108 ok=True (curr 0.4026 / cand 0.4122, ratio 0.6488→0.7045)

### candidate on default

| match | spell_f1_mean | macro_f1 | alive_dead_f1 | trans_f1_25 | mean_abs_latency_25 | n_gt_spells | n_pred_spells |
|:---|---:|---:|---:|---:|---:|---:|---:|
| J03WMX | 0.3809 | 0.8187 | 0.9290 | 0.3431 | 7.5314 | 511.0000 | 304.0000 |
| J03WN1 | 0.3637 | 0.8086 | 0.9280 | 0.3422 | 7.3851 | 500.0000 | 304.0000 |
| J03WOH | 0.3902 | 0.8033 | 0.9209 | 0.3439 | 8.7763 | 402.0000 | 292.0000 |
| J03WOY | 0.3890 | 0.7825 | 0.9084 | 0.3533 | 8.8833 | 456.0000 | 343.0000 |
| J03WPY | 0.3828 | 0.8135 | 0.9342 | 0.3533 | 7.2531 | 421.0000 | 302.0000 |
| J03WQQ | 0.4359 | 0.7991 | 0.9122 | 0.3780 | 9.1029 | 401.0000 | 324.0000 |
| J03WR9 | 0.3821 | 0.8156 | 0.9246 | 0.3399 | 8.1295 | 370.0000 | 257.0000 |
| mean | 0.3892 | 0.8059 | 0.9225 | 0.3505 | 8.1517 | 437.2857 | 303.7143 |
| se | 0.0084 | 0.0047 | 0.0035 | 0.0050 | 0.2932 | 20.1609 | 10.0918 |

## Step: v2 — v2 (calibrated + touch persistence, D-022)

- candidate: `v2` vs current `v2_base`
- default paired Δ spellF1: +0.0126±0.0023 (gain_ok=True, current 0.3892, candidate 0.4019, ratios 0.7002→0.7034)
- candidate decoder: lam_team 0.35, lam_dead 0.2, duration_stages {'H': 1, 'A': 1, 'D': 1}
- **accepted:** True
- stress `harder`: Δ +0.0100±0.0024 ok=True (curr 0.3576 / cand 0.3676, ratio 0.7150→0.7253)
- stress `easier`: Δ +0.0139±0.0040 ok=True (curr 0.4221 / cand 0.4360, ratio 0.6728→0.6906)
- stress `hidelo`: Δ +0.0103±0.0019 ok=True (curr 0.3745 / cand 0.3848, ratio 0.6979→0.7055)
- stress `hidehi`: Δ +0.0141±0.0043 ok=True (curr 0.4122 / cand 0.4263, ratio 0.7045→0.7044)

### candidate on default

| match | spell_f1_mean | macro_f1 | alive_dead_f1 | trans_f1_25 | mean_abs_latency_25 | n_gt_spells | n_pred_spells |
|:---|---:|---:|---:|---:|---:|---:|---:|
| J03WMX | 0.3840 | 0.8232 | 0.9321 | 0.3619 | 8.4409 | 511.0000 | 313.0000 |
| J03WN1 | 0.3776 | 0.8151 | 0.9281 | 0.3697 | 7.5745 | 500.0000 | 304.0000 |
| J03WOH | 0.4023 | 0.8153 | 0.9252 | 0.3874 | 8.8779 | 402.0000 | 295.0000 |
| J03WOY | 0.4113 | 0.7933 | 0.9118 | 0.3860 | 8.3938 | 456.0000 | 325.0000 |
| J03WPY | 0.4000 | 0.8222 | 0.9351 | 0.3720 | 7.3412 | 421.0000 | 299.0000 |
| J03WQQ | 0.4454 | 0.8050 | 0.9141 | 0.3801 | 9.2727 | 401.0000 | 322.0000 |
| J03WR9 | 0.3926 | 0.8147 | 0.9215 | 0.3445 | 8.2917 | 370.0000 | 275.0000 |
| mean | 0.4019 | 0.8127 | 0.9240 | 0.3717 | 8.3132 | 437.2857 | 304.7143 |
| se | 0.0084 | 0.0039 | 0.0033 | 0.0057 | 0.2560 | 20.1609 | 6.5273 |

## Step: dead_pooled — dead pooled (F3-04)

- candidate: `v2_dead_pooled` vs current `v2`
- default paired Δ spellF1: -0.0012±0.0010 (gain_ok=False, current 0.4019, candidate 0.4007, ratios 0.7034→0.7020)
- candidate decoder: lam_team 0.35, lam_dead 0.2, duration_stages {'H': 1, 'A': 1, 'D': 1}
- **accepted:** False
- stress `harder`: Δ -0.0013±0.0012 ok=False (curr 0.3676 / cand 0.3663, ratio 0.7253→0.7227)
- stress `easier`: Δ -0.0000±0.0009 ok=True (curr 0.4360 / cand 0.4360, ratio 0.6906→0.6881)
- stress `hidelo`: Δ -0.0010±0.0011 ok=True (curr 0.3848 / cand 0.3837, ratio 0.7055→0.7044)
- stress `hidehi`: Δ -0.0011±0.0010 ok=False (curr 0.4263 / cand 0.4252, ratio 0.7044→0.7024)

### candidate on default

| match | spell_f1_mean | macro_f1 | alive_dead_f1 | trans_f1_25 | mean_abs_latency_25 | n_gt_spells | n_pred_spells |
|:---|---:|---:|---:|---:|---:|---:|---:|
| J03WMX | 0.3787 | 0.8216 | 0.9306 | 0.3643 | 8.5372 | 511.0000 | 314.0000 |
| J03WN1 | 0.3781 | 0.8154 | 0.9278 | 0.3688 | 7.6364 | 500.0000 | 302.0000 |
| J03WOH | 0.3983 | 0.8143 | 0.9245 | 0.3820 | 8.7471 | 402.0000 | 297.0000 |
| J03WOY | 0.4087 | 0.7915 | 0.9092 | 0.3828 | 8.4398 | 456.0000 | 324.0000 |
| J03WPY | 0.4006 | 0.8211 | 0.9335 | 0.3672 | 7.2083 | 421.0000 | 299.0000 |
| J03WQQ | 0.4467 | 0.8042 | 0.9131 | 0.3800 | 9.3829 | 401.0000 | 319.0000 |
| J03WR9 | 0.3938 | 0.8159 | 0.9221 | 0.3469 | 8.1517 | 370.0000 | 274.0000 |
| mean | 0.4007 | 0.8120 | 0.9229 | 0.3703 | 8.3005 | 437.2857 | 304.1429 |
| se | 0.0088 | 0.0040 | 0.0034 | 0.0048 | 0.2718 | 20.1609 | 6.3674 |

## Step: team_gbdt — team GBDT (F1-07, D-023)

- candidate: `v2_gbdt` vs current `v2`
- default paired Δ spellF1: +0.0876±0.0098 (gain_ok=True, current 0.4019, candidate 0.4895, ratios 0.7034→0.6566)
- candidate decoder: lam_team 0.3, lam_dead 0.2, duration_stages {'H': 1, 'A': 1, 'D': 1}
- **accepted:** True
- stress `harder`: Δ +0.0900±0.0044 ok=True (curr 0.3676 / cand 0.4576, ratio 0.7253→0.6552)
- stress `easier`: Δ +0.0831±0.0057 ok=True (curr 0.4360 / cand 0.5191, ratio 0.6906→0.6585)
- stress `hidelo`: Δ +0.0811±0.0095 ok=True (curr 0.3848 / cand 0.4659, ratio 0.7055→0.6528)
- stress `hidehi`: Δ +0.0974±0.0096 ok=True (curr 0.4263 / cand 0.5237, ratio 0.7044→0.6580)

### candidate on default

| match | spell_f1_mean | macro_f1 | alive_dead_f1 | trans_f1_25 | mean_abs_latency_25 | n_gt_spells | n_pred_spells |
|:---|---:|---:|---:|---:|---:|---:|---:|
| J03WMX | 0.4541 | 0.8507 | 0.9313 | 0.4219 | 7.9670 | 511.0000 | 288.0000 |
| J03WN1 | 0.4785 | 0.8479 | 0.9310 | 0.4586 | 5.5913 | 500.0000 | 290.0000 |
| J03WOH | 0.4793 | 0.8435 | 0.9212 | 0.4586 | 6.9021 | 402.0000 | 254.0000 |
| J03WOY | 0.5144 | 0.8355 | 0.9138 | 0.4673 | 6.6900 | 456.0000 | 306.0000 |
| J03WPY | 0.5173 | 0.8587 | 0.9353 | 0.4723 | 6.7943 | 421.0000 | 271.0000 |
| J03WQQ | 0.4871 | 0.8256 | 0.9113 | 0.4322 | 7.0459 | 401.0000 | 306.0000 |
| J03WR9 | 0.4955 | 0.8433 | 0.9252 | 0.4444 | 6.1989 | 370.0000 | 275.0000 |
| mean | 0.4895 | 0.8436 | 0.9242 | 0.4508 | 6.7413 | 437.2857 | 284.2857 |
| se | 0.0083 | 0.0040 | 0.0035 | 0.0070 | 0.2779 | 20.1609 | 7.1870 |

## Step: dead_gbdt — dead GBDT (F3-08, D-026)

- candidate: `v2_gbdt_deadgbdt` vs current `v2_gbdt`
- default paired Δ spellF1: +0.0305±0.0014 (gain_ok=True, current 0.4895, candidate 0.5200, ratios 0.6566→0.6396)
- candidate decoder: lam_team 0.3, lam_dead 0.2, duration_stages {'H': 1, 'A': 1, 'D': 1}
- **accepted:** True
- stress `harder`: Δ +0.0337±0.0040 ok=True (curr 0.4576 / cand 0.4913, ratio 0.6552→0.6417)
- stress `easier`: Δ +0.0281±0.0022 ok=True (curr 0.5191 / cand 0.5472, ratio 0.6585→0.6419)
- stress `hidelo`: Δ +0.0355±0.0016 ok=True (curr 0.4659 / cand 0.5014, ratio 0.6528→0.6355)
- stress `hidehi`: Δ +0.0253±0.0043 ok=True (curr 0.5237 / cand 0.5490, ratio 0.6580→0.6550)

### candidate on default

| match | spell_f1_mean | macro_f1 | alive_dead_f1 | trans_f1_25 | mean_abs_latency_25 | n_gt_spells | n_pred_spells |
|:---|---:|---:|---:|---:|---:|---:|---:|
| J03WMX | 0.4775 | 0.8847 | 0.9661 | 0.4940 | 6.7878 | 511.0000 | 279.0000 |
| J03WN1 | 0.5093 | 0.8673 | 0.9548 | 0.5445 | 5.6434 | 500.0000 | 287.0000 |
| J03WOH | 0.5095 | 0.8677 | 0.9432 | 0.4941 | 5.8421 | 402.0000 | 252.0000 |
| J03WOY | 0.5455 | 0.8747 | 0.9536 | 0.5515 | 6.0642 | 456.0000 | 289.0000 |
| J03WPY | 0.5532 | 0.8942 | 0.9702 | 0.5701 | 5.4800 | 421.0000 | 263.0000 |
| J03WQQ | 0.5191 | 0.8642 | 0.9455 | 0.4742 | 6.4076 | 401.0000 | 291.0000 |
| J03WR9 | 0.5257 | 0.8796 | 0.9595 | 0.5286 | 5.8784 | 370.0000 | 276.0000 |
| mean | 0.5200 | 0.8761 | 0.9561 | 0.5224 | 6.0148 | 437.2857 | 276.7143 |
| se | 0.0095 | 0.0041 | 0.0038 | 0.0134 | 0.1709 | 20.1609 | 5.4890 |

## Step: post_midpoint — postprocess midpoint (F2-05 isolated)

- candidate: `v2_gbdt_deadgbdt_midpoint` vs current `v2_gbdt_deadgbdt`
- default paired Δ spellF1: +0.0000±0.0000 (gain_ok=False, current 0.5200, candidate 0.5200, ratios 0.6396→0.6396)
- candidate decoder: lam_team 0.3, lam_dead 0.2, duration_stages {'H': 1, 'A': 1, 'D': 1}
- **accepted:** False
- stress `harder`: Δ +0.0000±0.0000 ok=True (curr 0.4913 / cand 0.4913, ratio 0.6417→0.6417)
- stress `easier`: Δ +0.0000±0.0000 ok=True (curr 0.5472 / cand 0.5472, ratio 0.6419→0.6419)
- stress `hidelo`: Δ +0.0000±0.0000 ok=True (curr 0.5014 / cand 0.5014, ratio 0.6355→0.6355)
- stress `hidehi`: Δ +0.0000±0.0000 ok=True (curr 0.5490 / cand 0.5490, ratio 0.6550→0.6550)

### candidate on default

| match | spell_f1_mean | macro_f1 | alive_dead_f1 | trans_f1_25 | mean_abs_latency_25 | n_gt_spells | n_pred_spells |
|:---|---:|---:|---:|---:|---:|---:|---:|
| J03WMX | 0.4775 | 0.8847 | 0.9661 | 0.4940 | 6.7878 | 511.0000 | 279.0000 |
| J03WN1 | 0.5093 | 0.8673 | 0.9548 | 0.5445 | 5.6434 | 500.0000 | 287.0000 |
| J03WOH | 0.5095 | 0.8677 | 0.9432 | 0.4941 | 5.8421 | 402.0000 | 252.0000 |
| J03WOY | 0.5455 | 0.8747 | 0.9536 | 0.5515 | 6.0642 | 456.0000 | 289.0000 |
| J03WPY | 0.5532 | 0.8942 | 0.9702 | 0.5701 | 5.4800 | 421.0000 | 263.0000 |
| J03WQQ | 0.5191 | 0.8642 | 0.9455 | 0.4742 | 6.4076 | 401.0000 | 291.0000 |
| J03WR9 | 0.5257 | 0.8796 | 0.9595 | 0.5286 | 5.8784 | 370.0000 | 276.0000 |
| mean | 0.5200 | 0.8761 | 0.9561 | 0.5224 | 6.0148 | 437.2857 | 276.7143 |
| se | 0.0095 | 0.0041 | 0.0038 | 0.0134 | 0.1709 | 20.1609 | 5.4890 |

## Step: post_d_bridge_25 — postprocess d_bridge=25

- candidate: `v2_gbdt_deadgbdt_d_bridge_25` vs current `v2_gbdt_deadgbdt`
- default paired Δ spellF1: +0.0000±0.0000 (gain_ok=False, current 0.5200, candidate 0.5200, ratios 0.6396→0.6396)
- candidate decoder: lam_team 0.3, lam_dead 0.2, duration_stages {'H': 1, 'A': 1, 'D': 1}
- **accepted:** False
- stress `harder`: Δ +0.0000±0.0000 ok=True (curr 0.4913 / cand 0.4913, ratio 0.6417→0.6417)
- stress `easier`: Δ +0.0000±0.0000 ok=True (curr 0.5472 / cand 0.5472, ratio 0.6419→0.6419)
- stress `hidelo`: Δ +0.0000±0.0000 ok=True (curr 0.5014 / cand 0.5014, ratio 0.6355→0.6355)
- stress `hidehi`: Δ +0.0000±0.0000 ok=True (curr 0.5490 / cand 0.5490, ratio 0.6550→0.6550)

### candidate on default

| match | spell_f1_mean | macro_f1 | alive_dead_f1 | trans_f1_25 | mean_abs_latency_25 | n_gt_spells | n_pred_spells |
|:---|---:|---:|---:|---:|---:|---:|---:|
| J03WMX | 0.4775 | 0.8847 | 0.9661 | 0.4940 | 6.7878 | 511.0000 | 279.0000 |
| J03WN1 | 0.5093 | 0.8673 | 0.9548 | 0.5445 | 5.6434 | 500.0000 | 287.0000 |
| J03WOH | 0.5095 | 0.8677 | 0.9432 | 0.4941 | 5.8421 | 402.0000 | 252.0000 |
| J03WOY | 0.5455 | 0.8747 | 0.9536 | 0.5515 | 6.0642 | 456.0000 | 289.0000 |
| J03WPY | 0.5532 | 0.8942 | 0.9702 | 0.5701 | 5.4800 | 421.0000 | 263.0000 |
| J03WQQ | 0.5191 | 0.8642 | 0.9455 | 0.4742 | 6.4076 | 401.0000 | 291.0000 |
| J03WR9 | 0.5257 | 0.8796 | 0.9595 | 0.5286 | 5.8784 | 370.0000 | 276.0000 |
| mean | 0.5200 | 0.8761 | 0.9561 | 0.5224 | 6.0148 | 437.2857 | 276.7143 |
| se | 0.0095 | 0.0041 | 0.0038 | 0.0134 | 0.1709 | 20.1609 | 5.4890 |

## Step: post_d_bridge_50 — postprocess d_bridge=50

- candidate: `v2_gbdt_deadgbdt_d_bridge_50` vs current `v2_gbdt_deadgbdt`
- default paired Δ spellF1: -0.0000±0.0000 (gain_ok=False, current 0.5200, candidate 0.5199, ratios 0.6396→0.6393)
- candidate decoder: lam_team 0.3, lam_dead 0.2, duration_stages {'H': 1, 'A': 1, 'D': 1}
- **accepted:** False
- stress `harder`: Δ -0.0001±0.0001 ok=False (curr 0.4913 / cand 0.4912, ratio 0.6417→0.6411)
- stress `easier`: Δ +0.0003±0.0003 ok=True (curr 0.5472 / cand 0.5475, ratio 0.6419→0.6416)
- stress `hidelo`: Δ +0.0000±0.0000 ok=True (curr 0.5014 / cand 0.5014, ratio 0.6355→0.6352)
- stress `hidehi`: Δ -0.0000±0.0000 ok=False (curr 0.5490 / cand 0.5489, ratio 0.6550→0.6543)

### candidate on default

| match | spell_f1_mean | macro_f1 | alive_dead_f1 | trans_f1_25 | mean_abs_latency_25 | n_gt_spells | n_pred_spells |
|:---|---:|---:|---:|---:|---:|---:|---:|
| J03WMX | 0.4775 | 0.8847 | 0.9661 | 0.4940 | 6.7878 | 511.0000 | 279.0000 |
| J03WN1 | 0.5093 | 0.8673 | 0.9548 | 0.5445 | 5.6434 | 500.0000 | 287.0000 |
| J03WOH | 0.5095 | 0.8677 | 0.9432 | 0.4941 | 5.8421 | 402.0000 | 252.0000 |
| J03WOY | 0.5452 | 0.8744 | 0.9534 | 0.5485 | 6.1027 | 456.0000 | 288.0000 |
| J03WPY | 0.5532 | 0.8942 | 0.9702 | 0.5701 | 5.4800 | 421.0000 | 263.0000 |
| J03WQQ | 0.5191 | 0.8642 | 0.9455 | 0.4742 | 6.4076 | 401.0000 | 291.0000 |
| J03WR9 | 0.5257 | 0.8796 | 0.9595 | 0.5286 | 5.8784 | 370.0000 | 276.0000 |
| mean | 0.5199 | 0.8760 | 0.9561 | 0.5220 | 6.0203 | 437.2857 | 276.5714 |
| se | 0.0095 | 0.0041 | 0.0038 | 0.0133 | 0.1712 | 20.1609 | 5.4373 |

## Step: post_min_spell_13 — postprocess min_spell=13

- candidate: `v2_gbdt_deadgbdt_min_spell_13` vs current `v2_gbdt_deadgbdt`
- default paired Δ spellF1: +0.0000±0.0000 (gain_ok=False, current 0.5200, candidate 0.5200, ratios 0.6396→0.6396)
- candidate decoder: lam_team 0.3, lam_dead 0.2, duration_stages {'H': 1, 'A': 1, 'D': 1}
- **accepted:** False
- stress `harder`: Δ +0.0000±0.0000 ok=True (curr 0.4913 / cand 0.4913, ratio 0.6417→0.6417)
- stress `easier`: Δ +0.0000±0.0000 ok=True (curr 0.5472 / cand 0.5472, ratio 0.6419→0.6419)
- stress `hidelo`: Δ +0.0000±0.0000 ok=True (curr 0.5014 / cand 0.5014, ratio 0.6355→0.6355)
- stress `hidehi`: Δ +0.0000±0.0000 ok=True (curr 0.5490 / cand 0.5490, ratio 0.6550→0.6550)

### candidate on default

| match | spell_f1_mean | macro_f1 | alive_dead_f1 | trans_f1_25 | mean_abs_latency_25 | n_gt_spells | n_pred_spells |
|:---|---:|---:|---:|---:|---:|---:|---:|
| J03WMX | 0.4775 | 0.8847 | 0.9661 | 0.4940 | 6.7878 | 511.0000 | 279.0000 |
| J03WN1 | 0.5093 | 0.8673 | 0.9548 | 0.5445 | 5.6434 | 500.0000 | 287.0000 |
| J03WOH | 0.5095 | 0.8677 | 0.9432 | 0.4941 | 5.8421 | 402.0000 | 252.0000 |
| J03WOY | 0.5455 | 0.8747 | 0.9536 | 0.5515 | 6.0642 | 456.0000 | 289.0000 |
| J03WPY | 0.5532 | 0.8942 | 0.9702 | 0.5701 | 5.4800 | 421.0000 | 263.0000 |
| J03WQQ | 0.5191 | 0.8642 | 0.9455 | 0.4742 | 6.4076 | 401.0000 | 291.0000 |
| J03WR9 | 0.5257 | 0.8796 | 0.9595 | 0.5286 | 5.8784 | 370.0000 | 276.0000 |
| mean | 0.5200 | 0.8761 | 0.9561 | 0.5224 | 6.0148 | 437.2857 | 276.7143 |
| se | 0.0095 | 0.0041 | 0.0038 | 0.0134 | 0.1709 | 20.1609 | 5.4890 |

## Step: post_min_spell_25 — postprocess min_spell=25

- candidate: `v2_gbdt_deadgbdt_min_spell_25` vs current `v2_gbdt_deadgbdt`
- default paired Δ spellF1: -0.0001±0.0004 (gain_ok=False, current 0.5200, candidate 0.5199, ratios 0.6396→0.6380)
- candidate decoder: lam_team 0.3, lam_dead 0.2, duration_stages {'H': 1, 'A': 1, 'D': 1}
- **accepted:** False
- stress `harder`: Δ -0.0005±0.0008 ok=True (curr 0.4913 / cand 0.4909, ratio 0.6417→0.6397)
- stress `easier`: Δ -0.0007±0.0007 ok=True (curr 0.5472 / cand 0.5465, ratio 0.6419→0.6413)
- stress `hidelo`: Δ -0.0003±0.0003 ok=True (curr 0.5014 / cand 0.5011, ratio 0.6355→0.6348)
- stress `hidehi`: Δ +0.0005±0.0004 ok=True (curr 0.5490 / cand 0.5495, ratio 0.6550→0.6530)

### candidate on default

| match | spell_f1_mean | macro_f1 | alive_dead_f1 | trans_f1_25 | mean_abs_latency_25 | n_gt_spells | n_pred_spells |
|:---|---:|---:|---:|---:|---:|---:|---:|
| J03WMX | 0.4788 | 0.8847 | 0.9659 | 0.4939 | 6.8033 | 511.0000 | 276.0000 |
| J03WN1 | 0.5093 | 0.8673 | 0.9548 | 0.5445 | 5.6434 | 500.0000 | 287.0000 |
| J03WOH | 0.5095 | 0.8677 | 0.9432 | 0.4941 | 5.8421 | 402.0000 | 252.0000 |
| J03WOY | 0.5455 | 0.8747 | 0.9536 | 0.5515 | 6.0642 | 456.0000 | 289.0000 |
| J03WPY | 0.5513 | 0.8940 | 0.9702 | 0.5691 | 5.4779 | 421.0000 | 261.0000 |
| J03WQQ | 0.5191 | 0.8642 | 0.9455 | 0.4742 | 6.4076 | 401.0000 | 291.0000 |
| J03WR9 | 0.5257 | 0.8796 | 0.9595 | 0.5286 | 5.8784 | 370.0000 | 276.0000 |
| mean | 0.5199 | 0.8760 | 0.9561 | 0.5223 | 6.0167 | 437.2857 | 276.0000 |
| se | 0.0092 | 0.0041 | 0.0038 | 0.0134 | 0.1727 | 20.1609 | 5.5976 |

## Step: post_d_cross_25 — postprocess d_cross=25

- candidate: `v2_gbdt_deadgbdt_d_cross_25` vs current `v2_gbdt_deadgbdt`
- default paired Δ spellF1: +0.0000±0.0000 (gain_ok=False, current 0.5200, candidate 0.5200, ratios 0.6396→0.6396)
- candidate decoder: lam_team 0.3, lam_dead 0.2, duration_stages {'H': 1, 'A': 1, 'D': 1}
- **accepted:** False
- stress `harder`: Δ +0.0000±0.0000 ok=True (curr 0.4913 / cand 0.4913, ratio 0.6417→0.6417)
- stress `easier`: Δ +0.0000±0.0000 ok=True (curr 0.5472 / cand 0.5472, ratio 0.6419→0.6419)
- stress `hidelo`: Δ +0.0000±0.0000 ok=True (curr 0.5014 / cand 0.5014, ratio 0.6355→0.6355)
- stress `hidehi`: Δ +0.0000±0.0000 ok=True (curr 0.5490 / cand 0.5490, ratio 0.6550→0.6550)

### candidate on default

| match | spell_f1_mean | macro_f1 | alive_dead_f1 | trans_f1_25 | mean_abs_latency_25 | n_gt_spells | n_pred_spells |
|:---|---:|---:|---:|---:|---:|---:|---:|
| J03WMX | 0.4775 | 0.8847 | 0.9661 | 0.4940 | 6.7878 | 511.0000 | 279.0000 |
| J03WN1 | 0.5093 | 0.8673 | 0.9548 | 0.5445 | 5.6434 | 500.0000 | 287.0000 |
| J03WOH | 0.5095 | 0.8677 | 0.9432 | 0.4941 | 5.8421 | 402.0000 | 252.0000 |
| J03WOY | 0.5455 | 0.8747 | 0.9536 | 0.5515 | 6.0642 | 456.0000 | 289.0000 |
| J03WPY | 0.5532 | 0.8942 | 0.9702 | 0.5701 | 5.4800 | 421.0000 | 263.0000 |
| J03WQQ | 0.5191 | 0.8642 | 0.9455 | 0.4742 | 6.4076 | 401.0000 | 291.0000 |
| J03WR9 | 0.5257 | 0.8796 | 0.9595 | 0.5286 | 5.8784 | 370.0000 | 276.0000 |
| mean | 0.5200 | 0.8761 | 0.9561 | 0.5224 | 6.0148 | 437.2857 | 276.7143 |
| se | 0.0095 | 0.0041 | 0.0038 | 0.0134 | 0.1709 | 20.1609 | 5.4890 |

## Step: post_d_cross_50 — postprocess d_cross=50

- candidate: `v2_gbdt_deadgbdt_d_cross_50` vs current `v2_gbdt_deadgbdt`
- default paired Δ spellF1: +0.0000±0.0000 (gain_ok=False, current 0.5200, candidate 0.5200, ratios 0.6396→0.6396)
- candidate decoder: lam_team 0.3, lam_dead 0.2, duration_stages {'H': 1, 'A': 1, 'D': 1}
- **accepted:** False
- stress `harder`: Δ +0.0000±0.0000 ok=True (curr 0.4913 / cand 0.4913, ratio 0.6417→0.6417)
- stress `easier`: Δ +0.0000±0.0000 ok=True (curr 0.5472 / cand 0.5472, ratio 0.6419→0.6419)
- stress `hidelo`: Δ +0.0000±0.0000 ok=True (curr 0.5014 / cand 0.5014, ratio 0.6355→0.6355)
- stress `hidehi`: Δ +0.0000±0.0000 ok=True (curr 0.5490 / cand 0.5490, ratio 0.6550→0.6550)

### candidate on default

| match | spell_f1_mean | macro_f1 | alive_dead_f1 | trans_f1_25 | mean_abs_latency_25 | n_gt_spells | n_pred_spells |
|:---|---:|---:|---:|---:|---:|---:|---:|
| J03WMX | 0.4775 | 0.8847 | 0.9661 | 0.4940 | 6.7878 | 511.0000 | 279.0000 |
| J03WN1 | 0.5093 | 0.8673 | 0.9548 | 0.5445 | 5.6434 | 500.0000 | 287.0000 |
| J03WOH | 0.5095 | 0.8677 | 0.9432 | 0.4941 | 5.8421 | 402.0000 | 252.0000 |
| J03WOY | 0.5455 | 0.8745 | 0.9534 | 0.5479 | 6.0190 | 456.0000 | 289.0000 |
| J03WPY | 0.5532 | 0.8942 | 0.9702 | 0.5701 | 5.4800 | 421.0000 | 263.0000 |
| J03WQQ | 0.5191 | 0.8643 | 0.9457 | 0.4747 | 6.4076 | 401.0000 | 291.0000 |
| J03WR9 | 0.5257 | 0.8796 | 0.9595 | 0.5286 | 5.8784 | 370.0000 | 276.0000 |
| mean | 0.5200 | 0.8760 | 0.9561 | 0.5220 | 6.0083 | 437.2857 | 276.7143 |
| se | 0.0095 | 0.0041 | 0.0038 | 0.0132 | 0.1707 | 20.1609 | 5.4890 |

## Step: post_d_cross_75 — postprocess d_cross=75

- candidate: `v2_gbdt_deadgbdt_d_cross_75` vs current `v2_gbdt_deadgbdt`
- default paired Δ spellF1: -0.0003±0.0002 (gain_ok=False, current 0.5200, candidate 0.5196, ratios 0.6396→0.6396)
- candidate decoder: lam_team 0.3, lam_dead 0.2, duration_stages {'H': 1, 'A': 1, 'D': 1}
- **accepted:** False
- stress `harder`: Δ -0.0001±0.0003 ok=True (curr 0.4913 / cand 0.4912, ratio 0.6417→0.6417)
- stress `easier`: Δ -0.0002±0.0002 ok=False (curr 0.5472 / cand 0.5470, ratio 0.6419→0.6419)
- stress `hidelo`: Δ -0.0003±0.0002 ok=False (curr 0.5014 / cand 0.5011, ratio 0.6355→0.6355)
- stress `hidehi`: Δ -0.0001±0.0001 ok=False (curr 0.5490 / cand 0.5489, ratio 0.6550→0.6550)

### candidate on default

| match | spell_f1_mean | macro_f1 | alive_dead_f1 | trans_f1_25 | mean_abs_latency_25 | n_gt_spells | n_pred_spells |
|:---|---:|---:|---:|---:|---:|---:|---:|
| J03WMX | 0.4770 | 0.8849 | 0.9664 | 0.4945 | 6.7878 | 511.0000 | 279.0000 |
| J03WN1 | 0.5093 | 0.8673 | 0.9548 | 0.5445 | 5.6434 | 500.0000 | 287.0000 |
| J03WOH | 0.5095 | 0.8677 | 0.9432 | 0.4941 | 5.8421 | 402.0000 | 252.0000 |
| J03WOY | 0.5455 | 0.8740 | 0.9530 | 0.5443 | 6.0383 | 456.0000 | 289.0000 |
| J03WPY | 0.5526 | 0.8945 | 0.9706 | 0.5708 | 5.3840 | 421.0000 | 263.0000 |
| J03WQQ | 0.5179 | 0.8640 | 0.9455 | 0.4730 | 6.5286 | 401.0000 | 291.0000 |
| J03WR9 | 0.5257 | 0.8796 | 0.9595 | 0.5286 | 5.8784 | 370.0000 | 276.0000 |
| mean | 0.5196 | 0.8760 | 0.9561 | 0.5214 | 6.0146 | 437.2857 | 276.7143 |
| se | 0.0095 | 0.0042 | 0.0038 | 0.0132 | 0.1856 | 20.1609 | 5.4890 |

## Step: boundary_shift — boundary_shift_file (F2-05)

- candidate: `v2_gbdt_deadgbdt_boundary` vs current `v2_gbdt_deadgbdt`
- default paired Δ spellF1: -0.0148±0.0021 (gain_ok=False, current 0.5200, candidate 0.5052, ratios 0.6396→0.6396)
- candidate decoder: lam_team 0.3, lam_dead 0.2, duration_stages {'H': 1, 'A': 1, 'D': 1}
- **accepted:** False
- stress `harder`: Δ -0.0172±0.0024 ok=False (curr 0.4913 / cand 0.4741, ratio 0.6417→0.6417)
- stress `easier`: Δ -0.0142±0.0028 ok=False (curr 0.5472 / cand 0.5330, ratio 0.6419→0.6419)
- stress `hidelo`: Δ -0.0172±0.0024 ok=False (curr 0.5014 / cand 0.4842, ratio 0.6355→0.6355)
- stress `hidehi`: Δ -0.0150±0.0027 ok=False (curr 0.5490 / cand 0.5339, ratio 0.6550→0.6550)

### candidate on default

| match | spell_f1_mean | macro_f1 | alive_dead_f1 | trans_f1_25 | mean_abs_latency_25 | n_gt_spells | n_pred_spells |
|:---|---:|---:|---:|---:|---:|---:|---:|
| J03WMX | 0.4633 | 0.8801 | 0.9657 | 0.4960 | 8.2439 | 511.0000 | 279.0000 |
| J03WN1 | 0.4895 | 0.8605 | 0.9547 | 0.5365 | 7.2724 | 500.0000 | 287.0000 |
| J03WOH | 0.4881 | 0.8617 | 0.9433 | 0.4823 | 7.0245 | 402.0000 | 252.0000 |
| J03WOY | 0.5337 | 0.8696 | 0.9537 | 0.5411 | 7.5000 | 456.0000 | 289.0000 |
| J03WPY | 0.5345 | 0.8887 | 0.9699 | 0.5656 | 7.2097 | 421.0000 | 263.0000 |
| J03WQQ | 0.5133 | 0.8618 | 0.9455 | 0.4764 | 8.0000 | 401.0000 | 291.0000 |
| J03WR9 | 0.5139 | 0.8761 | 0.9594 | 0.5310 | 7.4619 | 370.0000 | 276.0000 |
| mean | 0.5052 | 0.8712 | 0.9560 | 0.5184 | 7.5303 | 437.2857 | 276.7143 |
| se | 0.0099 | 0.0041 | 0.0037 | 0.0127 | 0.1662 | 20.1609 | 5.4890 |

## Step: durations — semi-Markov durations (F2-08, D-027), λ re-swept

- candidate: `v2_gbdt_deadgbdt_nb` vs current `v2_gbdt_deadgbdt`
- default paired Δ spellF1: +0.0043±0.0034 (gain_ok=True, current 0.5200, candidate 0.5243, ratios 0.6396→0.6853)
- candidate decoder: lam_team 0.4, lam_dead 0.4, duration_stages {'H': 1, 'A': 1, 'D': 2}
- **accepted:** True
- stress `harder`: Δ +0.0089±0.0052 ok=True (curr 0.4913 / cand 0.5002, ratio 0.6417→0.7000)
- stress `easier`: Δ +0.0027±0.0021 ok=True (curr 0.5472 / cand 0.5498, ratio 0.6419→0.6806)
- stress `hidelo`: Δ +0.0045±0.0035 ok=True (curr 0.5014 / cand 0.5059, ratio 0.6355→0.6778)
- stress `hidehi`: Δ +0.0043±0.0044 ok=True (curr 0.5490 / cand 0.5533, ratio 0.6550→0.7009)

### candidate on default

| match | spell_f1_mean | macro_f1 | alive_dead_f1 | trans_f1_25 | mean_abs_latency_25 | n_gt_spells | n_pred_spells |
|:---|---:|---:|---:|---:|---:|---:|---:|
| J03WMX | 0.4758 | 0.8819 | 0.9649 | 0.5020 | 6.9291 | 511.0000 | 296.0000 |
| J03WN1 | 0.5095 | 0.8653 | 0.9528 | 0.5433 | 5.5181 | 500.0000 | 300.0000 |
| J03WOH | 0.5163 | 0.8715 | 0.9458 | 0.5052 | 5.6318 | 402.0000 | 272.0000 |
| J03WOY | 0.5478 | 0.8754 | 0.9546 | 0.5532 | 5.8974 | 456.0000 | 310.0000 |
| J03WPY | 0.5591 | 0.8906 | 0.9682 | 0.5724 | 5.6795 | 421.0000 | 288.0000 |
| J03WQQ | 0.5413 | 0.8693 | 0.9465 | 0.5071 | 6.4249 | 401.0000 | 315.0000 |
| J03WR9 | 0.5201 | 0.8763 | 0.9562 | 0.5213 | 5.9071 | 370.0000 | 293.0000 |
| mean | 0.5243 | 0.8758 | 0.9556 | 0.5292 | 5.9983 | 437.2857 | 296.2857 |
| se | 0.0106 | 0.0032 | 0.0032 | 0.0104 | 0.1911 | 20.1609 | 5.3927 |

## Step: hazard — hazard-guided transitions (F2-07, D-025, beta 2), λ re-swept

- candidate: `v2_gbdt_deadgbdt_nb_hzn2` vs current `v2_gbdt_deadgbdt_nb`
- default paired Δ spellF1: +0.0119±0.0043 (gain_ok=True, current 0.5243, candidate 0.5362, ratios 0.6853→0.7049)
- candidate decoder: lam_team 0.4, lam_dead 0.2, duration_stages {'H': 1, 'A': 1, 'D': 2}
- **accepted:** True
- stress `harder`: Δ +0.0125±0.0059 ok=True (curr 0.5002 / cand 0.5127, ratio 0.7000→0.7063)
- stress `easier`: Δ +0.0181±0.0053 ok=True (curr 0.5498 / cand 0.5680, ratio 0.6806→0.7028)
- stress `hidelo`: Δ +0.0102±0.0038 ok=True (curr 0.5059 / cand 0.5161, ratio 0.6778→0.6939)
- stress `hidehi`: Δ +0.0114±0.0037 ok=True (curr 0.5533 / cand 0.5647, ratio 0.7009→0.7198)

### candidate on default

| match | spell_f1_mean | macro_f1 | alive_dead_f1 | trans_f1_25 | mean_abs_latency_25 | n_gt_spells | n_pred_spells |
|:---|---:|---:|---:|---:|---:|---:|---:|
| J03WMX | 0.4893 | 0.8840 | 0.9662 | 0.5147 | 6.3206 | 511.0000 | 309.0000 |
| J03WN1 | 0.5192 | 0.8697 | 0.9559 | 0.5417 | 4.8116 | 500.0000 | 309.0000 |
| J03WOH | 0.5102 | 0.8668 | 0.9421 | 0.5069 | 5.5434 | 402.0000 | 273.0000 |
| J03WOY | 0.5670 | 0.8757 | 0.9529 | 0.5657 | 6.1857 | 456.0000 | 320.0000 |
| J03WPY | 0.5675 | 0.8930 | 0.9704 | 0.5787 | 5.1825 | 421.0000 | 298.0000 |
| J03WQQ | 0.5488 | 0.8662 | 0.9442 | 0.4957 | 5.9043 | 401.0000 | 330.0000 |
| J03WR9 | 0.5514 | 0.8819 | 0.9589 | 0.5310 | 5.3348 | 370.0000 | 296.0000 |
| mean | 0.5362 | 0.8768 | 0.9558 | 0.5335 | 5.6119 | 437.2857 | 305.0000 |
| se | 0.0114 | 0.0038 | 0.0040 | 0.0116 | 0.2084 | 20.1609 | 6.9625 |

## Step: fused — ball_source fused (F4-06, D-024)

- candidate: `v2_gbdt_deadgbdt_nb_hzn2_fused` vs current `v2_gbdt_deadgbdt_nb_hzn2`
- default paired Δ spellF1: +0.0037±0.0018 (gain_ok=True, current 0.5362, candidate 0.5399, ratios 0.7049→0.7105)
- candidate decoder: lam_team 0.4, lam_dead 0.2, duration_stages {'H': 1, 'A': 1, 'D': 2}
- **accepted:** True

### candidate on default

| match | spell_f1_mean | macro_f1 | alive_dead_f1 | trans_f1_25 | mean_abs_latency_25 | n_gt_spells | n_pred_spells |
|:---|---:|---:|---:|---:|---:|---:|---:|
| J03WMX | 0.4979 | 0.8848 | 0.9669 | 0.5234 | 6.2239 | 511.0000 | 314.0000 |
| J03WN1 | 0.5276 | 0.8725 | 0.9580 | 0.5524 | 4.7092 | 500.0000 | 312.0000 |
| J03WOH | 0.5081 | 0.8668 | 0.9424 | 0.5127 | 5.3919 | 402.0000 | 275.0000 |
| J03WOY | 0.5702 | 0.8769 | 0.9539 | 0.5706 | 6.0813 | 456.0000 | 322.0000 |
| J03WPY | 0.5694 | 0.8964 | 0.9714 | 0.5875 | 5.0936 | 421.0000 | 297.0000 |
| J03WQQ | 0.5567 | 0.8709 | 0.9465 | 0.5059 | 5.8248 | 401.0000 | 329.0000 |
| J03WR9 | 0.5492 | 0.8806 | 0.9565 | 0.5307 | 5.2489 | 370.0000 | 303.0000 |
| mean | 0.5399 | 0.8784 | 0.9565 | 0.5405 | 5.5105 | 437.2857 | 307.4286 |
| se | 0.0110 | 0.0038 | 0.0039 | 0.0116 | 0.2088 | 20.1609 | 6.7643 |

## Post-ladder control of the duration step (D-027 item 7, D-028 item 3)

The ladder rule compares a step only with the current candidate. The D-027 control (the same λ re-sweep with
1 stage) shows the gain of the duration step comes from the λ re-sweep alone. Paired Δ spellF1, 7 matches:

| Comparison | `default` | `harder` | `easier` | `hidelo` | `hidehi` |
|:---|---:|---:|---:|---:|---:|
| stages (λ 0.4/0.4) vs control (λ 0.4/0.2, 1 stage) | +0.0000±0.0025 | -0.0013±0.0035 | -0.0006±0.0020 | +0.0003±0.0026 | -0.0003±0.0020 |
| control vs current (`v2_gbdt_deadgbdt`) | +0.0043±0.0021 | +0.0102±0.0029 | +0.0033±0.0017 | +0.0042±0.0018 | +0.0046±0.0042 |
| ladder winner (D 2 stages) vs the same with 1 stage, tactical | -0.0003±0.0010 | -0.0002±0.0010 | +0.0000±0.0006 | -0.0005±0.0009 | -0.0011±0.0009 |

On the `fused` masters: winner with stages 0.5399, without stages **0.5406** (Δ -0.0007±0.0011).
`configs/final.yaml` drops the stages. The selection code now requires the duration step to beat this
control too (`control_build`); a fresh run would reject the stages and re-sweep the hazard λ on the base
without them.

## Local BAR-ATH checks (human)

Agents never run these. Record PASS/FAIL only. The fused master is
`<data>/processed_fused/master_features.parquet` (built by `make_submission` with this config).

- csv_valid (`python -m modeling.make_submission --config configs/final.yaml --skip-preprocessing`, SHA-256 recorded): **pending**
- check_public_window (`--pred outputs/final.csv --master <fused master>`): **pending**
- deadball_diagnostics (±40% plausibility; `--pred outputs/final.csv --master <fused master> --config configs/final.yaml`): **pending**
- stability dead default/hidelo/hidehi (<5% label change, `tools/dead_stability.py --config configs/final.yaml`, GBDT fits): **pending**
- dead_feature_shift (`--config configs/final.yaml --calibration configs/calib/dead_gbdt.yaml --master <fused master> --degrade fused`): **pending**
- gbdt_feature_shift (team GBDT, fused master, `--degrade fused`): **PASS** (D-025 item 10; the team section is unchanged)
- hazard_feature_shift (`hazard_nodir`, tactical and fused masters): **PASS** (D-025 item 10; the hazard does not depend on the dead or decoder sections)

## Skipped optional steps

- `team_tag_prior` (F1-05 item 8): 0 on IDSSE, so not part of the gain rule; its pre-registered BAR-ATH agreement check was not run, so it stays off.
