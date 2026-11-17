# Selection protocol (F2-06, Section 6)

Primary metric: spellF1 on degraded IDSSE (`default`). Acceptance: paired gain > 1 SE on `default`, and no stress mode (`harder`/`easier`/`hidelo`/`hidehi`) worsens spellF1 by more than 1 SE. `fused` is gain-only on `default` (D-024). BAR-ATH filters are local PASS/FAIL only (never used to rank).

**Winner:** `v2_gbdt_hzn2_fused` → written as `configs/final.yaml`.

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

## Step: post_midpoint — postprocess midpoint (F2-05 isolated)

- candidate: `v2_gbdt_midpoint` vs current `v2_gbdt`
- default paired Δ spellF1: +0.0000±0.0000 (gain_ok=False, current 0.4895, candidate 0.4895, ratios 0.6566→0.6566)
- **accepted:** False
- stress `harder`: Δ +0.0000±0.0000 ok=True (curr 0.4576 / cand 0.4576, ratio 0.6552→0.6552)
- stress `easier`: Δ +0.0000±0.0000 ok=True (curr 0.5191 / cand 0.5191, ratio 0.6585→0.6585)
- stress `hidelo`: Δ +0.0000±0.0000 ok=True (curr 0.4659 / cand 0.4659, ratio 0.6528→0.6528)
- stress `hidehi`: Δ +0.0000±0.0000 ok=True (curr 0.5237 / cand 0.5237, ratio 0.6580→0.6580)

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

## Step: post_d_bridge_25 — postprocess d_bridge=25

- candidate: `v2_gbdt_d_bridge_25` vs current `v2_gbdt`
- default paired Δ spellF1: +0.0000±0.0000 (gain_ok=False, current 0.4895, candidate 0.4895, ratios 0.6566→0.6566)
- **accepted:** False
- stress `harder`: Δ +0.0000±0.0000 ok=True (curr 0.4576 / cand 0.4576, ratio 0.6552→0.6552)
- stress `easier`: Δ +0.0000±0.0000 ok=True (curr 0.5191 / cand 0.5191, ratio 0.6585→0.6585)
- stress `hidelo`: Δ +0.0000±0.0000 ok=True (curr 0.4659 / cand 0.4659, ratio 0.6528→0.6528)
- stress `hidehi`: Δ +0.0000±0.0000 ok=True (curr 0.5237 / cand 0.5237, ratio 0.6580→0.6580)

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

## Step: post_d_bridge_50 — postprocess d_bridge=50

- candidate: `v2_gbdt_d_bridge_50` vs current `v2_gbdt`
- default paired Δ spellF1: +0.0000±0.0001 (gain_ok=False, current 0.4895, candidate 0.4895, ratios 0.6566→0.6560)
- **accepted:** False
- stress `harder`: Δ -0.0001±0.0002 ok=True (curr 0.4576 / cand 0.4575, ratio 0.6552→0.6539)
- stress `easier`: Δ +0.0001±0.0001 ok=True (curr 0.5191 / cand 0.5191, ratio 0.6585→0.6578)
- stress `hidelo`: Δ +0.0001±0.0002 ok=True (curr 0.4659 / cand 0.4660, ratio 0.6528→0.6515)
- stress `hidehi`: Δ -0.0001±0.0002 ok=True (curr 0.5237 / cand 0.5236, ratio 0.6580→0.6574)

### candidate on default

| match | spell_f1_mean | macro_f1 | alive_dead_f1 | trans_f1_25 | mean_abs_latency_25 | n_gt_spells | n_pred_spells |
|:---|---:|---:|---:|---:|---:|---:|---:|
| J03WMX | 0.4546 | 0.8510 | 0.9316 | 0.4227 | 7.9670 | 511.0000 | 287.0000 |
| J03WN1 | 0.4785 | 0.8479 | 0.9310 | 0.4586 | 5.5913 | 500.0000 | 290.0000 |
| J03WOH | 0.4793 | 0.8435 | 0.9212 | 0.4586 | 6.9021 | 402.0000 | 254.0000 |
| J03WOY | 0.5144 | 0.8355 | 0.9138 | 0.4673 | 6.6900 | 456.0000 | 306.0000 |
| J03WPY | 0.5173 | 0.8587 | 0.9353 | 0.4723 | 6.7943 | 421.0000 | 271.0000 |
| J03WQQ | 0.4867 | 0.8253 | 0.9111 | 0.4309 | 6.9590 | 401.0000 | 305.0000 |
| J03WR9 | 0.4955 | 0.8433 | 0.9252 | 0.4444 | 6.1989 | 370.0000 | 275.0000 |
| mean | 0.4895 | 0.8436 | 0.9242 | 0.4507 | 6.7289 | 437.2857 | 284.0000 |
| se | 0.0083 | 0.0041 | 0.0035 | 0.0070 | 0.2759 | 20.1609 | 7.1047 |

## Step: post_min_spell_13 — postprocess min_spell=13

- candidate: `v2_gbdt_min_spell_13` vs current `v2_gbdt`
- default paired Δ spellF1: +0.0000±0.0000 (gain_ok=False, current 0.4895, candidate 0.4895, ratios 0.6566→0.6566)
- **accepted:** False
- stress `harder`: Δ +0.0000±0.0000 ok=True (curr 0.4576 / cand 0.4576, ratio 0.6552→0.6552)
- stress `easier`: Δ +0.0000±0.0000 ok=True (curr 0.5191 / cand 0.5191, ratio 0.6585→0.6585)
- stress `hidelo`: Δ +0.0000±0.0000 ok=True (curr 0.4659 / cand 0.4659, ratio 0.6528→0.6528)
- stress `hidehi`: Δ +0.0000±0.0000 ok=True (curr 0.5237 / cand 0.5237, ratio 0.6580→0.6580)

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

## Step: post_min_spell_25 — postprocess min_spell=25

- candidate: `v2_gbdt_min_spell_25` vs current `v2_gbdt`
- default paired Δ spellF1: -0.0000±0.0004 (gain_ok=False, current 0.4895, candidate 0.4894, ratios 0.6566→0.6551)
- **accepted:** False
- stress `harder`: Δ -0.0005±0.0008 ok=True (curr 0.4576 / cand 0.4571, ratio 0.6552→0.6529)
- stress `easier`: Δ -0.0012±0.0010 ok=False (curr 0.5191 / cand 0.5178, ratio 0.6585→0.6569)
- stress `hidelo`: Δ -0.0003±0.0003 ok=True (curr 0.4659 / cand 0.4656, ratio 0.6528→0.6521)
- stress `hidehi`: Δ +0.0003±0.0002 ok=True (curr 0.5237 / cand 0.5240, ratio 0.6580→0.6558)

### candidate on default

| match | spell_f1_mean | macro_f1 | alive_dead_f1 | trans_f1_25 | mean_abs_latency_25 | n_gt_spells | n_pred_spells |
|:---|---:|---:|---:|---:|---:|---:|---:|
| J03WMX | 0.4558 | 0.8508 | 0.9313 | 0.4196 | 8.0048 | 511.0000 | 285.0000 |
| J03WN1 | 0.4785 | 0.8479 | 0.9310 | 0.4586 | 5.5913 | 500.0000 | 290.0000 |
| J03WOH | 0.4793 | 0.8435 | 0.9212 | 0.4586 | 6.9021 | 402.0000 | 254.0000 |
| J03WOY | 0.5144 | 0.8355 | 0.9138 | 0.4673 | 6.6900 | 456.0000 | 306.0000 |
| J03WPY | 0.5154 | 0.8585 | 0.9353 | 0.4711 | 6.7981 | 421.0000 | 269.0000 |
| J03WQQ | 0.4871 | 0.8256 | 0.9113 | 0.4322 | 7.0459 | 401.0000 | 306.0000 |
| J03WR9 | 0.4955 | 0.8433 | 0.9252 | 0.4444 | 6.1989 | 370.0000 | 275.0000 |
| mean | 0.4894 | 0.8436 | 0.9242 | 0.4503 | 6.7473 | 437.2857 | 283.5714 |
| se | 0.0080 | 0.0040 | 0.0035 | 0.0072 | 0.2819 | 20.1609 | 7.2534 |

## Step: post_d_cross_25 — postprocess d_cross=25

- candidate: `v2_gbdt_d_cross_25` vs current `v2_gbdt`
- default paired Δ spellF1: +0.0000±0.0000 (gain_ok=False, current 0.4895, candidate 0.4895, ratios 0.6566→0.6566)
- **accepted:** False
- stress `harder`: Δ +0.0000±0.0000 ok=True (curr 0.4576 / cand 0.4576, ratio 0.6552→0.6552)
- stress `easier`: Δ +0.0000±0.0000 ok=True (curr 0.5191 / cand 0.5191, ratio 0.6585→0.6585)
- stress `hidelo`: Δ +0.0000±0.0000 ok=True (curr 0.4659 / cand 0.4659, ratio 0.6528→0.6528)
- stress `hidehi`: Δ +0.0000±0.0000 ok=True (curr 0.5237 / cand 0.5237, ratio 0.6580→0.6580)

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

## Step: post_d_cross_50 — postprocess d_cross=50

- candidate: `v2_gbdt_d_cross_50` vs current `v2_gbdt`
- default paired Δ spellF1: -0.0002±0.0001 (gain_ok=False, current 0.4895, candidate 0.4893, ratios 0.6566→0.6566)
- **accepted:** False
- stress `harder`: Δ -0.0002±0.0001 ok=False (curr 0.4576 / cand 0.4574, ratio 0.6552→0.6552)
- stress `easier`: Δ +0.0000±0.0000 ok=True (curr 0.5191 / cand 0.5191, ratio 0.6585→0.6585)
- stress `hidelo`: Δ -0.0002±0.0001 ok=False (curr 0.4659 / cand 0.4657, ratio 0.6528→0.6528)
- stress `hidehi`: Δ +0.0000±0.0000 ok=True (curr 0.5237 / cand 0.5237, ratio 0.6580→0.6580)

### candidate on default

| match | spell_f1_mean | macro_f1 | alive_dead_f1 | trans_f1_25 | mean_abs_latency_25 | n_gt_spells | n_pred_spells |
|:---|---:|---:|---:|---:|---:|---:|---:|
| J03WMX | 0.4541 | 0.8501 | 0.9308 | 0.4207 | 7.9336 | 511.0000 | 288.0000 |
| J03WN1 | 0.4785 | 0.8476 | 0.9308 | 0.4591 | 5.6739 | 500.0000 | 290.0000 |
| J03WOH | 0.4793 | 0.8435 | 0.9214 | 0.4573 | 6.9223 | 402.0000 | 254.0000 |
| J03WOY | 0.5139 | 0.8350 | 0.9134 | 0.4663 | 6.7061 | 456.0000 | 306.0000 |
| J03WPY | 0.5168 | 0.8581 | 0.9347 | 0.4671 | 6.7573 | 421.0000 | 271.0000 |
| J03WQQ | 0.4871 | 0.8256 | 0.9113 | 0.4322 | 7.0459 | 401.0000 | 306.0000 |
| J03WR9 | 0.4955 | 0.8431 | 0.9250 | 0.4426 | 6.2162 | 370.0000 | 275.0000 |
| mean | 0.4893 | 0.8433 | 0.9239 | 0.4493 | 6.7508 | 437.2857 | 284.2857 |
| se | 0.0083 | 0.0040 | 0.0034 | 0.0068 | 0.2656 | 20.1609 | 7.1870 |

## Step: post_d_cross_75 — postprocess d_cross=75

- candidate: `v2_gbdt_d_cross_75` vs current `v2_gbdt`
- default paired Δ spellF1: -0.0006±0.0003 (gain_ok=False, current 0.4895, candidate 0.4889, ratios 0.6566→0.6566)
- **accepted:** False
- stress `harder`: Δ -0.0002±0.0001 ok=False (curr 0.4576 / cand 0.4574, ratio 0.6552→0.6552)
- stress `easier`: Δ -0.0003±0.0002 ok=False (curr 0.5191 / cand 0.5188, ratio 0.6585→0.6585)
- stress `hidelo`: Δ -0.0003±0.0002 ok=False (curr 0.4659 / cand 0.4656, ratio 0.6528→0.6528)
- stress `hidehi`: Δ -0.0001±0.0002 ok=True (curr 0.5237 / cand 0.5235, ratio 0.6580→0.6580)

### candidate on default

| match | spell_f1_mean | macro_f1 | alive_dead_f1 | trans_f1_25 | mean_abs_latency_25 | n_gt_spells | n_pred_spells |
|:---|---:|---:|---:|---:|---:|---:|---:|
| J03WMX | 0.4521 | 0.8492 | 0.9301 | 0.4184 | 8.0239 | 511.0000 | 288.0000 |
| J03WN1 | 0.4785 | 0.8476 | 0.9308 | 0.4591 | 5.6739 | 500.0000 | 290.0000 |
| J03WOH | 0.4793 | 0.8435 | 0.9214 | 0.4573 | 6.9223 | 402.0000 | 254.0000 |
| J03WOY | 0.5134 | 0.8345 | 0.9136 | 0.4645 | 6.6947 | 456.0000 | 306.0000 |
| J03WPY | 0.5168 | 0.8576 | 0.9343 | 0.4654 | 6.7902 | 421.0000 | 271.0000 |
| J03WQQ | 0.4866 | 0.8243 | 0.9104 | 0.4292 | 7.1031 | 401.0000 | 306.0000 |
| J03WR9 | 0.4955 | 0.8431 | 0.9250 | 0.4426 | 6.2162 | 370.0000 | 275.0000 |
| mean | 0.4889 | 0.8428 | 0.9237 | 0.4481 | 6.7749 | 437.2857 | 284.2857 |
| se | 0.0084 | 0.0041 | 0.0034 | 0.0070 | 0.2769 | 20.1609 | 7.1870 |

## Step: boundary_shift — boundary_shift_file (F2-05)

- candidate: `v2_gbdt_boundary` vs current `v2_gbdt`
- default paired Δ spellF1: -0.0142±0.0014 (gain_ok=False, current 0.4895, candidate 0.4752, ratios 0.6566→0.6566)
- **accepted:** False
- stress `harder`: Δ -0.0140±0.0022 ok=False (curr 0.4576 / cand 0.4436, ratio 0.6552→0.6552)
- stress `easier`: Δ -0.0141±0.0028 ok=False (curr 0.5191 / cand 0.5050, ratio 0.6585→0.6585)
- stress `hidelo`: Δ -0.0144±0.0014 ok=False (curr 0.4659 / cand 0.4515, ratio 0.6528→0.6528)
- stress `hidehi`: Δ -0.0139±0.0019 ok=False (curr 0.5237 / cand 0.5097, ratio 0.6580→0.6580)

### candidate on default

| match | spell_f1_mean | macro_f1 | alive_dead_f1 | trans_f1_25 | mean_abs_latency_25 | n_gt_spells | n_pred_spells |
|:---|---:|---:|---:|---:|---:|---:|---:|
| J03WMX | 0.4416 | 0.8469 | 0.9310 | 0.4239 | 9.3099 | 511.0000 | 288.0000 |
| J03WN1 | 0.4597 | 0.8413 | 0.9309 | 0.4526 | 7.6652 | 500.0000 | 290.0000 |
| J03WOH | 0.4634 | 0.8376 | 0.9210 | 0.4444 | 8.3670 | 402.0000 | 254.0000 |
| J03WOY | 0.4987 | 0.8307 | 0.9138 | 0.4612 | 8.4735 | 456.0000 | 306.0000 |
| J03WPY | 0.4994 | 0.8534 | 0.9350 | 0.4746 | 8.7238 | 421.0000 | 271.0000 |
| J03WQQ | 0.4775 | 0.8229 | 0.9113 | 0.4322 | 8.8010 | 401.0000 | 306.0000 |
| J03WR9 | 0.4862 | 0.8402 | 0.9249 | 0.4444 | 7.7634 | 370.0000 | 275.0000 |
| mean | 0.4752 | 0.8390 | 0.9240 | 0.4476 | 8.4434 | 437.2857 | 284.2857 |
| se | 0.0081 | 0.0038 | 0.0034 | 0.0065 | 0.2200 | 20.1609 | 7.1870 |

## Step: hazard — hazard-guided transitions (F2-07, D-025, beta 2)

- candidate: `v2_gbdt_hzn2` vs current `v2_gbdt`
- default paired Δ spellF1: +0.0141±0.0031 (gain_ok=True, current 0.4895, candidate 0.5036, ratios 0.6566→0.6861)
- **accepted:** True
- stress `harder`: Δ +0.0128±0.0043 ok=True (curr 0.4576 / cand 0.4704, ratio 0.6552→0.6775)
- stress `easier`: Δ +0.0144±0.0026 ok=True (curr 0.5191 / cand 0.5335, ratio 0.6585→0.6916)
- stress `hidelo`: Δ +0.0130±0.0027 ok=True (curr 0.4659 / cand 0.4789, ratio 0.6528→0.6839)
- stress `hidehi`: Δ +0.0139±0.0036 ok=True (curr 0.5237 / cand 0.5375, ratio 0.6580→0.6920)

### candidate on default

| match | spell_f1_mean | macro_f1 | alive_dead_f1 | trans_f1_25 | mean_abs_latency_25 | n_gt_spells | n_pred_spells |
|:---|---:|---:|---:|---:|---:|---:|---:|
| J03WMX | 0.4654 | 0.8527 | 0.9318 | 0.4416 | 7.2267 | 511.0000 | 302.0000 |
| J03WN1 | 0.4928 | 0.8515 | 0.9323 | 0.4635 | 4.7149 | 500.0000 | 302.0000 |
| J03WOH | 0.4766 | 0.8421 | 0.9201 | 0.4606 | 6.5561 | 402.0000 | 261.0000 |
| J03WOY | 0.5328 | 0.8356 | 0.9136 | 0.4919 | 6.7910 | 456.0000 | 318.0000 |
| J03WPY | 0.5331 | 0.8577 | 0.9351 | 0.4796 | 6.4817 | 421.0000 | 294.0000 |
| J03WQQ | 0.5104 | 0.8298 | 0.9125 | 0.4502 | 6.5096 | 401.0000 | 320.0000 |
| J03WR9 | 0.5139 | 0.8465 | 0.9254 | 0.4592 | 5.6598 | 370.0000 | 283.0000 |
| mean | 0.5036 | 0.8451 | 0.9244 | 0.4638 | 6.2771 | 437.2857 | 297.1429 |
| se | 0.0100 | 0.0038 | 0.0035 | 0.0065 | 0.3148 | 20.1609 | 7.7473 |

## Step: fused — ball_source fused (F4-06, D-024)

- candidate: `v2_gbdt_hzn2_fused` vs current `v2_gbdt_hzn2`
- default paired Δ spellF1: +0.0061±0.0020 (gain_ok=True, current 0.5036, candidate 0.5097, ratios 0.6861→0.6913)
- **accepted:** True

### candidate on default

| match | spell_f1_mean | macro_f1 | alive_dead_f1 | trans_f1_25 | mean_abs_latency_25 | n_gt_spells | n_pred_spells |
|:---|---:|---:|---:|---:|---:|---:|---:|
| J03WMX | 0.4701 | 0.8538 | 0.9343 | 0.4501 | 6.8783 | 511.0000 | 305.0000 |
| J03WN1 | 0.5066 | 0.8565 | 0.9368 | 0.4828 | 4.6423 | 500.0000 | 307.0000 |
| J03WOH | 0.4762 | 0.8439 | 0.9224 | 0.4737 | 6.3547 | 402.0000 | 265.0000 |
| J03WOY | 0.5347 | 0.8360 | 0.9138 | 0.4935 | 6.8612 | 456.0000 | 319.0000 |
| J03WPY | 0.5365 | 0.8604 | 0.9360 | 0.4912 | 6.3632 | 421.0000 | 294.0000 |
| J03WQQ | 0.5217 | 0.8362 | 0.9167 | 0.4604 | 6.3915 | 401.0000 | 319.0000 |
| J03WR9 | 0.5224 | 0.8515 | 0.9289 | 0.4652 | 5.6041 | 370.0000 | 287.0000 |
| mean | 0.5097 | 0.8483 | 0.9270 | 0.4738 | 6.1565 | 437.2857 | 299.4286 |
| se | 0.0102 | 0.0037 | 0.0036 | 0.0061 | 0.2989 | 20.1609 | 7.2764 |

## Local BAR-ATH checks (human)

Agents never run these. Record PASS/FAIL only.

`configs/final.yaml` has the same parameters as `configs/v2_gbdt_fused_hzn2.yaml` (same config hash
up to `name`), so the checks already run on that config (D-025, item 10) carry over; the CSV must
still be regenerated from `configs/final.yaml` and its SHA-256 compared.

- csv_valid: **PASS** (`v2_gbdt_fused_hzn2`, SHA-256 `ae80daec…`, D-025 item 10); regenerate from `final.yaml`: **pending**
- check_public_window: **PASS** (D-025 item 10)
- deadball_diagnostics (±40% plausibility, fused master): **PASS** (D-025 item 10)
- stability dead default/hidelo/hidehi (<5% label change, `tools/dead_stability.py`): **pending**
- gbdt_feature_shift (fused master, `--degrade fused`): **PASS** (D-025 item 10)
- hazard_feature_shift (`hazard_nodir`, tactical and fused masters): **PASS** (D-025 item 10)

## Skipped optional steps

- F3-08 dead GBDT (#83): still open (PR #152 not merged) — ladder continues without it.
- F2-08 semi-Markov durations (#126): still open — skipped.
- `team_tag_prior` (F1-05 item 8): 0 on IDSSE, so not part of the gain rule; its pre-registered BAR-ATH agreement check was not run, so it stays off.
