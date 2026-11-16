# Decoder sensitivity sweep (F2-05)

- base config: `configs/v1.yaml`
- decoder overlay: `configs/decoder_idsse.yaml`
- degrade (primary): `default`
- robust: True
- masters: `$MTAH_EXT_DIR` after F4-04 (`default` / `hidelo` / `hidehi`); labels SHA match `--verify`; parquet masters may differ by last bits across CPUs (documented in `docs/external_data.md`)

## Recommendation

**`configs/v1s.yaml`** keeps `rules_v1` evidence and the IDSSE decoder priors, with:

| knob | value | reason |
| :--- | :--- | :--- |
| `lam_team`, `lam_dead` | 0.4, 0.4 | best worst-case spellF1 on `{default, hidelo, hidehi}` among the λ plateau (default 0.3590±0.0074; worst case hidelo 0.3240) |
| `min_dwell` H/A, D | 25, 50 | plateau + robust pick on the λ-fixed grid (default 0.3601±0.0074; worst case 0.3253) |
| `mean_duration_s` scale | 1.0 | scale 1.0 beats 0.5 and 2.0 on the robust criterion |
| post-process | all off | stage 4 is informational only; isolated midpoint is +0.0005 and stays for F2-06 |
| `boundary_shift_file` | null in v1s | step 5 shows off→on 0.3601→0.3721 on `default`; the shift file is a Seção 6 candidate, not baked into v1s |

Spell ratio (pred/gt possessions) at the chosen decoder: **0.624** on `default`.

## Stage: lam

- selected: `lam|lam_dead=0.4|lam_team=0.4`
- spellF1 mean±SE: 0.3590±0.0074
- spell ratio (pred/gt): 0.6198
- worst-case mean across default,hidelo,hidehi: 0.3240
  - default: 0.3590±0.0074 (ratio 0.6198)
  - hidelo: 0.3240±0.0047 (ratio 0.6032)
  - hidehi: 0.4006±0.0105 (ratio 0.6441)

| id | stage | lam_team | lam_dead | mean | se | spell_ratio |
|:---|---:|---:|---:|---:|---:|---:|
| lam|lam_dead=0.05|lam_team=0.05 | lam | 0.0500 | 0.0500 | 0.2311 | 0.0127 | 0.2441 |
| lam|lam_dead=0.1|lam_team=0.05 | lam | 0.0500 | 0.1000 | 0.2503 | 0.0137 | 0.2717 |
| lam|lam_dead=0.2|lam_team=0.05 | lam | 0.0500 | 0.2000 | 0.2624 | 0.0122 | 0.3003 |
| lam|lam_dead=0.4|lam_team=0.05 | lam | 0.0500 | 0.4000 | 0.2651 | 0.0112 | 0.3260 |
| lam|lam_dead=0.8|lam_team=0.05 | lam | 0.0500 | 0.8000 | 0.2593 | 0.0106 | 0.3673 |
| lam|lam_dead=0.05|lam_team=0.1 | lam | 0.1000 | 0.0500 | 0.2743 | 0.0112 | 0.3089 |
| lam|lam_dead=0.1|lam_team=0.1 | lam | 0.1000 | 0.1000 | 0.2936 | 0.0113 | 0.3352 |
| lam|lam_dead=0.2|lam_team=0.1 | lam | 0.1000 | 0.2000 | 0.3034 | 0.0119 | 0.3602 |
| lam|lam_dead=0.4|lam_team=0.1 | lam | 0.1000 | 0.4000 | 0.3061 | 0.0106 | 0.3860 |
| lam|lam_dead=0.8|lam_team=0.1 | lam | 0.1000 | 0.8000 | 0.2964 | 0.0103 | 0.4268 |
| lam|lam_dead=0.05|lam_team=0.2 | lam | 0.2000 | 0.0500 | 0.3172 | 0.0095 | 0.4200 |
| lam|lam_dead=0.1|lam_team=0.2 | lam | 0.2000 | 0.1000 | 0.3353 | 0.0084 | 0.4405 |
| lam|lam_dead=0.2|lam_team=0.2 | lam | 0.2000 | 0.2000 | 0.3429 | 0.0103 | 0.4577 |
| lam|lam_dead=0.4|lam_team=0.2 | lam | 0.2000 | 0.4000 | 0.3463 | 0.0108 | 0.4818 |
| lam|lam_dead=0.8|lam_team=0.2 | lam | 0.2000 | 0.8000 | 0.3377 | 0.0087 | 0.5179 |
| lam|lam_dead=0.05|lam_team=0.4 | lam | 0.4000 | 0.0500 | 0.3308 | 0.0072 | 0.5754 |
| lam|lam_dead=0.1|lam_team=0.4 | lam | 0.4000 | 0.1000 | 0.3485 | 0.0070 | 0.5897 |
| lam|lam_dead=0.2|lam_team=0.4 | lam | 0.4000 | 0.2000 | 0.3563 | 0.0072 | 0.6013 |
| lam|lam_dead=0.4|lam_team=0.4 | lam | 0.4000 | 0.4000 | 0.3590 | 0.0074 | 0.6198 |
| lam|lam_dead=0.8|lam_team=0.4 | lam | 0.4000 | 0.8000 | 0.3550 | 0.0058 | 0.6421 |
| lam|lam_dead=0.05|lam_team=0.8 | lam | 0.8000 | 0.0500 | 0.3183 | 0.0077 | 0.8248 |
| lam|lam_dead=0.1|lam_team=0.8 | lam | 0.8000 | 0.1000 | 0.3292 | 0.0077 | 0.8366 |
| lam|lam_dead=0.2|lam_team=0.8 | lam | 0.8000 | 0.2000 | 0.3362 | 0.0074 | 0.8489 |
| lam|lam_dead=0.4|lam_team=0.8 | lam | 0.8000 | 0.4000 | 0.3400 | 0.0071 | 0.8513 |
| lam|lam_dead=0.8|lam_team=0.8 | lam | 0.8000 | 0.8000 | 0.3413 | 0.0075 | 0.8553 |

## Stage: min_dwell

- selected: `dwell|min_dwell_d=50|min_dwell_ha=25`
- spellF1 mean±SE: 0.3601±0.0074
- spell ratio (pred/gt): 0.6240
- worst-case mean across default,hidelo,hidehi: 0.3253
  - default: 0.3601±0.0074 (ratio 0.6240)
  - hidelo: 0.3253±0.0044 (ratio 0.6076)
  - hidehi: 0.4026±0.0104 (ratio 0.6488)

| id | stage | lam_team | lam_dead | min_dwell_ha | min_dwell_d | mean | se | spell_ratio |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| dwell|min_dwell_d=1|min_dwell_ha=1 | min_dwell | 0.4000 | 0.4000 | 1 | 1 | 0.3595 | 0.0074 | 0.6162 |
| dwell|min_dwell_d=25|min_dwell_ha=1 | min_dwell | 0.4000 | 0.4000 | 1 | 25 | 0.3598 | 0.0075 | 0.6165 |
| dwell|min_dwell_d=50|min_dwell_ha=1 | min_dwell | 0.4000 | 0.4000 | 1 | 50 | 0.3603 | 0.0075 | 0.6142 |
| dwell|min_dwell_d=1|min_dwell_ha=12 | min_dwell | 0.4000 | 0.4000 | 12 | 1 | 0.3588 | 0.0074 | 0.6195 |
| dwell|min_dwell_d=25|min_dwell_ha=12 | min_dwell | 0.4000 | 0.4000 | 12 | 25 | 0.3590 | 0.0074 | 0.6198 |
| dwell|min_dwell_d=50|min_dwell_ha=12 | min_dwell | 0.4000 | 0.4000 | 12 | 50 | 0.3596 | 0.0074 | 0.6175 |
| dwell|min_dwell_d=1|min_dwell_ha=25 | min_dwell | 0.4000 | 0.4000 | 25 | 1 | 0.3600 | 0.0074 | 0.6259 |
| dwell|min_dwell_d=25|min_dwell_ha=25 | min_dwell | 0.4000 | 0.4000 | 25 | 25 | 0.3595 | 0.0074 | 0.6263 |
| dwell|min_dwell_d=50|min_dwell_ha=25 | min_dwell | 0.4000 | 0.4000 | 25 | 50 | 0.3601 | 0.0074 | 0.6240 |

## Stage: duration_scale

- selected: `scale|duration_scale=1.0`
- spellF1 mean±SE: 0.3601±0.0074
- spell ratio (pred/gt): 0.6240
- worst-case mean across default,hidelo,hidehi: 0.3253
  - default: 0.3601±0.0074 (ratio 0.6240)
  - hidelo: 0.3253±0.0044 (ratio 0.6076)
  - hidehi: 0.4026±0.0104 (ratio 0.6488)

| id | stage | duration_scale | lam_team | lam_dead | mean | se | spell_ratio |
|:---|---:|---:|---:|---:|---:|---:|---:|
| scale|duration_scale=0.5 | duration_scale | 0.5000 | 0.4000 | 0.4000 | 0.3575 | 0.0068 | 0.6851 |
| scale|duration_scale=1.0 | duration_scale | 1.0000 | 0.4000 | 0.4000 | 0.3601 | 0.0074 | 0.6240 |
| scale|duration_scale=2.0 | duration_scale | 2.0000 | 0.4000 | 0.4000 | 0.3580 | 0.0080 | 0.5889 |

## Stage: postprocess

- best isolated (informational, not applied to v1s; see F2-06): `post|midpoint`
- spellF1 mean±SE: 0.3606±0.0073
- spell ratio (pred/gt): 0.6240

| id | stage | lam_team | lam_dead | mean | se | spell_ratio |
|:---|---:|---:|---:|---:|---:|---:|
| post|off | postprocess | 0.4000 | 0.4000 | 0.3601 | 0.0074 | 0.6240 |
| post|midpoint | postprocess | 0.4000 | 0.4000 | 0.3606 | 0.0073 | 0.6240 |
| post|d_bridge=25 | postprocess | 0.4000 | 0.4000 | 0.3601 | 0.0074 | 0.6240 |
| post|d_bridge=50 | postprocess | 0.4000 | 0.4000 | 0.3601 | 0.0074 | 0.6240 |
| post|min_spell=13 | postprocess | 0.4000 | 0.4000 | 0.3601 | 0.0074 | 0.6240 |
| post|min_spell=25 | postprocess | 0.4000 | 0.4000 | 0.3601 | 0.0074 | 0.6240 |
| post|d_cross=25 | postprocess | 0.4000 | 0.4000 | 0.3601 | 0.0074 | 0.6240 |
| post|d_cross=50 | postprocess | 0.4000 | 0.4000 | 0.3601 | 0.0074 | 0.6240 |
| post|d_cross=75 | postprocess | 0.4000 | 0.4000 | 0.3598 | 0.0074 | 0.6240 |

## Step 5: boundary shift (with / without `boundary_shift_file`)

- off: spell_f1_mean 0.3601±0.0074
- on: spell_f1_mean 0.3721±0.0089

### off

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

### on

| match | spell_f1_mean | macro_f1 | alive_dead_f1 | trans_f1_25 | mean_abs_latency_25 | n_gt_spells | n_pred_spells |
|:---|---:|---:|---:|---:|---:|---:|---:|
| J03WMX | 0.3366 | 0.7655 | 0.8694 | 0.3430 | 7.5380 | 511.0000 | 271.0000 |
| J03WN1 | 0.3958 | 0.7910 | 0.8950 | 0.3768 | 6.6378 | 500.0000 | 268.0000 |
| J03WOH | 0.3941 | 0.7832 | 0.8831 | 0.3562 | 6.1731 | 402.0000 | 271.0000 |
| J03WOY | 0.3436 | 0.7523 | 0.8719 | 0.3565 | 7.2130 | 456.0000 | 275.0000 |
| J03WPY | 0.3723 | 0.7690 | 0.8832 | 0.3704 | 6.4667 | 421.0000 | 271.0000 |
| J03WQQ | 0.3874 | 0.7439 | 0.8472 | 0.3392 | 7.9281 | 401.0000 | 298.0000 |
| J03WR9 | 0.3749 | 0.7722 | 0.8725 | 0.3627 | 5.8288 | 370.0000 | 236.0000 |
| mean | 0.3721 | 0.7682 | 0.8746 | 0.3578 | 6.8265 | 437.2857 | 270.0000 |
| se | 0.0089 | 0.0062 | 0.0057 | 0.0051 | 0.2869 | 20.1609 | 6.8522 |

