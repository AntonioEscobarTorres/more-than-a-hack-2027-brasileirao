# Team Logit Calibration Report (F1-05)

- variant written to the calibration file: `touch`

## Touch thresholds (leave-one-match-out log-loss of the `touch` ball model)

- abrupt-change condition required: `False`
- grid: `r_touch_m` [1.5, 2.0, 3.0], `touch_min_frames` [2, 3, 5], `touch_acc_mps2` [8.0], `touch_dir_change_rad` [0.5]
- selected: `r_touch_m` = `1.5`, `touch_min_frames` = `5`, `touch_acc_mps2` = `8.0`, `touch_dir_change_rad` = `0.5`

| Rank | `r_touch_m` | `touch_min_frames` | `touch_acc_mps2` | `touch_dir_change_rad` | LOMO log-loss |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.5 | 5 | 8.0 | 0.5 | 0.46198 |
| 2 | 1.5 | 3 | 8.0 | 0.5 | 0.46627 |
| 3 | 1.5 | 2 | 8.0 | 0.5 | 0.46734 |
| 4 | 2.0 | 5 | 8.0 | 0.5 | 0.46828 |
| 5 | 2.0 | 3 | 8.0 | 0.5 | 0.47253 |

## Variant: `base`

- ball `C`: `0.1`
- gap `C`: `0.01`
- negative signs among f1..f6: `['f1', 'f5', 'f6']`

### Coefficients (`all`)

| Model | Feature | Scale | Weight |
|:---|:---|---:|---:|
| ball | `f1` | 0.8232 | -0.1440 |
| ball | `f2` | 0.6036 | +0.9042 |
| ball | `f3` | 0.7131 | +1.9512 |
| ball | `f4` | 0.5398 | +0.7556 |
| ball | `f5` | 0.8228 | -0.0593 |
| ball | `f6` | 0.8167 | -1.3376 |
| ball | `f7` | 0.4921 | -0.3051 |
| ball | `f8` | 0.7660 | -0.1651 |
| gap | `ev_coll_adv` | 3.5556 | +0.7624 |
| gap | `ev_coll_spread` | 89.3976 | +1.0442 |
| gap | `ev_coll_speed` | 0.5015 | -0.5238 |

### Out-of-fold Log-Loss and AUC

| Match | Log-Loss | AUC | Ball Log-Loss | Ball AUC | Gap Log-Loss | Gap AUC |
|:---|---:|---:|---:|---:|---:|---:|
| J03WMX | 0.4871 | 0.8434 | 0.4670 | 0.8564 | 0.5629 | 0.7914 |
| J03WN1 | 0.5045 | 0.8313 | 0.4926 | 0.8389 | 0.5669 | 0.7851 |
| J03WOH | 0.5031 | 0.8295 | 0.5028 | 0.8263 | 0.5042 | 0.8387 |
| J03WOY | 0.5216 | 0.8158 | 0.5276 | 0.8095 | 0.4946 | 0.8476 |
| J03WPY | 0.4795 | 0.8351 | 0.4788 | 0.8319 | 0.4824 | 0.8462 |
| J03WQQ | 0.4751 | 0.8551 | 0.4819 | 0.8453 | 0.4529 | 0.8809 |
| J03WR9 | 0.4432 | 0.8700 | 0.4308 | 0.8734 | 0.4923 | 0.8467 |

### Decoder Evaluation (`degrade=default`)

- mean `spell_f1_mean`: `0.3828 ± 0.0103`
- mean `spell_count_ratio`: `0.5889`
- mean `boundary_shift_frames`: `-3.03`

| Match | spell_f1_mean | Count Ratio (Pred/GT) | Boundary Shift (frames) |
|:---|---:|---:|---:|
| J03WMX | 0.3756 | 0.5401 | -5.44 |
| J03WN1 | 0.3422 | 0.5080 | -1.90 |
| J03WOH | 0.3875 | 0.5871 | +0.91 |
| J03WOY | 0.3906 | 0.5921 | -9.23 |
| J03WPY | 0.3875 | 0.5914 | -0.68 |
| J03WQQ | 0.4310 | 0.6708 | -4.15 |
| J03WR9 | 0.3656 | 0.6324 | -0.70 |

### Decoder Evaluation (`degrade=hidelo`)

- mean `spell_f1_mean`: `0.3656 ± 0.0118`
- mean `spell_count_ratio`: `0.5928`
- mean `boundary_shift_frames`: `+0.69`

| Match | spell_f1_mean | Count Ratio (Pred/GT) | Boundary Shift (frames) |
|:---|---:|---:|---:|
| J03WMX | 0.3673 | 0.5577 | -2.09 |
| J03WN1 | 0.3099 | 0.4820 | -0.03 |
| J03WOH | 0.3652 | 0.6020 | +6.76 |
| J03WOY | 0.3753 | 0.5987 | -4.03 |
| J03WPY | 0.3629 | 0.5867 | +5.96 |
| J03WQQ | 0.4168 | 0.6658 | -1.11 |
| J03WR9 | 0.3622 | 0.6568 | -0.61 |

### Decoder Evaluation (`degrade=hidehi`)

- mean `spell_f1_mean`: `0.4033 ± 0.0103`
- mean `spell_count_ratio`: `0.5868`
- mean `boundary_shift_frames`: `-6.09`

| Match | spell_f1_mean | Count Ratio (Pred/GT) | Boundary Shift (frames) |
|:---|---:|---:|---:|
| J03WMX | 0.3857 | 0.5342 | -7.15 |
| J03WN1 | 0.3675 | 0.5020 | -6.36 |
| J03WOH | 0.4112 | 0.5945 | -4.44 |
| J03WOY | 0.4082 | 0.6009 | -10.27 |
| J03WPY | 0.4030 | 0.5914 | -3.74 |
| J03WQQ | 0.4550 | 0.6683 | -4.91 |
| J03WR9 | 0.3926 | 0.6162 | -5.78 |

## Variant: `bridge`

- ball `C`: `0.1`
- gap `C`: `0.01`
- negative signs among f1..f6: `['f2', 'f6']`

### Coefficients (`all`)

| Model | Feature | Scale | Weight |
|:---|:---|---:|---:|
| ball | `f1` | 0.8232 | +0.2562 |
| ball | `f2` | 0.6036 | -0.3334 |
| ball | `f3` | 0.7131 | +1.2518 |
| ball | `f4` | 0.5398 | +0.0656 |
| ball | `f5` | 0.8228 | +0.2152 |
| ball | `f6` | 0.8167 | -1.1338 |
| ball | `f7` | 0.4921 | -0.5398 |
| ball | `f8` | 0.7660 | -0.0080 |
| ball | `f10_15` | 1.1769 | -0.3183 |
| ball | `f11_15` | 0.5673 | -1.0778 |
| ball | `f10_35` | 1.0426 | +2.7981 |
| ball | `f11_35` | 0.4972 | +1.3762 |
| gap | `ev_coll_adv` | 3.5556 | +0.7550 |
| gap | `ev_coll_spread` | 89.3976 | +0.9531 |
| gap | `ev_coll_speed` | 0.5015 | -0.5448 |
| gap | `f10_15` | 0.1372 | +0.0319 |
| gap | `f11_15` | 0.0319 | +0.0193 |
| gap | `f10_35` | 0.1919 | +0.6587 |
| gap | `f11_35` | 0.0588 | +0.1016 |

### Out-of-fold Log-Loss and AUC

| Match | Log-Loss | AUC | Ball Log-Loss | Ball AUC | Gap Log-Loss | Gap AUC |
|:---|---:|---:|---:|---:|---:|---:|
| J03WMX | 0.3585 | 0.9196 | 0.3065 | 0.9423 | 0.5552 | 0.8005 |
| J03WN1 | 0.4164 | 0.8921 | 0.3953 | 0.9036 | 0.5271 | 0.8193 |
| J03WOH | 0.3720 | 0.9135 | 0.3507 | 0.9237 | 0.4619 | 0.8672 |
| J03WOY | 0.4097 | 0.8950 | 0.3933 | 0.9040 | 0.4827 | 0.8560 |
| J03WPY | 0.3444 | 0.9208 | 0.3197 | 0.9321 | 0.4604 | 0.8641 |
| J03WQQ | 0.3507 | 0.9245 | 0.3270 | 0.9331 | 0.4269 | 0.8955 |
| J03WR9 | 0.3124 | 0.9365 | 0.2763 | 0.9493 | 0.4542 | 0.8734 |

### Decoder Evaluation (`degrade=default`)

- mean `spell_f1_mean`: `0.3654 ± 0.0071`
- mean `spell_count_ratio`: `0.5045`
- mean `boundary_shift_frames`: `+0.29`

| Match | spell_f1_mean | Count Ratio (Pred/GT) | Boundary Shift (frames) |
|:---|---:|---:|---:|
| J03WMX | 0.3699 | 0.4560 | -6.18 |
| J03WN1 | 0.3335 | 0.4200 | -0.48 |
| J03WOH | 0.3644 | 0.4801 | +5.82 |
| J03WOY | 0.3675 | 0.5373 | -7.70 |
| J03WPY | 0.3714 | 0.4964 | +8.79 |
| J03WQQ | 0.3962 | 0.5661 | -0.42 |
| J03WR9 | 0.3547 | 0.5757 | +2.20 |

### Decoder Evaluation (`degrade=hidelo`)

- mean `spell_f1_mean`: `0.3508 ± 0.0081`
- mean `spell_count_ratio`: `0.5122`
- mean `boundary_shift_frames`: `+3.92`

| Match | spell_f1_mean | Count Ratio (Pred/GT) | Boundary Shift (frames) |
|:---|---:|---:|---:|
| J03WMX | 0.3564 | 0.4716 | -4.95 |
| J03WN1 | 0.3070 | 0.4200 | +3.29 |
| J03WOH | 0.3510 | 0.4826 | +10.68 |
| J03WOY | 0.3542 | 0.5526 | -1.79 |
| J03WPY | 0.3553 | 0.5107 | +11.45 |
| J03WQQ | 0.3783 | 0.5611 | +5.22 |
| J03WR9 | 0.3530 | 0.5865 | +3.52 |

### Decoder Evaluation (`degrade=hidehi`)

- mean `spell_f1_mean`: `0.3791 ± 0.0073`
- mean `spell_count_ratio`: `0.4889`
- mean `boundary_shift_frames`: `-1.73`

| Match | spell_f1_mean | Count Ratio (Pred/GT) | Boundary Shift (frames) |
|:---|---:|---:|---:|
| J03WMX | 0.3668 | 0.4364 | -6.21 |
| J03WN1 | 0.3546 | 0.4100 | -5.39 |
| J03WOH | 0.3715 | 0.4652 | +3.90 |
| J03WOY | 0.3943 | 0.5417 | -9.79 |
| J03WPY | 0.3872 | 0.4798 | +5.46 |
| J03WQQ | 0.4110 | 0.5461 | +2.27 |
| J03WR9 | 0.3685 | 0.5432 | -2.37 |

## Variant: `touch`

- ball `C`: `0.1`
- gap `C`: `0.01`
- negative signs among f1..f6: `['f1', 'f6']`

### Coefficients (`all`)

| Model | Feature | Scale | Weight |
|:---|:---|---:|---:|
| ball | `f1` | 0.8232 | -0.0492 |
| ball | `f2` | 0.6036 | +0.9716 |
| ball | `f3` | 0.7131 | +1.1196 |
| ball | `f4` | 0.5398 | +0.2540 |
| ball | `f5` | 0.8228 | +0.0092 |
| ball | `f6` | 0.8167 | -0.8897 |
| ball | `f7` | 0.4921 | -0.7063 |
| ball | `f8` | 0.7660 | +0.0368 |
| ball | `f9` | 0.5953 | +1.2895 |
| gap | `ev_coll_adv` | 3.5556 | +0.7624 |
| gap | `ev_coll_spread` | 89.3976 | +1.0442 |
| gap | `ev_coll_speed` | 0.5015 | -0.5238 |

### Out-of-fold Log-Loss and AUC

| Match | Log-Loss | AUC | Ball Log-Loss | Ball AUC | Gap Log-Loss | Gap AUC |
|:---|---:|---:|---:|---:|---:|---:|
| J03WMX | 0.4669 | 0.8523 | 0.4414 | 0.8668 | 0.5629 | 0.7914 |
| J03WN1 | 0.4949 | 0.8367 | 0.4812 | 0.8435 | 0.5669 | 0.7851 |
| J03WOH | 0.4892 | 0.8370 | 0.4857 | 0.8348 | 0.5042 | 0.8387 |
| J03WOY | 0.5007 | 0.8278 | 0.5020 | 0.8221 | 0.4946 | 0.8476 |
| J03WPY | 0.4595 | 0.8471 | 0.4546 | 0.8451 | 0.4824 | 0.8462 |
| J03WQQ | 0.4576 | 0.8666 | 0.4591 | 0.8579 | 0.4529 | 0.8809 |
| J03WR9 | 0.4265 | 0.8781 | 0.4098 | 0.8822 | 0.4923 | 0.8467 |

### Decoder Evaluation (`degrade=default`)

- mean `spell_f1_mean`: `0.3966 ± 0.0080`
- mean `spell_count_ratio`: `0.5618`
- mean `boundary_shift_frames`: `+0.50`

| Match | spell_f1_mean | Count Ratio (Pred/GT) | Boundary Shift (frames) |
|:---|---:|---:|---:|
| J03WMX | 0.3716 | 0.4892 | -3.41 |
| J03WN1 | 0.3672 | 0.4880 | +0.52 |
| J03WOH | 0.3994 | 0.5547 | +0.28 |
| J03WOY | 0.4133 | 0.5811 | -3.07 |
| J03WPY | 0.4061 | 0.5511 | +4.18 |
| J03WQQ | 0.4250 | 0.6334 | -1.25 |
| J03WR9 | 0.3934 | 0.6351 | +6.23 |

### Decoder Evaluation (`degrade=hidelo`)

- mean `spell_f1_mean`: `0.3801 ± 0.0094`
- mean `spell_count_ratio`: `0.5667`
- mean `boundary_shift_frames`: `+3.88`

| Match | spell_f1_mean | Count Ratio (Pred/GT) | Boundary Shift (frames) |
|:---|---:|---:|---:|
| J03WMX | 0.3646 | 0.5049 | +0.54 |
| J03WN1 | 0.3331 | 0.4700 | +2.32 |
| J03WOH | 0.3804 | 0.5771 | +4.53 |
| J03WOY | 0.4000 | 0.5833 | +1.94 |
| J03WPY | 0.3858 | 0.5439 | +8.81 |
| J03WQQ | 0.4067 | 0.6284 | +2.62 |
| J03WR9 | 0.3902 | 0.6595 | +6.37 |

### Decoder Evaluation (`degrade=hidehi`)

- mean `spell_f1_mean`: `0.4140 ± 0.0090`
- mean `spell_count_ratio`: `0.5613`
- mean `boundary_shift_frames`: `-2.50`

| Match | spell_f1_mean | Count Ratio (Pred/GT) | Boundary Shift (frames) |
|:---|---:|---:|---:|
| J03WMX | 0.3752 | 0.4853 | -5.71 |
| J03WN1 | 0.3957 | 0.4900 | -2.29 |
| J03WOH | 0.4247 | 0.5721 | -5.39 |
| J03WOY | 0.4329 | 0.5987 | -4.16 |
| J03WPY | 0.4228 | 0.5439 | +1.88 |
| J03WQQ | 0.4440 | 0.6309 | -1.99 |
| J03WR9 | 0.4027 | 0.6081 | +0.15 |

## Variant: `bridge_touch`

- ball `C`: `0.1`
- gap `C`: `0.01`
- negative signs among f1..f6: `['f2', 'f4', 'f6']`

### Coefficients (`all`)

| Model | Feature | Scale | Weight |
|:---|:---|---:|---:|
| ball | `f1` | 0.8232 | +0.3462 |
| ball | `f2` | 0.6036 | -0.2483 |
| ball | `f3` | 0.7131 | +0.5052 |
| ball | `f4` | 0.5398 | -0.2866 |
| ball | `f5` | 0.8228 | +0.2777 |
| ball | `f6` | 0.8167 | -0.7928 |
| ball | `f7` | 0.4921 | -0.8377 |
| ball | `f8` | 0.7660 | +0.1544 |
| ball | `f9` | 0.5953 | +1.0494 |
| ball | `f10_15` | 1.1769 | -0.2180 |
| ball | `f11_15` | 0.5673 | -1.1483 |
| ball | `f10_35` | 1.0426 | +2.7003 |
| ball | `f11_35` | 0.4972 | +1.3754 |
| gap | `ev_coll_adv` | 3.5556 | +0.7550 |
| gap | `ev_coll_spread` | 89.3976 | +0.9531 |
| gap | `ev_coll_speed` | 0.5015 | -0.5448 |
| gap | `f10_15` | 0.1372 | +0.0319 |
| gap | `f11_15` | 0.0319 | +0.0193 |
| gap | `f10_35` | 0.1919 | +0.6587 |
| gap | `f11_35` | 0.0588 | +0.1016 |

### Out-of-fold Log-Loss and AUC

| Match | Log-Loss | AUC | Ball Log-Loss | Ball AUC | Gap Log-Loss | Gap AUC |
|:---|---:|---:|---:|---:|---:|---:|
| J03WMX | 0.3494 | 0.9233 | 0.2949 | 0.9462 | 0.5552 | 0.8005 |
| J03WN1 | 0.4116 | 0.8952 | 0.3897 | 0.9068 | 0.5271 | 0.8193 |
| J03WOH | 0.3644 | 0.9172 | 0.3413 | 0.9277 | 0.4619 | 0.8672 |
| J03WOY | 0.3970 | 0.9013 | 0.3778 | 0.9112 | 0.4827 | 0.8560 |
| J03WPY | 0.3351 | 0.9253 | 0.3083 | 0.9366 | 0.4604 | 0.8641 |
| J03WQQ | 0.3419 | 0.9281 | 0.3155 | 0.9376 | 0.4269 | 0.8955 |
| J03WR9 | 0.3051 | 0.9391 | 0.2672 | 0.9520 | 0.4542 | 0.8734 |

### Decoder Evaluation (`degrade=default`)

- mean `spell_f1_mean`: `0.3754 ± 0.0108`
- mean `spell_count_ratio`: `0.4997`
- mean `boundary_shift_frames`: `+1.06`

| Match | spell_f1_mean | Count Ratio (Pred/GT) | Boundary Shift (frames) |
|:---|---:|---:|---:|
| J03WMX | 0.3680 | 0.4423 | -2.83 |
| J03WN1 | 0.3224 | 0.4120 | -1.95 |
| J03WOH | 0.3770 | 0.4701 | +3.80 |
| J03WOY | 0.3897 | 0.5373 | -5.66 |
| J03WPY | 0.3911 | 0.4893 | +11.62 |
| J03WQQ | 0.4140 | 0.5661 | +2.03 |
| J03WR9 | 0.3658 | 0.5811 | +0.40 |

### Decoder Evaluation (`degrade=hidelo`)

- mean `spell_f1_mean`: `0.3648 ± 0.0103`
- mean `spell_count_ratio`: `0.5088`
- mean `boundary_shift_frames`: `+4.22`

| Match | spell_f1_mean | Count Ratio (Pred/GT) | Boundary Shift (frames) |
|:---|---:|---:|---:|
| J03WMX | 0.3610 | 0.4638 | -2.82 |
| J03WN1 | 0.3113 | 0.4160 | +2.56 |
| J03WOH | 0.3642 | 0.4751 | +7.71 |
| J03WOY | 0.3757 | 0.5526 | -0.36 |
| J03WPY | 0.3791 | 0.5036 | +14.53 |
| J03WQQ | 0.4000 | 0.5611 | +5.96 |
| J03WR9 | 0.3626 | 0.5892 | +1.97 |

### Decoder Evaluation (`degrade=hidehi`)

- mean `spell_f1_mean`: `0.3897 ± 0.0106`
- mean `spell_count_ratio`: `0.4879`
- mean `boundary_shift_frames`: `-1.03`

| Match | spell_f1_mean | Count Ratio (Pred/GT) | Boundary Shift (frames) |
|:---|---:|---:|---:|
| J03WMX | 0.3693 | 0.4286 | -4.20 |
| J03WN1 | 0.3470 | 0.4040 | -7.04 |
| J03WOH | 0.3871 | 0.4652 | +3.91 |
| J03WOY | 0.4153 | 0.5482 | -8.49 |
| J03WPY | 0.4058 | 0.4751 | +9.11 |
| J03WQQ | 0.4270 | 0.5511 | +3.53 |
| J03WR9 | 0.3762 | 0.5432 | -4.01 |

