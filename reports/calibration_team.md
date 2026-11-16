# Team Logit Calibration Report (F1-05)

- variant written to the calibration file: `touch`

## Touch thresholds (leave-one-match-out log-loss of the `touch` ball model)

- grid: `r_touch_m` [1.5, 2.0, 3.0], `touch_min_frames` [2, 3, 5], `touch_acc_mps2` [5.0, 8.0, 12.0], `touch_dir_change_rad` [0.3, 0.5, 0.8]
- selected: `r_touch_m` = `1.5`, `touch_min_frames` = `5`, `touch_acc_mps2` = `5.0`, `touch_dir_change_rad` = `0.3`

| Rank | `r_touch_m` | `touch_min_frames` | `touch_acc_mps2` | `touch_dir_change_rad` | LOMO log-loss |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.5 | 5 | 5.0 | 0.3 | 0.46194 |
| 2 | 1.5 | 5 | 5.0 | 0.5 | 0.46194 |
| 3 | 1.5 | 5 | 5.0 | 0.8 | 0.46194 |
| 4 | 1.5 | 5 | 8.0 | 0.3 | 0.46194 |
| 5 | 1.5 | 5 | 8.0 | 0.5 | 0.46194 |

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

- mean `spell_f1_mean`: `0.3646 ± 0.0052`
- mean `spell_count_ratio`: `0.5149`
- mean `boundary_shift_frames`: `-5.84`

| Match | spell_f1_mean | Count Ratio (Pred/GT) | Boundary Shift (frames) |
|:---|---:|---:|---:|
| J03WMX | 0.3554 | 0.4912 | -9.99 |
| J03WN1 | 0.3440 | 0.4420 | -5.12 |
| J03WOH | 0.3686 | 0.5224 | -0.25 |
| J03WOY | 0.3670 | 0.5132 | -6.36 |
| J03WPY | 0.3607 | 0.5249 | -4.72 |
| J03WQQ | 0.3887 | 0.5835 | -9.48 |
| J03WR9 | 0.3681 | 0.5270 | -4.95 |

### Decoder Evaluation (`degrade=hidelo`)

- mean `spell_f1_mean`: `0.3342 ± 0.0063`
- mean `spell_count_ratio`: `0.5013`
- mean `boundary_shift_frames`: `-2.17`

| Match | spell_f1_mean | Count Ratio (Pred/GT) | Boundary Shift (frames) |
|:---|---:|---:|---:|
| J03WMX | 0.3407 | 0.4775 | -4.89 |
| J03WN1 | 0.3066 | 0.4040 | -2.91 |
| J03WOH | 0.3298 | 0.5299 | +12.27 |
| J03WOY | 0.3385 | 0.5110 | -4.03 |
| J03WPY | 0.3227 | 0.4988 | +3.75 |
| J03WQQ | 0.3592 | 0.5636 | -9.35 |
| J03WR9 | 0.3418 | 0.5243 | -10.04 |

### Decoder Evaluation (`degrade=hidehi`)

- mean `spell_f1_mean`: `0.4017 ± 0.0094`
- mean `spell_count_ratio`: `0.5457`
- mean `boundary_shift_frames`: `-7.92`

| Match | spell_f1_mean | Count Ratio (Pred/GT) | Boundary Shift (frames) |
|:---|---:|---:|---:|
| J03WMX | 0.3746 | 0.5108 | -12.23 |
| J03WN1 | 0.3833 | 0.4820 | -8.12 |
| J03WOH | 0.4179 | 0.5572 | -4.95 |
| J03WOY | 0.4119 | 0.5504 | -5.70 |
| J03WPY | 0.3829 | 0.5558 | -6.95 |
| J03WQQ | 0.4449 | 0.6234 | -7.60 |
| J03WR9 | 0.3965 | 0.5405 | -9.89 |

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

- mean `spell_f1_mean`: `0.3483 ± 0.0061`
- mean `spell_count_ratio`: `0.4261`
- mean `boundary_shift_frames`: `-2.41`

| Match | spell_f1_mean | Count Ratio (Pred/GT) | Boundary Shift (frames) |
|:---|---:|---:|---:|
| J03WMX | 0.3604 | 0.4051 | -10.28 |
| J03WN1 | 0.3202 | 0.3640 | +1.74 |
| J03WOH | 0.3410 | 0.4005 | +6.70 |
| J03WOY | 0.3522 | 0.4496 | -5.97 |
| J03WPY | 0.3714 | 0.4299 | +2.45 |
| J03WQQ | 0.3435 | 0.4663 | -6.14 |
| J03WR9 | 0.3492 | 0.4676 | -5.37 |

### Decoder Evaluation (`degrade=hidelo`)

- mean `spell_f1_mean`: `0.3188 ± 0.0074`
- mean `spell_count_ratio`: `0.4161`
- mean `boundary_shift_frames`: `+0.98`

| Match | spell_f1_mean | Count Ratio (Pred/GT) | Boundary Shift (frames) |
|:---|---:|---:|---:|
| J03WMX | 0.3335 | 0.4012 | -9.72 |
| J03WN1 | 0.2816 | 0.3380 | +4.68 |
| J03WOH | 0.3183 | 0.4005 | +18.44 |
| J03WOY | 0.3234 | 0.4539 | -2.46 |
| J03WPY | 0.3438 | 0.4204 | +8.32 |
| J03WQQ | 0.3106 | 0.4389 | -2.41 |
| J03WR9 | 0.3207 | 0.4595 | -10.01 |

### Decoder Evaluation (`degrade=hidehi`)

- mean `spell_f1_mean`: `0.3818 ± 0.0067`
- mean `spell_count_ratio`: `0.4429`
- mean `boundary_shift_frames`: `-3.54`

| Match | spell_f1_mean | Count Ratio (Pred/GT) | Boundary Shift (frames) |
|:---|---:|---:|---:|
| J03WMX | 0.3689 | 0.4090 | -12.62 |
| J03WN1 | 0.3523 | 0.3920 | -5.42 |
| J03WOH | 0.3790 | 0.4204 | +6.76 |
| J03WOY | 0.3952 | 0.4759 | -5.91 |
| J03WPY | 0.4026 | 0.4394 | -1.88 |
| J03WQQ | 0.3960 | 0.4963 | -2.11 |
| J03WR9 | 0.3786 | 0.4676 | -3.61 |

## Variant: `touch`

- ball `C`: `0.1`
- gap `C`: `0.01`
- negative signs among f1..f6: `['f1', 'f6']`

### Coefficients (`all`)

| Model | Feature | Scale | Weight |
|:---|:---|---:|---:|
| ball | `f1` | 0.8232 | -0.0459 |
| ball | `f2` | 0.6036 | +0.9709 |
| ball | `f3` | 0.7131 | +1.1189 |
| ball | `f4` | 0.5398 | +0.2524 |
| ball | `f5` | 0.8228 | +0.0080 |
| ball | `f6` | 0.8167 | -0.8892 |
| ball | `f7` | 0.4921 | -0.7070 |
| ball | `f8` | 0.7660 | +0.0370 |
| ball | `f9` | 0.5952 | +1.2914 |
| gap | `ev_coll_adv` | 3.5556 | +0.7624 |
| gap | `ev_coll_spread` | 89.3976 | +1.0442 |
| gap | `ev_coll_speed` | 0.5015 | -0.5238 |

### Out-of-fold Log-Loss and AUC

| Match | Log-Loss | AUC | Ball Log-Loss | Ball AUC | Gap Log-Loss | Gap AUC |
|:---|---:|---:|---:|---:|---:|---:|
| J03WMX | 0.4669 | 0.8523 | 0.4414 | 0.8668 | 0.5629 | 0.7914 |
| J03WN1 | 0.4951 | 0.8365 | 0.4815 | 0.8433 | 0.5669 | 0.7851 |
| J03WOH | 0.4893 | 0.8370 | 0.4857 | 0.8348 | 0.5042 | 0.8387 |
| J03WOY | 0.5007 | 0.8278 | 0.5020 | 0.8221 | 0.4946 | 0.8476 |
| J03WPY | 0.4595 | 0.8471 | 0.4546 | 0.8451 | 0.4824 | 0.8462 |
| J03WQQ | 0.4575 | 0.8666 | 0.4589 | 0.8579 | 0.4529 | 0.8809 |
| J03WR9 | 0.4261 | 0.8783 | 0.4093 | 0.8825 | 0.4923 | 0.8467 |

### Decoder Evaluation (`degrade=default`)

- mean `spell_f1_mean`: `0.3823 ± 0.0063`
- mean `spell_count_ratio`: `0.4882`
- mean `boundary_shift_frames`: `-1.55`

| Match | spell_f1_mean | Count Ratio (Pred/GT) | Boundary Shift (frames) |
|:---|---:|---:|---:|
| J03WMX | 0.3545 | 0.4442 | -10.00 |
| J03WN1 | 0.3638 | 0.4360 | +0.22 |
| J03WOH | 0.3846 | 0.4876 | +6.90 |
| J03WOY | 0.3901 | 0.5000 | -1.54 |
| J03WPY | 0.3994 | 0.4941 | -0.68 |
| J03WQQ | 0.3915 | 0.5287 | -6.26 |
| J03WR9 | 0.3922 | 0.5270 | +0.52 |

### Decoder Evaluation (`degrade=hidelo`)

- mean `spell_f1_mean`: `0.3521 ± 0.0066`
- mean `spell_count_ratio`: `0.4703`
- mean `boundary_shift_frames`: `+1.59`

| Match | spell_f1_mean | Count Ratio (Pred/GT) | Boundary Shift (frames) |
|:---|---:|---:|---:|
| J03WMX | 0.3493 | 0.4364 | -3.21 |
| J03WN1 | 0.3172 | 0.3920 | +2.59 |
| J03WOH | 0.3466 | 0.4900 | +13.73 |
| J03WOY | 0.3683 | 0.4934 | +1.39 |
| J03WPY | 0.3571 | 0.4632 | +4.22 |
| J03WQQ | 0.3589 | 0.5037 | -4.49 |
| J03WR9 | 0.3671 | 0.5135 | -3.13 |

### Decoder Evaluation (`degrade=hidehi`)

- mean `spell_f1_mean`: `0.4152 ± 0.0099`
- mean `spell_count_ratio`: `0.5195`
- mean `boundary_shift_frames`: `-4.22`

| Match | spell_f1_mean | Count Ratio (Pred/GT) | Boundary Shift (frames) |
|:---|---:|---:|---:|
| J03WMX | 0.3668 | 0.4599 | -12.86 |
| J03WN1 | 0.3978 | 0.4760 | -2.26 |
| J03WOH | 0.4356 | 0.5373 | -2.38 |
| J03WOY | 0.4318 | 0.5439 | -1.88 |
| J03WPY | 0.4208 | 0.5083 | -1.16 |
| J03WQQ | 0.4418 | 0.5761 | -4.35 |
| J03WR9 | 0.4120 | 0.5351 | -4.63 |

## Variant: `bridge_touch`

- ball `C`: `0.1`
- gap `C`: `0.01`
- negative signs among f1..f6: `['f2', 'f4', 'f6']`

### Coefficients (`all`)

| Model | Feature | Scale | Weight |
|:---|:---|---:|---:|
| ball | `f1` | 0.8232 | +0.3474 |
| ball | `f2` | 0.6036 | -0.2485 |
| ball | `f3` | 0.7131 | +0.5066 |
| ball | `f4` | 0.5398 | -0.2856 |
| ball | `f5` | 0.8228 | +0.2748 |
| ball | `f6` | 0.8167 | -0.7922 |
| ball | `f7` | 0.4921 | -0.8362 |
| ball | `f8` | 0.7660 | +0.1536 |
| ball | `f9` | 0.5952 | +1.0482 |
| ball | `f10_15` | 1.1769 | -0.2174 |
| ball | `f11_15` | 0.5673 | -1.1486 |
| ball | `f10_35` | 1.0426 | +2.6993 |
| ball | `f11_35` | 0.4972 | +1.3752 |
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
| J03WMX | 0.3493 | 0.9233 | 0.2949 | 0.9462 | 0.5552 | 0.8005 |
| J03WN1 | 0.4119 | 0.8950 | 0.3901 | 0.9067 | 0.5271 | 0.8193 |
| J03WOH | 0.3644 | 0.9172 | 0.3414 | 0.9277 | 0.4619 | 0.8672 |
| J03WOY | 0.3970 | 0.9013 | 0.3778 | 0.9112 | 0.4827 | 0.8560 |
| J03WPY | 0.3350 | 0.9253 | 0.3083 | 0.9366 | 0.4604 | 0.8641 |
| J03WQQ | 0.3419 | 0.9281 | 0.3155 | 0.9376 | 0.4269 | 0.8955 |
| J03WR9 | 0.3049 | 0.9392 | 0.2670 | 0.9521 | 0.4542 | 0.8734 |

### Decoder Evaluation (`degrade=default`)

- mean `spell_f1_mean`: `0.3603 ± 0.0098`
- mean `spell_count_ratio`: `0.4262`
- mean `boundary_shift_frames`: `-0.81`

| Match | spell_f1_mean | Count Ratio (Pred/GT) | Boundary Shift (frames) |
|:---|---:|---:|---:|
| J03WMX | 0.3618 | 0.3933 | -7.15 |
| J03WN1 | 0.3083 | 0.3520 | -0.60 |
| J03WOH | 0.3555 | 0.4080 | +8.30 |
| J03WOY | 0.3747 | 0.4539 | -3.70 |
| J03WPY | 0.3927 | 0.4347 | +6.10 |
| J03WQQ | 0.3628 | 0.4738 | -4.80 |
| J03WR9 | 0.3661 | 0.4676 | -3.80 |

### Decoder Evaluation (`degrade=hidelo`)

- mean `spell_f1_mean`: `0.3340 ± 0.0099`
- mean `spell_count_ratio`: `0.4158`
- mean `boundary_shift_frames`: `+2.19`

| Match | spell_f1_mean | Count Ratio (Pred/GT) | Boundary Shift (frames) |
|:---|---:|---:|---:|
| J03WMX | 0.3391 | 0.3875 | -6.49 |
| J03WN1 | 0.2820 | 0.3360 | +4.29 |
| J03WOH | 0.3333 | 0.4030 | +19.09 |
| J03WOY | 0.3526 | 0.4627 | -2.71 |
| J03WPY | 0.3658 | 0.4181 | +11.26 |
| J03WQQ | 0.3297 | 0.4464 | -2.17 |
| J03WR9 | 0.3354 | 0.4568 | -7.94 |

### Decoder Evaluation (`degrade=hidehi`)

- mean `spell_f1_mean`: `0.3932 ± 0.0110`
- mean `spell_count_ratio`: `0.4442`
- mean `boundary_shift_frames`: `-2.63`

| Match | spell_f1_mean | Count Ratio (Pred/GT) | Boundary Shift (frames) |
|:---|---:|---:|---:|
| J03WMX | 0.3681 | 0.3973 | -9.89 |
| J03WN1 | 0.3420 | 0.3800 | -6.08 |
| J03WOH | 0.3937 | 0.4303 | +7.70 |
| J03WOY | 0.4183 | 0.4868 | -4.50 |
| J03WPY | 0.4190 | 0.4466 | +1.89 |
| J03WQQ | 0.4172 | 0.5062 | -2.08 |
| J03WR9 | 0.3941 | 0.4622 | -5.44 |

