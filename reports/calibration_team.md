# Team Logit Calibration Report (F1-05)

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

### Decoder Evaluation (`degrade=hidelo`)

- mean `spell_f1_mean`: `0.3342 ± 0.0063`
- mean `spell_count_ratio`: `0.5013`
- mean `boundary_shift_frames`: `-2.17`

### Decoder Evaluation (`degrade=hidehi`)

- mean `spell_f1_mean`: `0.4017 ± 0.0094`
- mean `spell_count_ratio`: `0.5457`
- mean `boundary_shift_frames`: `-7.92`

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

### Decoder Evaluation (`degrade=hidelo`)

- mean `spell_f1_mean`: `0.3188 ± 0.0074`
- mean `spell_count_ratio`: `0.4161`
- mean `boundary_shift_frames`: `+0.98`

### Decoder Evaluation (`degrade=hidehi`)

- mean `spell_f1_mean`: `0.3818 ± 0.0067`
- mean `spell_count_ratio`: `0.4429`
- mean `boundary_shift_frames`: `-3.54`

## Variant: `touch`

- ball `C`: `0.1`
- gap `C`: `0.01`
- negative signs among f1..f6: `['f6']`

### Coefficients (`all`)

| Model | Feature | Scale | Weight |
|:---|:---|---:|---:|
| ball | `f1` | 0.8232 | +0.0322 |
| ball | `f2` | 0.6036 | +0.9293 |
| ball | `f3` | 0.7131 | +0.9080 |
| ball | `f4` | 0.5398 | +0.4470 |
| ball | `f5` | 0.8228 | +0.0877 |
| ball | `f6` | 0.8167 | -1.0835 |
| ball | `f7` | 0.4921 | -0.5165 |
| ball | `f8` | 0.7660 | -0.0810 |
| ball | `f9` | 0.6506 | +1.0754 |
| gap | `ev_coll_adv` | 3.5556 | +0.7624 |
| gap | `ev_coll_spread` | 89.3976 | +1.0442 |
| gap | `ev_coll_speed` | 0.5015 | -0.5238 |

### Out-of-fold Log-Loss and AUC

| Match | Log-Loss | AUC | Ball Log-Loss | Ball AUC | Gap Log-Loss | Gap AUC |
|:---|---:|---:|---:|---:|---:|---:|
| J03WMX | 0.4772 | 0.8476 | 0.4545 | 0.8614 | 0.5629 | 0.7914 |
| J03WN1 | 0.4996 | 0.8337 | 0.4868 | 0.8405 | 0.5669 | 0.7851 |
| J03WOH | 0.4966 | 0.8332 | 0.4948 | 0.8307 | 0.5042 | 0.8387 |
| J03WOY | 0.5116 | 0.8212 | 0.5154 | 0.8150 | 0.4946 | 0.8476 |
| J03WPY | 0.4692 | 0.8414 | 0.4664 | 0.8384 | 0.4824 | 0.8462 |
| J03WQQ | 0.4661 | 0.8614 | 0.4702 | 0.8520 | 0.4529 | 0.8809 |
| J03WR9 | 0.4336 | 0.8754 | 0.4187 | 0.8797 | 0.4923 | 0.8467 |

### Decoder Evaluation (`degrade=default`)

- mean `spell_f1_mean`: `0.3778 ± 0.0054`
- mean `spell_count_ratio`: `0.5050`
- mean `boundary_shift_frames`: `-4.20`

### Decoder Evaluation (`degrade=hidelo`)

- mean `spell_f1_mean`: `0.3460 ± 0.0059`
- mean `spell_count_ratio`: `0.4897`
- mean `boundary_shift_frames`: `-1.37`

### Decoder Evaluation (`degrade=hidehi`)

- mean `spell_f1_mean`: `0.4101 ± 0.0078`
- mean `spell_count_ratio`: `0.5363`
- mean `boundary_shift_frames`: `-6.13`

## Variant: `bridge_touch`

- ball `C`: `0.1`
- gap `C`: `0.01`
- negative signs among f1..f6: `['f2', 'f4', 'f6']`

### Coefficients (`all`)

| Model | Feature | Scale | Weight |
|:---|:---|---:|---:|
| ball | `f1` | 0.8232 | +0.4097 |
| ball | `f2` | 0.6036 | -0.2988 |
| ball | `f3` | 0.7131 | +0.1688 |
| ball | `f4` | 0.5398 | -0.1446 |
| ball | `f5` | 0.8228 | +0.3423 |
| ball | `f6` | 0.8167 | -0.8946 |
| ball | `f7` | 0.4921 | -0.7268 |
| ball | `f8` | 0.7660 | +0.0693 |
| ball | `f9` | 0.6506 | +0.9975 |
| ball | `f10_15` | 1.1769 | -0.0938 |
| ball | `f11_15` | 0.5673 | -1.1788 |
| ball | `f10_35` | 1.0426 | +2.6864 |
| ball | `f11_35` | 0.4972 | +1.3885 |
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
| J03WMX | 0.3529 | 0.9220 | 0.2994 | 0.9447 | 0.5552 | 0.8005 |
| J03WN1 | 0.4124 | 0.8945 | 0.3907 | 0.9062 | 0.5271 | 0.8193 |
| J03WOH | 0.3673 | 0.9158 | 0.3449 | 0.9261 | 0.4619 | 0.8672 |
| J03WOY | 0.4023 | 0.8988 | 0.3843 | 0.9084 | 0.4827 | 0.8560 |
| J03WPY | 0.3392 | 0.9234 | 0.3134 | 0.9346 | 0.4604 | 0.8641 |
| J03WQQ | 0.3448 | 0.9270 | 0.3193 | 0.9362 | 0.4269 | 0.8955 |
| J03WR9 | 0.3071 | 0.9386 | 0.2697 | 0.9515 | 0.4542 | 0.8734 |

### Decoder Evaluation (`degrade=default`)

- mean `spell_f1_mean`: `0.3571 ± 0.0079`
- mean `spell_count_ratio`: `0.4254`
- mean `boundary_shift_frames`: `-1.47`

### Decoder Evaluation (`degrade=hidelo`)

- mean `spell_f1_mean`: `0.3290 ± 0.0081`
- mean `spell_count_ratio`: `0.4137`
- mean `boundary_shift_frames`: `+1.16`

### Decoder Evaluation (`degrade=hidehi`)

- mean `spell_f1_mean`: `0.3908 ± 0.0084`
- mean `spell_count_ratio`: `0.4422`
- mean `boundary_shift_frames`: `-2.58`

