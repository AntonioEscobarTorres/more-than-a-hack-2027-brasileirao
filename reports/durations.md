# Semi-Markov durations: `decoder.duration_stages` (F2-08)

Decision: [D-027](../docs/decisions.md). Task: [`docs/tasks/F2-08.md`](../docs/tasks/F2-08.md).

Everything here comes from the public IDSSE matches: the calibration uses their true labels, and the
selection step scores the degraded IDSSE masters (`holdout = match`). BAR-ATH was never read.

## Summary

- **H and A: one stage (geometric) in every base config and every fold.** Possession lengths on
  IDSSE are 2–3 times *more* spread than a geometric with the same mean, and their mode (1–2 s) sits
  right after `min_dwell`. With the mean held fixed, more stages only shrink the spread, so the
  likelihood falls by 400–600 nats already at `r = 2`. The premise of the task (a geometric mode at
  `min_dwell` is unrealistic) does not hold for possessions.
- **D: one stage for `v1s` (`min_dwell` 50), two for `v2_gbdt` (`min_dwell` 25)**, the same in
  all 7 leave-one-match-out folds. Dead-ball lengths have a mode at 8–12 s, away from the minimum.
- **`v1s` (the base named in the specification):** the calibrated stages are all 1, so the step is
  the identity: the decoder is unchanged bit for bit and there is nothing to evaluate.
- **`v2_gbdt`:** two D stages at the committed λ (0.3, 0.2) *lose* (−0.0029 ± 0.0009). With λ re-swept
  (F2-05 stage 1) they pick λ (0.4, 0.4) and gain +0.0034 ± 0.0024 over `v2_gbdt`, with every stress
  mode positive: **accepted by the F2-06 rule**, as `configs/v2_gbdt_nb.yaml`.
- **Control:** re-sweeping λ the same way with one stage picks the same λ (0.4, 0.4) and gains
  nothing (−0.0009 ± 0.0025). At that λ the two D stages add +0.0043 ± 0.0011 (`default`), positive
  in all five modes. The gain is the stages', not the λ re-sweep's.
- **Caveat:** the stage effect depends on `lam_dead` (λ grid below). With `lam_dead ≥ 0.4` the
  stages mostly help (heavier dead evidence creates short spurious D spells that split possessions,
  and the stages make them costly); with `lam_dead ≤ 0.2` they mostly hurt. The gain is modest
  (1.4 EP against `v2_gbdt`), and λ (0.4, 0.4) is the best of 25 points of the stage arm, so it is
  slightly optimistic; the control went through the same selection.
- Not evaluated: the combination with the hazard of F2-07 (D-025, `v2_gbdt_hz`). The code supports
  it (the hazard modulates the exits of the last stage; tested against a dense reference), but the
  stages and the hazard were each chosen on `v2_gbdt` alone; combining them is a step of F2-06.

Commands:

```bash
python -m modeling.calibration.durations --config configs/v1s.yaml \
    --out configs/calib/durations.yaml --report <section.md> --histogram
python -m modeling.calibration.durations --config configs/v2_gbdt.yaml \
    --out configs/calib/durations_v2_gbdt.yaml --report <section.md>
python -m modeling.scoring.durations_sweep --config configs/v1s.yaml \
    --calib configs/calib/durations.yaml --report <section.md>
python -m modeling.scoring.durations_sweep --config configs/v2_gbdt.yaml \
    --calib configs/calib/durations_v2_gbdt.yaml --report <section.md> --extra-d 3 4
```

The second sweep takes about an hour (≈ 80 evaluations of the 7 matches).

## 1. Calibration of `duration_stages` on `configs/v1s.yaml`

True IDSSE segment lengths (7 matches, per half, X never forms a segment). Mean and `min_dwell` from the config; every `r` is scored on the segments with `L ≥ m − 1 + 8` and the likelihood truncated there.

| State | m (frames) | mean (frames) | segments | kept (L ≥ floor) | median L | Var Y / Var geometric | r (all) |
|---|---:|---:|---:|---:|---:|---:|---:|
| H | 25 | 191.0 | 1539 | 1276 | 90 | 2.99 | 1 |
| A | 25 | 170.3 | 1522 | 1270 | 93 | 2.44 | 1 |
| D | 50 | 630.5 | 713 | 702 | 491 | 0.91 | 1 |

Log-likelihood by `r` (all matches; Δ against the best):

| State | r = 1 | r = 2 | r = 3 | r = 4 | r = 6 | r = 8 |
|---|---:|---:|---:|---:|---:|---:|
| H | -8023.1 (+0.0) | -8596.9 (-573.8) | -9365.1 (-1342.0) | -10240.1 (-2217.0) | -12197.6 (-4174.5) | -14391.6 (-6368.6) |
| A | -7800.3 (+0.0) | -8208.2 (-407.9) | -8800.2 (-999.8) | -9493.3 (-1692.9) | -11069.8 (-3269.5) | -12850.4 (-5050.0) |
| D | -5172.9 (+0.0) | -5188.4 (-15.6) | -5317.3 (-144.4) | -5494.2 (-321.4) | -5919.5 (-746.6) | -6398.6 (-1225.7) |

Leave-one-match-out choice:

| Held-out match | H | A | D |
|---|---:|---:|---:|
| J03WMX | 1 | 1 | 1 |
| J03WN1 | 1 | 1 | 1 |
| J03WOH | 1 | 1 | 1 |
| J03WOY | 1 | 1 | 1 |
| J03WPY | 1 | 1 | 1 |
| J03WQQ | 1 | 1 | 1 |
| J03WR9 | 1 | 1 | 1 |

### IDSSE segment lengths by state (all configs)

```text
H (n = 1539, 1 s bins, # = 7.2 segments)
  0–1 |  202 ############################
  1–2 |  289 ########################################
  2–3 |  189 ##########################
  3–4 |  135 ###################
  4–5 |  120 #################
  5–6 |   63 #########
  6–7 |   60 ########
  7–8 |   51 #######
  8–9 |   54 #######
 9–10 |   43 ######
10–11 |   45 ######
11–12 |   19 ###
12–13 |   22 ###
13–14 |   25 ###
14–15 |   15 ##
  ≥15 |  207 #############################
```

```text
A (n = 1522, 1 s bins, # = 6.2 segments)
  0–1 |  182 ##############################
  1–2 |  246 ########################################
  2–3 |  210 ##################################
  3–4 |  168 ###########################
  4–5 |  112 ##################
  5–6 |   99 ################
  6–7 |   71 ############
  7–8 |   44 #######
  8–9 |   44 #######
 9–10 |   40 #######
10–11 |   33 #####
11–12 |   27 ####
12–13 |   27 ####
13–14 |   22 ####
14–15 |   22 ####
  ≥15 |  175 ############################
```

```text
D (n = 713, 4 s bins, # = 2.4 segments)
  0–4 |   38 ################
  4–8 |   74 ###############################
 8–12 |   95 ########################################
12–16 |   73 ###############################
16–20 |   83 ###################################
20–24 |   69 #############################
24–28 |   45 ###################
28–32 |   57 ########################
32–36 |   36 ###############
36–40 |   27 ###########
40–44 |   24 ##########
44–48 |   17 #######
48–52 |   11 #####
52–56 |    9 ####
56–60 |   10 ####
  ≥60 |   45 ###################
```


## 2. Calibration of `duration_stages` on `configs/v2_gbdt.yaml`

True IDSSE segment lengths (7 matches, per half, X never forms a segment). Mean and `min_dwell` from the config; every `r` is scored on the segments with `L ≥ m − 1 + 8` and the likelihood truncated there.

| State | m (frames) | mean (frames) | segments | kept (L ≥ floor) | median L | Var Y / Var geometric | r (all) |
|---|---:|---:|---:|---:|---:|---:|---:|
| H | 12 | 191.0 | 1539 | 1407 | 90 | 2.44 | 1 |
| A | 12 | 170.3 | 1522 | 1402 | 93 | 1.96 | 1 |
| D | 25 | 630.5 | 713 | 710 | 491 | 0.84 | 2 |

Log-likelihood by `r` (all matches; Δ against the best):

| State | r = 1 | r = 2 | r = 3 | r = 4 | r = 6 | r = 8 |
|---|---:|---:|---:|---:|---:|---:|
| H | -8787.4 (+0.0) | -9330.8 (-543.5) | -10088.4 (-1301.1) | -10960.4 (-2173.0) | -12918.5 (-4131.1) | -15108.6 (-6321.3) |
| A | -8566.5 (+0.0) | -8957.5 (-391.0) | -9554.3 (-987.8) | -10261.8 (-1695.3) | -11881.6 (-3315.1) | -13717.6 (-5151.1) |
| D | -5253.8 (-9.1) | -5244.7 (+0.0) | -5349.9 (-105.3) | -5503.3 (-258.7) | -5880.5 (-635.8) | -6309.1 (-1064.4) |

Leave-one-match-out choice:

| Held-out match | H | A | D |
|---|---:|---:|---:|
| J03WMX | 1 | 1 | 2 |
| J03WN1 | 1 | 1 | 2 |
| J03WOH | 1 | 1 | 2 |
| J03WOY | 1 | 1 | 2 |
| J03WPY | 1 | 1 | 2 |
| J03WQQ | 1 | 1 | 2 |
| J03WR9 | 1 | 1 | 2 |

## 3. Selection step on `configs/v1s.yaml`

- calibration file: `configs/calib/durations.yaml`, `all` = H 1, A 1, D 1
- the calibrated stages equal the config's (`duration_stages` all 1): the decoder is unchanged bit for bit (tests/test_modeling_negbin.py), so the step is the identity and nothing was evaluated.

## 4. Selection step on `configs/v2_gbdt.yaml`

- calibration file: `configs/calib/durations_v2_gbdt.yaml`, `all` = H 1, A 1, D 2
- λ re-swept with the stages (F2-05 stage 1, plateau + worst case over default / hidelo / hidehi): `lam_team` 0.4, `lam_dead` 0.4
- control, λ re-swept the same way with the base stages: `lam_team` 0.4, `lam_dead` 0.4

Configs: `base` = `v2_gbdt` as committed; `stages` = base + calibrated stages; `stages_lam` = `stages` with λ re-swept; `base_lam` = base with λ re-swept (control).

| Config | Mode | spellF1 | EP | Pred/true possessions | Δ vs base | EP(Δ) |
|---|---|---:|---:|---:|---:|---:|
| base | default | 0.4895 | 0.0083 | 0.657 | — | — |
| base | harder | 0.4576 | 0.0100 | 0.655 | — | — |
| base | easier | 0.5191 | 0.0052 | 0.658 | — | — |
| base | hidelo | 0.4659 | 0.0085 | 0.653 | — | — |
| base | hidehi | 0.5237 | 0.0094 | 0.658 | — | — |
| stages | default | 0.4865 | 0.0085 | 0.647 | -0.0029 | 0.0009 |
| stages | harder | 0.4588 | 0.0100 | 0.644 | +0.0011 | 0.0009 |
| stages | easier | 0.5171 | 0.0058 | 0.650 | -0.0020 | 0.0008 |
| stages | hidelo | 0.4647 | 0.0082 | 0.646 | -0.0012 | 0.0008 |
| stages | hidehi | 0.5205 | 0.0097 | 0.650 | -0.0032 | 0.0009 |
| stages_lam | default | 0.4928 | 0.0078 | 0.704 | +0.0034 | 0.0024 |
| stages_lam | harder | 0.4651 | 0.0119 | 0.709 | +0.0075 | 0.0036 |
| stages_lam | easier | 0.5259 | 0.0052 | 0.703 | +0.0068 | 0.0032 |
| stages_lam | hidelo | 0.4698 | 0.0087 | 0.710 | +0.0039 | 0.0026 |
| stages_lam | hidehi | 0.5280 | 0.0097 | 0.710 | +0.0044 | 0.0025 |
| base_lam | default | 0.4885 | 0.0073 | 0.718 | -0.0009 | 0.0025 |
| base_lam | harder | 0.4627 | 0.0120 | 0.726 | +0.0051 | 0.0047 |
| base_lam | easier | 0.5227 | 0.0047 | 0.717 | +0.0036 | 0.0036 |
| base_lam | hidelo | 0.4672 | 0.0087 | 0.727 | +0.0013 | 0.0027 |
| base_lam | hidehi | 0.5244 | 0.0092 | 0.722 | +0.0007 | 0.0026 |

F2-06 rule (paired gain > 1 EP on default; no stress mode worse by > 1 EP); against `base` unless named:

- `stages`: Δ default -0.0029 ± 0.0009; gain: no; stress modes worse by > 1 EP: easier, hidelo, hidehi → **rejected**
- `stages_lam`: Δ default +0.0034 ± 0.0024; gain: yes; stress modes worse by > 1 EP: none → **accepted**
- `base_lam`: Δ default -0.0009 ± 0.0025; gain: no; stress modes worse by > 1 EP: none → **rejected**
- `stages_lam vs base_lam`: Δ default +0.0043 ± 0.0011; gain: yes; stress modes worse by > 1 EP: none → **accepted**

Paired Δ `stages_lam` − `base_lam` per mode (the stage effect at re-swept λ): default +0.0043 ± 0.0011; harder +0.0025 ± 0.0015; easier +0.0032 ± 0.0010; hidelo +0.0027 ± 0.0009; hidehi +0.0037 ± 0.0007

λ grid on default, with the stages and with the base stages (control):

| lam_team | lam_dead | spellF1 (stages) | EP | Pred/true | spellF1 (control) | EP | Pred/true |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.05 | 0.3614 | 0.0158 | 0.379 | 0.3736 | 0.0150 | 0.389 |
| 0.05 | 0.1 | 0.3784 | 0.0155 | 0.406 | 0.3874 | 0.0138 | 0.420 |
| 0.05 | 0.2 | 0.3885 | 0.0116 | 0.434 | 0.3895 | 0.0116 | 0.448 |
| 0.05 | 0.4 | 0.3880 | 0.0110 | 0.470 | 0.3880 | 0.0112 | 0.490 |
| 0.05 | 0.8 | 0.3802 | 0.0108 | 0.517 | 0.3769 | 0.0108 | 0.544 |
| 0.1 | 0.05 | 0.4115 | 0.0113 | 0.456 | 0.4206 | 0.0103 | 0.468 |
| 0.1 | 0.1 | 0.4331 | 0.0101 | 0.484 | 0.4341 | 0.0098 | 0.491 |
| 0.1 | 0.2 | 0.4410 | 0.0069 | 0.509 | 0.4444 | 0.0065 | 0.527 |
| 0.1 | 0.4 | 0.4391 | 0.0059 | 0.551 | 0.4392 | 0.0057 | 0.569 |
| 0.1 | 0.8 | 0.4317 | 0.0051 | 0.598 | 0.4259 | 0.0066 | 0.625 |
| 0.2 | 0.05 | 0.4364 | 0.0090 | 0.552 | 0.4445 | 0.0083 | 0.562 |
| 0.2 | 0.1 | 0.4708 | 0.0075 | 0.576 | 0.4736 | 0.0071 | 0.585 |
| 0.2 | 0.2 | 0.4794 | 0.0054 | 0.600 | 0.4816 | 0.0057 | 0.609 |
| 0.2 | 0.4 | 0.4806 | 0.0042 | 0.629 | 0.4738 | 0.0048 | 0.645 |
| 0.2 | 0.8 | 0.4673 | 0.0063 | 0.672 | 0.4615 | 0.0068 | 0.695 |
| 0.4 | 0.05 | 0.4092 | 0.0110 | 0.673 | 0.4188 | 0.0107 | 0.680 |
| 0.4 | 0.1 | 0.4654 | 0.0090 | 0.675 | 0.4716 | 0.0095 | 0.682 |
| 0.4 | 0.2 | 0.4855 | 0.0095 | 0.684 | 0.4860 | 0.0091 | 0.696 |
| 0.4 | 0.4 | 0.4928 | 0.0078 | 0.704 | 0.4885 | 0.0073 | 0.718 |
| 0.4 | 0.8 | 0.4842 | 0.0083 | 0.746 | 0.4790 | 0.0086 | 0.767 |
| 0.8 | 0.05 | 0.3740 | 0.0072 | 0.872 | 0.3775 | 0.0071 | 0.874 |
| 0.8 | 0.1 | 0.4233 | 0.0080 | 0.826 | 0.4279 | 0.0076 | 0.831 |
| 0.8 | 0.2 | 0.4716 | 0.0069 | 0.806 | 0.4707 | 0.0071 | 0.813 |
| 0.8 | 0.4 | 0.4858 | 0.0069 | 0.812 | 0.4864 | 0.0072 | 0.819 |
| 0.8 | 0.8 | 0.4859 | 0.0058 | 0.835 | 0.4822 | 0.0068 | 0.852 |

Informational only, never used for a choice: other D stage counts with the base λ (default):

| r_D | spellF1 | EP | Pred/true possessions | Δ vs base | EP(Δ) |
|---:|---:|---:|---:|---:|---:|
| 3 | 0.4868 | 0.0092 | 0.639 | -0.0027 | 0.0019 |
| 4 | 0.4855 | 0.0094 | 0.634 | -0.0039 | 0.0022 |
