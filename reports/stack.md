# Second stage over the evidence (F2-09, D-029)

All numbers are spellF1 on the seven degraded IDSSE matches, every component evaluated with the
match left out of its fit. Nothing here reads the provided match; the BAR-ATH checks are local
PASS/FAIL runs (end of this file).

## Why: where `configs/final.yaml` of D-028 lost its score

Oracles on the `fused` masters with the D-028 decoder (only the named evidence replaced by the
truth, ±6 for the team, ±8 for the dead ball):

| evidence replaced by the truth | spellF1 |
|:---|---:|
| none (D-028 `final.yaml`) | 0.5406 |
| dead ball | 0.5928 |
| team | 0.9063 |
| both | 0.9709 |

True transitions on the 7 matches, and the share found by the D-028 output within 12 / 25 frames
(nearest predicted transition of the same type):

| type | n | ≤ 12 frames | ≤ 25 frames |
|:---|---:|---:|---:|
| team → team (open play) | 2351 | 0.29 | 0.37 |
| team → D | 710 | 0.57 | 0.65 |
| D → team | 699 | 0.65 | 0.75 |

Recall at IoU 0.5 by true possession length: 0.04 below 1 s, 0.19 for 1-2 s, 0.38 for 2-5 s,
0.66 for 5-10 s, 0.86 for 10-20 s, 0.93 above 20 s. The loss is the timing of team changes in open
play, not the dead ball.

## What

1. **GBDT hazard** (`hazard.kind: gbdt`, `modeling/evidence/hazard_gbdt.py`): same target as F1-08
   (team change within 12 frames), 46 swap-even features: the F1-08 `hazard_nodir` features (no
   `ball_dir_change`, D-025) at windows 12/25/50, products / differences / magnitudes of the
   backward and forward EMAs of the pass-bridge control signal (tau 10-200 frames) and of the
   touch feature (tau 25-200), backward/forward EMAs of the control mass, and past/future `d_diff`
   means over 100/200 frames. Out-of-fold AUC on `default` 0.813 (F1-08 logistic: 0.712).
2. **Team second stage** (`stack`, `modeling/evidence/stack.py`): GBDT over the first-stage
   `team_logit` (past/future means over 6-200 frames) and the hazard (past/future maxima over
   12/50 frames), antisymmetric by construction (`(g(X) - g(SX)) / 2`).
3. **Dead-ball second stage** (`dead_stack`): GBDT over the F3-08 `d_fit` (before the tag prior,
   which is added back afterwards: IDSSE has no tags) and the hazard, in windows. Out-of-fold AUC
   (usable frames, `default`) 0.986 → 0.992, better on all 7 folds.
4. Decoder `beta`, `lam_team`, `lam_dead` re-swept on a grid fixed before the run
   (`modeling/scoring/stack_sweep.py`; plateau at 1 SE on `default`, then best worst case over
   `default`/`hidelo`/`hidehi`).

Nesting: the fold model of match `m` is trained on the other matches `j`, with first-stage inputs
of `j` from models trained without `m` and `j` (21 pair models per first stage). A scratch check
with the same features gave 0.5615 (`fused`) nested against 0.5580 without the nesting: no sign of
leakage.

## Ladder (re-swept decoder at every step, `reports/stack_sweep*.md`)

| step | default | hidelo | hidehi | fused | harder | easier |
|:---|---:|---:|---:|---:|---:|---:|
| D-028 `final.yaml` (`configs/final_d028.yaml`) | 0.5365 | 0.5166 | 0.5658 | 0.5406 | 0.5129 | 0.5680 |
| + GBDT hazard + team second stage (`stack_sweep_team.md`) | 0.5568 | 0.5375 | 0.5912 | 0.5612 | 0.5339 | 0.5823 |
| + dead-ball second stage = **`final.yaml`** (`stack_sweep.md`) | **0.5672** | **0.5461** | **0.5984** | **0.5724** | **0.5405** | **0.5963** |
| rejected: + control EMAs in the first stage (`stack_sweep_ema.md`) | 0.5653 | 0.5453 | 0.5983 | 0.5713 | 0.5398 | 0.5973 |

The hazard alone (scratch, before the second stage) gave about +0.005 on `fused`; the second stage
without the hazard as input made the score worse, so the two only work together.

## Final against D-028 (`reports/external/final__*.md`, `final_d028__*.md`)

Paired over the 7 matches (new − D-028):

| mode | spellF1 | Δ ± SE | matches better | macroF1 Δ |
|:---|---:|---:|---:|---:|
| fused | 0.5724 | +0.0318 ± 0.0072 | 7/7 | +0.0124 |
| default | 0.5672 | +0.0307 ± 0.0068 | 7/7 | +0.0124 |
| hidelo | 0.5460 | +0.0294 ± 0.0062 | 7/7 | +0.0117 |
| hidehi | 0.5984 | +0.0325 ± 0.0064 | 7/7 | +0.0104 |
| harder | 0.5405 | +0.0276 ± 0.0079 | 7/7 | +0.0147 |
| easier | 0.5963 | +0.0283 ± 0.0030 | 7/7 | +0.0127 |

Predicted possessions per match: 310 (D-028: 308) for 437 true ones on `fused`.

Decoder of `final.yaml`: `beta` 1.5, `lam_team` 0.4, `lam_dead` 0.4 (D-028: hazard `beta` 2 on the
logistic hazard, 0.4 / 0.2); everything else unchanged.

## Reproduction (IDSSE only)

```bash
python -m modeling.calibration.stack_gbdt --config configs/final_d028.yaml --jobs 7
python -m modeling.calibration.dead_stack --config configs/final.yaml --jobs 5
python -m modeling.scoring.stack_sweep --config configs/final.yaml --jobs 7 --write
python -m modeling.scoring.external_eval --config configs/final.yaml --degrade fused
```

## Local checks on BAR-ATH (a person ran them on 17/11; PASS/FAIL only)

- `python -m modeling.make_submission --config configs/final.yaml --skip-preprocessing`: config
  `a84e8120…`, CSV SHA-256 `25064d90ff2808ffb325f43d51253a5dd58be765d70c2ddee6d608f5fb9f0503`
- `tools/check_public_window.py --pred outputs/final.csv --master <fused master>`: **PASS** (all 7)
- `tools/deadball_diagnostics.py --pred outputs/final.csv --master <fused master> --config configs/final.yaml`:
  **PASS** (informative (iv) and (v) too)
- `tools/stack_feature_shift.py --master <fused master>`: **PASS** on (a)-(d) and (f), informative (e) PASS
- `tools/dead_stability.py --config configs/final.yaml`: **PASS** on the three pairs
- unchanged inputs, already PASS for D-028: `gbdt_feature_shift` (team first stage), `dead_feature_shift` (dead GBDT).

Plan B (`configs/final_d028.yaml`, CSV `f9e363ea…`) was not needed.
