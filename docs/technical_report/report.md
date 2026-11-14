# Frame-level Ball Possession from Automatic Tracking: Technical Report

**Team:** `<team_name>` · **Challenge:** More than a Hack 2027, Phase 1 (Ball Possession) ·
**Match:** FC Barcelona (home, `H`) vs Athletic Club (away, `A`), LaLiga 2024-25, matchday 2

> **Draft status.** Sections marked *To be completed* are filled in by the front that owns the
> corresponding task, as part of that task's Definition of Done. The report is finalised in F5-07.
> No number in this report comes from the public 5-minute window, and no match data appears in it:
> every result is computed on external data.

## 1. Approach and why

### 1.1 Overview

We label every frame of the tactical video as `H`, `A`, `D` (dead ball) or `X` (outside the graded
halves) with a three-state hidden Markov model decoded offline:

1. **Preprocessing** turns the provided assets into one table with one row per video frame
   (Section 1.2).
2. **Two per-frame evidences** are computed from that table, both as log-likelihood ratios:
   `team_logit` (home versus away in possession) and `dead_logit` (dead versus live ball). A value of
   zero means "no evidence".
3. **A Viterbi decoder** with a minimum duration per state turns the two evidences into the globally
   best label sequence, separately for each half.

The reasons for this design:

- **The ranking metric scores possessions, not frames.** spellF1 matches whole possession intervals,
  so a label that flickers, or a boundary that arrives late, costs far more than a few wrong frames.
  A global decoder produces no flicker and, being offline and bidirectional, no phase lag.
- **Missing evidence must mean persistence.** The ball is missing in 41% of the graded frames
  (DATA.md). With likelihood-ratio evidences, a frame with no ball contributes zero and the previous
  state persists, which is exactly the challenge rule that a loose ball stays with the last team
  that had it.
- **Few, interpretable parameters, all chosen on other matches.** No label of the provided match is
  available for training or model selection, so every parameter is either a physical prior or is
  calibrated on external data (Section 3). A small model transfers across domains more safely than a
  large one.

*To be completed by F1, F3 and F2 (tasks F1-02/F1-05, F3-02/F3-04, F2-02/F2-06): the final form of the
two evidences and of the decoder, with the selected configuration.*

### 1.2 Preprocessing

One command, `python -m preprocessing.run_all`, builds `master_features.parquet`: 199,319 rows (one
per tactical video frame) and 60 columns, specified in `SCHEMA.md` v1.0. It runs in about 25 s on a
CPU and is deterministic: regenerated from scratch in a clean environment, every output file has the
same SHA-256.

Principles that hold for every step:

- **Observed and inferred values are kept apart.** Every interpolated or removed sample is flagged,
  and `NaN` always means "no evidence". It is never replaced by zero.
- **Offline and bidirectional.** No causal filters, because boundary delay is expensive under spellF1.
- **Nothing crosses halftime.** Smoothing, interpolation, derivatives and searches run inside each
  half. Outside the graded halves every tracking-derived column is `NaN`.
- **Team aggregates, not individuals.** The 46 tracking slots change identity during the match, so
  features are aggregated per team.
- **Parameters are physical priors** (table below), kept in one file (`preprocessing/params.py`).
  None was tuned on the public window, and none is tied to specific frames.

The steps:

1. **Load and align.** The automatic tracking data (ATD) is read and indexed by video frame. The
   graded halves come from the constants published by the organisers.
2. **Metres and orientation.** Positions are scaled to the 105.3 × 68.0 m pitch. The attacking
   direction is inferred per half from the team positions just after kick-off, and the pitch is
   rotated so that the home team always attacks towards increasing x. Velocities are rotated too.
3. **Ball.**
   - A jump filter keeps chains of mutually consistent observations (a step is consistent if it is
     reachable at 38 m/s plus 1 m of tracker noise). It bridges over blocks of phantom detections and
     keeps long disjoint chains, so a legitimate discontinuity never discards real ball data.
   - Confidence columns are computed after the filter: observed, removed, filled, gap length, and
     frames since the last and until the next observation.
   - Only gaps of at most 12 frames (0.5 s) are linearly interpolated. Longer gaps stay `NaN`.
   - Velocity, speed, acceleration and direction change are derived from a zero-phase smoothed
     trajectory (Savitzky–Golay, 7 frames, order 2), only on observed samples. They are `NaN` on
     interpolated frames.
4. **Players.** Single-frame jumps above 12 m/s mark a slot break, and nothing is derived across a
   break. Isolated outlier positions are removed. Same-team duplicates closer than 1 m are removed.
5. **Team aggregates.** Centroid, spread, mean speed, number of still players (below 0.5 m/s) and
   number of tracked players, per team.
6. **Proximity to the ball.** Per team: distance from the nearest player to the ball, and their
   difference; player counts within 2, 3 and 5 m; a continuous pressure term `Σ exp(−d / 3 m)`; and
   the relative and radial speed between the ball and the nearest player. Pitch features: ball out
   of the pitch (1 m margin) and distance to the touchline and to the goal line.
7. **Smart Tagging.** Set-piece intervals (type, team, and frames to the next and since the previous
   one), counts of phase tags favouring each team, and the pre-kick-off mask of each half. The
   kick-off frames are the published ones; the tags are only checked for consistency with them.
8. **Master table.** The intermediate tables are joined and validated against the frozen schema
   (names, dtypes, `NaN` rules).

| Parameter | Value | Justification |
| :--- | :--- | :--- |
| Ball maximum speed | 38 m/s | physical upper bound of a very hard shot |
| Ball position tolerance | 1 m | tracker position noise |
| Jump-filter window | 30 observations | bridges gaps at bounded cost |
| Longest interpolated ball gap | 12 frames (0.5 s) | short, nearly continuous segment |
| Minimum ball speed for direction change | 1.0 m/s | avoids spurious angles on a nearly still ball |
| Player maximum speed | 12 m/s | above human sprint speed |
| Duplicate distance (same team) | 1 m | two players cannot share one spot |
| Still-player speed | 0.5 m/s | slow walking |
| Orientation window | 25 frames (1 s) | players are still in their own half after kick-off |
| Count radii | 2, 3 and 5 m | scale of duels and of ball control |
| Pressure decay | 3 m | continuous scale of immediate pressure |
| Out-of-pitch margin | 1 m | position noise near the lines |

The same preprocessing code, called in memory, is applied to the external matches (Section 3), so the
model sees the same features in both domains.

## 2. Assets used and dropped

| Asset | Status | How it is used, or why it is not |
| :--- | :--- | :--- |
| Tactical ATD (automatic tracking) | **Used** | The basis of everything: ball and player positions, and every feature of the master table. |
| Smart Tagging | **Used as weak evidence** | Set-piece tags contribute low-weight terms to the dead-ball evidence only. They never decide a label. |
| Tactical homography | **Not in the model** | Diagnostics only. Its reprojection error is kept as an optional column and is deliberately excluded from the features. |
| Tactical video | **Not in the model** | Watched by team members to check the pipeline (for example the attacking direction in each half). Nothing seen in it became a label, a correction or a frame-specific constant. |
| Broadcast video | **Not in the model** | Not frame-aligned to the tactical video, with cuts, replays and zooms. |
| Broadcast tracking | **Exploratory** | *To be completed by F4 (tasks F4-05/F4-06): "used" or "tested and dropped", with the reason.* |

Why each decision was taken:

- **Smart Tagging is weak evidence.** The tags are automatic and have low recall for restarts, every
  throw-in is tagged with no team (DATA.md), and a set-piece interval extends beyond the restart
  itself. A high weight would delay the start of the next possession after every tagged restart, so
  dead-ball detection relies mainly on the kinematics of the tracking. The phase tags are intervals
  of play, not possessions; their per-team counts are kept in the master table but are not model
  inputs.
- **Homography and videos.** The tracking already carries the positions the model needs. Detecting
  the ball in the video ourselves was out of scope for the time available, and a vision model would
  make bit-exact CPU reproduction harder. The reprojection error does not exist for the external
  matches and is tied to match time, so using it as a feature would let a model memorise the moment
  of the match.
- **Broadcast tracking.** The tactical ball is missing in 41% of the graded frames, and the broadcast
  camera may see some of them. This is evaluated as a separate track with a go/no-go gate, and is
  accepted only through the protocol of Section 5.

## 3. External data

**Dataset.** The only external data is the IDSSE dataset: seven complete matches of the German
Bundesliga and 2. Bundesliga, season 2022/23, with official positional data at 25 fps. For every
frame it provides the ball status (alive or dead) and the team in possession.

> Bassek, M., Rein, R., Weber, H., Memmert, D. (2025). *An integrated dataset of spatiotemporal and
> event data in elite soccer.* Scientific Data 12, 195. https://doi.org/10.1038/s41597-025-04505-y

- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0). Its use was approved by the
  organisers (official issue #16).
- **Source:** the public Hugging Face mirror `pysport/idsse-data`, at the pinned revision
  `a715a38dfbaf5f58e431727c2b78d174101a703c`. File names, sizes and SHA-256 values are listed in
  `docs/external_data.md`.
- **Labels: who made them and how much.** The per-frame ball status and possession labels are part of
  the published dataset. We labelled nothing, by hand or otherwise. In total there are 1,002,644
  labelled frames (about 11.1 hours): 41–52% dead ball per match, the rest split between the two teams
  (`reports/idsse_overview.md`). Live frames without a valid possession value would be excluded; there
  are none.
- **No label comes from the provided match.**

**What it is used for.** Calibrating the two evidences, estimating the transition statistics of the
decoder (state durations and exit probabilities, `reports/idsse_label_stats.md`), sweeping the decoder
parameters and selecting the final variant (Section 5).

**How it is degraded.** IDSSE tracking is corrected and complete, with the ball present in nearly every
frame, while the provided ATD is automatic. Before use, each match is therefore degraded to resemble
the ATD and is then passed through the same preprocessing as the provided match:

- **Ball:** removed while it is in the air or inside a crowd of players, plus additional gaps drawn
  from a gap-length profile, until the visible fraction reaches 59.3%, the figure published in DATA.md
  for the graded frames; position noise; short phantom detections.
- **Players:** tracks are dropped and restored until the mean number of tracked players reaches 20.5
  (DATA.md); position noise; short duplicated tracks.
- All random draws use fixed seeds, so the degraded matches are reproducible.

*To be completed by F4 (task F4-04): the realism check of the degradation and the `harder` and `easier`
stress variants.*

## 4. Possession rules implemented

*To be completed by F1 (team evidence), F3 (dead ball and restarts) and F2 (decoder). Each item states
the rule and where it is implemented.*

- **Loose ball: persistence.** *(F1, F2)*
- **Challenges and duels.** A touch without control, a deflection or an aerial duel does not change
  possession. *(F1)*
- **Restarts and dead ball.** A missing ball is not treated as a dead ball. Every ball gap longer
  than 0.5 s is classified from the last frame before it and the first frame after it:
  - *exit*: the ball was out of the pitch, or within 3 m of a line and moving towards it, or it
    reappears in a restart position (on a touchline, or inside the goal area);
  - *high ball*: it is not an exit, the ball was faster than 8 m/s and the gap lasts at most 4 s.
    Possession continues;
  - *undetermined*: anything else.

  Only an exit is strong evidence of a dead ball. The other dead-ball features are the fraction of
  players standing still, the mean player speed, a ball at rest for at least 1 s, a fast ball inside
  the pitch (evidence of live play) and the set-piece tags. All thresholds are physical priors kept
  in the configuration. Implemented in `modeling/deadball/features.py`.

  The dead-ball evidence is the sum of the terms below, clipped to ±12. A term whose input is
  missing contributes exactly zero. These weights are priors; none was fitted on any data.

  | Evidence | Contribution |
  | :--- | :--- |
  | Ball outside the pitch | +4.0 |
  | Ball gap: exit / high ball / undetermined | +3.0 / −1.0 / +0.5 |
  | Fraction of players standing still | 0 up to 30%, rising to +3.0 at 70% |
  | Mean player speed | +1.5 below 0.5 m/s, 0 from 1.5 to 2.5 m/s, −2.0 above 4.5 m/s |
  | Ball at rest for at least 1 s | +2.0 |
  | Fast ball inside the pitch | −3.0 |
  | Set-piece tag active | +1.0 |
  | Approaching a tagged set piece | up to +1.0, decaying over 10 s |
  | Just after a tagged set piece | down to −1.0, decaying over 3 s |
  | Within 60 s before a tagged kick-off after a goal | +2.0 |

  The dead ball ends, and the next possession starts, when live evidence returns: a fast ball
  inside the pitch, players running again and the end of a tagged set piece all push the evidence
  negative, and the decoder then switches to the team in control. The tag weights are low on
  purpose, because a tagged interval extends past the restart itself. Implemented in
  `modeling/deadball/dead.py`.
  *To be completed by F3 (task F3-04): the weights calibrated on IDSSE, if that variant is selected
  by the protocol of Section 5. The tag terms always keep the priors above, because IDSSE has no
  tags.*
- **Before kick-off.** The graded block of the second half opens before its kick-off: those 2,934
  frames are forced to `D` by a mask that the decoder cannot override. The first half starts at the
  kick-off itself, so it has no such frames. The mask is the `pre_kickoff` column of the master
  table, built from the kick-off frames published by the organisers (FAQ.md); nothing is inferred
  from the tracking. Implemented in `modeling/deadball/force.py`.
- **No `X` inside the halves.** `X` is written only outside the graded halves. *(F2)*

## 5. Validation protocol

The protocol was fixed before any result existed.

- **Where.** Only on the seven degraded IDSSE matches. Every calibrated component is evaluated with the
  evaluated match left out of its fit (leave-one-match-out).
- **Criterion.** Mean spellF1 over the seven matches; external macroF1 breaks ties.
- **Parsimony rule.** Candidates are tried from the simplest to the most complex. A step is accepted
  only if its mean paired gain is larger than one standard error of that gain. Otherwise the simpler
  candidate stays.
- **Stress test.** An accepted step must also not lose more than one standard error on a harder and on
  an easier degradation, because the degradation is only an approximation of the provided tracking.
- **The public window was never used to choose anything.** On the provided match there are only
  pass/fail checks: a valid file, a correctness check on the public window (alignment, format, label
  direction), and plausibility bands for label statistics derived from IDSSE.
- **Plausibility filter.** For each half of the provided match, three statistics of the predicted
  labels (the fraction of dead-ball frames, possessions per minute and mean possession duration) must
  fall within the range observed over the 14 IDSSE halves, widened by 40% on each side
  (`tools/deadball_diagnostics.py`). The check uses no ground truth and only eliminates: a candidate
  that fails it is replaced by the best previous one that passes, and it never ranks candidates or
  tunes anything. Two further checks are reported for information only: how much of the ball
  out-of-pitch time is predicted as live play, and how many predicted dead-ball segments have no
  dead-ball evidence at all.

*To be completed by F2 (task F2-06), from `reports/selection.md`: the table of candidates with
per-match scores, paired gains and the decision at each step, including the variants that were tested
and dropped and the reason for each.*

## 6. Limitations

*To be completed by all fronts; edited by F5 (task F5-07). Known items:*

- **Domain shift.** Parameters are calibrated on corrected tracking from another provider and league,
  degraded synthetically. The degradation approximates the automatic tracking but does not reproduce it.
- **Possession convention.** The possession labels of IDSSE may follow a different convention from the
  challenge ground truth on short touches and duels.
- **Tag recall.** Many restarts have no tag, so dead-ball detection depends mostly on kinematics, which
  are weakest when the ball is not tracked.
- **Ball jump filter.** Two known edge cases of the filter were deliberately left unchanged.
  *(F3: task F3-07 measures how often they occur.)*
- **What we would fix with more time.** *(All fronts.)*

## 7. Reproduction

*To be completed by F5 (tasks F5-06 and F5-09): final command, tag and SHA-256.*

```bash
# Python 3.12; exact dependency pins
pip install -r requirements-modeling.txt

# Unpack the match assets into ./data, or point MTAH_DATA_DIR at them
python -m preprocessing.run_all
python -m modeling.make_submission --config configs/final.yaml --out outputs/<team_name>.csv
python evaluation/validate_submission.py outputs/<team_name>.csv
```

- **Versions:** Python 3.12, numpy 2.0.2, pandas 2.3.3, pyarrow 21.0.0, scipy 1.13.1, kloppy 3.18.0,
  pyyaml 6.0.2, scikit-learn 1.5.2.
- **Determinism:** CPU inference, single thread, fixed seeds, explicit `float64` sums, stable sorts.
- **SHA-256 of the submitted file:** *(F5)*
- **External data is not needed to reproduce the submission.** The command above uses only the provided
  match and the committed configuration and weights. IDSSE is needed only to redo the calibration,
  whose code is in the repository (`modeling/external`, `modeling/calibration`, `modeling/scoring`).

## 8. How the provided game was used (compliance)

- **No human labels and no training on the provided match.** No frame was annotated by hand, and no
  parameter, weight or threshold was fitted or chosen with labels of this match, including algorithmic
  pseudo-labels.
- **Unlabelled statistics, for the degradation only.** Unlabelled statistics of the provided tracking
  are used only to make the degradation of IDSSE realistic (ball visibility and tracked-player targets,
  and optionally the ball gap-length profile). This is unsupervised use, allowed by RULES.md.
- **Published constants.** The graded blocks of each half and the kick-off frames are the values
  published by the organisers (DATA.md, FAQ.md).
- **Public window.** Used only for pass/fail correctness checks, never to tune or select anything. Its
  score is not reported anywhere.
- **No frame-specific rules.** No rule or constant is tied to specific frames of the match, other than
  the published constants above.
- **Cloud LLMs.** No match data, extract or screenshot was sent to cloud-hosted language models. Coding
  assistants worked only with synthetic fixtures and with the public IDSSE data; every step that reads
  the match data was run locally by a team member.

## References

- Bassek, M., Rein, R., Weber, H., Memmert, D. (2025). An integrated dataset of spatiotemporal and
  event data in elite soccer. *Scientific Data* 12, 195. https://doi.org/10.1038/s41597-025-04505-y
  (CC BY 4.0)
- Metrica Sports and Barça Innovation Hub (2026). *More than a Hack 2027: Frame-level Ball Possession.*
  https://github.com/metrica-sports/more-than-a-hack-2027
