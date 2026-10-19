# Semantic Evaluator Calibration & Discrimination Findings

This document reports how the semantic evaluators in [`semantic/`](../semantic/)
behave on real data: how their thresholds are (mis)calibrated, and how well each
signal actually separates good episodes from failures. It exists because the
audit's semantic checks ship with thresholds that were set by assumption, not
measured — and because an evaluator that produces a plausible-looking curve is
not the same as one that detects failures.

**Scope and honesty note.** All numbers below come from a single dataset
(`DerekLX/lerobot_derek_depth`), a single task (`Table Clean`), one camera
(`observation.images.global`), and 25 episodes. There are no human success/failure
labels, so failures were *injected* into known-good episodes as a proxy for ground
truth. Treat these as directional findings on a small sample, not population
statistics. See [Limitations](#limitations).

## Summary

1. **The progress-regression threshold (fixed `0.15`) is miscalibrated.** On 25
   presumably-good episodes it flags 7; a dataset-relative threshold — the repo's
   own `robust_upper_threshold`, already used for action/state jumps — flags 0.
2. **CLIP `stage_temperature=100` caused most of the false positives.** Lowering it
   to 30 drops the semantic false-positive rate from 32% to 0% with no loss of
   discrimination. The default is now 30.
3. **Global-view CLIP alignment does not discriminate failures on this task**
   (AUC ≈ 0.5 across injected failures). A static overhead camera carries little
   progress signal.
4. **Proprioceptive motion is a strong, cheap detector** of physical/temporal
   corruption (AUC 1.0 for frozen and scrambled episodes, 0.91 for truncated) — but
   is blind to reversed/mislabeled episodes.
5. **No single evaluator suffices; the ensemble should combine them** — with the
   explicit caveat that a conservative `min`-combine inherits every child's false
   positives.

## Setup

| | |
| --- | --- |
| Dataset | `DerekLX/lerobot_derek_depth`, `Table Clean`, 30 fps, `so_follower` 6-DoF |
| Episodes | 25 (indices 0–24, the span of `videos/observation.images.global/chunk-000/file-000.mp4`) |
| CLIP evaluator | open_clip ViT-B/32 (laion2b), 32 frames, stage-anchored, median smoothing |
| Motion evaluator | model-free; `observation.state` trajectory read via HF column projection |
| Metric | threshold-free **AUC** = P(good ranks better than failure); 1.0 perfect, 0.5 chance |

Failures were injected by manipulating an episode's real frames (for CLIP) or its
real state trajectory (for motion) and scoring the variant through the same path:

| Injection | Simulates |
| --- | --- |
| `truncate` (first 50%) | task abandoned early |
| `reverse` | robot un-does the task |
| `freeze` | robot never moves |
| `shuffle` | temporally scrambled / spliced |
| `wrong_task` / `wrong_motion` | mislabeled episode (video or motion from a different task) |

## Finding 1 — the progress-regression threshold is miscalibrated

The consistency check `semantic_progress_regression_review` fires when an episode's
largest frame-to-frame progress drop reaches a **fixed 0.15**. Across the 25
episodes the max-drop distribution is:

```
median 0.121   MAD 0.024   90th pct 0.194   max 0.212
```

The threshold sits *inside* this distribution — i.e. below the noise floor of raw
CLIP curves — so it flags ordinary variation:

| Threshold | Basis | Episodes flagged |
| --- | --- | --- |
| **0.15** | fixed constant | **7 / 25** |
| 0.24 | `median + 5·MAD` | 0 / 25 |
| 0.61 | `robust_upper_threshold` (median + 20·MAD, or q99) | 0 / 25 |

Every episode here is published training data, presumably successful, so a 28%
regression-flag rate is over-firing rather than genuine detection.

**Recommendation.** Make the progress-regression check **dataset-relative**, exactly
like the action- and state-jump checks that sit beside it in
`pipeline/data_quality_audit.py`. Those already call
`utils/quality_utils.robust_upper_threshold` over the population; applying the same
function to per-episode max-drop lands near 0.24 here and flags none of the 25. The
fix reuses existing code and doctrine rather than adding anything.

## Finding 2 — CLIP `stage_temperature` drove the false positives

CLIP cosine similarities are small (~0.15–0.35). The stage-anchored curve applies a
softmax over stages scaled by `stage_temperature`; at 100 the softmax is nearly
one-hot, so the curve becomes brittle argmax-noise that trips the regression check.
Sweeping the temperature (25 episodes, false-positive rate on the good baseline):

| `stage_temperature` | Good false-positive rate |
| --- | --- |
| 100 (old default) | **0.32** |
| 30 (new default) | **0.00** |
| 10 | 0.00 |

Lowering the temperature eliminates the false positives **without** changing
discrimination (Finding 3 holds at every temperature). The evaluator default is now
`stage_temperature = 30`. Findings 1 and 2 are complementary fixes: the temperature
change quiets the raw signal, and the dataset-relative threshold is the principled
long-term calibration.

A related design note surfaced during prompt tuning: stage prompts that describe an
**end-state** ("a cleared tidy table") *inverted* the progress curve, because a
cleaning task looks most "finished" before manipulation begins. Prompts that
describe the **action in progress** ("a robot arm reaching toward a cluttered
table") produce a correctly rising curve. Prompt wording is itself a calibration
surface.

## Finding 3 — global-view CLIP does not discriminate failures

Injected failures scored through the CLIP evaluator, AUC on the final progress
score (at the new default temperature 30; other temperatures within ±0.02):

| Injection | AUC | Verdict |
| --- | --- | --- |
| `truncate` | 0.49 | chance |
| `reverse` | 0.50 | chance |
| `freeze` | 0.50 | chance |
| `shuffle` | 0.53 | near chance |
| `wrong_task` | 0.02 | score moves the *wrong* way |

Mean final progress is ~0.68 for good, truncated, reversed, and frozen episodes
alike — a frozen robot is indistinguishable from a good episode. The cause is the
signal, not the threshold: a fixed overhead view of a tabletop barely changes across
the task, so per-frame CLIP embeddings carry little progress information, and
per-frame scoring is order-blind in aggregate. `wrong_task` scoring *higher* than
good reflects the softmax pegging an arbitrary stage when no stage matches — a sign
the alignment is not reading task semantics here.

## Finding 4 — proprioceptive motion is a strong, cheap detector

The same injections applied to the state trajectory and scored through the motion
evaluator:

| Injection | AUC (success) | AUC (path length) | Detected @ 0.5 |
| --- | --- | --- | --- |
| good (baseline) | — | — | FPR 0.00 |
| `freeze` | **1.00** | **1.00** | 1.00 |
| `shuffle` | **1.00** | **1.00** | 1.00 |
| `truncate` | 0.43 | **0.91** | 0.00 |
| `reverse` | 0.50 | 0.50 | 0.00 |
| `wrong_motion` | 0.50 | 0.50 | 0.00 |

Motion perfectly separates frozen and temporally scrambled episodes (via the
"did it move" and direction-consistency terms), and separates truncated episodes via
absolute path length — though truncation needs a *dataset-relative* path-length
threshold, since a truncated trajectory normalizes to a normal-looking success
score on its own. Motion is honestly blind to `reverse` (kinematically
direction-symmetric) and `wrong_motion` (a different valid episode's motion looks
valid). It requires no model, no GPU, and no video download.

Two design flaws were found and fixed while calibrating the motion evaluator, both
surfaced by tests:

- Jerk normalized by the trajectory's own median step cannot detect a *uniform*
  scramble (both scale together) → replaced with scale-invariant **direction
  consistency** (cosine between consecutive step vectors), which drives to ≈ −0.44
  under shuffle.
- "Fraction of frames moving" penalized normal teleop pauses (good FPR 0.32) →
  changed to "did it move at all" (total path above a floor), giving good FPR 0.00.

## Finding 5 — the ensemble, and its tradeoff

The signals are complementary: motion catches what a static camera cannot (physical
plausibility), and alignment/VLM signals are needed for semantic errors motion
cannot see. The [`ensemble_evaluator`](../semantic/ensemble_evaluator.py) combines
them behind one interface with a conservative `min`-combine and graceful degradation
(a child that cannot run is skipped). End-to-end on real episode 0:

```
good            combined 0.72  = min(motion 0.94, clip 0.72)   -> pass
freeze_motion   combined 0.00  = min(motion 0.00, clip 0.72)   -> caught
shuffle_motion  combined 0.00  = min(motion 0.00, clip 0.72)   -> caught
no-motion       combined 0.72  (motion skips, clip carries)    -> graceful
```

**Tradeoff to calibrate.** `min`-combine maximizes failure recall but inherits every
child's false positives. Because CLIP good scores range 0.46–0.94, a `min`-combine
inherits CLIP's occasional low score as a false positive that motion alone would not
produce. Options: down-weight or drop the weak child, calibrate the child first, or
use `mean`/`weighted` combine. This is a deliberate operating-point choice, not a
default to accept blindly.

## Recommendations

1. **Progress-regression check:** replace the fixed `0.15` with
   `robust_upper_threshold` over the dataset's max-drop distribution
   (`pipeline/data_quality_audit.py`), matching the action/state-jump checks.
2. **Truncation:** add a dataset-relative check on trajectory path length (again via
   `robust_upper_threshold`); the per-episode success score does not catch it.
3. **Evaluator choice for manipulation data:** prefer motion (or an ensemble led by
   motion) over global-view CLIP; reserve alignment for semantic checks and wrist or
   task-relevant camera views.
4. **`reverse` is unresolved:** neither single-frame signal catches a physically
   valid but reversed episode. This needs a temporal model (sequence-aware VLM or a
   directed goal signal) and is the clearest next research target.
5. **Thresholds are parameters, not constants:** derive them from the dataset or from
   labeled episodes; the two fixed constants examined here (`0.15`, `temperature 100`)
   were both mis-set.

## Limitations

- **n = 25, single task, single camera.** Distribution statistics (Finding 1) and
  AUCs (Findings 3–4) are directional on a small sample.
- **No human labels.** Failures are *synthetic injections* into presumably-good
  episodes; the "good" baseline is assumed, not confirmed. Injected failures are a
  proxy for, not a substitute for, labeled real failures.
- **Some injections overlap other audit layers.** `freeze` and `truncate` would also
  be caught by structural checks (frozen-video, frame-count) and the contributor's
  motion/gripper check; the purest test of *semantic* discrimination is `reverse` and
  `shuffle`.
- **`min`-combine tradeoff (Finding 5) is not yet operating-point-tuned.**

## Reproduction

The evaluators and their tests are in [`semantic/`](../semantic/) and
`tests/test_{motion,clip_alignment,ensemble}_evaluator.py`,
`tests/test_semantic_smoothing.py`. The discrimination and calibration harnesses
used for this report are experiment scripts (not part of the shipped package):
they decode each episode's frames / read its state, inject the failure variants,
score through the evaluators, and compute the AUC and threshold tables above.
Running them requires the real dataset, `ffmpeg`, and — for the CLIP paths —
`torch` + `open_clip` on a GPU; the motion paths need only numpy and
`huggingface_hub`.
