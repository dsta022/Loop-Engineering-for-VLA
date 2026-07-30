# Semantic evaluators

Concrete `SemanticEvaluator` backends for the data quality audit. The audit and
its model-free consistency checks (`utils/semantic_consistency.py`) never import
this package — it is loaded **only** when an audit is invoked with an explicit
evaluator spec:

```powershell
python .\pipeline\data_quality_audit.py .\lerobot_derek_depth `
  --semantic-only `
  --semantic-evaluator semantic.motion_evaluator:create_evaluator `
  --semantic-config .\configs\semantic_evaluator.motion.example.json `
  --semantic-review-threshold 0.5 `
  --semantic-pass-threshold 0.9
```

The evaluator's job is to score *whether the robot did the task*, producing a
`progress_curve` and/or a `success_probability`. Semantic evaluation can only
route an episode to **pass** or **review**, never **drop** — model-based judgment
is not trusted for automatic deletion. The `progress_curve` an evaluator returns
is what activates the audit's progress-shape and stage-timeline checks; without an
evaluator those checks stay dormant.

## What's here

| Path | Purpose |
| --- | --- |
| [motion_evaluator.py](motion_evaluator.py) | model-free proprioceptive evaluator (reads action/state) |
| [clip_alignment_evaluator.py](clip_alignment_evaluator.py) | CLIP / SigLIP frame↔text alignment evaluator |
| [ensemble_evaluator.py](ensemble_evaluator.py) | combine several evaluators behind one interface |
| [smoothing.py](smoothing.py) | temporal smoothing filters for per-frame signals |
| [frame_sampler.py](frame_sampler.py) | decode an episode's video time-segment via ffmpeg |
| [backends.py](backends.py) | vision-language embedding backend protocol + open_clip |

Heavy dependencies (`torch`, `open_clip`) are imported **lazily** inside the
backend classes, so importing this package, running the base audit, and running
the unit tests never require a model runtime. The motion evaluator is pure numpy.

## The three evaluators

### `motion_evaluator` — proprioceptive, model-free

Reads the arm's own action/state trajectory (exposed on `SemanticEpisode.motion`
as `EpisodeMotion`). No model, no GPU, no video. Produces:

- `progress_curve` — normalized cumulative path length (monotone 0→1);
- `success_probability` — *moved* × *direction-consistency*, where "moved" is
  "did it move at all" (total path above a floor, so normal teleop pauses are not
  penalized) and direction-consistency is the mean cosine between consecutive step
  vectors (scale-invariant, so a uniform temporal scramble drives it toward 0).

Config: `still_fraction`, `jerk_multiple`, `curve_points`, `min_total_path`.

### `clip_alignment_evaluator` — vision-language alignment

Samples frames from the episode's video segment, embeds frames and text with an
open_clip backend, and turns per-frame image/text similarity into a progress
curve. **Stage-anchored** when the episode carries a Task Profile (each frame is
scored against every stage's `observable_condition`, weighted by cumulative stage
weight); **instruction-only** min-max fallback otherwise. Deterministic given
fixed frames and backend; a fixed smoothing filter denoises the raw signal.

Requires `torch` + `open_clip` in the run environment and `ffmpeg` on `PATH`.
Config: `model_name`, `pretrained`, `device`, `batch_size`, `num_frames`,
`frame_size`, `stage_temperature`, `smoothing`, `smoothing_window`,
`prompt_template`.

### `ensemble_evaluator` — combine the above

Runs several child evaluators (built from their own `module:attribute` specs) and
combines their scores. `combine="min"` (default) is conservative — any child that
sees a problem pulls the episode toward review. A child that cannot run on an
episode (no video, no action stream) is **skipped**, not fatal, so the audit keeps
working with whatever signals are available. `curve_from` selects which child
feeds the progress curve.

Config: `combine` (`min`/`mean`/`weighted`), `curve_from`, `require`, and a
`children` list of `{name, spec, config, weight}`.

## Choosing an evaluator

A discrimination study on `lerobot_derek_depth` (25 episodes, real frames /
trajectories corrupted into known failures, separation measured by AUC where
1.0 = perfect and 0.5 = chance) found the signals are complementary:

| Injected failure | CLIP (global view) | Motion |
| --- | --- | --- |
| frozen robot | 0.48 | **1.00** |
| temporally shuffled | 0.52 | **1.00** |
| truncated (task abandoned) | 0.48 | **0.91** (via path length) |
| reversed (task undone) | 0.48 | 0.50 |
| wrong goal / mislabeled | ~chance | 0.50 |

Motion is a far stronger detector of physical and temporal corruption on this
manipulation data; a static overhead camera carries little progress signal, so
global-view CLIP is near chance. Neither single-frame signal catches a *reversed*
episode (physically valid, semantically wrong) — that needs a temporal model.
The practical recommendation is the **ensemble**: motion for cheap physical
plausibility, alignment for semantic checks, combined conservatively.

Note the tradeoff: `combine="min"` maximizes failure recall but inherits every
child's false positives, so pair a noisy child with threshold calibration or a
lower weight.

## Config examples

- [`configs/semantic_evaluator.motion.example.json`](../configs/semantic_evaluator.motion.example.json)
- [`configs/semantic_evaluator.clip.example.json`](../configs/semantic_evaluator.clip.example.json)
- [`configs/semantic_evaluator.ensemble.example.json`](../configs/semantic_evaluator.ensemble.example.json)

## Tests

Model-free, no GPU, no video — fake backends and scripted trajectories cover the
scoring paths, smoothing, combination logic, and graceful degradation:

```powershell
python -m unittest tests.test_motion_evaluator tests.test_clip_alignment_evaluator `
  tests.test_semantic_smoothing tests.test_ensemble_evaluator
```
