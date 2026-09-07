# Plugin contracts

Every extension point of the toolkit is a small Python object loaded with
`module:attribute` syntax. This page lists the exact data classes each plugin
receives and returns, the validation the toolkit applies, and the built-in
implementations you can start from. Source of truth: the modules named in each
section.

## How plugins are loaded

`module:attribute` is resolved with `importlib.import_module(module)` followed by
`getattr(module, attribute)`. The attribute may be:

| Attribute is | The toolkit does | Configuration reaches it as |
| --- | --- | --- |
| a class | instantiates it | keyword arguments (`Cls(**config)`) |
| an object with the plugin method | uses it as-is | ignored |
| any other callable | calls it as a factory | one dictionary (`factory(config)`) |

Configuration files (`--semantic-config`, `--annotator-config`, trigger
`config_path`, plugin `init` blocks) must contain exactly one JSON object.
Missing files mean an empty configuration. The module must be importable from
the process that runs the stage: put it on `PYTHONPATH`, install it, or drop it
next to the toolkit packages. Loader functions: `utils.semantic_evaluator.load_evaluator`,
`utils.language_annotation_profile.load_annotator`,
`utils.task_description_parser.load_parser`,
`policy_improvement.capture.load_window_trigger`,
`policy_improvement.plugins.load_plugin`.

## Semantic evaluator

Selected by `--semantic-evaluator` on `pipeline/data_quality_audit.py`.
Contracts live in `utils/semantic_evaluator.py`.

### Input: `SemanticEpisode`

| Field | Type | Meaning |
| --- | --- | --- |
| `dataset_root` | `Path` | dataset directory being audited |
| `episode_index` | `int` | episode number |
| `length` | `int` | number of frames |
| `tasks` | `tuple[str, ...]` | distinct task texts of the episode; the audit only calls the evaluator when there is exactly one |
| `videos` | `tuple[VideoSegment, ...]` | one entry per camera (or only the Task Profile's `preferred_video_keys`) |
| `task_profile` | `dict \| None` | the matched Task Profile as `to_dict()`, or `None` without `--task-profile-dir` |
| `motion` | `EpisodeMotion \| None` | action/state streams already loaded by the audit |

`VideoSegment(key, path, from_timestamp, to_timestamp)`: the episode occupies
`[from_timestamp, to_timestamp)` seconds of `path`; recorded datasets start at
0, aggregated files do not.

`EpisodeMotion(action, state, fps, gripper_index)`: frame-major `numpy` arrays
of shape `(length, dim)` or `None` when a column is absent; `gripper_index` is
the configured `--motion-gripper-index` or `None`. `gripper_series()` returns
the gripper column when it can.

### Output: `SemanticResult` or an equivalent mapping

| Key | Type | Rule |
| --- | --- | --- |
| `progress_curve` | sequence of floats | each value finite and within `[0, 1]`; optional |
| `success_probability` | float | finite, within `[0, 1]`; optional but required when `progress_curve` is empty |
| `failure_stage` | str | optional; empty means "infer from the curve or the Task Profile" |
| `score_samples` | sequence of floats | optional repeated scorings in `[0, 1]`; their mean fills a missing `success_probability`, their std is the score uncertainty |
| `score_std` | float | optional explicit uncertainty, non-negative; wins over `score_samples` |
| `details` | JSON-serializable mapping | optional; `details["counterfactual_margin"]` must be within `[-1, 1]`; `details["stage_timeline"]` may name a stage per progress point or give `{"stage", "start_index", "end_index"}` spans |

`score` is `success_probability` when present, otherwise the last progress
value. An exception from `evaluate()` sends the episode to review with
`semantic_evaluation_failed_review:<ExceptionName>`; it never approves data.

### How the audit uses the result

- Score bands: below `--semantic-failure-threshold` (default 0.5) is
  `semantic_low_score_review`; below `--semantic-pass-threshold` (default 0.9)
  is `semantic_uncertain_score_review`; otherwise semantic pass. A Task Profile
  may override both thresholds.
- Reliability contracts: uncertainty at or above
  `--semantic-uncertainty-threshold` (0.15) gives
  `semantic_score_unstable_review`; a counterfactual margin below
  `--semantic-counterfactual-margin-threshold` (0.10) gives
  `semantic_counterfactual_margin_low_review`; with `--semantic-per-view`, a
  per-camera score gap at or above `--semantic-view-disagreement-threshold`
  (0.30) gives `semantic_view_disagreement_review`.
- Progress-curve shape and stage-timeline checks read the curve and the Task
  Profile; they record `not_checked` entries when either is absent.

### Built-in: `semantic_backends.vla_checkpoint_evaluator:create_evaluator`

Scores instruction-conditioned action consistency with a lerobot policy
checkpoint. Configuration keys: `policy_path` (required), `device`, `anchors`
(6), `eval_horizon` (10), `sample_splits` (3), `max_counterfactuals` (3),
`camera_map` (`{dataset_camera: policy_camera}`), `static_epsilon`. Returns
`success_probability`, `score_samples`, and `details.counterfactual_margin`; no
progress curve. Needs `pip install lerobot` in the process that runs the
semantic stage. Two protocols let you swap the runtime without touching the
scoring: `ActionChunkPredictor.predict_chunk(observation, task, horizon, *,
features, action_names) -> ndarray[horizon, action_dim]` and
`FrameReader.read_frame(path, seconds, height, width) -> uint8 RGB frame`.

## Task description parser

Selected in Task Semantic Studio. Contract in `utils/task_description_parser.py`:

```python
class MyParser:
    name = "my_parser"
    def parse(self, description: str, task_name: str = "", task_key: str = "")\
            -> TaskSemanticProfile | Mapping[str, Any]: ...
```

The result is normalized with `TaskSemanticProfile.from_dict` and validated:
`task_key` matches `[a-z0-9][a-z0-9_-]*`; `task_name`, `instruction`, at least
one stage, and at least one success criterion are required; stage names are
unique; every stage has `description` and `observable_condition` and a positive
`weight`; `review_threshold <= pass_threshold` when both are set. Profiles are
matched against dataset task text by exact normalized equality of `task_key`,
`task_name`, `instruction`, or any alias. Built-in:
`utils.task_description_parser:HeuristicTaskDescriptionParser` (regular
expressions, Chinese and English).

## Trajectory language annotator

Selected by `--annotator` on `pipeline/trajectory_language_enrichment.py`.
Contracts live in `utils/language_annotation_profile.py`.

### Input: `LanguageAnnotationEpisode`

Same fields as `SemanticEpisode` plus `semantic_context`: the episode's quality
report row as a dictionary (`quality_status`, `structural_status`,
`semantic_status`, `semantic_score`, `semantic_success_probability`,
`semantic_failure_stage`, `semantic_progress_curve`, `review_reasons`,
`metrics`). Exactly one task is guaranteed.

### Output: `TrajectoryLanguageAnnotation` or an equivalent mapping

```python
{
  "subtasks": [{"start_frame": 0, "end_frame": 45, "text": "...", "stage": "approach", "confidence": 0.9}],
  "events": [{"frame_index": 46, "event": "gripper_closing", "description": "...", "end_frame": None, "confidence": None}],
  "state_descriptions": [{"start_frame": 0, "end_frame": 46, "text": "...", "level": "state", "confidence": None}],
  "summary": "...",
  "failure_description": "",
  "details": {}
}
```

Validation: frames are episode-local and within `[0, length)`; subtasks are
ordered and non-overlapping; events are ordered by `frame_index`;
`state_descriptions` are ordered by `start_frame`; every `text`, `event`,
`description`, and the `summary` are non-empty; confidences are within
`[0, 1]`; at least one time-aligned label exists; `details` is JSON
serializable; when a Task Profile is configured, every non-empty subtask
`stage` must be one of its stage names. Aliases accepted on input: `start`/`end`
for frame bounds, `subtask` for text, `stage_key` for stage, `frame` for
`frame_index`, `descriptions` for `state_descriptions`, `trajectory_summary`
for `summary`.

After validation the enrichment step compares interior subtask boundaries with
kinematic change points (gripper level changes and arm near-stops); fewer than
half aligned within `--boundary-tolerance-frames` records
`language_subtask_boundary_unaligned_review` in
`subtask_boundary_alignment.jsonl`.

### Built-in: `language_backends.kinematic_annotator:create_annotator`

Segments the episode at kinematic change points, labels segments with the
Task Profile's stages in order (merging surplus segments), emits a
`gripper_level_change` event per gripper transition, and writes templated
state descriptions. Configuration keys: `min_segment_frames` (5),
`gripper_index`, `max_segments`. Use it as a baseline or as the skeleton a
language-model annotator refines.

## Window trigger (feedback capture)

Configured under `feedback.automatic_trigger.plugin` in the feedback record
YAML; only valid in `event_buffer` mode. Contract in `policy_improvement/capture.py`:

```python
class MyWindowEvaluator:
    name = "my_window_evaluator"
    def evaluate_window(self, packets: Sequence[CapturedPacket]) -> TriggerSignal | Mapping[str, Any]: ...
```

`CapturedPacket(timestamp, payload, size_bytes)` wraps the recorder's
`DatasetPacket` (observation, executed and policy actions, phase). The result
mapping has `triggered` (bool), `source` (`manual`, `deterministic`,
`semantic`, `scheduled`; defaults to `semantic`), optional finite `score`,
`reason`, and JSON-serializable `details`. A triggered signal must name a
source. Evaluation runs in one background thread every
`interval_frames`; a trigger saves the buffered window plus
`post_trigger_seconds`, it never transfers robot control, and an exception
becomes a non-triggered signal with `reason="trigger_evaluator_error"`.

## Policy runner (feedback capture)

`record/recorder/policy_runner.py`:

```python
class PolicyRunner(Protocol):
    def reset(self) -> None: ...
    def select_action(self, observation: dict[str, Any], *, task: str) -> dict[str, Any]: ...
```

`observation` holds `<motor>.pos` floats, camera frames under the camera name,
and `<camera>_depth` arrays; the returned mapping must contain `<motor>.pos`
targets. Built-ins: `LeRobotPolicyRunner` (lazy `lerobot` import, selected when
`policy.path` is set) and `StubPolicyRunner` (holds position, used by `--mock`).
Pass your own runner to `recorder.feedback.run_feedback_loop` to drive another
policy stack.

## Handover adapter (feedback capture)

`policy_improvement/intervention.py`:

```python
class HandoverAdapter(Protocol):
    def prepare_takeover(self, robot, teleop, last_action) -> None: ...
    def begin_human_control(self, robot, teleop) -> None: ...
    def end_human_control(self, robot, teleop) -> None: ...
    def resume_autonomous(self, robot, teleop) -> None: ...
```

`prepare_takeover` runs while the follower holds its last commanded position; it
must return only when the leader is verified to match the follower, and raise
otherwise (the takeover is then cancelled and autonomous control continues).
Built-ins, selected with `feedback.handover`:

| `handover` | Adapter | Requires | Knobs |
| --- | --- | --- | --- |
| `capability` (default) | `CapabilityHandoverAdapter` | a leader with `send_feedback` and `feedback_features`; passive leaders are rejected unless `require_feedback_alignment: false` | `synchronization_seconds` |
| `position_match` | `PositionMatchHandoverAdapter` | any leader with `get_action()`; the operator moves it by hand until every joint is within tolerance | `handover_tolerance` (3.0, action units), `handover_timeout_s` (30) |

Neither built-in has been validated on every arm; verify emergency-stop
behaviour on your hardware before enabling takeover.

## Policy iteration: trainer, evaluator, deployer

Selected by `plugins.trainer|evaluator|deployer.spec` in the iteration config
read by `pipeline/policy_iteration.py`. Contracts in `policy_improvement/plugins.py`.

| Request | Fields | Result | Fields and rules |
| --- | --- | --- | --- |
| `TrainingRequest` | `iteration_id`, `base_policy`, `training_dataset: Path`, `output_dir: Path`, `config` | `TrainingResult` | `checkpoint` (non-empty str), `metrics` (finite numbers), `details` (JSON) |
| `EvaluationRequest` | `iteration_id`, `checkpoint`, `evaluation_dataset: Path \| None`, `config` | `EvaluationResult` | `metrics` (non-empty, finite numbers), `details` (JSON) |
| `DeploymentRequest` | `iteration_id`, `checkpoint`, `target`, `config` | `DeploymentResult` | `deployment_id` and `target` non-empty; `target` must equal the request's |

`plugins.<name>.init` is passed to the class or factory; `plugins.<name>.request`
becomes `request.config`. Neither block is copied into
`iteration_manifest.json`, so credentials stay out of the audit trail.

`PromotionPolicy` reads `metrics["success_rate"]`, `metrics["intervention_rate"]`,
and `metrics["critical_failures"]`; a missing or out-of-range metric blocks
promotion, as does a missing human approval file (`{"iteration_id": ...,
"approved": true}`) when `require_human_approval` is set. Optional
`baseline_success_rate` and `min_success_rate_delta` require an improvement
over the previous policy.

### Reference backends (`policy_backends/`)

| Spec | Behaviour | `init` keys |
| --- | --- | --- |
| `policy_backends.command_trainer:create_trainer` | runs a command line; placeholders `{iteration_id}`, `{base_policy}`, `{training_dataset}`, `{output_dir}` and every `request` key are formatted into `command`, `checkpoint`, and `metrics_file` (double literal braces as `{{`) | `command` (list or string), `checkpoint`, `metrics_file`, `cwd`, `env` |
| `policy_backends.offline_evaluator:create_evaluator` | scores `evaluation_dataset` with the VLA checkpoint evaluator; `success_rate` is the fraction of episodes at or above `pass_threshold` and is an offline proxy; measured `intervention_rate` and `critical_failures` must come from `extra_metrics` | `pass_threshold` (0.9), `evaluator_config`, `extra_metrics`, `max_episodes` |
| `policy_backends.filesystem_deployer:create_deployer` | copies the checkpoint to `<target_root>/<target>/<iteration_id>` and atomically rewrites `<target_root>/<target>/current.json` | `target_root`, `overwrite` (false) |

See `configs/policy_iteration.example.json` for a complete configuration.

## Feedback sidecar files

Written by `FeedbackSidecarWriter` (`policy_improvement/contracts.py`) next to,
never inside, the dataset:

| File | Content |
| --- | --- |
| `session_manifest.json` | schema version, session id, source dataset and policy, capture mode, fps, episode list, frame count |
| `frame_feedback.parquet` | one row per frame: `episode_index`, `frame_index`, `timestamp`, `phase` (`autonomous`, `syncing`, `recovery`, `correction`, `paused`), `intervention_id` (-1 outside interventions), `trigger_source`, `policy_action_json`, `executed_action_json`, `training_eligible`, `details_json` |
| `episode_feedback.jsonl` | per episode: `length`, `source_policy`, `capture_mode`, `task`, `outcome`, `failure_stage`, `intervention_count`, `review_status` (`pending`, `approved`, `rejected`), `notes`, `details` |
| `intervention_segments.jsonl` | per intervention: ordered `recovery_*` and `correction_*` frame ranges, `trigger_source`, `outcome`, `details` |

Invariants enforced on write and on load: frame indices are contiguous and
episode-local, human-controlled frames carry an intervention id and an executed
action, only recovery and correction frames may be training-eligible, and
segment ranges match the per-frame phases exactly.
