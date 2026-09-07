[![Loop Engineering for VLAnything website: Better data. Better policies.](assets/readme/site_hero.png)](https://dsta022.github.io/Loop-Engineering-for-VLA/)

# Loop Engineering for VLAnything

[![Website](https://img.shields.io/badge/Website-dsta022.github.io-111111)](https://dsta022.github.io/Loop-Engineering-for-VLA/)
[![Tests](https://github.com/dsta022/Loop-Engineering-for-VLA/actions/workflows/tests.yml/badge.svg)](https://github.com/dsta022/Loop-Engineering-for-VLA/actions/workflows/tests.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-DerekLX%2Flerobot__derek__depth-yellow)](https://huggingface.co/datasets/DerekLX/lerobot_derek_depth)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![LeRobot](https://img.shields.io/badge/Data-LeRobot%20RGB%2FRGB--D-green)](https://github.com/huggingface/lerobot)

Website: https://dsta022.github.io/Loop-Engineering-for-VLA/.

## Multimodal VLA Dataset Toolkit

An end-to-end toolkit for collecting, merging, auditing, enriching, and
iteratively improving multimodal LeRobot datasets for vision-language-action
models. It supports standard RGB-only LeRobot datasets and RGB-D datasets with
lossless depth sidecars, and it treats RGB video, robot actions, states,
metadata, task semantics, language annotations, and human feedback as one
auditable engineering loop: **Capture · Audit · Enrich · Improve**.

The offline pipeline never imports LeRobot or camera SDKs. The recorder in
`record/` is a self-contained, framework-free data collection layer that talks
to SO-100/SO-101 arms and cameras directly and writes LeRobot v3.0 datasets:

- `record/rgb_record/`: RGB-only recording.
- `record/rgbd_record/`: Orbbec / Intel RealSense RGB-D recording, plus
  human-guided policy feedback.
- `record/visuo-tactile_record/`: visuo-tactile recording, in progress.

Recorded datasets flow straight into the merge, audit, clean, enrich, feedback,
and upload steps below. Three guiding rules run through every stage:

1. **Conservative decisions.** Only objective structural, numerical, or decoding
   failures become `drop`; semantic and threshold-based findings are `review`;
   a check with missing preconditions records `not_checked` instead of passing.
2. **Sidecars, never in-place edits.** Depth, feedback provenance, language,
   training weights, and reports live outside the dataset; source data is
   read-only.
3. **Pluggable, model-free by default.** Evaluators, annotators, triggers, and
   the train/evaluate/deploy trio load by `module:attribute`; the base install
   downloads no weights.

## Dataset

The merged RGB-D VLA dataset is published on Hugging Face as
[DerekLX/lerobot_derek_depth](https://huggingface.co/datasets/DerekLX/lerobot_derek_depth).
Its root contains the LeRobot subdirectories directly:

```text
data/
depth_sidecar/
meta/
videos/
```

## Dataset Preview

Samples from one `pick_up_cups_dataset` episode with two cups in the scene.

![Dataset preview: global robot collection view, wrist camera view, and front depth visualization](assets/readme/three_view_preview.png)

Left to right: global robot collection view, wrist camera view, and front depth visualization.

<video src="assets/readme/task_demo.mp4" controls width="720"></video>

If the video does not render in your Markdown viewer, open [task_demo.mp4](assets/readme/task_demo.mp4) directly.

| Front camera, 9 sampled episodes | Wrist camera, 9 sampled episodes | Front depth, 9 sampled episodes |
| --- | --- | --- |
| ![Front camera episode matrix](assets/readme/front_episode_matrix.jpg) | ![Wrist camera episode matrix](assets/readme/wrist_episode_matrix.jpg) | ![Front depth episode matrix](assets/readme/depth_episode_matrix.png) |

## Repository Layout

```text
pipeline/               # entry points: merge, audit, clean, feedback build,
                        #   language enrichment, policy iteration, HF upload
record/                 # framework-free recorder: recorder/ library, camera_profiles,
                        #   rgb_record/, rgbd_record/ (+ feedback), visuo-tactile_record/
tools/                  # studios, depth visualization, completeness check,
                        #   threshold calibration
utils/                  # shared audit, metadata, media, and semantic helpers
semantic_backends/      # built-in --semantic-evaluator plugins (VLA checkpoint scoring)
language_backends/      # built-in --annotator plugins (kinematic segment annotator)
policy_backends/        # reference trainer / evaluator / deployer plugins
policy_improvement/     # feedback capture, intervention, and policy iteration contracts
tests/                  # unit tests with synthetic datasets
configs/                # example configuration files
docs/                   # project website (GitHub Pages source), plugin contracts,
                        #   audit strategy, and the operating playbook
.github/workflows/      # tests on Linux and Windows; GitHub Pages deployment
assets/                 # README media
data/                   # local datasets (not tracked by git)
```

## Getting Started

Python 3.12 is recommended. Create an environment, install the project in
editable mode, and add `ffmpeg`/`ffprobe` for video decoding and trimming:

```powershell
conda create -n vla_data_check python=3.12 -y
conda activate vla_data_check
conda install -c conda-forge ffmpeg -y
python -m pip install -e ".[record]"
```

The editable install exposes console commands and lets every script run by
path from any directory:

| Command | Script | Purpose |
| --- | --- | --- |
| `vla-merge` | `pipeline/dataset_merge.py` | merge datasets into one real dataset |
| `vla-audit` | `pipeline/data_quality_audit.py` | conservative quality audit |
| `vla-clean` | `pipeline/dataset_clean_build.py` | build a clean dataset from a drop list |
| `vla-enrich` | `pipeline/trajectory_language_enrichment.py` | language sidecars for approved episodes |
| `vla-feedback-build` | `pipeline/feedback_dataset_build.py` | weighted feedback training dataset |
| `vla-iterate` | `pipeline/policy_iteration.py` | train, evaluate, promote, deploy |
| `vla-push` | `pipeline/hf_push.py` | upload to Hugging Face |
| `vla-completeness` | `tools/dataset_completeness_check.py` | structure-only check |
| `vla-calibrate` | `tools/threshold_calibration.py` | review-reason precision and threshold recommendation |
| `vla-depth-vis` | `tools/depth_visualize.py` | depth PNG statistics and previews |
| `vla-task-studio`, `vla-language-studio` | `tools/*_studio.py` | local windows for profiles and enrichment |

Optional extras: `.[realsense]`, `.[orbbec]` for camera SDKs and `.[policy]`
for the lerobot runtime used by the VLA checkpoint evaluator, the feedback
recorder's in-loop policy, and the offline policy evaluator. Faster Hugging
Face uploads: `$env:HF_XET_HIGH_PERFORMANCE="1"`.

## Record Data

Camera hardware facts live once in `record/camera_profiles.yaml`; select a set
at record time with `--camera_profile=<name>`, override any field inline
(`--dataset.num_episodes=5`), or add `--mock` for an offline dry run with
synthetic cameras and joints. Calibration uses lerobot's cache location, so
arms calibrated with lerobot work unchanged; add `--calibrate` to a teleoperate
command to calibrate a new arm. Recording keys: Right = end episode, Left =
re-record, Esc = stop. Set `display_data: true` to open a live preview window
per camera (depth is colorized next to RGB; needs a GUI build of OpenCV).
Recording into an existing dataset root is refused unless you pass
`--resume=true`, which appends episodes.

### RGB

```powershell
python .\record\rgbd_record\find_cameras.py opencv
python .\record\rgb_record\rgb_teleoperate.py --config_path=.\record\rgb_record\configs\head_wrist_rgb_teleoperate.yaml
python .\record\rgb_record\rgb_record.py --config_path=.\record\rgb_record\configs\head_wrist_rgb_record.yaml
```

Depth-enabled cameras are rejected on this path. The output is a standard
LeRobot RGB dataset with `data/`, `meta/`, and `videos/`.

### RGB-D (Orbbec / RealSense)

```powershell
pip install pyorbbecsdk2      # and/or: pip install pyrealsense2
python .\record\rgbd_record\find_cameras.py orbbec
python .\record\rgbd_record\find_cameras.py realsense
python .\record\rgbd_record\record.py --config_path=.\record\rgbd_record\configs\record_rgbd.yaml
```

Supported backends are `orbbec` (Femto Bolt), `intelrealsense` (D405 / D435 /
D435i), and `opencv`. `record_rgbd_realsense.yaml` is a ready all-RealSense
rig. Every camera with `use_depth: true` writes lossless uint16 PNG depth
frames under `depth_sidecar/` with a manifest at
`meta/rgbd_vla_depth_recording.json`. See [record/README.md](record/README.md).

## Merge Datasets

```powershell
python ./pipeline/dataset_merge.py --src-root . --out-dir .\lerobot_derek_depth --dataset-glob "*_dataset" --copy-mode copy
```

The merge rewrites episode, frame, and task indices, metadata, parquet
tables, video references, and depth sidecars into one physical dataset with
real files. Episode metadata split across several
`meta/episodes/chunk-*/file-*.parquet` shards is read as a whole. Add
`--overwrite` to replace an existing output.

## Audit Dataset Quality

```powershell
python ./pipeline/data_quality_audit.py .\lerobot_derek_depth `
  --depth-check header --video-check decode --video-content-check sample `
  --sample-frames 8 --depth-sample-frames 8 --workers 4
```

Deterministic checks cover dataset metadata, episode ranges, index and
timestamp integrity, action/state numerics, depth PNG structure and content,
RGB video existence, ffprobe dimensions and codec, full decode, and sampled
black/white/near-constant/frozen content confined to each episode's own time
range. Distribution outliers, exact trajectory duplicates, and low motion are
review-only. `--workers` runs ffprobe, decode, and content sampling for unique
video files in parallel before the episode loop; verdicts are unchanged.

Outputs in `quality_audit_reports/<dataset_name>/` (or `--out-dir`):

```text
quality_report.csv      # one row per episode: status, reasons, not_checked, metrics_json
quality_summary.json    # global status, thresholds, audit_script_version, fingerprint
keep_episodes.txt / review_episodes.txt / drop_episodes.txt
```

Semantic evaluation is expensive, so run the audit in two stages during
iteration: a deterministic pass first, then `--semantic-only` on top of the
saved report:

```powershell
python ./pipeline/data_quality_audit.py .\lerobot_derek_depth --semantic-only `
  --semantic-evaluator my_quality_backend:create_evaluator `
  --semantic-failure-threshold 0.5 --semantic-pass-threshold 0.9
```

The rerun preserves every structural finding, replaces the previous
`semantic_*` results, and records `audit_mode: semantic_reuse`. Reuse is refused when the dataset fingerprint, episode set, or
lengths changed, and `--feedback-sidecar` is rejected in that mode. Run one
final single-pass audit before publishing. The full six-stage procedure is in
[docs/audit_operations_playbook.md](docs/audit_operations_playbook.md); the
decision policy is in [docs/data_quality_audit_strategy.md](docs/data_quality_audit_strategy.md).

## Task Semantic Profiles

```powershell
python ./tools/task_semantic_studio.py
```

The studio turns a detailed task description into a validated, editable
profile: task key, name, instruction, aliases, objects, ordered stages with
observable conditions and weights, success and failure criteria, preferred
video keys, and optional threshold overrides. Profiles are saved as JSON and
passed to the audit and the enrichment step with `--task-profile-dir`. With a
profile directory configured, an episode whose task text matches no profile,
or that contains several distinct tasks, is sent to review. The built-in
parser is a conservative heuristic; install a stronger parser as a
`module:attribute` plugin.

## Semantic Evaluation

An optional evaluator receives the task text, episode-scoped video segments,
the Task Profile, and the action/state streams, and returns a progress curve
and/or a success probability:

```python
class MySemanticEvaluator:
    name = "my_video_language_evaluator"

    def evaluate(self, episode):
        return {"progress_curve": [0.05, 0.30, 0.62], "success_probability": 0.41,
                "failure_stage": "grasp", "details": {"counterfactual_margin": 0.35}}
```

```powershell
python ./pipeline/data_quality_audit.py .\lerobot_derek_depth `
  --semantic-evaluator my_quality_backend:create_evaluator --semantic-config .\semantic_evaluator.json `
  --semantic-failure-threshold 0.5 --semantic-pass-threshold 0.9 --task-profile-dir .\task_profiles
```

Decision bands are conservative: below the failure threshold is
`semantic_low_score_review`, below the pass threshold is
`semantic_uncertain_score_review`, and only scores at or above the pass
threshold can enter `keep`. Evaluator errors go to review, structural drops
skip inference, and three optional reliability contracts demote unstable
scores: sampling dispersion (`score_samples` / `score_std`), the
counterfactual margin over deliberately corrupted instructions, and per-view
agreement with `--semantic-per-view`. Every field, rule, and threshold is
documented in [docs/plugins.md](docs/plugins.md).

**Built-in backend.** `semantic_backends.vla_checkpoint_evaluator:create_evaluator`
turns a trained lerobot policy checkpoint into a data-quality scorer. At
sampled anchors the policy predicts an action chunk from the camera frames,
joint state, and task text; the score measures how much better than a
hold-position baseline it reproduces the demonstrated actions (1.0 reproduces
the demonstration). The same anchors are re-scored under corrupted
instructions to fill the counterfactual margin.

```powershell
python ./pipeline/data_quality_audit.py .\lerobot_derek_depth --semantic-only `
  --semantic-evaluator semantic_backends.vla_checkpoint_evaluator:create_evaluator `
  --semantic-config .\configs\vla_checkpoint_evaluator.example.json
```

It needs the policy runtime (`pip install -e ".[policy]"`), so run this stage
in the training environment. A low score can also mean the episode is out of
distribution for the current policy; calibrate the thresholds before trusting
automatic passes.

## Model-Free Consistency Checks

A second group of semantic checks needs no model. Each is review-only and
records `not_checked` when its inputs are missing:

- Progress-curve shape (needs an evaluator): `semantic_progress_regression_review`,
  `semantic_progress_idle_tail_review`, `semantic_progress_idle_head_review`.
- Stage ordering (needs an evaluator and a Task Profile):
  `semantic_stage_order_violation_review`, `semantic_stage_skipped_review`,
  `semantic_stage_unknown_review`, `semantic_stage_dwell_outlier_review`.
- Action/language consistency (needs nothing): `semantic_motion_gripper_pattern_review`
  compares the verb in the task text with the observed gripper actuation, and
  `semantic_motion_vertical_direction_review` checks directional verbs against
  `--motion-vertical-index`. The gripper channel is auto-detected or set with
  `--motion-gripper-index`; `--motion-check none` turns the group off.

Counts per check are written to the `semantic_consistency` block of
`quality_summary.json` so each check's precision can be measured on its own.

## Calibrate Thresholds

Review reasons and semantic bands are conservative defaults, not calibrated
classifiers. After a human has labelled a set of episodes, measure them:

```powershell
python ./tools/threshold_calibration.py `
  --quality-report .\quality_audit_reports\lerobot_derek_depth\quality_report.csv `
  --labels .\lerobot_derek_depth_labels.csv
```

The labels file is a CSV with `episode_index` and a `label` column (`reject`,
`fail`, `bad`, `drop`, `0`, `false`, `no` mean rejected). The tool prints the
precision and recall of every review and drop reason, the keep/review/drop
confusion against the labels, and recommends `--semantic-failure-threshold`
and `--semantic-pass-threshold` values that meet `--precision-target` and
`--pass-false-rate`. Results are written to `calibration_report.json`.

## Trajectory Language Enrichment

The third layer describes what happens inside an approved trajectory. It runs
after quality review so annotation compute is spent only on usable data:

```powershell
python ./pipeline/trajectory_language_enrichment.py .\lerobot_derek_depth `
  --task-profile-dir .\task_profiles `
  --annotator language_backends.kinematic_annotator:create_annotator `
  --approved-episodes .\approved_episodes.txt
```

By default only `keep` episodes are processed; `--approved-episodes` adds
human-approved `review` episodes, and `drop` is always rejected. The quality
report defaults to `quality_audit_reports/<dataset_name>/quality_report.csv`,
and its `quality_summary.json` must belong to the selected dataset. Outputs are
external sidecars under `language_enrichment_outputs/<dataset_name>/`:
per-episode annotation JSON files, `language_annotations.jsonl`,
`subtasks.jsonl`, `events.jsonl`, `state_descriptions.jsonl`,
`subtask_boundary_alignment.jsonl`, and `enrichment_manifest.json`.

An annotator returns time-aligned subtasks, events, state descriptions, and a
summary; every annotation is validated for frame bounds, ordering,
non-overlap, required text, confidence ranges, and Task Profile stage names,
and subtask boundaries are checked against kinematic change points
(`language_subtask_boundary_unaligned_review`). The built-in kinematic
annotator segments episodes at gripper transitions and arm near-stops and
labels them with Task Profile stages; treat it as a baseline for a
language-model annotator. `python ./tools/trajectory_language_studio.py` opens
a window for the same settings. Contract details: [docs/plugins.md](docs/plugins.md).

## Closed-Loop Policy Improvement

The fourth layer collects targeted human feedback from states a deployed policy
visits, keeps control provenance outside the dataset, and gates the next
policy on evidence plus human approval.

### Collect human-guided feedback

```powershell
python ./record/rgbd_record/feedback_record.py `
  --config_path=./record/rgbd_record/configs/record_rgbd_feedback.yaml `
  --policy.path=./outputs/train/run/checkpoints/last/pretrained_model
```

Keys: `i` pauses the policy, aligns the teleoperator, and begins Recovery;
`c` marks the first Correction frame; `i` again completes the intervention,
saves the episode, and returns control to the policy; `esc` requests a clean
stop that never cuts an intervention in half. Capture modes in the config:
`corrections_only` (recommended; only Recovery/Correction frames are stored and
training-eligible), `event_buffer` (keeps a bounded pre-event window; `h`
saves a manual event; an optional `automatic_trigger` plugin can flag windows
in a background thread without transferring control), and `continuous`.
Autonomous frames are never expert targets.

Handover to the human is verified before control transfers. `handover:
capability` drives an actuated leader to the follower pose; `handover:
position_match` supports a passive leader: the follower holds still while the
operator moves the leader by hand, and control transfers only once every joint
is within `handover_tolerance`. Policy inference sits behind
`recorder.policy_runner.PolicyRunner`; the default adapter runs any lerobot
checkpoint, and `--mock` uses a stub policy.

Each session writes `feedback_outputs/<dataset>/<session>/` with
`session_manifest.json`, `frame_feedback.parquet`, `episode_feedback.jsonl`, and
`intervention_segments.jsonl`.

### Audit and build the feedback dataset

```powershell
python ./pipeline/data_quality_audit.py ./data/so100_rgbd_feedback `
  --feedback-sidecar ./feedback_outputs/so100_rgbd_feedback/session-001
python ./pipeline/feedback_dataset_build.py `
  --base-dataset ./data/original_clean_dataset `
  --feedback-dataset ./data/so100_rgbd_feedback `
  --feedback-sidecar ./feedback_outputs/so100_rgbd_feedback/session-001 `
  --quality-report ./quality_audit_reports/so100_rgbd_feedback/quality_report.csv `
  --approved-episodes ./approved_feedback_episodes.txt `
  --language-enrichment ./language_enrichment_outputs/so100_rgbd_feedback `
  --out-dir ./iteration_datasets/iteration-001
```

The audit validates sidecar alignment (episode lengths, contiguous frames,
intervention ranges, phase labels, training eligibility) and sends new
`pending` feedback to review until a human approval list exists. The builder
accepts corrections-only episodes, refuses episodes with autonomous frames
until they are segmented, and writes `training_sidecars/training_weights.parquet`
(base demonstrations 1.0, Recovery 1.0, Correction 2.0 by default) plus a
manifest confirming that no autonomous failure action became an expert target.

### Train, evaluate, promote, deploy

```powershell
python ./pipeline/policy_iteration.py ./configs/policy_iteration.example.json
```

Three plugins own the model-specific work. The reference backends in
`policy_backends/` run a configurable training command, score the checkpoint
offline on a held-out dataset with the VLA checkpoint evaluator, and publish
it into a target directory with an atomic `current.json` pointer. Promotion
requires the configured success rate, intervention rate, and critical-failure
limits plus a separate human approval file; deployment is skipped when any
gate fails, and `iteration_manifest.json` records every stage. The offline
evaluator's `success_rate` is a proxy computed from demonstrations; supply
measured `intervention_rate` and `critical_failures` from real trials.

## Check Completeness

```powershell
python ./tools/dataset_completeness_check.py .\lerobot_derek_depth
```

Runs the structural checks only. Useful right after a merge or clean build.

## Build a Clean Dataset

```powershell
python ./pipeline/dataset_clean_build.py .\lerobot_derek_depth `
  --drop-list .\quality_audit_reports\lerobot_derek_depth\drop_episodes.txt `
  --out-dir .\lerobot_derek_depth_clean --video-mode trim-reencode
```

The source dataset is preserved. `trim-reencode` physically removes dropped
video segments; `copy-referenced` is faster but may keep unreferenced segments
inside copied files. The output's `cleaning_manifest.json` records the source,
the kept, reviewed, and dropped episodes with their reasons, the index
mappings, and the audit version, time, and thresholds that produced the drop
list.

## Upload to Hugging Face

```powershell
$env:HF_TOKEN="hf_your_write_token"          # or: huggingface-cli login
$env:HF_XET_HIGH_PERFORMANCE="1"
python ./pipeline/hf_push.py --dry-run
python ./pipeline/hf_push.py --dataset-dir .\lerobot_derek_depth_clean --dataset-name lerobot_derek_depth_clean
```

The token comes from `HF_TOKEN` or the cached login; it is never stored in the
repository. Uploads validate the dataset layout, refuse junction, symlink, or
reparse paths, and use resumable large-folder uploads when available.

## Visualize Depth

```powershell
python ./tools/depth_visualize.py .\lerobot_derek_depth\depth_sidecar --stats
python ./tools/depth_visualize.py .\lerobot_derek_depth\depth_sidecar --out .\depth_vis
```

## Run Tests

```powershell
python -m unittest discover -s tests
python -m unittest discover -s record/recorder/tests -t .
```

Both suites use synthetic datasets only; tests that need `ffmpeg` skip
themselves when it is missing. The same suites run on Linux and Windows in
GitHub Actions for every push and pull request.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup, the
conventions (conservative decisions, sidecars, framework-free core), and the
pull-request checklist. Changes are tracked in [CHANGELOG.md](CHANGELOG.md).

## Dataset Citation

```bibtex
@dataset{dereklx_lerobot_derek_depth_2026,
  author    = {DerekLX},
  title     = {lerobot_derek_depth},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/DerekLX/lerobot_derek_depth}
}
```

## License

This project is released under the [MIT License](LICENSE).
