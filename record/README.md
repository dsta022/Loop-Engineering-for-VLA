# Recorder — framework-free LeRobot data collection

This folder collects SO-100 / SO-101 robot demonstrations and writes standard
**LeRobot v3.0** datasets **without depending on the lerobot framework**.

There is **no "copy this into a lerobot checkout" step** anymore. Everything the
recorder needs — Feetech motor control, camera capture (OpenCV / RealSense /
Orbbec), and the LeRobot v3.0 dataset writer — lives here in [recorder/](recorder/)
and reuses the toolkit's own lerobot-free helpers in [../utils/](../utils/) and
[../policy_improvement/](../policy_improvement/). Recorded datasets flow straight
into this toolkit's merge / audit / clean / upload pipeline.

## What's here

| Path | Purpose |
| --- | --- |
| [recorder/](recorder/) | self-contained recording library (cameras, motors, robot/teleop, dataset writer, record loop) |
| [rgb_record/](rgb_record/) | RGB-only recording + teleoperation entrypoints and configs |
| [rgbd_record/](rgbd_record/) | RGB-D recording, feedback recording, and configs |
| [visuo-tactile_record/](visuo-tactile_record/) | planned vision + touch path (docs only) |
| [camera_profiles.yaml](camera_profiles.yaml) / [camera_profiles.py](camera_profiles.py) | shared camera inventory + `--camera_profile` resolver |

## Install

```powershell
python -m pip install -r record/requirements.txt
```

Base install is **torch-free**. Add only what your hardware needs:

```powershell
pip install pyrealsense2        # Intel RealSense
pip install pyorbbecsdk2        # Orbbec
pip install huggingface_hub     # push_to_hub upload
pip install lerobot             # ONLY for feedback_record's in-loop policy inference
```

`ffmpeg` / `ffprobe` must be on `PATH` (or in your conda env's `Library/bin`) for
video encoding.

## Quick start

Calibration is stored in lerobot's own location
(`~/.cache/huggingface/lerobot/calibration/...`), so arms already calibrated with
lerobot work unchanged. To calibrate from scratch:

```powershell
python ./record/rgb_record/rgb_teleoperate.py --config_path=./record/rgb_record/configs/head_wrist_rgb_teleoperate.yaml --calibrate
```

Teleoperate (leader → follower preview, no recording):

```powershell
python ./record/rgb_record/rgb_teleoperate.py --config_path=./record/rgb_record/configs/head_wrist_rgb_teleoperate.yaml
```

Record an RGB dataset:

```powershell
python ./record/rgb_record/rgb_record.py --config_path=./record/rgb_record/configs/head_wrist_rgb_record.yaml
```

Record an RGB-D dataset (depth stored as uint16 PNG sidecars):

```powershell
python ./record/rgbd_record/record.py --config_path=./record/rgbd_record/configs/record_rgbd.yaml
```

During recording: **Right arrow** ends the episode, **Left arrow** re-records it,
**Esc** stops the session. Without a keyboard/`pynput`, episodes end by time
(`dataset.episode_time_s`).

### Camera selection

Edit hardware facts once in [camera_profiles.yaml](camera_profiles.yaml) and pick a
set at record time instead of editing the config's `cameras:` block:

```powershell
python ./record/rgbd_record/record.py `
  --config_path=./record/rgbd_record/configs/record_rgbd.yaml `
  --camera_profile=realsense_d435_front_d405_wrist
```

Any field can also be overridden inline, e.g. `--dataset.num_episodes=5
--dataset.single_task="Pick up the cup"`.

### Feedback (human-guided policy improvement)

```powershell
python ./record/rgbd_record/feedback_record.py `
  --config_path=./record/rgbd_record/configs/record_rgbd_feedback.yaml `
  --policy.path=./outputs/train/run/checkpoints/last/pretrained_model
```

Teleoperated capture, intervention labelling, and the feedback sidecar are fully
framework-free. Only the in-loop policy inference uses a policy runtime, isolated
behind `recorder.policy_runner.PolicyRunner` (default: a lazy `lerobot` adapter —
supply your own runner for any other policy stack). See
[rgbd_record/README.md](rgbd_record/README.md).

## Offline dry-run (no hardware)

Add `--mock` to any entrypoint to synthesize cameras and joints and exercise the
full pipeline (useful for CI and format checks):

```powershell
python ./record/rgbd_record/record.py --config_path=./record/rgbd_record/configs/record_rgbd.yaml --mock --dataset.root=./outputs/mock --dataset.episode_time_s=1
python ./pipeline/data_quality_audit.py ./outputs/mock
```

## Dataset output

A standard LeRobot v3.0 dataset:

```text
meta/info.json  meta/tasks.parquet  meta/stats.json  meta/episodes/...
data/chunk-000/file-000.parquet         # per-episode state/action + depth struct
videos/<key>/chunk-000/file-000.mp4     # per-episode RGB video
depth_sidecar/<key>/episode-XXXXXX/frame-XXXXXX.png   # uint16 depth (RGB-D only)
```

After recording, run this toolkit's merge, audit, clean, and upload steps on the
dataset root.

## Tests

```powershell
python -m unittest discover -s record/recorder/tests -t record
```
