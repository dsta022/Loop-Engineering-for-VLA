# RGB-D recording (framework-free)

RGB-D demonstration capture for SO-100 / SO-101 arms with Orbbec or Intel
RealSense cameras. No lerobot dependency and no "copy into a lerobot checkout"
step — run the entrypoints directly from this repository.

RGB streams are stored as mp4 video; every camera with `use_depth: true` produces
an `<name>_depth` observation saved as a lossless **uint16 PNG sidecar** under
`depth_sidecar/`.

## Supported cameras

| Camera | config `type` | Depth | Notes |
| --- | --- | --- | --- |
| Orbbec Femto Bolt | `orbbec` | independent `depth_width/height/fps`, optional align-to-color | `pip install pyorbbecsdk2` |
| RealSense D435 / D435i | `intelrealsense` | shares color `width/height/fps` | 848x480@30 is the depth sweet spot |
| RealSense D405 | `intelrealsense` | shares color `width/height/fps` | near-range (~7cm-50cm), ideal wrist cam |
| UVC webcam | `opencv` | none | RGB only |

## Install

```powershell
python -m pip install -r record/requirements.txt
pip install pyorbbecsdk2      # and/or: pip install pyrealsense2
```

Find cameras and set serials/ports in the configs:

```powershell
python ./record/rgbd_record/find_cameras.py orbbec
python ./record/rgbd_record/find_cameras.py realsense
```

## Record

```powershell
python ./record/rgbd_record/record.py --config_path=./record/rgbd_record/configs/record_rgbd.yaml
```

All-RealSense rig (D435 front + D405 wrist):

```powershell
python ./record/rgbd_record/record.py --config_path=./record/rgbd_record/configs/record_rgbd_realsense.yaml
```

Pick cameras from the shared inventory instead of editing the config:

```powershell
python ./record/rgbd_record/record.py `
  --config_path=./record/rgbd_record/configs/record_rgbd.yaml `
  --camera_profile=realsense_d435_front_d405_wrist
```

Add `--mock` for an offline dry-run. Recording keys: **Right** = end episode,
**Left** = re-record, **Esc** = stop.

## Feedback (human-guided policy improvement)

```powershell
python ./record/rgbd_record/feedback_record.py `
  --config_path=./record/rgbd_record/configs/record_rgbd_feedback.yaml `
  --policy.path=./outputs/train/run/checkpoints/last/pretrained_model
```

The safe default is `feedback.mode: corrections_only`. Press `i` to take over and
begin **Recovery**, `c` to mark **Correction**, and `i` again to complete and hand
control back to the policy. Feedback provenance (control phase, intervention
ranges, the policy's proposed action) is written to a sidecar **outside** the
dataset (`feedback_outputs/...`), so autonomous failure actions never become
expert targets.

### Policy inference is pluggable

Teleoperated capture, intervention labelling, and the sidecar are 100%
framework-free. Only the in-loop policy inference needs a policy runtime, behind
`recorder.policy_runner.PolicyRunner`:

- **default**: `LeRobotPolicyRunner` lazily imports `lerobot` to run any lerobot
  checkpoint (`pip install lerobot`). Nothing else in the recorder imports lerobot.
- **custom**: implement `PolicyRunner` (`reset()` + `select_action(obs, task)`) to
  drive any other policy stack with zero lerobot dependency.
- **offline**: `--mock` uses a `StubPolicyRunner` (hold-position), useful for dry runs.

> Safe autonomous → human handover needs an actuated leader with position
> feedback. A passive SO leader requires
> `feedback.require_feedback_alignment: false` (accepting a manual-alignment
> warning) or a hardware-specific handover adapter.

Three capture modes (in `configs/record_rgbd_feedback.yaml`): `corrections_only`
(human frames only), `event_buffer` (retain a pre-trigger window), `continuous`
(retain autonomous frames as analysis-only data).

## Output

```text
data/           meta/           videos/
depth_sidecar/observation.images.front_depth/episode-000000/frame-000000.png
meta/rgbd_vla_depth_recording.json
```

A standard LeRobot v3.0 dataset. Visualize recorded depth:

```powershell
python ./tools/depth_visualize.py <dataset_root>/depth_sidecar --stats
```

Then run the toolkit's merge / audit / clean / upload steps.
