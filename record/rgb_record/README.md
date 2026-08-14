# RGB recording (framework-free)

RGB-only demonstration capture for SO-100 / SO-101 arms. No lerobot dependency and
no "copy into a lerobot checkout" step — run the entrypoints directly from this
repository. Depth-enabled cameras/profiles are rejected here; use
[../rgbd_record/](../rgbd_record/) for depth.

Default cameras are two RGB webcams: `head` (fixed) and `wrist` (wrist-mounted).

## Install

```powershell
python -m pip install -r record/requirements.txt
```

`ffmpeg`/`ffprobe` must be available (PATH or conda `Library/bin`). Find OpenCV
camera indices, then set them in the configs:

```powershell
python ./record/rgbd_record/find_cameras.py opencv
```

Edit camera index/resolution/fps and robot/leader serial ports in:

```text
rgb_record/configs/head_wrist_rgb_record.yaml
rgb_record/configs/head_wrist_rgb_teleoperate.yaml
```

## Calibrate

Uses lerobot's calibration location, so existing calibrations load unchanged. To
calibrate a new arm:

```powershell
python ./record/rgb_record/rgb_teleoperate.py `
  --config_path=./record/rgb_record/configs/head_wrist_rgb_teleoperate.yaml --calibrate
```

## Teleoperate

```powershell
python ./record/rgb_record/rgb_teleoperate.py `
  --config_path=./record/rgb_record/configs/head_wrist_rgb_teleoperate.yaml
```

## Record

```powershell
python ./record/rgb_record/rgb_record.py `
  --config_path=./record/rgb_record/configs/head_wrist_rgb_record.yaml
```

Override any field inline, e.g. `--dataset.num_episodes=5
--dataset.single_task="Pick up the cup"`. Select cameras from the shared inventory
with `--camera_profile=rgb_head_wrist` (see [../camera_profiles.yaml](../camera_profiles.yaml)).
Add `--mock` for an offline dry-run.

Recording keys: **Right** = end episode, **Left** = re-record, **Esc** = stop.

## Output

A standard LeRobot v3.0 RGB dataset (`data/`, `meta/`, `videos/`) with no
`depth_sidecar/`. Run the toolkit's merge / audit / clean / upload steps next.

## Operational notes

- Keep camera names (`head`, `wrist`), fps, resolution, action space, and robot IDs
  stable across datasets you intend to merge.
- Set `dataset.push_to_hub: false` to audit and clean locally before uploading.
