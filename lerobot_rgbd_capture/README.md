# LeRobot Orbbec RGB-D Capture

This folder contains the LeRobot-side code needed to collect RGB-D data with an Orbbec Femto Bolt camera.

It extends the data pipeline upstream of merge, audit, clean, and upload by adding a real robot data collection path.

## Scope

This folder is only for data collection. It adds Orbbec RGB-D capture support, recording scripts, and lossless depth sidecar storage helpers.

It does not contain this project's depth model training code, depth model inference code, evaluation config, or pretrained model weights. The recorder follows LeRobot's normal recording interfaces, but the included project-local Python package only provides RGB-D data helpers.

## Where To Put It In LeRobot

Copy the contents of this folder into the root of a `huggingface/lerobot` checkout. The target is the directory that contains LeRobot's own `pyproject.toml` and `src/lerobot/`.

Do not copy this folder as:

```text
<lerobot_root>/lerobot_rgbd_capture/
```

Merge these contents into the LeRobot repo root instead:

```powershell
$LEROBOT_ROOT="<path-to-lerobot>"
Copy-Item -Recurse -Force .\lerobot_rgbd_capture\examples $LEROBOT_ROOT\
Copy-Item -Recurse -Force .\lerobot_rgbd_capture\src $LEROBOT_ROOT\
Copy-Item -Force .\lerobot_rgbd_capture\pyproject.toml $LEROBOT_ROOT\
```

After copying, the expected layout is:

```text
<lerobot_root>/examples/rgbd_vla_depth/
<lerobot_root>/src/lerobot/cameras/orbbec/
<lerobot_root>/src/lerobot/scripts/
<lerobot_root>/src/lerobot/robots/so_follower/
<lerobot_root>/pyproject.toml
```

Install from the LeRobot repo root:

```powershell
cd $LEROBOT_ROOT
python -m pip install -e ".[feetech,orbbec,rgbd_vla]" -e .\examples\rgbd_vla_depth
```

## What It Provides

- `src/lerobot/cameras/orbbec/`: Orbbec RGB-D camera backend.
- `src/lerobot/scripts/`: patched LeRobot commands for finding cameras, recording, calibration, and teleoperation.
- `src/lerobot/robots/so_follower/`: SO follower support for RGB-D observations.
- `examples/rgbd_vla_depth/record.py`: RGB-D recording entrypoint.
- `examples/rgbd_vla_depth/configs/record_rgbd.yaml`: example Orbbec + wrist camera recording config.
- `examples/rgbd_vla_depth/src/lerobot_policy_rgbd_vla_depth/data/`: helpers for depth feature naming and uint16 PNG sidecar storage.

## Dataset Output

The recorder keeps RGB streams in standard LeRobot video storage and saves depth frames as lossless uint16 PNG files:

```text
data/
videos/
depth_sidecar/observation.images.front_depth/episode-000000/frame-000000.png
meta/rgbd_vla_depth_recording.json
```

## Typical Entry Points

Find the Orbbec camera:

```powershell
lerobot-find-cameras orbbec
```

Record RGB-D data:

```powershell
python .\examples\rgbd_vla_depth\record.py --config_path=.\examples\rgbd_vla_depth\configs\record_rgbd.yaml
```

Visualize recorded depth after collection from this repository root:

```powershell
python .\visualize_lerobot_depth.py <dataset_root>\depth_sidecar --stats
```

The main config to edit before recording is:

```text
examples/rgbd_vla_depth/configs/record_rgbd.yaml
```

Important fields in `record_rgbd.yaml`:

```yaml
robot:
  cameras:
    front:
      type: orbbec
      serial_number_or_name: "YOUR_FEMTO_BOLT_SERIAL"
      use_depth: true
      align_depth_to_color: true
dataset:
  repo_id: YOUR_HF_USERNAME/so100_rgbd_dataset
  fps: 30
  num_episodes: 10
  episode_time_s: 60
```
