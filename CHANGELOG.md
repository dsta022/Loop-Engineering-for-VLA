# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] - 2026-09-07

### Added

- `pyproject.toml` packaging with `vla-*` console commands; `pip install -e .`
  is now the supported setup.
- Test workflow running both unittest suites on Linux and Windows.
- `docs/plugins.md` documenting every plugin contract, and a project website
  under `docs/` deployed by GitHub Pages.
- Reference backends: `language_backends.kinematic_annotator`
  (rule-based trajectory language), `policy_backends.command_trainer`,
  `policy_backends.offline_evaluator`, and `policy_backends.filesystem_deployer`.
- `tools/threshold_calibration.py` measuring the precision and recall of every
  review reason and recommending semantic failure/pass thresholds from a
  human-labelled episode list.
- `--workers` for the quality audit: ffprobe, ffmpeg decode, and content
  sampling of unique video files run in parallel before the episode loop.
- Recorder `--resume=true` appends to an existing dataset instead of
  overwriting it, and a new run now refuses to overwrite a dataset root.
- Recorder `display_data: true` opens live OpenCV preview windows for every
  camera, including a colorized depth stream.
- `PositionMatchHandoverAdapter` (`feedback.handover: position_match`) for
  passive leader arms: control transfers only once every leader joint matches
  the follower within a tolerance.
- Multi-shard episode metadata (`meta/episodes/chunk-*/file-*.parquet`) is read
  by the audit, merge, clean build, feedback build, and upload validation.
- Audit summaries record `audit_script_version` and `audit_time_utc`; the
  cleaning manifest records the audit version, time, thresholds, and the
  reviewed episode list required by the strategy document.
- `CONTRIBUTING.md`, `CHANGELOG.md`, issue and pull-request templates,
  `.gitattributes`, and a License section in the README.

### Changed

- Every `pipeline/` and `tools/` script can be run by path
  (`python ./pipeline/dataset_merge.py`) without installing the project.
- Depth PNG decoding uses OpenCV when available and falls back to the
  pure-Python reference decoder.
- `hf_push.py` reads the token from `HF_TOKEN` or the cached
  `huggingface-cli login`; the in-source token constant is gone.
- Unused optional dependencies (`av`, `imageio`, `jsonlines`, `rerun-sdk`)
  were removed from `requirements.txt`.
- README shortened; procedural detail moved to `docs/`.

### Fixed

- Video content sampling failed on ffmpeg builds that removed the `-vsync`
  option; the audit now uses `-fps_mode vfr` and falls back to `-vsync vfr`
  on ffmpeg older than 5.1.
- Task Semantic Studio launched a non-existent script path when running the
  audit from the window.
- The studios no longer require `tkinter` just to import their command
  builders.

## [0.1.0] - 2026-08-25

Initial public release: framework-free RGB/RGB-D recorder, dataset merge,
conservative quality audit with optional semantic evaluation, clean build,
trajectory language enrichment, human feedback capture, policy iteration
contracts, and Hugging Face upload.
