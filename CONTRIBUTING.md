# Contributing

Thanks for helping improve Loop Engineering for VLAnything. This page covers the
development setup, the conventions the code base follows, and what a pull
request needs.

## Development setup

```powershell
conda create -n vla_data_check python=3.12 -y
conda activate vla_data_check
conda install -c conda-forge ffmpeg -y
python -m pip install -e ".[record]"
```

The editable install exposes the `vla-*` console commands and makes every
`python ./pipeline/<script>.py` invocation work from any directory. Robot-side
SDKs (`pyrealsense2`, `pyorbbecsdk2`) and the policy runtime (`lerobot`) are
optional extras and are never imported by the offline pipeline.

## Running the tests

```powershell
python -m unittest discover -s tests
python -m unittest discover -s record/recorder/tests -t .
```

Both suites use synthetic datasets only. Tests that need `ffmpeg` skip
themselves when it is missing, so install it to get full coverage. CI runs both
suites on Linux and Windows for every push and pull request.

## Conventions

- **Conservative audit policy.** Only objective, deterministic failures become
  `drop`. Semantic, statistical, and threshold-based findings are `review`.
  A check whose preconditions are missing records a `not_checked` entry
  instead of passing silently.
- **Sidecars, never in-place edits.** Depth, feedback provenance, language
  annotations, training weights, and audit reports live outside the LeRobot
  dataset. Source datasets are read-only; tools write new directories.
- **Plugins load by `module:attribute`.** Evaluators, annotators, triggers,
  parsers, and the policy-iteration trio follow the contracts in
  [docs/plugins.md](docs/plugins.md). A class receives its configuration as
  keyword arguments, a factory receives one dictionary, an instance is used
  as-is.
- **Framework-free core.** `pipeline/`, `utils/`, `policy_improvement/`, and
  `record/recorder` must not import `lerobot` or camera SDKs at module import
  time. Heavy runtimes are imported lazily inside the adapter that needs them.
- **Reports are machine-readable.** New review or drop reasons use stable
  snake_case tags; semantic reasons carry the `semantic_` prefix so
  `--semantic-only` can replace them on a rerun.

## Pull requests

1. Open an issue first for larger changes so the design can be discussed.
2. Keep pull requests focused; unrelated refactors are easier to review apart.
3. Add or extend a synthetic-dataset test for every behaviour change.
4. Update the README, `docs/`, and `CHANGELOG.md` when flags, outputs, or
   contracts change.
5. Bump `utils/version.py` for user-visible changes; the audit summary and
   cleaning manifest record that version.
