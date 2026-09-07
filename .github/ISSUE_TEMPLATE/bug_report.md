---
name: Bug report
about: Something in the pipeline, recorder, or audit behaves incorrectly
labels: bug
---

## What happened

Describe the behaviour and the exact command you ran.

```powershell
python ./pipeline/data_quality_audit.py ./my_dataset ...
```

## What you expected

## Environment

- OS and Python version:
- Toolkit version (`python -c "import utils.version as v; print(v.__version__)"`):
- ffmpeg / ffprobe version (`ffmpeg -version | head -1`):
- Camera / robot hardware, if the recorder is involved:

## Dataset details

- Layout: recorded (aggregated parquet) / merged (one file per episode) / other
- `meta/info.json` totals (episodes, frames, tasks) and the feature keys
- Attach `quality_summary.json` and the relevant `quality_report.csv` rows when the audit is involved

## Logs

Paste the full traceback or the audit output here.
