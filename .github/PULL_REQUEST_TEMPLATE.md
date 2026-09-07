## Summary

What changes and why.

## Checklist

- [ ] `python -m unittest discover -s tests` passes
- [ ] `python -m unittest discover -s record/recorder/tests -t .` passes
- [ ] New checks are review-only unless the failure is objective, and report `not_checked` when their preconditions are absent
- [ ] Source datasets are never modified; new outputs are sidecars or new directories
- [ ] README / docs / CHANGELOG updated when behaviour or CLI flags change
