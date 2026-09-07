"""Run unittest discovery and surface every failure as a GitHub Actions annotation.

Usage: python .github/scripts/run_tests.py <start_dir> [<top_level_dir>]

The exit code and console output match ``python -m unittest discover``; in
addition each failing or erroring test is reported with an ``::error`` workflow
command so the failure and the tail of its traceback appear as check-run
annotations, which are readable without opening the raw log.
"""

from __future__ import annotations

import os
import sys
import unittest


def _escape(value: str, *, property_value: bool = False) -> str:
    escaped = value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    if property_value:
        escaped = escaped.replace(":", "%3A").replace(",", "%2C")
    return escaped


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: run_tests.py <start_dir> [<top_level_dir>]", file=sys.stderr)
        return 2
    start_dir = argv[1]
    top_level_dir = argv[2] if len(argv) > 2 else None
    # `python -m unittest` puts the working directory on sys.path; do the same so
    # `from tests.<module> import ...` inside the suites keeps working.
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    suite = unittest.TestLoader().discover(start_dir=start_dir, top_level_dir=top_level_dir)
    result = unittest.TextTestRunner(verbosity=1).run(suite)

    problems = [("FAIL", test, trace) for test, trace in result.failures]
    problems += [("ERROR", test, trace) for test, trace in result.errors]
    for kind, test, trace in problems:
        lines = [line.rstrip() for line in trace.strip().splitlines() if line.strip()]
        tail = " | ".join(lines[-8:])[:1500]
        title = _escape(f"{kind}: {test.id()}"[:250], property_value=True)
        print(f"::error title={title}::{_escape(tail)}")
    if problems:
        print(f"::error::{len(problems)} test problem(s) in {start_dir}; see annotations above")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
