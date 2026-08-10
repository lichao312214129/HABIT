#!/usr/bin/env python
"""
Smoke-run lightweight Examples gallery scripts (synthetic data only).

Skips demos that require demo_data/, heavy radiomics, or interactive viewers
unless HABIT_NO_VIEW=1 is already set. Exit code is non-zero on first failure.

Run from the repository root::

    set HABIT_NO_VIEW=1
    python docs/source/examples/scripts/_smoke_examples.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent

# Keep this list fast and offline (synthetic / no demo_data).
LIGHTWEIGHT = (
    "data_from_arrays_demo.py",
    "habitat_atomic_ops_demo.py",
    "habitat_custom_pipeline_demo.py",
    "provenance_methods_demo.py",
    "plugin_entry_points_demo.py",
    "two_step_habitat_quickstart.py",
    "one_step_habitat_demo.py",
    "direct_pooling_habitat_demo.py",
    "fault_tolerance_demo.py",
    "persistence_demo.py",
    "apply_saved_model_demo.py",
    "feature_composition_demo.py",
    "tabular_ml_quickstart.py",
)


def main() -> int:
    """Execute each lightweight demo; return the number of failures."""
    os.environ.setdefault("HABIT_NO_VIEW", "1")
    python = sys.executable
    failures = 0
    for name in LIGHTWEIGHT:
        path = SCRIPTS_DIR / name
        if not path.is_file():
            print(f"[MISSING] {name}")
            failures += 1
            continue
        print(f"[RUN] {name}")
        completed = subprocess.run(
            [python, str(path)],
            cwd=str(path.parents[3]),
            check=False,
        )
        if completed.returncode != 0:
            print(f"[FAIL] {name} exit={completed.returncode}")
            failures += 1
        else:
            print(f"[OK]   {name}")
    print(f"Done. failures={failures}")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
