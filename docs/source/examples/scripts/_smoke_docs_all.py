#!/usr/bin/env python
"""
Full docs smoke: every Examples gallery script + key CLI entry points.

Exit non-zero if any script/CLI check fails. Run from the repository root::

    set HABIT_NO_VIEW=1
    python docs/source/examples/scripts/_smoke_docs_all.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
SCRIPTS = Path(__file__).resolve().parent
PY = sys.executable

# All runnable gallery scripts (exclude private helpers).
SKIP = {
    "_smoke_examples.py",
    "_smoke_docs_all.py",
    "_example_roi.py",
    "_habitat_eye_check.py",
    "_make_example_thumbnails.py",
}


def _run(cmd: list[str], *, cwd: Path = REPO, label: str) -> bool:
    """Run a command; print OK/FAIL. Return True on success."""
    print(f"[RUN] {label}")
    completed = subprocess.run(cmd, cwd=str(cwd), check=False)
    if completed.returncode != 0:
        print(f"[FAIL] {label} exit={completed.returncode}")
        return False
    print(f"[OK]   {label}")
    return True


def main() -> int:
    """Smoke all example scripts and a small CLI surface."""
    os.environ["HABIT_NO_VIEW"] = "1"
    failures = 0

    scripts = sorted(
        p.name
        for p in SCRIPTS.glob("*.py")
        if p.name not in SKIP and not p.name.startswith("_")
    )
    # Include quickstarts named without _demo suffix.
    for extra in ("two_step_habitat_quickstart.py", "tabular_ml_quickstart.py"):
        if (SCRIPTS / extra).is_file() and extra not in scripts:
            scripts.append(extra)
    scripts = sorted(set(scripts))

    for name in scripts:
        ok = _run([PY, str(SCRIPTS / name)], label=f"script:{name}")
        if not ok:
            failures += 1

    cli_checks = [
        ([PY, "-m", "habit", "--help"], "cli:habit --help"),
        ([PY, "-m", "habit", "get-habitat", "--help"], "cli:get-habitat --help"),
        ([PY, "-m", "habit", "view", "--help"], "cli:view --help"),
        ([PY, "-m", "habit", "extract", "--help"], "cli:extract --help"),
        ([PY, "-m", "habit", "check-config", "--help"], "cli:check-config --help"),
    ]
    yaml_demo = REPO / "config" / "habitat" / "config_habitat_two_step.yaml"
    if yaml_demo.is_file():
        cli_checks.append(
            (
                [PY, "-m", "habit", "check-config", "-c", str(yaml_demo)],
                "cli:check-config two_step.yaml",
            )
        )

    for cmd, label in cli_checks:
        if not _run(cmd, label=label):
            failures += 1

    # Public API import smoke for symbols taught in docs.
    api_smoke = (
        "from habit import ("
        "cohort_from_directory, one_step_habitat, two_step_habitat, "
        "extract_graph_features, local_entropy_map, HabitatSpec, Spec, Stage"
        "); "
        "from habit.viz import ("
        "plot_habitat_overlay, plot_partition_triptych, "
        "plot_habitat_volume_fractions, plot_msi_matrix, plot_ith_summary, "
        "plot_cluster_validation_from_report, plot_habitat_label_compare, "
        "plot_voxel_texture_slice, plot_habitat_graph_network_2d"
        "); "
        "print('api_ok')"
    )
    if not _run([PY, "-c", api_smoke], label="api:public imports"):
        failures += 1

    print(f"Done. failures={failures} / scripts={len(scripts)}")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
