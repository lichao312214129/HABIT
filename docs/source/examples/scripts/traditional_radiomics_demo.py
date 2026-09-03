#!/usr/bin/env python
"""
Standalone traditional radiomics via ``habit.recipes.traditional_radiomics``.

Assembles the config dict in Python (no YAML). Requires
``demo_data/preprocessed/`` and PyRadiomics; pass
``--dry-run`` to validate the config without running extraction.

The extractor always opens a ``multiprocessing.Pool`` (even at
``n_processes: 1``), so on Windows this script must use
``if __name__ == "__main__"``.

This script accompanies ``docs/source/examples/feature_extraction.rst``.

Run from the repository root (py310)::

    python docs/source/examples/scripts/traditional_radiomics_demo.py --dry-run
    python docs/source/examples/scripts/traditional_radiomics_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

from habit.recipes import traditional_radiomics

# Repository root: docs/source/examples/scripts/<this file> -> parents[4]
REPO_ROOT: Path = Path(__file__).resolve().parents[4]
IMAGING_ROOT: Path = REPO_ROOT / "demo_data" / "preprocessed"
MODALITIES: Tuple[str, ...] = ("pre_contrast", "LAP", "PVP", "delay_3min")
OUT_DIR: Path = REPO_ROOT / "demo_data" / "results" / "examples" / "traditional_radiomics_docs"


dry_run: bool = "--dry-run" in sys.argv

config: Dict[str, Any] = {
    "paths": {
        "images_folder": str(IMAGING_ROOT),
        "out_dir": str(OUT_DIR),
    },
    "processing": {
        "n_processes": 1,
        "process_image_types": list(MODALITIES),
    },
    "export": {
        "export_by_image_type": True,
        "export_combined": True,
    },
}

print(f"Image root: {config['paths']['images_folder']}")
print(f"Output: {config['paths']['out_dir']}")
print(f"Modalities: {MODALITIES}")

if not IMAGING_ROOT.is_dir():
    raise SystemExit(
        f"demo_data not found at {IMAGING_ROOT}\n"
        "Obtain demo_data/ locally or swap images_folder for your cohort."
    )

if dry_run:
    print("Dry-run OK: config dict assembled.")
else:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = traditional_radiomics(config)
    print(f"Workflow output: {result.output_dir}")

print("Done.")
