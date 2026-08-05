#!/usr/bin/env python
# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Manual example: DICOM sort / rename via ``habit.recipes.sort_dicom``.

Pure Python API — sort parameters built as a dict in code.

Run from repository root (py310)::

    E:\\conda\\mconda\\envs\\py310\\python.exe tests/examples/manual_sort_dicom.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict

from habit.recipes import sort_dicom

from tests.examples.demo_paths import DCM2NIIX, DICOM_ROOT, EXAMPLE_OUT

DRY_RUN: bool = "--dry-run" in sys.argv
OUT_DIR: Path = EXAMPLE_OUT / "sorted_dicom"

if not DICOM_ROOT.is_dir():
    raise SystemExit(f"DICOM demo data not found: {DICOM_ROOT}")
if not DCM2NIIX.is_file():
    raise SystemExit(f"dcm2niix executable not found: {DCM2NIIX}")

config: Dict[str, Any] = {
    "data_dir": str(DICOM_ROOT),
    "out_dir": str(OUT_DIR),
    "dcm2niix_path": str(DCM2NIIX),
    "f": "subj_%n_%g_%x/%s_%d/%r_%o.dcm",
    "extra_args": [],
}

print(f"DICOM root: {config['data_dir']}")
print(f"Output: {config['out_dir']}")
print(f"dcm2niix: {config['dcm2niix_path']}")

if DRY_RUN:
    print("Dry-run OK: config dict and prerequisites validated.")
else:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("manual_sort_dicom")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    result = sort_dicom(config, logger=logger)
    print(f"Workflow output directory: {result.output_dir}")
    if result.manifest_path:
        print(f"Run manifest: {result.manifest_path}")

print("Done.")
