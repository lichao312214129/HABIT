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
Manual example: standalone traditional radiomics via ``habit.recipes.traditional_radiomics``.

Pure Python API — radiomics parameters built as a dict in code.

Run from repository root (py310)::

    E:\\conda\\mconda\\envs\\py310\\python.exe tests/examples/manual_traditional_radiomics.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict

from habit.recipes import traditional_radiomics

from tests.examples.demo_paths import EXAMPLE_OUT, IMAGING_ROOT, MODALITIES

DRY_RUN: bool = "--dry-run" in sys.argv
OUT_DIR: Path = EXAMPLE_OUT / "traditional_radiomics"

config: Dict[str, Any] = {
    "paths": {
        "images_folder": str(IMAGING_ROOT),
        "out_dir": str(OUT_DIR),
    },
    "processing": {
        "n_processes": 2,
        "process_image_types": list(MODALITIES),
    },
    "export": {
        "export_by_image_type": True,
        "export_combined": True,
    },
}

print(f"Image root: {config['paths']['images_folder']}")
print(f"Output: {config['paths']['out_dir']}")

if DRY_RUN:
    print("Dry-run OK: config dict assembled.")
else:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("manual_radiomics")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    result = traditional_radiomics(config, logger=logger)
    print(f"Workflow output directory: {result.output_dir}")

print("Done.")
