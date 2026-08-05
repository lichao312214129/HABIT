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
Manual example: batch image preprocessing via ``habit.recipes.preprocess_images``.

Pure Python API — preprocessing parameters built as a dict in code.

Run from repository root (py310)::

    E:\\conda\\mconda\\envs\\py310\\python.exe tests/examples/manual_preprocess.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict

from habit.recipes import preprocess_images

from tests.examples.demo_paths import EXAMPLE_OUT, IMAGING_ROOT, MODALITIES

DRY_RUN: bool = "--dry-run" in sys.argv
OUT_DIR: Path = EXAMPLE_OUT / "preprocess"

# --- build preprocessing config dict in code (no YAML file) ---
config: Dict[str, Any] = {
    "data_dir": str(IMAGING_ROOT),
    "out_dir": str(OUT_DIR),
    "auto_select_first_file": True,
    "processes": 2,
    "preprocessing": {
        "resample": {
            "images": list(MODALITIES),
            "target_spacing": [3.0, 3.0, 3.0],
            "img_mode": "bilinear",
        },
    },
}

print(f"Input root: {config['data_dir']}")
print(f"Output: {config['out_dir']}")

if DRY_RUN:
    print("Dry-run OK: config dict assembled.")
else:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("manual_preprocess")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    result = preprocess_images(config, logger=logger)
    print(f"Workflow output directory: {result.output_dir}")
    if result.manifest_path:
        print(f"Run manifest: {result.manifest_path}")

print("Done.")
