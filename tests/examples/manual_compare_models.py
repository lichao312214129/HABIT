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
Manual example: multi-model comparison via ``habit.recipes.compare_models``.

Prerequisite: prediction CSVs from prior ML runs under ``demo_data/results/ml/``.

Run from repository root (py310)::

    E:\\conda\\mconda\\envs\\py310\\python.exe tests/examples/manual_compare_models.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from habit.recipes import compare_models

from tests.examples.demo_paths import DEMO_DATA, EXAMPLE_OUT

OUT_DIR: Path = EXAMPLE_OUT / "model_comparison"
RADIOMICS_PRED: Path = DEMO_DATA / "results" / "ml" / "radiomics" / "all_prediction_results.csv"
CLINICAL_PRED: Path = DEMO_DATA / "results" / "ml" / "clinical" / "all_prediction_results.csv"

for path, label in [(RADIOMICS_PRED, "radiomics"), (CLINICAL_PRED, "clinical")]:
    if not path.is_file():
        raise SystemExit(
            f"{label} prediction CSV not found: {path}\n"
            "Run the ML CLI workflows first to generate prediction files."
        )

config: Dict[str, Any] = {
    "output_dir": str(OUT_DIR),
    "files_config": [
        {
            "path": str(RADIOMICS_PRED),
            "model_name": "radiomics",
            "subject_id_col": "subject_id",
            "label_col": "label",
            "prediction_col": "prediction",
        },
        {
            "path": str(CLINICAL_PRED),
            "model_name": "clinical",
            "subject_id_col": "subject_id",
            "label_col": "label",
            "prediction_col": "prediction",
        },
    ],
}

print(f"Output: {config['output_dir']}")
for entry in config["files_config"]:
    print(f"  model {entry['model_name']}: {entry['path']}")

OUT_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("manual_compare_models")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

result = compare_models(config, logger=logger, output_dir=str(OUT_DIR))
print(f"Workflow output directory: {result.output_dir}")
if result.data:
    print(f"Comparison metrics keys: {list(result.data.keys())}")
print("Done.")
