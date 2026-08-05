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
Manual example: legacy YAML dispatch via ``habit.recipes.run_from_yaml``.

This is the **only** example that loads a YAML file — it demonstrates the
optional bridge for v0.1 configs. All other scripts build specs in Python.

Run from repository root (py310)::

    E:\\conda\\mconda\\envs\\py310\\python.exe tests/examples/manual_run_from_yaml.py
"""

from __future__ import annotations

import logging
from pathlib import Path

from habit.recipes import run_from_yaml

from tests.examples.demo_paths import EXAMPLE_OUT, ML_DATA, REPO_ROOT

# Minimal patched YAML written locally (paths only — algorithm params stay in YAML)
SOURCE_YAML: Path = REPO_ROOT / "config" / "auxiliary" / "config_icc_demo.yaml"
OUT_DIR: Path = EXAMPLE_OUT / "run_from_yaml"
PATCHED_YAML: Path = OUT_DIR / "config_icc_patched.yaml"

if not (ML_DATA / "breast_cancer_dataset.csv").is_file():
    raise SystemExit(f"ML demo data not found under: {ML_DATA}")

OUT_DIR.mkdir(parents=True, exist_ok=True)
text: str = SOURCE_YAML.read_text(encoding="utf-8")
text = text.replace(
    "../../demo_data/results/icc/icc_radiomics.json",
    str((OUT_DIR / "icc_radiomics.json").as_posix()),
)
text = text.replace(
    "../../demo_data/ml_data/breast_cancer_dataset.csv",
    str((ML_DATA / "breast_cancer_dataset.csv").as_posix()),
)
text = text.replace(
    "../../demo_data/ml_data/breast_cancer_dataset_retest_simulated.csv",
    str((ML_DATA / "breast_cancer_dataset_retest_simulated.csv").as_posix()),
)
PATCHED_YAML.write_text(text, encoding="utf-8")

logger = logging.getLogger("manual_run_from_yaml")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

print(f"Config: {PATCHED_YAML}")
print(f"Output directory: {OUT_DIR}")

result = run_from_yaml(PATCHED_YAML, workflow="icc", logger=logger)
print(f"Workflow output directory: {result.output_dir}")
print(f"ICC result JSON: {result.artifacts.get('icc_result')}")
print("Done.")
