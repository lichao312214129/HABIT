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
Manual example: test-retest habitat label mapping via ``habit.recipes.test_retest_analysis``.

Prerequisite: two-step habitat outputs under ``demo_data/results/habitat_two_step/``.
Uses the same feature table for test/retest (identity mapping smoke test).

Run from repository root (py310)::

    E:\\conda\\mconda\\envs\\py310\\python.exe tests/examples/manual_test_retest.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from habit.recipes import test_retest_analysis

from tests.examples.demo_paths import DEMO_DATA, EXAMPLE_OUT

OUT_DIR: Path = EXAMPLE_OUT / "test_retest_remapped"
HABITAT_DIR: Path = DEMO_DATA / "results" / "habitat_two_step"
HABITATS_PARQUET: Path = HABITAT_DIR / "habitats.parquet"

if not HABITATS_PARQUET.is_file():
    raise SystemExit(
        f"Habitat feature table not found: {HABITATS_PARQUET}\n"
        "Run manual_habitat_two_step.py first."
    )

config: Dict[str, Any] = {
    "test_habitat_table": str(HABITATS_PARQUET),
    "retest_habitat_table": str(HABITATS_PARQUET),
    "input_dir": str(HABITAT_DIR),
    "out_dir": str(OUT_DIR),
    "similarity_method": "pearson",
    "processes": 2,
}

print(f"Test table: {config['test_habitat_table']}")
print(f"Habitat maps: {config['input_dir']}")
print(f"Output: {config['out_dir']}")

OUT_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("manual_test_retest")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

result = test_retest_analysis(config, logger=logger)
print(f"Label mapping: {result.data}")
print(f"Remapped maps directory: {result.output_dir}")
print("Done.")
