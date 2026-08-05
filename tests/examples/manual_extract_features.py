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
Manual example: habitat feature extraction via ``habit.recipes.extract_habitat_features``.

Prerequisite: habitat maps under ``demo_data/results/habitat_two_step/``
(run ``manual_habitat_two_step.py`` first).

Run from repository root (py310)::

    E:\\conda\\mconda\\envs\\py310\\python.exe tests/examples/manual_extract_features.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from habit.recipes import extract_habitat_features

from tests.examples.demo_paths import DEMO_DATA, EXAMPLE_OUT, IMAGING_ROOT

OUT_DIR: Path = EXAMPLE_OUT / "extract_features"
HABITAT_MAPS_DIR: Path = DEMO_DATA / "results" / "habitat_two_step"

if not HABITAT_MAPS_DIR.is_dir():
    raise SystemExit(
        f"Habitat maps not found: {HABITAT_MAPS_DIR}\n"
        "Run manual_habitat_two_step.py first."
    )

# --- build extraction config dict in code ---
config: Dict[str, Any] = {
    "raw_img_folder": str(IMAGING_ROOT),
    "habitats_map_folder": str(HABITAT_MAPS_DIR),
    "out_dir": str(OUT_DIR),
    "n_processes": 2,
    "habitat_pattern": "*_habitats.nrrd",
    "feature_types": ["traditional", "non_radiomics", "whole_habitat", "each_habitat", "msi", "ith_score"],
}

print(f"Image root: {config['raw_img_folder']}")
print(f"Habitat maps: {config['habitats_map_folder']}")
print(f"Output: {config['out_dir']}")

OUT_DIR.mkdir(parents=True, exist_ok=True)
result = extract_habitat_features(config)
print(f"Workflow output directory: {result.output_dir}")
if result.artifacts:
    for key, value in result.artifacts.items():
        print(f"  artifact[{key}]: {value}")
print("Done.")
