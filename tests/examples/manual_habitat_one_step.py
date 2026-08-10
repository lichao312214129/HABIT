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
Manual example: one-step habitat analysis via ``habit.recipes.Study``.

Pure Python API — HabitatSpec built in code, cohort loaded from demo_data paths.

Run from repository root (py310)::

    E:\\conda\\mconda\\envs\\py310\\python.exe tests/examples/manual_habitat_one_step.py
"""

from __future__ import annotations

from pathlib import Path

from habit.adapters import DirectoryDataSource
from habit.recipes import Study
from habit.spec.specs import HabitatSpec, Spec

from tests.examples.demo_paths import EXAMPLE_OUT, IMAGING_ROOT, MODALITIES

OUT_DIR: Path = EXAMPLE_OUT / "habitat_one_step"

# --- build HabitatSpec in code (supervoxelizer=None selects one-step design) ---
spec = HabitatSpec(
    name="habitat_one_step",
    voxel_feature_extractor=Spec(name="raw", params={"modalities": list(MODALITIES)}),
    supervoxelizer=None,
    habitat_model_fitter=Spec(
        name="kmeans",
        params={
            "min_habitats": 2,
            "max_habitats": 10,
            "validation": "elbow",
            "max_iter": 300,
            "n_init": 10,
        },
    ),
    habitat_assigner=Spec(name="nearest_centroid"),
    voxel_feature_preprocessors=(
        Spec(name="winsorize", params={"winsor_limits": (0.05, 0.05), "across_features": False}),
        Spec(name="minmax", params={"across_features": False}),
    ),
    random_seed=42,
)

cohort = DirectoryDataSource(IMAGING_ROOT, modalities=list(MODALITIES), roi=MODALITIES[0]).load()
print(f"Loaded cohort: {len(cohort)} subjects from {IMAGING_ROOT}")

result = Study(spec=spec, design="one_step").fit_predict(cohort)
OUT_DIR.mkdir(parents=True, exist_ok=True)
result.save(OUT_DIR)

print(f"Saved {len(result.habitat_maps)} habitat maps to: {OUT_DIR}")
print("Done.")
