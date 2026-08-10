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
Manual example: apply a saved habitat model via ``Study.from_model(...).predict(...)``.

Train in memory, save/reload ``HabitatModel``, then project onto the same cohort.

Run from repository root (py310)::

    E:\\conda\\mconda\\envs\\py310\\python.exe tests/examples/manual_habitat_apply_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from habit.adapters import DirectoryDataSource
from habit.contracts.habitat import HabitatModel
from habit.recipes import Study
from habit.spec.specs import HabitatSpec, Spec

from tests.examples.demo_paths import EXAMPLE_OUT, IMAGING_ROOT, MODALITIES

DRY_RUN: bool = "--dry-run" in sys.argv
OUT_DIR: Path = EXAMPLE_OUT / "habitat_apply_model"

spec = HabitatSpec(
    name="habitat_two_step",
    voxel_feature_extractor=Spec(name="raw", params={"modalities": list(MODALITIES)}),
    supervoxelizer=Spec(
        name="kmeans",
        params={"n_supervoxels": 50, "max_iter": 300, "n_init": 10},
    ),
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
    cohort_feature_preprocessors=(
        Spec(name="binning", params={"n_bins": 10, "bin_strategy": "uniform", "across_features": False}),
    ),
    random_seed=42,
)

cohort = DirectoryDataSource(IMAGING_ROOT, modalities=list(MODALITIES), roi=MODALITIES[0]).load()
print(f"Loaded cohort: {len(cohort)} subjects from {IMAGING_ROOT}")

if DRY_RUN:
    print("Dry-run OK: spec, cohort and imports validated.")
else:
    train_result = Study(spec=spec, design="two_step").fit_predict(cohort)
    assert train_result.habitat_model is not None

    model_dir: Path = OUT_DIR / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    archive: Path = train_result.habitat_model.save(model_dir / "demo.habitatmodel")
    print(f"Saved HabitatModel archive: {archive}")

    reloaded: HabitatModel = HabitatModel.load(archive)
    predict_result = Study.from_model(reloaded, spec).predict(cohort)

    predict_dir: Path = OUT_DIR / "predict"
    predict_dir.mkdir(parents=True, exist_ok=True)
    predict_result.save(predict_dir)

    mismatches: int = sum(
        1
        for train_map, predict_map in zip(train_result.habitat_maps, predict_result.habitat_maps)
        if not np.array_equal(train_map.label_array, predict_map.label_array)
    )
    print(f"Predict output: {predict_dir}")
    print(f"Label mismatches vs training: {mismatches} / {len(train_result.habitat_maps)}")

print("Done.")
