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
"""GPU vs CPU voxel radiomics timing on resample_02 real cohort."""

import time
from pathlib import Path

from habit.adapters import DirectoryDataSource
from habit.domain.voxel_features.voxel_radiomics import VoxelRadiomicsFeatures

from tests.examples.demo_paths import REPO_ROOT

ROOT = Path(__file__).resolve().parents[2] / ".cursor" / "test" / "resample_02"
MODS = ("T1", "T2")
ROI = "T1"
PARAMS = str(REPO_ROOT / "config" / "radiomics" / "params_voxel_radiomics.yaml")

cohort = DirectoryDataSource(ROOT, modalities=MODS, roi=ROI).load()[:10]
print(f"{len(cohort)} subjects from {ROOT}")

kw = dict(modalities=MODS, roi=ROI, params_file=PARAMS, kernel_radius=3, voxel_batch=1000)
gpu_ext = VoxelRadiomicsFeatures(**kw, use_torch_radiomics="auto")
cpu_ext = VoxelRadiomicsFeatures(**kw, use_torch_radiomics=False)

gpu_total = cpu_total = 0.0
for s in cohort:
    print(f"Processing {s.subject_id}...")
    n = int((s.mask(ROI).data > 0).sum())
    t0 = time.perf_counter()
    gpu_ext(s)
    g = time.perf_counter() - t0
    t0 = time.perf_counter()
    cpu_ext(s)
    c = time.perf_counter() - t0
    gpu_total += g
    cpu_total += c
    print(f"{s.subject_id} roi={n:5d}  GPU {g:6.1f}s  CPU {c:6.1f}s  {c / g:4.1f}x")

print(f"TOTAL  GPU {gpu_total:6.1f}s  CPU {cpu_total:6.1f}s  {cpu_total / gpu_total:4.1f}x")
