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
"""In-memory step inspection for habitat pipelines (L3).

Only depends on :mod:`habit.contracts`. Recorders never touch the filesystem;
directory sinks are a later adapters concern.
"""

from __future__ import annotations

from habit.contracts.inspection import (
    STEP_HABITAT_FEATURES,
    STEP_HABITAT_MAP,
    STEP_NAMES,
    STEP_SUPERVOXELS_DESCRIBED,
    STEP_SUPERVOXELS_PARTITION,
    STEP_SUPERVOXELS_POSTPROCESSED,
    STEP_SUPERVOXELS_PREPROCESSED,
    STEP_UNITS_COHORT_PREPROCESSED,
    STEP_VOXEL_FEATURES_PREPROCESSED,
    STEP_VOXEL_FEATURES_RAW,
    StepObserver,
    StepRecord,
)
from habit.inspection.recorder import StepRecorder

__all__ = [
    "STEP_HABITAT_FEATURES",
    "STEP_HABITAT_MAP",
    "STEP_NAMES",
    "STEP_SUPERVOXELS_DESCRIBED",
    "STEP_SUPERVOXELS_PARTITION",
    "STEP_SUPERVOXELS_POSTPROCESSED",
    "STEP_SUPERVOXELS_PREPROCESSED",
    "STEP_UNITS_COHORT_PREPROCESSED",
    "STEP_VOXEL_FEATURES_PREPROCESSED",
    "STEP_VOXEL_FEATURES_RAW",
    "StepObserver",
    "StepRecord",
    "StepRecorder",
]
