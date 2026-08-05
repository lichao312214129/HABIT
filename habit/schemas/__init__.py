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
"""L5 YAML configuration schemas for CLI and check-config.

The canonical Pydantic models live in :mod:`habit.schemas.workflows`; this
package is the **CLI-facing import surface** so ``habit.commands`` never
reaches into ``habit.core`` for schema types alone. Execution paths already
call ``habit.recipes.*``; only configuration loading uses these models.
"""

from __future__ import annotations

from habit.schemas.workflows.dicom_sort import DicomSortConfig
from habit.schemas.workflows.habitat import (
    DROPPING_PREPROCESSING_METHODS,
    FeatureExtractionConfig,
    HabitatAnalysisConfig,
    RadiomicsConfig,
)
from habit.schemas.workflows.icc import ICCConfig
from habit.schemas.workflows.ml import (
    MLConfig,
    ModelComparisonConfig,
    TestRetestConfig,
)
from habit.schemas.workflows.preprocessing import PreprocessingConfig

__all__ = [
    "DROPPING_PREPROCESSING_METHODS",
    "DicomSortConfig",
    "FeatureExtractionConfig",
    "HabitatAnalysisConfig",
    "ICCConfig",
    "MLConfig",
    "ModelComparisonConfig",
    "PreprocessingConfig",
    "RadiomicsConfig",
    "TestRetestConfig",
]
