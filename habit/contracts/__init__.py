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
"""L2 domain data contracts: the vocabulary every HABIT layer speaks.

The guiding rule is that each type must be explainable in the language of
habitat imaging research -- an abstraction that can only be justified in
software terms does not belong here. Nothing in this package knows about
YAML, output directories, run modes, CLI, or logging.
"""

from __future__ import annotations

from habit.contracts.geometry import Geometry
from habit.contracts.image import ArrayImageRef, ImageRef, ImageVolume, MaskVolume
from habit.contracts.subject import (
    Cohort,
    CohortFingerprint,
    Subject,
    cohort_from_directory,
)
from habit.contracts.provenance import Provenance, software_fingerprint
from habit.contracts.habitat import (
    HabitatMap,
    HabitatModel,
    Supervoxelization,
    VoxelFeatureField,
)
from habit.contracts.table import FeatureTable
from habit.contracts.manifest import RunManifest, StudyResult
from habit.contracts.ops import (
    CohortOperator,
    DataSource,
    ExecutionBackend,
    ResultWriter,
    SubjectOperator,
    SubjectResult,
)

__all__ = [
    "Geometry",
    "ImageRef",
    "ImageVolume",
    "MaskVolume",
    "ArrayImageRef",
    "Subject",
    "Cohort",
    "CohortFingerprint",
    "cohort_from_directory",
    "Provenance",
    "software_fingerprint",
    "VoxelFeatureField",
    "Supervoxelization",
    "HabitatMap",
    "HabitatModel",
    "FeatureTable",
    "RunManifest",
    "StudyResult",
    "SubjectOperator",
    "CohortOperator",
    "SubjectResult",
    "ExecutionBackend",
    "DataSource",
    "ResultWriter",
]
