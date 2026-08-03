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
"""Built-in habitat feature extractors and their registry."""

from __future__ import annotations

from habit.domain.habitat_features.ith import IthHabitatFeatures, IthHabitatFeaturesParams
from habit.domain.habitat_features.msi import MsiHabitatFeatures, MsiHabitatFeaturesParams
from habit.domain.habitat_features.registry import HabitatFeatureExtractorRegistry
from habit.domain.habitat_features.volume import (
    HabitatVolumeFeatures,
    HabitatVolumeFeaturesParams,
)

__all__ = [
    "IthHabitatFeatures",
    "IthHabitatFeaturesParams",
    "MsiHabitatFeatures",
    "MsiHabitatFeaturesParams",
    "HabitatVolumeFeatures",
    "HabitatVolumeFeaturesParams",
    "HabitatFeatureExtractorRegistry",
]
