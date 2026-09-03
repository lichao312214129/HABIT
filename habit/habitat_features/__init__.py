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

from typing import Any

from habit.habitat_features.compare import (
    HabitatFeatureComparison,
    HabitatFeaturePanel,
    compare_habitat_features,
    to_habitat_feature_panel,
)
from habit.habitat_features.each_habitat import (
    EachHabitatRadiomicsFeatures,
)
from habit.habitat_features.graph import (
    GraphHabitatFeatures,
)
from habit.habitat_features.ith import IthHabitatFeatures
from habit.habitat_features.msi import MsiHabitatFeatures
from habit.habitat_features.non_radiomics import (
    NonRadiomicsHabitatFeatures,
)
from habit.habitat_features.registry import HabitatFeatureExtractorRegistry
from habit.habitat_features.traditional import (
    TraditionalRadiomicsHabitatFeatures,
)
from habit.habitat_features.volume import (
    HabitatVolumeFeatures,
)
from habit.habitat_features.whole_habitat import (
    WholeHabitatRadiomicsFeatures,
)
from habit._protocols import HabitatFeatureExtractor

__all__ = [
    "GraphHabitatFeatures",
    "IthHabitatFeatures",
    "MsiHabitatFeatures",
    "NonRadiomicsHabitatFeatures",
    "TraditionalRadiomicsHabitatFeatures",
    "WholeHabitatRadiomicsFeatures",
    "EachHabitatRadiomicsFeatures",
    "HabitatFeaturePanel",
    "HabitatFeatureComparison",
    "to_habitat_feature_panel",
    "compare_habitat_features",
    "HabitatVolumeFeatures",
    "HabitatFeatureExtractorRegistry",
    "HabitatFeatureExtractor",
    "HabitatFeatureTree",
    "build_habitat_extractor",
]


def __getattr__(name: str) -> Any:
    """Lazily expose the shared tree wrapper after registry bootstrap."""
    if name in {"HabitatFeatureTree", "build_habitat_extractor"}:
        from habit._feature_trees import HabitatFeatureTree, build_habitat_extractor

        return {
            "HabitatFeatureTree": HabitatFeatureTree,
            "build_habitat_extractor": build_habitat_extractor,
        }[name]
    raise AttributeError(name)
