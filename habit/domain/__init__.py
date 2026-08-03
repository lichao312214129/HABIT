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
"""L3 domain layer: the five protocols and their built-in implementations.

Importing this package registers every built-in component in its domain
registry, so ``<Registry>.available()`` and ``list_plugins(domain=...)``
reflect the full built-in surface after a single import. Heavy third-party
libraries (scikit-learn, scikit-image) are imported lazily inside method
bodies, keeping this package cheap to import.
"""

from __future__ import annotations

from habit.domain.protocols import (
    HabitatAssigner,
    HabitatFeatureExtractor,
    HabitatModelFitter,
    Seedable,
    Supervoxelizer,
    VoxelFeatureExtractor,
)
from habit.domain.pipeline import SubjectPipeline
from habit.domain.voxel_features import (
    RawVoxelFeatures,
    RawVoxelFeaturesParams,
    VoxelFeatureExtractorRegistry,
)
from habit.domain.supervoxel import (
    SlicSupervoxelizer,
    SlicSupervoxelizerParams,
    SupervoxelizerRegistry,
)
from habit.domain.habitat_model import (
    GmmHabitatModelFitter,
    GmmHabitatModelFitterParams,
    HabitatModelFitterRegistry,
    KMeansHabitatModelFitter,
    KMeansHabitatModelFitterParams,
)
from habit.domain.assignment import (
    HabitatAssignerRegistry,
    NearestCentroidAssigner,
    NearestCentroidAssignerParams,
)
from habit.domain.habitat_features import (
    HabitatFeatureExtractorRegistry,
    HabitatVolumeFeatures,
    HabitatVolumeFeaturesParams,
    IthHabitatFeatures,
    IthHabitatFeaturesParams,
    MsiHabitatFeatures,
    MsiHabitatFeaturesParams,
)

__all__ = [
    # The five domain protocols plus the seeding contract.
    "VoxelFeatureExtractor",
    "Supervoxelizer",
    "HabitatModelFitter",
    "HabitatAssigner",
    "HabitatFeatureExtractor",
    "Seedable",
    # Composition.
    "SubjectPipeline",
    # Built-in voxel feature extractors.
    "RawVoxelFeatures",
    "RawVoxelFeaturesParams",
    "VoxelFeatureExtractorRegistry",
    # Built-in supervoxelizers.
    "SlicSupervoxelizer",
    "SlicSupervoxelizerParams",
    "SupervoxelizerRegistry",
    # Built-in habitat model fitters.
    "KMeansHabitatModelFitter",
    "KMeansHabitatModelFitterParams",
    "GmmHabitatModelFitter",
    "GmmHabitatModelFitterParams",
    "HabitatModelFitterRegistry",
    # Built-in habitat assigners.
    "NearestCentroidAssigner",
    "NearestCentroidAssignerParams",
    "HabitatAssignerRegistry",
    # Built-in habitat feature extractors.
    "MsiHabitatFeatures",
    "MsiHabitatFeaturesParams",
    "IthHabitatFeatures",
    "IthHabitatFeaturesParams",
    "HabitatVolumeFeatures",
    "HabitatVolumeFeaturesParams",
    "HabitatFeatureExtractorRegistry",
]
