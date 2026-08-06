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
"""Built-in supervoxel feature extractors and their registry."""

from __future__ import annotations

from habit.domain.supervoxel_features._base import aggregate_voxel_means
from habit.domain.supervoxel_features.mean import (
    MeanVoxelFeatures,
    MeanVoxelFeaturesParams,
)
from habit.domain.supervoxel_features.radiomics import (
    SupervoxelRadiomicsFeatures,
    SupervoxelRadiomicsFeaturesParams,
)
from habit.domain.supervoxel_features.registry import (
    SupervoxelFeatureExtractorRegistry,
)
from habit.domain.supervoxel_features.statistics import (
    MeanSupervoxelFeatures,
    MeanSupervoxelFeaturesParams,
    PercentileSupervoxelFeatures,
    PercentileSupervoxelFeaturesParams,
    StdSupervoxelFeatures,
    StdSupervoxelFeaturesParams,
)

__all__ = [
    "MeanSupervoxelFeatures",
    "MeanSupervoxelFeaturesParams",
    "MeanVoxelFeatures",
    "MeanVoxelFeaturesParams",
    "PercentileSupervoxelFeatures",
    "PercentileSupervoxelFeaturesParams",
    "StdSupervoxelFeatures",
    "StdSupervoxelFeaturesParams",
    "SupervoxelRadiomicsFeatures",
    "SupervoxelRadiomicsFeaturesParams",
    "SupervoxelFeatureExtractorRegistry",
    "aggregate_voxel_means",
]
