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
"""Built-in voxel feature extractors and their registry."""

from __future__ import annotations

from habit.domain.voxel_features._base import (
    aligned_image,
    build_voxel_field,
    roi_voxels,
)
from habit.domain.voxel_features.concat import (
    ConcatVoxelFeatures,
    ConcatVoxelFeaturesParams,
)
from habit.domain.voxel_features.extract import extract_voxel_texture
from habit.domain.voxel_features.expression import (
    ExpressionVoxelFeatures,
    ExpressionVoxelFeaturesParams,
)
from habit.domain.voxel_features.kinetic import (
    KineticVoxelFeatures,
    KineticVoxelFeaturesParams,
)
from habit.domain.voxel_features.local_entropy import (
    LocalEntropyVoxelFeatures,
    LocalEntropyVoxelFeaturesParams,
)
from habit.domain.voxel_features.raw import RawVoxelFeatures, RawVoxelFeaturesParams
from habit.domain.voxel_features.registry import VoxelFeatureExtractorRegistry
from habit.domain.voxel_features.cache import (
    load_cached_voxel_field,
    save_cached_voxel_field,
    voxel_radiomics_cache_key,
    voxel_volume_fingerprint,
)
from habit.domain.voxel_features.voxel_radiomics import (
    VoxelRadiomicsFeatures,
    VoxelRadiomicsFeaturesParams,
)

__all__ = [
    "aligned_image",
    "build_voxel_field",
    "roi_voxels",
    "ConcatVoxelFeatures",
    "ConcatVoxelFeaturesParams",
    "extract_voxel_texture",
    "ExpressionVoxelFeatures",
    "ExpressionVoxelFeaturesParams",
    "KineticVoxelFeatures",
    "KineticVoxelFeaturesParams",
    "LocalEntropyVoxelFeatures",
    "LocalEntropyVoxelFeaturesParams",
    "RawVoxelFeatures",
    "RawVoxelFeaturesParams",
    "load_cached_voxel_field",
    "save_cached_voxel_field",
    "voxel_radiomics_cache_key",
    "voxel_volume_fingerprint",
    "VoxelRadiomicsFeatures",
    "VoxelRadiomicsFeaturesParams",
    "VoxelFeatureExtractorRegistry",
]
