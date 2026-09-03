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

from typing import Any

from habit.voxel_features._base import (
    aligned_image,
    build_voxel_field,
    roi_voxels,
)
from habit.voxel_features.concat import (
    ConcatVoxelFeatures,
)
from habit.voxel_features.extract import extract_voxel_texture
from habit.voxel_features.expression import (
    ExpressionVoxelFeatures,
)
from habit.voxel_features.kinetic import (
    KineticVoxelFeatures,
)
from habit.voxel_features.local_entropy import (
    LocalEntropyVoxelFeatures,
)
from habit.voxel_features.raw import RawVoxelFeatures
from habit.voxel_features.registry import VoxelFeatureExtractorRegistry
from habit.voxel_features.cache import (
    load_cached_voxel_field,
    save_cached_voxel_field,
    voxel_radiomics_cache_key,
    voxel_volume_fingerprint,
)
from habit.voxel_features.voxel_radiomics import (
    VoxelRadiomicsFeatures,
)
from habit._protocols import VoxelFeatureExtractor

__all__ = [
    "aligned_image",
    "build_voxel_field",
    "roi_voxels",
    "ConcatVoxelFeatures",
    "extract_voxel_texture",
    "ExpressionVoxelFeatures",
    "KineticVoxelFeatures",
    "LocalEntropyVoxelFeatures",
    "RawVoxelFeatures",
    "load_cached_voxel_field",
    "save_cached_voxel_field",
    "voxel_radiomics_cache_key",
    "voxel_volume_fingerprint",
    "VoxelRadiomicsFeatures",
    "VoxelFeatureExtractorRegistry",
    "VoxelFeatureExtractor",
    "VoxelFeatureTree",
    "build_voxel_extractor",
]


def __getattr__(name: str) -> Any:
    """Lazily expose the tree wrapper after all feature registries load."""
    if name in {"VoxelFeatureTree", "build_voxel_extractor"}:
        from habit._feature_trees import VoxelFeatureTree, build_voxel_extractor

        return {
            "VoxelFeatureTree": VoxelFeatureTree,
            "build_voxel_extractor": build_voxel_extractor,
        }[name]
    raise AttributeError(name)
