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
"""Supervoxel partitioning and per-supervoxel feature capabilities."""

from __future__ import annotations

from typing import Any

from habit.supervoxel.feature_clustering import (
    GmmSupervoxelizer,
    KMeansSupervoxelizer,
)
from habit.supervoxel.registry import SupervoxelizerRegistry
from habit.supervoxel.slic import SlicSupervoxelizer
from habit.supervoxel.features_base import aggregate_voxel_means
from habit.supervoxel.features_registry import SupervoxelFeatureExtractorRegistry
from habit.supervoxel.mean import MeanVoxelFeatures
from habit.supervoxel.radiomics import (
    SupervoxelRadiomicsFeatures,
)
from habit.supervoxel.statistics import (
    MeanSupervoxelFeatures,
    PercentileSupervoxelFeatures,
    StdSupervoxelFeatures,
)
from habit._protocols import SupervoxelFeatureExtractor, Supervoxelizer

__all__ = [
    "SlicSupervoxelizer",
    "KMeansSupervoxelizer",
    "GmmSupervoxelizer",
    "SupervoxelizerRegistry",
    "MeanSupervoxelFeatures",
    "MeanVoxelFeatures",
    "PercentileSupervoxelFeatures",
    "StdSupervoxelFeatures",
    "SupervoxelRadiomicsFeatures",
    "SupervoxelFeatureExtractorRegistry",
    "aggregate_voxel_means",
    "Supervoxelizer",
    "SupervoxelFeatureExtractor",
    "SupervoxelFeatureTree",
    "build_supervoxel_extractor",
]


def __getattr__(name: str) -> Any:
    """Lazily expose the shared tree wrapper after registry bootstrap."""
    if name in {"SupervoxelFeatureTree", "build_supervoxel_extractor"}:
        from habit._feature_trees import (
            SupervoxelFeatureTree,
            build_supervoxel_extractor,
        )

        return {
            "SupervoxelFeatureTree": SupervoxelFeatureTree,
            "build_supervoxel_extractor": build_supervoxel_extractor,
        }[name]
    raise AttributeError(name)
