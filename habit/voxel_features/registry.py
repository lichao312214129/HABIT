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
"""Registry for the ``voxel_feature_extractor`` plugin domain."""

from __future__ import annotations

from typing import Type

from habit._protocols import VoxelFeatureExtractor
from habit.registry.core import ComponentRegistry

__all__ = ["VoxelFeatureExtractorRegistry"]


class VoxelFeatureExtractorRegistry(ComponentRegistry[Type[VoxelFeatureExtractor]]):
    """Name-to-implementation registry for voxel feature extractors."""

    domain = "voxel_feature_extractor"
    kind = "voxel feature extractor"
