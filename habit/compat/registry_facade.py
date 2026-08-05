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
"""Merged registry facade for the legacy ``feature_extractors`` plugin domain."""

from __future__ import annotations

from typing import Any, List, Optional, Type

__all__ = ["LegacyFeatureExtractorRegistryFacade"]


class LegacyFeatureExtractorRegistryFacade:
    """
    Route legacy ``feature_extractors`` names to v1 voxel/supervoxel registries.

    The v0.1 ``FeatureExtractorRegistry`` mixed voxel- and supervoxel-stage
    extractors in one table. v1 splits them across
    ``VoxelFeatureExtractorRegistry`` and ``SupervoxelFeatureExtractorRegistry``;
    this facade preserves the old combined name list for plugin discovery.
    """

    kind = "feature extractor"

    @classmethod
    def _voxel_registry(cls) -> Type[Any]:
        from habit.domain.voxel_features.registry import VoxelFeatureExtractorRegistry

        return VoxelFeatureExtractorRegistry

    @classmethod
    def _supervoxel_registry(cls) -> Type[Any]:
        from habit.domain.supervoxel_features.registry import (
            SupervoxelFeatureExtractorRegistry,
        )

        return SupervoxelFeatureExtractorRegistry

    @classmethod
    def _registry_for(cls, name: str) -> Type[Any]:
        normalized = str(name).lower()
        voxel = cls._voxel_registry()
        if normalized in {entry.lower() for entry in voxel.available()}:
            return voxel
        supervoxel = cls._supervoxel_registry()
        if normalized in {entry.lower() for entry in supervoxel.available()}:
            return supervoxel
        raise ValueError(
            f"Feature extractor '{name}' is not registered in the v1 registries. "
            f"Available: {sorted(cls.available())}."
        )

    @classmethod
    def available(cls) -> List[str]:
        names = set(cls._voxel_registry().available())
        names.update(cls._supervoxel_registry().available())
        return sorted(names)

    @classmethod
    def get(cls, name: str) -> Any:
        return cls._registry_for(name).get(name)

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> Any:
        return cls._registry_for(name).create(name, **kwargs)

    @classmethod
    def get_params_model(cls, name: str) -> Optional[Type[Any]]:
        return cls._registry_for(name).get_params_model(name)

    @classmethod
    def register_params_model(cls, name: str, model: Type[Any]) -> None:
        cls._registry_for(name).register_params_model(name, model)
