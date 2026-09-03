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
"""Legacy v0.1 plugin factory accessors (L1 compat)."""

from __future__ import annotations

from typing import Any, Type

__all__ = [
    "get_legacy_feature_extractor_registry",
    "get_legacy_metric_registry",
    "get_legacy_model_factory",
]


def get_legacy_model_factory() -> Type[Any]:
    """Return the v1 classifier registry (v0.1 ``ModelFactory`` alias surface)."""
    from habit.classification.registry import ClassifierRegistry

    return ClassifierRegistry


def get_legacy_metric_registry() -> Type[Any]:
    """Return the v1 metric registry (v0.1 ``MetricRegistry`` alias surface)."""
    from habit.evaluation.registry import MetricRegistry

    return MetricRegistry


def get_legacy_feature_extractor_registry() -> Type[Any]:
    """Return the merged v1 clustering-feature extractor registry facade."""
    from habit.compat.registry_facade import LegacyFeatureExtractorRegistryFacade

    return LegacyFeatureExtractorRegistryFacade
