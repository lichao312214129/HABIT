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
"""Thin re-exports of remaining v0.1 engines (deliberate debt).

The public API and recipes must not import ``habit.core.*`` directly once the
v1.0 refactor completes. During the transition this module is the single
compatibility facade; concrete workflow runners and registry gates live in
``habit.compat.*`` submodules so this file stays free of ``habit.core`` imports.

Deleting ``habit.core`` is blocked until every compat runner has a v1 recipe
or domain replacement and the fast/full golden gates stay green.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Type

__all__ = [
    "get_legacy_feature_extractor_registry",
    "get_legacy_metric_registry",
    "get_legacy_model_factory",
]


def get_legacy_model_factory() -> Type[Any]:
    """Return the v1 classifier registry (v0.1 ``ModelFactory`` alias surface)."""
    from habit.compat.plugin_registries import get_legacy_model_factory as _get

    return _get()


def get_legacy_metric_registry() -> Type[Any]:
    """Return the v1 metric registry (v0.1 ``MetricRegistry`` alias surface)."""
    from habit.compat.plugin_registries import get_legacy_metric_registry as _get

    return _get()


def get_legacy_feature_extractor_registry() -> Type[Any]:
    """Return the v0.1 ``FeatureExtractorRegistry`` class."""
    from habit.compat.plugin_registries import (
        get_legacy_feature_extractor_registry as _get,
    )

    return _get()
