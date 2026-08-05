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
"""Optional graph habitat-feature plugin loader (L1 compat)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

__all__ = ["ensure_graph_plugin_available", "load_graph_feature_config"]


class GraphFeatureConfig(BaseModel):
    """Minimal validated config for the optional graph habitat-feature plugin."""

    enabled: bool = Field(default=True)


def ensure_graph_plugin_available() -> None:
    """
    Raise ``ValueError`` when the graph plugin is requested but not installed.
    """
    from habit.compat.plugin_registries import get_legacy_habitat_feature_factory

    HabitatFeatureFactory = get_legacy_habitat_feature_factory()
    if HabitatFeatureFactory.get("graph") is None:
        raise ValueError(
            "feature_types includes 'graph' but the graph feature plugin is "
            "not installed. Graph topology features are only available in "
            "the private HABIT-v2 distribution."
        )


def load_graph_feature_config(graph_data: Any) -> GraphFeatureConfig:
    """
    Validate graph plugin configuration.

    Args:
        graph_data: Mapping parsed from the YAML ``graph:`` block.

    Returns:
        Validated ``GraphFeatureConfig`` instance.
    """
    ensure_graph_plugin_available()
    return GraphFeatureConfig.model_validate(graph_data or {})
