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

"""Deprecated shim for the v0.1 graph feature-plugin config loader (L1 compat).

Graph topology features are built into HABIT as the ``graph`` domain feature
family (:class:`habit.domain.habitat_features.GraphHabitatFeatures`), and the
``graph:`` YAML block is validated as
:class:`habit.schemas.workflows.habitat.GraphFeatureBlock` by
:func:`habit.api.habitat.load_feature_extraction_config` without touching the
compat layer. These wrappers keep the historical import paths working for the
deprecation period and will be removed in a future release.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict

from habit.domain.habitat_features.graph import GraphHabitatFeaturesParams
from habit.schemas.workflows.habitat import GraphFeatureBlock

__all__ = ["ensure_graph_plugin_available", "load_graph_feature_config"]

_DEPRECATION_MESSAGE = (
    "habit.compat.graph_plugin is deprecated and will be removed in a future "
    "release; the graph feature family is built into the domain registry and "
    "its YAML block is validated by "
    "habit.api.habitat.load_feature_extraction_config."
)

#: Process-local flag so the deprecation warning is emitted once per module.
_WARNED: bool = False


def _warn_deprecated_once() -> None:
    """Emit the module deprecation warning on first use only."""
    global _WARNED
    if _WARNED:
        return
    _WARNED = True
    warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=3)


def ensure_graph_plugin_available() -> None:
    """
    Deprecated no-op kept for backward compatibility.

    Graph topology features shipped as an optional v0.1 plugin; they are now a
    built-in domain feature family, so there is nothing left to check.
    """
    _warn_deprecated_once()
    return None


def load_graph_feature_config(graph_data: Any) -> GraphHabitatFeaturesParams:
    """
    Validate a v0.1 ``graph:`` YAML block as domain extractor parameters.

    .. deprecated::
        Use :func:`habit.api.habitat.load_feature_extraction_config`, which
        preserves the block's visualization settings instead of dropping them.

    Args:
        graph_data: Mapping parsed from the YAML ``graph:`` block. Extraction
            parameter names match
            :class:`~habit.domain.habitat_features.GraphHabitatFeaturesParams`
            one-to-one; visualization keys are validated (through
            :class:`~habit.schemas.workflows.habitat.GraphFeatureBlock`) but
            not part of the returned extraction-only model.

    Returns:
        Validated ``GraphHabitatFeaturesParams`` instance.
    """
    _warn_deprecated_once()
    block = GraphFeatureBlock.model_validate(dict(graph_data or {}))
    extraction_data: Dict[str, Any] = {
        field: getattr(block, field)
        for field in GraphHabitatFeaturesParams.model_fields
    }
    return GraphHabitatFeaturesParams.model_validate(extraction_data)
