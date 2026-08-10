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

"""Tests for the deprecated compat graph-loader shims.

The shims keep the v0.1 import paths working (with a ``DeprecationWarning``)
while the real implementation lives in :mod:`habit.api.habitat`. They are
deleted once the deprecation period ends, together with these tests.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any, Dict

import pytest


def _fresh_import(module_name: str) -> Any:
    """Re-import a shim module with its warn-once flag reset."""
    for name in list(sys.modules):
        if name == module_name:
            del sys.modules[name]
    return importlib.import_module(module_name)


def _extract_mapping() -> Dict[str, Any]:
    """Minimal feature-extraction mapping with a graph block."""
    return {
        "raw_img_folder": "raw",
        "habitats_map_folder": "habitats",
        "out_dir": "features",
        "feature_types": ["graph"],
        "graph": {"distance_threshold": 7.0, "visualize": True},
    }


@pytest.mark.unit
def test_compat_feature_extraction_loader_warns_and_delegates() -> None:
    """The loader shim warns once, then returns the new path's objects."""
    module = _fresh_import("habit.compat.feature_extraction_loader")

    with pytest.warns(DeprecationWarning, match="habit.api.habitat"):
        config, plugins = module.parse_feature_extraction_config(
            _extract_mapping()
        )

    from habit.schemas.workflows.habitat import (
        FeatureExtractionConfig,
        GraphFeatureBlock,
    )

    assert isinstance(config, FeatureExtractionConfig)
    assert isinstance(plugins["graph"], GraphFeatureBlock)
    # The shim preserves the visualization settings (it delegates to the new
    # implementation instead of dropping keys like the v0.1 filter did).
    assert plugins["graph"].visualize is True
    assert plugins["graph"].distance_threshold == 7.0


@pytest.mark.unit
def test_compat_loader_warns_only_once() -> None:
    """The deprecation warning fires on first use, not on every call."""
    module = _fresh_import("habit.compat.feature_extraction_loader")

    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module.parse_feature_extraction_config(_extract_mapping())
        module.plugin_configs_for_feature_types(["graph"])
        module.parse_feature_extraction_config(_extract_mapping())

    deprecation = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecation) == 1


@pytest.mark.unit
def test_compat_graph_plugin_warns_and_returns_extraction_params() -> None:
    """The graph plugin shim keeps its extraction-only return contract."""
    module = _fresh_import("habit.compat.graph_plugin")

    with pytest.warns(DeprecationWarning, match="domain registry"):
        params = module.load_graph_feature_config(
            {"distance_threshold": 9.0, "visualize": True, "n_workers": 2}
        )

    from habit.domain.habitat_features.graph import GraphHabitatFeaturesParams

    assert isinstance(params, GraphHabitatFeaturesParams)
    assert params.distance_threshold == 9.0
    # Visualization keys are validated but not part of the extraction model.
    assert not hasattr(params, "visualize")


@pytest.mark.unit
def test_compat_graph_plugin_ensure_is_deprecated_noop() -> None:
    """The shim's availability check warns on first use and never raises."""
    module = _fresh_import("habit.compat.graph_plugin")

    with pytest.warns(DeprecationWarning, match="domain registry"):
        module.ensure_graph_plugin_available()

    # Warn-once is module-level: any second call stays silent.
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module.ensure_graph_plugin_available()
        module.load_graph_feature_config({"distance_threshold": 3.0})
    assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]


@pytest.mark.unit
def test_engine_ensure_graph_plugin_available_is_deprecated_noop() -> None:
    """The v0.1 factory's graph check warns and never raises anymore."""
    module = _fresh_import(
        "habit.compat.engines.habitat_extraction.feature_registry"
    )
    module._GRAPH_ENSURE_WARNED = False

    with pytest.warns(DeprecationWarning, match="domain feature family"):
        module.ensure_graph_plugin_available()

    # Second call stays silent (warn-once) and still does not raise.
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module.ensure_graph_plugin_available()
    assert not [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ]


@pytest.mark.unit
def test_engine_validate_feature_types_points_graph_to_domain() -> None:
    """The legacy factory's error text no longer references HABIT-v2."""
    from habit.compat.engines.habitat_extraction.feature_registry import (
        validate_feature_types,
    )

    with pytest.raises(ValueError, match="domain") as excinfo:
        validate_feature_types(["definitely_unknown_family"])
    assert "HABIT-v2" not in str(excinfo.value)
