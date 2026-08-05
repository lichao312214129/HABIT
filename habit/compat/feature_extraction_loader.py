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
"""Load feature-extraction YAML with optional private plugin sections (L1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, Union

from habit.schemas.workflows.habitat import FeatureExtractionConfig
from habit.utils.config_loader import load_config

__all__ = [
    "load_feature_extraction_config_from_file",
    "parse_feature_extraction_config",
    "plugin_configs_for_feature_types",
]


def parse_feature_extraction_config(
    raw: Union[Dict[str, Any], FeatureExtractionConfig],
) -> Tuple[FeatureExtractionConfig, Dict[str, Any]]:
    """
    Split a feature-extraction config into shared schema and plugin configs.

    The ``graph:`` YAML block is stripped before validating
    ``FeatureExtractionConfig`` so the public schema stays plugin-free.

    Args:
        raw: Parsed YAML dict or an already validated config object.

    Returns:
        Tuple of (validated FeatureExtractionConfig, plugin_configs mapping).
    """
    if isinstance(raw, FeatureExtractionConfig):
        return raw, plugin_configs_for_feature_types(raw.feature_types)

    data = dict(raw)
    graph_data = data.pop("graph", None)
    cfg = FeatureExtractionConfig.model_validate(data)
    plugin_configs: Dict[str, Any] = {}

    if graph_data is not None:
        plugin_configs["graph"] = _load_graph_config(graph_data)
    elif "graph" in cfg.feature_types:
        plugin_configs["graph"] = _load_graph_config({})

    return cfg, plugin_configs


def plugin_configs_for_feature_types(
    feature_types: list[str],
) -> Dict[str, Any]:
    """
    Build default plugin configs when only feature_types are known.

    Args:
        feature_types: Requested extraction feature type names.

    Returns:
        Plugin name to config object mapping (may be empty).
    """
    plugin_configs: Dict[str, Any] = {}
    if "graph" in feature_types:
        plugin_configs["graph"] = _load_graph_config({})
    return plugin_configs


def _ensure_graph_plugin_available() -> None:
    """
    Raise ValueError when the graph plugin is requested but not installed.

    Graph handlers still live in the optional v0.1 habitat_features package;
    this check delegates through ``habit.compat.graph_plugin`` so this loader
    stays free of direct ``habit.core`` imports.
    """
    from habit.compat.graph_plugin import ensure_graph_plugin_available

    ensure_graph_plugin_available()


def _load_graph_config(graph_data: Any) -> Any:
    """Load graph plugin config or raise when the plugin is unavailable."""
    from habit.compat.graph_plugin import load_graph_feature_config

    _ensure_graph_plugin_available()
    return load_graph_feature_config(graph_data)


def load_feature_extraction_config_from_file(
    config_path: Union[str, Path],
) -> Tuple[FeatureExtractionConfig, Dict[str, Any]]:
    """
    Load and validate a feature extraction YAML including plugin sections.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Tuple of (FeatureExtractionConfig, plugin_configs mapping).
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    config_dict = load_config(str(path), resolve_paths=True)
    return parse_feature_extraction_config(config_dict)
