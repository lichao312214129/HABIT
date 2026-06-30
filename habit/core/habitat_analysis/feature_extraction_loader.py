"""Load feature extraction YAML with optional private plugin sections."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, Union

from habit.core.common.configs.loader import load_config
from habit.core.habitat_analysis.config_schemas import FeatureExtractionConfig
from habit.core.habitat_analysis.feature_registry import ensure_graph_plugin_available


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


def _load_graph_config(graph_data: Any) -> Any:
    """Load graph plugin config or raise when the plugin is unavailable."""
    ensure_graph_plugin_available()
    from habit.core.habitat_analysis.habitat_features.graph_features.config import (
        GraphFeatureConfig,
    )

    return GraphFeatureConfig.model_validate(graph_data)


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
