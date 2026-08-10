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

"""Deprecated shim for the feature-extraction plugin config loader (L1 compat).

The real implementation lives in :mod:`habit.api.habitat`
(:func:`~habit.api.habitat.load_feature_extraction_config` /
:func:`~habit.api.habitat.build_feature_extraction_config`), which validates
the optional ``graph:`` YAML block without touching the compat layer. These
wrappers keep the historical import paths working for the deprecation period
and will be removed in a future release.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from habit.schemas.workflows.habitat import FeatureExtractionConfig, GraphFeatureBlock

__all__ = [
    "load_feature_extraction_config_from_file",
    "parse_feature_extraction_config",
    "plugin_configs_for_feature_types",
]

_DEPRECATION_MESSAGE = (
    "habit.compat.feature_extraction_loader is deprecated and will be removed "
    "in a future release; use "
    "habit.api.habitat.load_feature_extraction_config / "
    "habit.api.habitat.build_feature_extraction_config instead."
)

#: Process-local flag so the deprecation warning is emitted once per module,
#: not on every call (the v0.1 configurators call these wrappers per run).
_WARNED: bool = False


def _warn_deprecated_once() -> None:
    """Emit the module deprecation warning on first use only."""
    global _WARNED
    if _WARNED:
        return
    _WARNED = True
    warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=3)


def parse_feature_extraction_config(
    raw: Union[Dict[str, Any], FeatureExtractionConfig],
) -> Tuple[FeatureExtractionConfig, Dict[str, Any]]:
    """
    Split a feature-extraction config into shared schema and plugin configs.

    .. deprecated::
        Use :func:`habit.api.habitat.build_feature_extraction_config`.

    Args:
        raw: Parsed YAML dict or an already validated config object.

    Returns:
        Tuple of (validated FeatureExtractionConfig, plugin_configs mapping).
    """
    _warn_deprecated_once()
    from habit.api.habitat import build_feature_extraction_config

    return build_feature_extraction_config(raw)


def plugin_configs_for_feature_types(
    feature_types: List[str],
) -> Dict[str, Any]:
    """
    Build default plugin configs when only feature_types are known.

    .. deprecated::
        Use :func:`habit.api.habitat.build_feature_extraction_config` on a
        full mapping, which derives the same defaults.

    Args:
        feature_types: Requested extraction feature type names.

    Returns:
        Plugin name to config object mapping (may be empty).
    """
    _warn_deprecated_once()
    plugin_configs: Dict[str, Any] = {}
    if "graph" in feature_types:
        plugin_configs["graph"] = GraphFeatureBlock()
    return plugin_configs


def load_feature_extraction_config_from_file(
    config_path: Union[str, Path],
) -> Tuple[FeatureExtractionConfig, Dict[str, Any]]:
    """
    Load and validate a feature extraction YAML including plugin sections.

    .. deprecated::
        Use :func:`habit.api.habitat.load_feature_extraction_config`.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Tuple of (FeatureExtractionConfig, plugin_configs mapping).
    """
    _warn_deprecated_once()
    from habit.api.habitat import load_feature_extraction_config

    return load_feature_extraction_config(config_path)
