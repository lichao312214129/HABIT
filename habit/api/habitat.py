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
"""Public habitat segmentation and feature-extraction API."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Tuple, Union, cast

import pandas as pd

from habit.api.contracts import WorkflowResult, coerce_config
from habit.api.provenance import create_run_manifest, write_run_manifest

if TYPE_CHECKING:
    from habit.schemas.workflows.habitat import (
        FeatureExtractionConfig,
        HabitatAnalysisConfig,
        RadiomicsConfig,
    )

__all__ = [
    "HabitatAnalysisConfig",
    "FeatureExtractionConfig",
    "RadiomicsConfig",
    "apply_habitat_cli_overrides",
    "build_feature_extraction_config",
    "load_feature_extraction_config",
    "run_habitat_analysis",
    "run_feature_extraction",
    "run_radiomics",
]


def __getattr__(name: str) -> Any:
    if name in {
        "HabitatAnalysisConfig",
        "FeatureExtractionConfig",
        "RadiomicsConfig",
    }:
        from habit.schemas.workflows import habitat as habitat_schemas

        return getattr(habitat_schemas, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def apply_habitat_cli_overrides(
    config: Union["HabitatAnalysisConfig", Mapping[str, Any]],
    *,
    mode: Optional[str] = None,
    pipeline_path: Optional[str] = None,
    debug: bool = False,
    resume: bool = False,
) -> "HabitatAnalysisConfig":
    """
    Validate a config object or mapping before applying CLI-compatible overrides.

    Args:
        config: Validated habitat config or a dictionary accepted by its schema.
        mode: Optional override for ``run_mode``.
        pipeline_path: Optional override for the fitted pipeline path.
        debug: Whether to enable debug logging.
        resume: Whether to resume checkpointed work.

    Returns:
        The validated config instance with the requested overrides.
    """
    from habit.schemas.workflows.habitat import HabitatAnalysisConfig

    validated_config = coerce_config(config, HabitatAnalysisConfig)
    if debug:
        validated_config.debug = True
    if mode:
        validated_config.run_mode = mode
    if pipeline_path:
        validated_config.pipeline_path = pipeline_path
    if resume:
        validated_config.resume = True
    return validated_config


def run_habitat_analysis(
    config: Union["HabitatAnalysisConfig", Mapping[str, Any]],
    logger: Optional[logging.Logger] = None,
) -> WorkflowResult[pd.DataFrame]:
    """
    Run habitat train or predict workflow from a validated config object.

    Args:
        config: Validated config or dictionary accepted by
            :class:`~habit.schemas.workflows.habitat.HabitatAnalysisConfig`.
        logger: Optional logger passed to the core runner.

    Returns:
        Result table in ``data`` and declared output artifacts.
    """
    from habit.recipes.yaml_runner import run_habitat_config
    from habit.schemas.workflows.habitat import HabitatAnalysisConfig

    validated_config = coerce_config(config, HabitatAnalysisConfig)
    results = run_habitat_config(validated_config, logger=logger)
    pipeline_path = Path(validated_config.out_dir) / "habitat_model.habitatmodel"
    artifacts = {"pipeline": pipeline_path} if pipeline_path.is_file() else {}
    manifest = create_run_manifest(
        "habitat_analysis",
        validated_config,
        metadata={"run_mode": validated_config.run_mode},
    )
    manifest_path = write_run_manifest(manifest, validated_config.out_dir)
    return WorkflowResult(
        data=results,
        output_dir=validated_config.out_dir,
        artifacts=artifacts,
        metadata={
            "run_mode": validated_config.run_mode,
            "config_hash": manifest.config_hash,
            "habit_version": manifest.habit_version,
        },
        run_id=manifest.run_id,
        manifest_path=manifest_path,
    )


def load_feature_extraction_config(
    config_path: Union[str, Path],
) -> Tuple["FeatureExtractionConfig", Dict[str, Any]]:
    """
    Load feature-extraction YAML, including optional feature-plugin settings.

    Unlike ``FeatureExtractionConfig.from_file``, this function preserves and
    validates plugin-specific sections such as ``graph`` (validated as
    :class:`~habit.schemas.workflows.habitat.GraphFeatureBlock`, including its
    visualization settings).  Pass the returned ``plugin_configs`` to
    :func:`run_feature_extraction` for behavior identical to the
    ``habit extract-features`` CLI command.

    Args:
        config_path: Path to a feature-extraction YAML configuration file.

    Returns:
        Tuple containing the validated shared config and plugin config mapping.
    """
    from habit.utils.config_loader import load_config

    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    config_dict = load_config(str(path), resolve_paths=True)
    return build_feature_extraction_config(config_dict)


def build_feature_extraction_config(
    config: Union["FeatureExtractionConfig", Mapping[str, Any]],
) -> Tuple["FeatureExtractionConfig", Dict[str, Any]]:
    """
    Validate an in-memory feature-extraction dictionary, including plugins.

    Use this function when the configuration is constructed in Python and may
    contain plugin-specific sections such as ``graph``.  It is the dictionary
    equivalent of :func:`load_feature_extraction_config`.

    The ``graph:`` block is stripped before validating the shared schema (so
    ``FeatureExtractionConfig`` stays plugin-free) and validated separately as
    :class:`~habit.schemas.workflows.habitat.GraphFeatureBlock`; validation
    errors point at the offending ``graph`` field.

    Args:
        config: Validated shared config or a complete feature-extraction mapping.

    Returns:
        Tuple containing the validated shared config and plugin config mapping.
    """
    from habit.schemas.workflows.habitat import (
        FeatureExtractionConfig,
        GraphFeatureBlock,
    )

    if isinstance(config, FeatureExtractionConfig):
        return config, _plugin_configs_for_feature_types(config.feature_types)

    data = dict(config)
    graph_data = data.pop("graph", None)
    validated = FeatureExtractionConfig.model_validate(data)
    plugin_configs: Dict[str, Any] = {}
    if graph_data is not None:
        plugin_configs["graph"] = GraphFeatureBlock.model_validate(graph_data)
    elif "graph" in validated.feature_types:
        # Defaults apply when the family is requested without a settings block.
        plugin_configs["graph"] = GraphFeatureBlock()
    return validated, plugin_configs


def _plugin_configs_for_feature_types(
    feature_types: Any,
) -> Dict[str, Any]:
    """
    Build default plugin configs when only the feature-type names are known.

    Args:
        feature_types: Requested extraction feature type names.

    Returns:
        Plugin name to default config object mapping (may be empty).
    """
    from habit.schemas.workflows.habitat import GraphFeatureBlock

    if "graph" in list(feature_types or []):
        return {"graph": GraphFeatureBlock()}
    return {}


def run_feature_extraction(
    config: Union["FeatureExtractionConfig", Mapping[str, Any]],
    logger: Optional[logging.Logger] = None,
    plugin_configs: Optional[Mapping[str, Any]] = None,
) -> WorkflowResult[None]:
    """
    Extract habitat-map features from a validated config object.

    Delegates to :func:`habit.recipes.features.extract_habitat_features`, the
    domain-native recipe used by ``habit extract``. Optional plugins that are
    not registered in the domain registry still fall back to the compat
    analyzer inside the recipe.

    Args:
        config: Validated config or dictionary accepted by
            :class:`~habit.schemas.workflows.habitat.FeatureExtractionConfig`.
        logger: Optional logger passed to the recipe.
        plugin_configs: Optional settings returned by
            :func:`load_feature_extraction_config` or
            :func:`build_feature_extraction_config` for plugin-backed features.

    Returns:
        A result with the feature output directory in ``artifacts``.
    """
    from habit.recipes.features import extract_habitat_features

    # ``follow_imports = "skip"`` makes the recipe's return type Any to mypy,
    # so restate it here rather than letting warn_return_any fire.
    return cast(
        WorkflowResult[None],
        extract_habitat_features(
            config,
            plugin_configs=plugin_configs,
            logger=logger,
        ),
    )


def run_radiomics(
    config: Union["RadiomicsConfig", Mapping[str, Any]],
    logger: Optional[logging.Logger] = None,
) -> WorkflowResult[None]:
    """
    Run standalone radiomics extraction from a validated config object.

    Args:
        config: Validated config or dictionary accepted by
            :class:`~habit.schemas.workflows.habitat.RadiomicsConfig`.
        logger: Optional logger passed to the core runner.

    Returns:
        A result with the radiomics output directory in ``artifacts``.
    """
    from habit.recipes.radiomics import traditional_radiomics
    from habit.schemas.workflows.habitat import RadiomicsConfig

    validated_config = coerce_config(config, RadiomicsConfig)
    return traditional_radiomics(validated_config, logger=logger)
