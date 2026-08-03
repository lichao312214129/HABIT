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
    from habit.core.habitat_analysis.config_schemas import (
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
        from habit.core.habitat_analysis import config_schemas

        return getattr(config_schemas, name)
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
    from habit.core.habitat_analysis.config_schemas import HabitatAnalysisConfig
    from habit.core.habitat_analysis.run import (
        apply_habitat_cli_overrides as _apply_habitat_cli_overrides,
    )

    validated_config = coerce_config(config, HabitatAnalysisConfig)
    return _apply_habitat_cli_overrides(
        validated_config,
        mode=mode,
        pipeline_path=pipeline_path,
        debug=debug,
        resume=resume,
    )


def run_habitat_analysis(
    config: Union["HabitatAnalysisConfig", Mapping[str, Any]],
    logger: Optional[logging.Logger] = None,
) -> WorkflowResult[pd.DataFrame]:
    """
    Run habitat train or predict workflow from a validated config object.

    Args:
        config: Validated config or dictionary accepted by
            :class:`~habit.core.habitat_analysis.config_schemas.HabitatAnalysisConfig`.
        logger: Optional logger passed to the core runner.

    Returns:
        Result table in ``data`` and declared output artifacts.
    """
    from habit.core.habitat_analysis.config_schemas import HabitatAnalysisConfig
    from habit.core.habitat_analysis.run import run_habitat_analysis_from_config

    validated_config = coerce_config(config, HabitatAnalysisConfig)
    results = run_habitat_analysis_from_config(validated_config, logger=logger)
    pipeline_path = Path(validated_config.out_dir) / "habitat_pipeline.pkl"
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
    validates plugin-specific sections such as ``graph``.  Pass the returned
    ``plugin_configs`` to :func:`run_feature_extraction` for behavior identical
    to the ``habit extract-features`` CLI command.

    Args:
        config_path: Path to a feature-extraction YAML configuration file.

    Returns:
        Tuple containing the validated shared config and plugin config mapping.
    """
    from habit.core.habitat_analysis.feature_extraction_loader import (
        load_feature_extraction_config_from_file,
    )

    return cast(
        Tuple["FeatureExtractionConfig", Dict[str, Any]],
        load_feature_extraction_config_from_file(config_path),
    )


def build_feature_extraction_config(
    config: Union["FeatureExtractionConfig", Mapping[str, Any]],
) -> Tuple["FeatureExtractionConfig", Dict[str, Any]]:
    """
    Validate an in-memory feature-extraction dictionary, including plugins.

    Use this function when the configuration is constructed in Python and may
    contain plugin-specific sections such as ``graph``.  It is the dictionary
    equivalent of :func:`load_feature_extraction_config`.

    Args:
        config: Validated shared config or a complete feature-extraction mapping.

    Returns:
        Tuple containing the validated shared config and plugin config mapping.
    """
    from habit.core.habitat_analysis.feature_extraction_loader import (
        parse_feature_extraction_config,
    )

    if isinstance(config, Mapping):
        return cast(
            Tuple["FeatureExtractionConfig", Dict[str, Any]],
            parse_feature_extraction_config(dict(config)),
        )
    return cast(
        Tuple["FeatureExtractionConfig", Dict[str, Any]],
        parse_feature_extraction_config(config),
    )


def run_feature_extraction(
    config: Union["FeatureExtractionConfig", Mapping[str, Any]],
    logger: Optional[logging.Logger] = None,
    plugin_configs: Optional[Mapping[str, Any]] = None,
) -> WorkflowResult[None]:
    """
    Extract habitat-map features from a validated config object.

    Args:
        config: Validated config or dictionary accepted by
            :class:`~habit.core.habitat_analysis.config_schemas.FeatureExtractionConfig`.
        logger: Optional logger passed to the core runner.
        plugin_configs: Optional settings returned by
            :func:`load_feature_extraction_config` or
            :func:`build_feature_extraction_config` for plugin-backed features.

    Returns:
        A result with the feature output directory in ``artifacts``.
    """
    from habit.core.habitat_analysis.run import run_feature_extraction_from_config

    if isinstance(config, Mapping):
        validated_config, inferred_plugin_configs = build_feature_extraction_config(
            config
        )
        resolved_plugin_configs: Optional[Dict[str, Any]] = (
            dict(plugin_configs)
            if plugin_configs is not None
            else inferred_plugin_configs
        )
    else:
        from habit.core.habitat_analysis.config_schemas import FeatureExtractionConfig

        validated_config = coerce_config(config, FeatureExtractionConfig)
        resolved_plugin_configs = (
            dict(plugin_configs) if plugin_configs is not None else None
        )
    run_feature_extraction_from_config(
        validated_config,
        logger=logger,
        plugin_configs=resolved_plugin_configs,
    )
    manifest = create_run_manifest(
        "feature_extraction",
        validated_config,
        metadata={"plugins": sorted((resolved_plugin_configs or {}).keys())},
    )
    manifest_path = write_run_manifest(manifest, validated_config.out_dir)
    return WorkflowResult(
        output_dir=validated_config.out_dir,
        metadata={
            "config_hash": manifest.config_hash,
            "habit_version": manifest.habit_version,
        },
        run_id=manifest.run_id,
        manifest_path=manifest_path,
    )


def run_radiomics(
    config: Union["RadiomicsConfig", Mapping[str, Any]],
    logger: Optional[logging.Logger] = None,
) -> WorkflowResult[None]:
    """
    Run standalone radiomics extraction from a validated config object.

    Args:
        config: Validated config or dictionary accepted by
            :class:`~habit.core.habitat_analysis.config_schemas.RadiomicsConfig`.
        logger: Optional logger passed to the core runner.

    Returns:
        A result with the radiomics output directory in ``artifacts``.
    """
    from habit.core.habitat_analysis.config_schemas import RadiomicsConfig
    from habit.core.habitat_analysis.run import run_radiomics_from_config

    validated_config = coerce_config(config, RadiomicsConfig)
    run_radiomics_from_config(validated_config, logger=logger)
    output_dir = validated_config.out_dir or validated_config.paths.out_dir
    manifest = create_run_manifest("radiomics", validated_config)
    manifest_path = write_run_manifest(manifest, output_dir)
    return WorkflowResult(
        output_dir=output_dir,
        metadata={
            "config_hash": manifest.config_hash,
            "habit_version": manifest.habit_version,
        },
        run_id=manifest.run_id,
        manifest_path=manifest_path,
    )
