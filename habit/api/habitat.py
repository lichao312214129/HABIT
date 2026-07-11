# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""Public habitat segmentation and feature-extraction API."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import pandas as pd

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
    if name == "apply_habitat_cli_overrides":
        from habit.core.habitat_analysis.run import apply_habitat_cli_overrides

        return apply_habitat_cli_overrides
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def run_habitat_analysis(
    config: "HabitatAnalysisConfig",
    logger: Optional[logging.Logger] = None,
) -> Optional[pd.DataFrame]:
    """
    Run habitat train or predict workflow from a validated config object.

    Args:
        config: Loaded habitat analysis configuration.
        logger: Optional logger passed to the core runner.

    Returns:
        Results dataframe when the core runner returns one; otherwise ``None``.
    """
    from habit.core.habitat_analysis.run import run_habitat_analysis_from_config

    return run_habitat_analysis_from_config(config, logger=logger)


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

    return load_feature_extraction_config_from_file(config_path)


def run_feature_extraction(
    config: "FeatureExtractionConfig",
    logger: Optional[logging.Logger] = None,
    plugin_configs: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Extract habitat-map features from a validated config object.

    Args:
        config: Loaded feature-extraction configuration.
        logger: Optional logger passed to the core runner.
        plugin_configs: Optional settings returned by
            :func:`load_feature_extraction_config` for plugin-backed features.
    """
    from habit.core.habitat_analysis.run import run_feature_extraction_from_config

    run_feature_extraction_from_config(
        config,
        logger=logger,
        plugin_configs=plugin_configs,
    )


def run_radiomics(
    config: "RadiomicsConfig",
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Run standalone radiomics extraction from a validated config object.

    Args:
        config: Loaded radiomics configuration.
        logger: Optional logger passed to the core runner.
    """
    from habit.core.habitat_analysis.run import run_radiomics_from_config

    run_radiomics_from_config(config, logger=logger)
