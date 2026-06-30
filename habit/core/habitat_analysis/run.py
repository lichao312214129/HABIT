# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""Programmatic entry points for habitat analysis and feature extraction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from habit.core.habitat_analysis.config_schemas import (
    FeatureExtractionConfig,
    HabitatAnalysisConfig,
    RadiomicsConfig,
)
from habit.core.habitat_analysis.configurator import HabitatConfigurator
from habit.utils.log_utils import get_module_logger

_LOG = get_module_logger(__name__)


def apply_habitat_cli_overrides(
    config: HabitatAnalysisConfig,
    *,
    mode: Optional[str] = None,
    pipeline_path: Optional[str] = None,
    debug: bool = False,
    resume: bool = False,
) -> HabitatAnalysisConfig:
    """
    Apply CLI-style overrides onto a loaded habitat config.

    Args:
        config: Base habitat analysis configuration.
        mode: Optional override for ``run_mode`` (``train`` or ``predict``).
        pipeline_path: Optional override for ``pipeline_path``.
        debug: When True, force ``config.debug = True``.
        resume: When True, force ``config.resume = True``.

    Returns:
        The same config instance (mutated in place).
    """
    if debug:
        config.debug = True
    if mode:
        config.run_mode = mode
    if pipeline_path:
        config.pipeline_path = pipeline_path
    if resume:
        config.resume = True
    return config


def run_habitat_analysis_from_config(
    config: HabitatAnalysisConfig,
    *,
    logger: Optional[logging.Logger] = None,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run habitat segmentation in train or predict mode.

    Predict mode requires a valid ``pipeline_path`` on the config.

    Args:
        config: Validated habitat analysis configuration.
        logger: Optional logger for the configurator.
        output_dir: Optional output directory override for the configurator.

    Returns:
        Habitat clustering results dataframe.

    Raises:
        ValueError: When predict mode is requested without ``pipeline_path``.
        FileNotFoundError: When the pipeline file does not exist.
    """
    log = logger or _LOG
    out = output_dir or str(config.out_dir)
    configurator = HabitatConfigurator(config=config, logger=log, output_dir=out)
    analysis = configurator.create_habitat_analysis()

    if config.run_mode == "predict":
        if not config.pipeline_path:
            raise ValueError(
                "In 'predict' mode, pipeline_path is required in the YAML "
                "or via CLI override."
            )
        resolved_pipeline = Path(config.pipeline_path)
        if not resolved_pipeline.exists():
            raise FileNotFoundError(
                f"Pipeline file not found: {resolved_pipeline}"
            )
        return analysis.predict(
            pipeline_path=str(resolved_pipeline),
            save_results_csv=config.save_results_csv,
        )

    return analysis.fit(save_results_csv=config.save_results_csv)


def run_feature_extraction_from_config(
    config: FeatureExtractionConfig,
    *,
    plugin_configs: Optional[dict] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Run habitat feature extraction from a validated config.

    Args:
        config: Validated feature-extraction configuration.
        plugin_configs: Optional private plugin configuration mapping.
        logger: Optional logger for the configurator.

    Raises:
        Exception: Propagates extractor failures.
    """
    log = logger or _LOG
    configurator = HabitatConfigurator(
        config=config,
        logger=log,
        plugin_configs=plugin_configs,
    )
    extractor = configurator.create_feature_extractor()
    log.info("Executing feature extraction")
    extractor.run(
        feature_types=config.feature_types,
        n_habitats=config.n_habitats,
    )
    log.info("Feature extraction completed")


def run_radiomics_from_config(
    config: RadiomicsConfig,
    *,
    logger: Optional[logging.Logger] = None,
    output_dir: Optional[str] = None,
) -> None:
    """
    Run traditional radiomics extraction from a validated config.

    Args:
        config: Validated radiomics configuration.
        logger: Optional logger for the configurator.
        output_dir: Optional output directory override.

    Raises:
        Exception: Propagates extractor failures.
    """
    log = logger or _LOG
    out = output_dir or str(config.out_dir or config.paths.out_dir)
    configurator = HabitatConfigurator(config=config, logger=log, output_dir=out)
    extractor = configurator.create_radiomics_extractor()
    log.info("Executing radiomics extraction")
    extractor.extract_features()
    log.info("Radiomics extraction completed")
