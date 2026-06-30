# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""Programmatic entry points for machine-learning workflows."""

from __future__ import annotations

import logging
from typing import Optional

from habit.core.machine_learning.config_schemas import MLConfig, ModelComparisonConfig
from habit.core.machine_learning.configurator import MLConfigurator
from habit.utils.log_utils import get_module_logger

_LOG = get_module_logger(__name__)


def apply_ml_mode_override(config: MLConfig, mode: Optional[str]) -> MLConfig:
    """
    Apply a CLI ``--mode`` override and re-validate cross-field invariants.

    Args:
        config: Loaded ML configuration.
        mode: Optional ``train`` or ``predict`` override.

    Returns:
        Config instance (re-validated when mode changes).
    """
    if mode is None or config.run_mode == mode:
        return config
    return MLConfig.model_validate({**config.model_dump(), "run_mode": mode})


def run_ml_from_config(
    config: MLConfig,
    *,
    logger: Optional[logging.Logger] = None,
    output_dir: Optional[str] = None,
) -> None:
    """
    Run holdout ML training or prediction.

    Args:
        config: Validated ML configuration (``run_mode`` selects train/predict).
        logger: Optional logger for the configurator.
        output_dir: Optional output directory override.

    Raises:
        Exception: Propagates workflow failures.
    """
    log = logger or _LOG
    out = output_dir or str(config.output)
    configurator = MLConfigurator(config=config, logger=log, output_dir=out)
    workflow = configurator.create_ml_workflow()
    log.info("Running ML workflow (mode=%s)", config.run_mode)
    workflow.run()
    log.info("ML workflow completed (mode=%s)", config.run_mode)


def run_kfold_from_config(
    config: MLConfig,
    *,
    logger: Optional[logging.Logger] = None,
    output_dir: Optional[str] = None,
) -> None:
    """
    Run K-fold cross-validation.

    Args:
        config: Validated ML configuration (must have ``run_mode='train'``).
        logger: Optional logger for the configurator.
        output_dir: Optional output directory override.

    Raises:
        ValueError: When ``run_mode`` is not ``train``.
        Exception: Propagates workflow failures.
    """
    if config.run_mode != "train":
        raise ValueError("K-fold cross-validation requires run_mode='train'.")
    log = logger or _LOG
    out = output_dir or str(config.output)
    configurator = MLConfigurator(config=config, logger=log, output_dir=out)
    workflow = configurator.create_kfold_workflow()
    log.info("Running K-fold cross-validation")
    workflow.run()
    log.info("K-fold cross-validation completed")


def run_model_comparison_from_config(
    config: ModelComparisonConfig,
    *,
    logger: Optional[logging.Logger] = None,
    output_dir: Optional[str] = None,
) -> None:
    """
    Run multi-model comparison (plots, metrics, DeLong tests).

    Args:
        config: Validated model-comparison configuration.
        logger: Optional logger for the configurator.
        output_dir: Optional output directory override.

    Raises:
        Exception: Propagates comparison workflow failures.
    """
    log = logger or _LOG
    out = output_dir or str(config.output_dir)
    configurator = MLConfigurator(config=config, logger=log, output_dir=out)
    comparison = configurator.create_model_comparison()
    log.info("Running model comparison")
    comparison.run()
    log.info("Model comparison completed")
