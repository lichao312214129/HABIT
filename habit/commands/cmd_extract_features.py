# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""Feature extraction command implementation."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from habit.cli_commands.common import (
    echo_success,
    exit_with_error,
)
from habit.core.habitat_analysis.feature_extraction_loader import (
    load_feature_extraction_config_from_file,
)
from habit.core.habitat_analysis.run import run_feature_extraction_from_config
from habit.utils.log_utils import setup_logger


def run_extract_features(config_file: str) -> None:
    """
    Run habitat feature extraction pipeline.

    Args:
        config_file: Path to configuration YAML file.
    """
    config, plugin_configs = load_feature_extraction_config_from_file(config_file)

    output_dir = Path(config.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_level = logging.DEBUG if config.debug else logging.INFO
    logger = setup_logger(
        name="cli.extract_features",
        output_dir=output_dir,
        log_filename="processing.log",
        level=log_level,
    )

    msg = f"Starting habitat feature extraction with config: {config_file}"
    logger.info(msg)
    click.echo(msg)

    try:
        run_feature_extraction_from_config(
            config,
            plugin_configs=plugin_configs,
            logger=logger,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Feature extraction failed: %s", exc, exc_info=True)
        exit_with_error(f"Error: {exc}")

    echo_success("Feature extraction completed successfully!")
