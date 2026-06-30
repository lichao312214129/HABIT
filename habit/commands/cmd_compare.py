# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""Model comparison command implementation."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from habit.cli_commands.common import (
    echo_success,
    exit_with_error,
    load_config_or_exit,
)
from habit.core.machine_learning.config_schemas import ModelComparisonConfig
from habit.core.machine_learning.run import run_model_comparison_from_config
from habit.utils.log_utils import setup_logger


def run_compare(config_file: str) -> None:
    """
    Run model comparison analysis.

    Args:
        config_file: Path to configuration YAML file.
    """
    config = load_config_or_exit(ModelComparisonConfig, config_file)
    click.echo(f"Loaded configuration from: {config_file}")

    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        name="cli.compare",
        output_dir=output_dir,
        log_filename="processing.log",
        level=logging.INFO,
    )

    msg = f"Starting model comparison with config: {config_file}"
    logger.info(msg)
    click.echo(msg)

    try:
        run_model_comparison_from_config(
            config,
            logger=logger,
            output_dir=str(output_dir),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Model comparison failed: %s", exc, exc_info=True)
        exit_with_error(f"Error: {exc}")

    logger.info("Model comparison completed successfully")
    echo_success("Model comparison completed successfully!")
