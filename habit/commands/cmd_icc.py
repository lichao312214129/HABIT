# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""ICC analysis command implementation."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from habit.cli_commands.common import (
    echo_success,
    exit_with_error,
    load_config_or_exit,
)
from habit.core.machine_learning.feature_selectors.icc.config import ICCConfig
from habit.core.machine_learning.feature_selectors.icc.icc import (
    run_icc_analysis_from_config,
)
from habit.utils.log_utils import setup_logger


def run_icc(config_file: str) -> None:
    """
    Run ICC analysis from a configuration file.

    Args:
        config_file: Path to the configuration YAML file.
    """
    config = load_config_or_exit(ICCConfig, config_file)

    output_path = Path(config.output.path)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        name="habit.icc",
        output_dir=str(output_dir),
        log_filename="icc_analysis.log",
        level=logging.DEBUG if config.debug else logging.INFO,
    )

    logger.info("Successfully loaded config: %s", config_file)
    click.echo(f"Starting ICC analysis with config: {config_file}")

    try:
        run_icc_analysis_from_config(config)
    except Exception as exc:  # noqa: BLE001
        logger.error("ICC analysis failed: %s", exc, exc_info=True)
        exit_with_error(f"Error: {exc}")

    echo_success("ICC analysis completed successfully!")
    logger.info("ICC analysis process finished.")
