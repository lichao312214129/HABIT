# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""Traditional radiomics extraction command implementation."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from habit.cli_commands.common import (
    echo_success,
    exit_with_error,
    load_config_or_exit,
)
from habit.core.habitat_analysis.config_schemas import RadiomicsConfig
from habit.core.habitat_analysis.run import run_radiomics_from_config
from habit.utils.log_utils import setup_logger


def run_radiomics(config_file: str) -> None:
    """
    Run traditional radiomics feature extraction.

    Args:
        config_file: Path to configuration YAML file.
    """
    config = load_config_or_exit(RadiomicsConfig, config_file)

    output_dir = Path(config.out_dir or config.paths.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        name="cli.radiomics",
        output_dir=output_dir,
        log_filename="radiomics_extraction.log",
        level=logging.INFO,
    )

    msg = f"Starting traditional radiomics extraction with config: {config_file}"
    logger.info(msg)
    click.echo(msg)

    try:
        run_radiomics_from_config(
            config,
            logger=logger,
            output_dir=str(output_dir),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Radiomics extraction failed: %s", exc, exc_info=True)
        exit_with_error(f"Error: {exc}")

    echo_success("Radiomics extraction completed successfully!")
