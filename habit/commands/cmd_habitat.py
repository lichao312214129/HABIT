# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""Habitat analysis command implementation."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import click

from habit.cli_commands.common import (
    echo_error,
    echo_success,
    exit_with_error,
    load_config_or_exit,
)
from habit.core.habitat_analysis.config_schemas import HabitatAnalysisConfig
from habit.core.habitat_analysis.run import (
    apply_habitat_cli_overrides,
    run_habitat_analysis_from_config,
)
from habit.utils.log_utils import setup_logger, stop_queue_listener


def run_habitat(
    config_file: str,
    debug_mode: bool,
    mode: Optional[str],
    pipeline_path: Optional[str],
    resume: bool = False,
    exit_on_error: bool = True,
) -> None:
    """
    Run habitat analysis pipeline in train or predict mode.

    Args:
        config_file: Path to configuration YAML file.
        debug_mode: Whether to enable debug mode.
        mode: Override run mode (``train`` or ``predict``).
        pipeline_path: Override pipeline path for prediction.
        resume: Resume train run from individual-level checkpoint.
        exit_on_error: When True (CLI default), call ``sys.exit(1)`` on failure.
            GUI callers should pass False so exceptions propagate.
    """
    config = load_config_or_exit(HabitatAnalysisConfig, config_file)
    click.echo(f"Loaded configuration from: {config_file}")

    apply_habitat_cli_overrides(
        config,
        mode=mode,
        pipeline_path=pipeline_path,
        debug=debug_mode,
        resume=resume,
    )

    output_path = Path(config.out_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    log_level = logging.DEBUG if config.debug else logging.INFO
    logger = setup_logger(
        name="cli.habitat",
        output_dir=output_path,
        log_filename="habitat_analysis.log",
        level=log_level,
    )

    logger.info("==== Starting Habitat Analysis ====")
    logger.info("Config file: %s", config_file)
    logger.info("Full configuration: %s", config.model_dump())
    logger.info("=====================================")

    click.echo("Starting habitat analysis...")
    click.echo(f"  Mode: {config.run_mode}")
    click.echo(f"  Output directory: {config.out_dir}")
    if config.run_mode != "predict" and config.resume:
        click.echo("  Resume: enabled (skip checkpointed subjects)")
    if config.run_mode == "predict":
        click.echo(f"  Pipeline path: {config.pipeline_path or 'auto'}")
    click.echo(f"  Log file at: {output_path / 'habitat_analysis.log'}")

    try:
        run_habitat_analysis_from_config(
            config,
            logger=logger,
            output_dir=str(output_path),
        )
        logger.info("Habitat analysis completed successfully")
        echo_success("Habitat analysis completed successfully!")
    except Exception as exc:  # noqa: BLE001
        logger.error("Error during habitat analysis: %s", exc, exc_info=True)
        echo_error(f"Error: {exc}")
        if exit_on_error:
            sys.exit(1)
        raise
    finally:
        stop_queue_listener()
