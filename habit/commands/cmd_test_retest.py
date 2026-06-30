# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""Test-retest reproducibility analysis command implementation."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from habit.cli_commands.common import (
    echo_success,
    exit_with_error,
    load_config_or_exit,
)
from habit.core.habitat_analysis.configurator import HabitatConfigurator
from habit.core.machine_learning.config_schemas import TestRetestConfig
from habit.core.machine_learning.feature_selectors.icc.habitat_test_retest_mapper import (
    batch_process_files,
    configure_logging,
    find_habitat_mapping,
)
from habit.utils.log_utils import setup_logger


def run_test_retest(config_file: str) -> None:
    """
    Run test-retest reproducibility analysis.

    Args:
        config_file: Path to configuration YAML file.
    """
    config = load_config_or_exit(TestRetestConfig, config_file)

    output_dir = Path(config.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        name="cli.test_retest",
        output_dir=output_dir,
        log_filename="test_retest.log",
        level=logging.DEBUG if config.debug else logging.INFO,
    )

    msg = f"Starting test-retest analysis with config: {config_file}"
    logger.info(msg)
    click.echo(msg)

    try:
        configurator = HabitatConfigurator(
            config=config,
            logger=logger,
            output_dir=str(output_dir),
        )
        cfg = configurator.create_test_retest_analyzer()
        configure_logging(output_dir, cfg.debug)

        click.echo("Computing habitat mapping between test and retest data...")
        habitat_mapping = find_habitat_mapping(
            cfg.test_habitat_table,
            cfg.retest_habitat_table,
            cfg.features,
            cfg.similarity_method,
        )

        click.echo("Habitat mapping:")
        for retest_label, test_label in habitat_mapping.items():
            click.echo(f"  Retest Habitat {retest_label} -> Test Habitat {test_label}")
        logger.info("Habitat mapping: %s", habitat_mapping)

        click.echo(f"Processing files using {cfg.processes} processes...")
        batch_process_files(
            cfg.input_dir,
            habitat_mapping,
            cfg.out_dir,
            cfg.processes,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Test-retest analysis failed: %s", exc, exc_info=True)
        exit_with_error(f"Error: {exc}")

    logger.info("Test-retest analysis completed successfully")
    echo_success("Test-retest analysis completed successfully!")
