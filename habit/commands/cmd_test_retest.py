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
"""Test-retest reproducibility analysis command implementation.

L5 wiring only: parse the v0.1 YAML through the public schema, hand the
validated object to the L4 recipe, echo the discovered label mapping, and
surface success/failure. No ``habit.core`` imports live here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click

from habit.api.analysis import TestRetestConfig
from habit.commands.common import (
    echo_success,
    exit_with_error,
    load_config_or_exit,
)
from habit.recipes.test_retest import test_retest_analysis
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
        click.echo("Computing habitat mapping between test and retest data...")
        click.echo(f"Processing files using {config.processes} processes...")
        result = test_retest_analysis(config, logger=logger)

        habitat_mapping = result.data or {}
        click.echo("Habitat mapping:")
        for retest_label, test_label in habitat_mapping.items():
            click.echo(f"  Retest Habitat {retest_label} -> Test Habitat {test_label}")
    except Exception as exc:  # noqa: BLE001
        logger.error("Test-retest analysis failed: %s", exc, exc_info=True)
        exit_with_error(f"Error: {exc}")

    logger.info("Test-retest analysis completed successfully")
    echo_success("Test-retest analysis completed successfully!")
