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
"""ICC analysis command implementation.

L5 wiring only: parse the v0.1 YAML through the public schema, hand the
validated object to the L4 recipe, and surface success/failure. No
``habit.core`` imports live here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click

from habit.api.analysis import ICCConfig
from habit.commands.common import (
    echo_success,
    exit_with_error,
    load_config_or_exit,
)
from habit.recipes.icc import icc_analysis
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
        icc_analysis(config)
    except Exception as exc:  # noqa: BLE001
        logger.error("ICC analysis failed: %s", exc, exc_info=True)
        exit_with_error(f"Error: {exc}")

    echo_success("ICC analysis completed successfully!")
    logger.info("ICC analysis process finished.")
