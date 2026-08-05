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
"""Traditional radiomics extraction command implementation.

L5 wiring only: validate the v0.1 YAML schema, hand the object to the L4
recipe, and surface success/failure. No ``habit.core.run`` imports live here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click

from habit.commands.common import (
    echo_success,
    exit_with_error,
    load_config_or_exit,
)
from habit.schemas import RadiomicsConfig
from habit.recipes.features import traditional_radiomics
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
        traditional_radiomics(config, logger=logger)
    except Exception as exc:  # noqa: BLE001
        logger.error("Radiomics extraction failed: %s", exc, exc_info=True)
        exit_with_error(f"Error: {exc}")

    echo_success("Radiomics extraction completed successfully!")
