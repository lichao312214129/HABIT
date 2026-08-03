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
"""Feature extraction command implementation."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from habit.commands.common import (
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
