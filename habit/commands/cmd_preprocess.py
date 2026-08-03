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
"""Image preprocessing command implementation."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from habit.commands.common import (
    echo_fatal,
    echo_success,
    exit_with_error,
    load_config_or_exit,
    log_platform_info,
)
from habit.core.preprocessing.config_schemas import PreprocessingConfig
from habit.core.preprocessing.run import run_preprocess_from_config
from habit.utils.log_utils import setup_logger, stop_queue_listener


def run_preprocess(config_path: str) -> None:
    """
    Run image preprocessing pipeline.

    Args:
        config_path: Path to configuration YAML file.
    """
    try:
        config = load_config_or_exit(PreprocessingConfig, config_path)
        output_dir = Path(config.out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger = setup_logger(
            name="cli.preprocessing",
            output_dir=output_dir,
            log_filename="processing.log",
            level=logging.INFO,
        )
        log_platform_info(logger, config_path)

        msg = f"Starting image preprocessing with config: {config_path}"
        logger.info(msg)
        click.echo(msg)

        try:
            run_preprocess_from_config(config, logger=logger)
        except Exception as exc:  # noqa: BLE001
            logger.error("Preprocessing failed: %s", exc, exc_info=True)
            exit_with_error(f"Error: {exc}")
        finally:
            stop_queue_listener()

        success_msg = "Image preprocessing completed successfully!"
        logger.info(success_msg)
        echo_success(success_msg)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        echo_fatal(exc)
        sys.exit(1)
