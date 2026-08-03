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
"""CLI for standalone DICOM sort via dcm2niix."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from habit.commands.common import (
    echo_fatal,
    echo_success,
    exit_with_error,
    load_config_or_exit,
    log_platform_info,
    safe_echo,
)
from habit.core.dicom_sort import DicomSortConfig, run_dicom_sort
from habit.utils.log_utils import setup_logger, stop_queue_listener


def run_sort_dicom(config_path: str) -> None:
    """
    Load ``DicomSortConfig`` and run a single dcm2niix sort.

    Args:
        config_path: Path to YAML configuration file.
    """
    try:
        config = load_config_or_exit(DicomSortConfig, config_path)
        output_dir = Path(config.output_dir or config.out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger = setup_logger(
            name="cli.dicom_sort",
            output_dir=output_dir,
            log_filename="processing.log",
            level=logging.INFO,
        )
        log_platform_info(logger, config_path)

        msg = f"Starting DICOM sort with config: {config_path}"
        logger.info(msg)
        safe_echo(msg)

        try:
            run_dicom_sort(config, logger=logger)
        except Exception as exc:  # noqa: BLE001
            logger.error("DICOM sort failed: %s", exc, exc_info=True)
            exit_with_error(f"Error: {exc}")
        else:
            success_msg = "DICOM sorting completed successfully!"
            logger.info(success_msg)
            echo_success(success_msg)
        finally:
            stop_queue_listener()
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        echo_fatal(exc)
        sys.exit(1)
