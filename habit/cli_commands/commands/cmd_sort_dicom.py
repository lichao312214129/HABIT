"""
CLI for standalone DICOM sort via dcm2niix (separate from ``habit preprocess``).
"""

from __future__ import annotations

import logging
import platform
import sys
import traceback
from pathlib import Path

import click

from habit.core.dicom_sort import DicomSortConfig, run_dicom_sort
from habit.utils.log_utils import setup_logger


def run_sort_dicom(config_path: str) -> None:
    """
    Load ``DicomSortConfig`` and run a single dcm2niix sort.

    Args:
        config_path: Path to YAML configuration file.
    """
    if not config_path:
        click.echo("Error: --config is required", err=True)
        sys.exit(1)

    if not Path(config_path).is_file():
        click.echo(f"Error: Configuration file not found: {config_path}", err=True)
        sys.exit(1)

    try:
        config = DicomSortConfig.from_file(config_path)
        output_dir = Path(config.output_dir or config.out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger = setup_logger(
            name="cli.dicom_sort",
            output_dir=output_dir,
            log_filename="processing.log",
            level=logging.INFO,
        )

        logger.info(f"Python version: {sys.version}")
        logger.info(f"Platform: {platform.platform()}")
        logger.info(f"Using configuration file: {config_path}")

        msg = f"Starting DICOM sort with config: {config_path}"
        logger.info(msg)
        click.echo(msg)

        try:
            run_dicom_sort(config, logger=logger)
        except Exception as e:
            error_msg = f"DICOM sort failed: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            click.echo(error_msg, err=True)
            sys.exit(1)

        success_msg = "DICOM sorting completed successfully!"
        logger.info(success_msg)
        click.secho(f"✓ {success_msg}", fg="green")

    except Exception as e:
        click.echo(f"Fatal error: {e}", err=True)
        click.echo(traceback.format_exc(), err=True)
        sys.exit(1)
