# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""Shared helpers for HABIT CLI command implementations."""

from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Optional, Type, TypeVar

import click

from habit.utils.log_utils import setup_logger

TConfig = TypeVar("TConfig")

CONFIG_REQUIRED_MSG: str = (
    "Error: Configuration file is required. Use --config / -c."
)


def require_config_path(config_path: Optional[str]) -> str:
    """
    Validate that a config path was provided on the CLI.

    Args:
        config_path: Path from Click (may be None when optional at decorator level).

    Returns:
        Non-empty config path string.

    Raises:
        SystemExit: When ``config_path`` is missing or blank.
    """
    if not config_path or not str(config_path).strip():
        exit_with_error(CONFIG_REQUIRED_MSG)
    return str(config_path)


def load_config_or_exit(config_cls: Type[TConfig], config_path: str) -> TConfig:
    """
    Load a typed config via ``config_cls.from_file`` with uniform CLI errors.

    Args:
        config_cls: Pydantic config class exposing ``from_file``.
        config_path: Path to the YAML configuration file.

    Returns:
        Validated config instance.

    Raises:
        SystemExit: On load/validation failure.
    """
    path = require_config_path(config_path)
    if not Path(path).is_file():
        exit_with_error(f"Error: Configuration file not found: {path}")
    try:
        return config_cls.from_file(path)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        exit_with_error(f"Error: Failed to load configuration: {exc}")


def echo_error(message: str) -> None:
    """Print a CLI error line to stderr."""
    click.echo(message, err=True)


def echo_success(message: str) -> None:
    """Print a standardized success line."""
    click.secho(f"✓ {message}", fg="green")


def exit_with_error(message: str, *, exit_code: int = 1) -> None:
    """
    Echo an error and terminate the process.

    Args:
        message: User-facing error text.
        exit_code: Process exit code (default 1).
    """
    echo_error(message)
    sys.exit(exit_code)


def run_cli_job(
    *,
    logger_name: str,
    output_dir: Path,
    log_filename: str,
    level: int = logging.INFO,
    start_message: str,
    job: Any,
    success_message: str,
) -> None:
    """
    Run a core job with consistent logging and CLI feedback.

    Args:
        logger_name: Logger name passed to ``setup_logger``.
        output_dir: Directory for log files (created if missing).
        log_filename: Log file basename under ``output_dir``.
        level: Logging level.
        start_message: Echoed when the job starts.
        job: Zero-argument callable executed inside the try block.
        success_message: Echoed on success (prefixed with ✓).

    Raises:
        SystemExit: When ``job`` raises.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(
        name=logger_name,
        output_dir=output_dir,
        log_filename=log_filename,
        level=level,
    )
    logger.info(start_message)
    click.echo(start_message)
    try:
        job()
    except Exception as exc:  # noqa: BLE001
        logger.error("Job failed: %s", exc, exc_info=True)
        exit_with_error(f"Error: {exc}")
    logger.info(success_message)
    echo_success(success_message)


def log_platform_info(logger: logging.Logger, config_path: str) -> None:
    """
    Log Python/platform metadata at CLI startup.

    Args:
        logger: Active CLI logger.
        config_path: Configuration file path in use.
    """
    import platform

    logger.info("Python version: %s", sys.version)
    logger.info("Platform: %s", platform.platform())
    logger.info("Using configuration file: %s", config_path)


def echo_fatal(exc: BaseException) -> None:
    """Echo a fatal error with traceback (config/bootstrap failures)."""
    echo_error(f"Fatal error: {exc}")
    echo_error(traceback.format_exc())
