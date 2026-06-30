# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""Machine learning command implementation."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from habit.cli_commands.common import (
    echo_success,
    exit_with_error,
    load_config_or_exit,
)
from habit.core.machine_learning.config_schemas import MLConfig
from habit.core.machine_learning.run import (
    apply_ml_mode_override,
    run_kfold_from_config,
    run_ml_from_config,
)
from habit.utils.log_utils import setup_logger


def run_ml(config_path: str, mode: str) -> None:
    """
    Run the ML pipeline (training or prediction).

    Args:
        config_path: Path to configuration YAML file.
        mode: Operation mode (``train`` or ``predict``).
    """
    if mode not in ("train", "predict"):
        exit_with_error(
            f"Error: invalid --mode {mode!r} (expected 'train' or 'predict')."
        )

    config = load_config_or_exit(MLConfig, config_path)
    click.echo(f"Loaded configuration from: {config_path}")
    config = apply_ml_mode_override(config, mode)

    output_dir = Path(config.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_filename = "processing.log" if mode == "train" else "prediction.log"
    logger_name = "cli.ml" if mode == "train" else "cli.ml.predict"
    logger = setup_logger(
        name=logger_name,
        output_dir=output_dir,
        log_filename=log_filename,
        level=logging.INFO,
    )
    logger.info(
        "Starting machine learning pipeline (mode=%s) with config: %s",
        mode,
        config_path,
    )
    logger.info("Full configuration: %s", config.model_dump())

    click.echo(f"Initialising machine learning pipeline (mode={mode})...")
    try:
        run_ml_from_config(config, logger=logger, output_dir=str(output_dir))
    except Exception as exc:  # noqa: BLE001
        logger.error("Workflow failed: %s", exc, exc_info=True)
        exit_with_error(f"Error during {mode}: {exc}")

    if mode == "train":
        echo_success("Training completed successfully!")
    else:
        echo_success("Prediction completed successfully!")


def run_kfold(config_file: str) -> None:
    """
    Run K-fold cross-validation pipeline.

    Args:
        config_file: Path to configuration YAML file.
    """
    config = load_config_or_exit(MLConfig, config_file)
    click.echo(f"Loaded configuration from: {config_file}")

    output_dir = Path(config.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        name="cli.kfold",
        output_dir=output_dir,
        log_filename="kfold_cv.log",
        level=logging.INFO,
    )
    logger.info("Starting K-fold cross-validation with config: %s", config_file)
    logger.info("Full configuration: %s", config.model_dump())
    click.echo(f"Starting K-fold cross-validation with config: {config_file}")

    try:
        click.echo("Initialising machine learning pipeline...")
        run_kfold_from_config(config, logger=logger, output_dir=str(output_dir))
    except ValueError as exc:
        exit_with_error(f"Error: {exc}")
    except Exception as exc:  # noqa: BLE001
        logger.error("K-fold failed: %s", exc, exc_info=True)
        exit_with_error(f"Error: {exc}")

    echo_success("K-fold cross-validation completed successfully!")
