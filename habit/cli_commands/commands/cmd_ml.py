"""
Machine learning command implementation.

Handles model training, prediction, and k-fold cross-validation. Train and
predict share a single :class:`MLConfig` schema and a single workflow class
(``MachineLearningWorkflow``). The CLI ``--mode`` flag is mirrored into
``config.run_mode`` for users who don't want to set it in YAML.
"""

import logging
import os
import sys
from pathlib import Path

import click

from habit.core.machine_learning.config_schemas import MLConfig
from habit.utils.log_utils import setup_logger


def run_ml(config_path: str, mode: str) -> None:
    """
    Run the ML pipeline (training or prediction).

    Args:
        config_path: Path to configuration YAML file.
        mode: Operation mode (``'train'`` or ``'predict'``). Overrides the
            ``run_mode`` field inside the config.
    """
    from habit.core.common.configurators import MLConfigurator

    if mode not in ('train', 'predict'):
        click.echo(
            f"Error: invalid --mode {mode!r} (expected 'train' or 'predict').",
            err=True,
        )
        sys.exit(1)

    try:
        config = MLConfig.from_file(config_path)
        click.echo(f"Loaded configuration from: {config_path}")
    except Exception as e:
        click.echo(f"Error: Failed to load configuration: {e}", err=True)
        sys.exit(1)

    # CLI flag wins over the YAML value so callers can flip without editing
    # the file. Re-validate to enforce the cross-field invariants.
    if config.run_mode != mode:
        config = MLConfig.model_validate({**config.model_dump(), 'run_mode': mode})

    output_dir = Path(config.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_filename = 'processing.log' if mode == 'train' else 'prediction.log'
    logger_name = 'cli.ml' if mode == 'train' else 'cli.ml.predict'
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
    logger.info(
        "Full configuration: %s",
        config.model_dump() if hasattr(config, 'model_dump') else config.dict(),
    )

    click.echo(f"Initialising machine learning pipeline (mode={mode})...")
    try:
        configurator = MLConfigurator(
            config=config, logger=logger, output_dir=str(output_dir)
        )
        workflow = configurator.create_ml_workflow()
    except Exception as e:
        logger.error(f"Failed to initialise service: {e}", exc_info=True)
        click.echo(f"Error: Failed to initialise service: {e}", err=True)
        sys.exit(1)

    try:
        workflow.run()
    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)
        click.echo(f"Error during {mode}: {e}", err=True)
        sys.exit(1)

    if mode == 'train':
        click.secho("OK Training completed successfully!", fg='green')
    else:
        click.secho("OK Prediction completed successfully!", fg='green')


def run_kfold(config_file: str) -> None:
    """
    Run K-fold cross-validation pipeline.

    Args:
        config_file: Path to configuration YAML file.
    """
    from habit.core.common.configurators import MLConfigurator

    if not config_file:
        click.echo("Error: Configuration file is required. Use --config option.", err=True)
        sys.exit(1)

    if not os.path.exists(config_file):
        click.echo(f"Error: Configuration file not found: {config_file}", err=True)
        sys.exit(1)

    try:
        config = MLConfig.from_file(config_file)
        click.echo(f"Loaded configuration from: {config_file}")
    except Exception as e:
        click.echo(f"Error: Failed to load configuration: {e}", err=True)
        sys.exit(1)

    if config.run_mode != 'train':
        click.echo(
            "Error: K-fold cross-validation requires run_mode='train'.",
            err=True,
        )
        sys.exit(1)

    try:
        output_dir = Path(config.output)
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        click.echo(f"Error: Failed to create output directory: {e}", err=True)
        sys.exit(1)

    logger = setup_logger(
        name='cli.kfold',
        output_dir=output_dir,
        log_filename='kfold_cv.log',
        level=logging.INFO,
    )
    logger.info(f"Starting K-fold cross-validation with config: {config_file}")
    logger.info(
        "Full configuration: %s",
        config.model_dump() if hasattr(config, 'model_dump') else config.dict(),
    )
    click.echo(f"Starting K-fold cross-validation with config: {config_file}")

    try:
        click.echo("Initialising machine learning pipeline...")
        configurator = MLConfigurator(
            config=config, logger=logger, output_dir=str(output_dir)
        )
        workflow = configurator.create_kfold_workflow()
        workflow.run()
        logger.info("K-fold cross-validation completed successfully")
        click.secho("OK K-fold cross-validation completed successfully!", fg='green')
    except Exception as e:
        logger.error(f"Error during K-fold cross-validation: {e}", exc_info=True)
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
