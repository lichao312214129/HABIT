"""
ICC analysis command implementation.
Performs Intraclass Correlation Coefficient analysis.
"""

import logging
import sys
from pathlib import Path

import click


def run_icc(config_file: str) -> None:
    """
    Run the ICC (Intraclass Correlation Coefficient) analysis from a config
    file.

    This function is a thin wrapper. It loads the config into a validated
    :class:`ICCConfig`, sets up logging, and delegates to
    :func:`run_icc_analysis_from_config`.

    Args:
        config_file (str): Path to the configuration YAML file.
    """
    from habit.core.machine_learning.feature_selectors.icc.config import ICCConfig
    from habit.core.machine_learning.feature_selectors.icc.icc import (
        run_icc_analysis_from_config,
    )
    from habit.utils.log_utils import setup_logger

    if not config_file:
        click.echo("Error: Configuration file is required. Use --config option.", err=True)
        sys.exit(1)

    try:
        config = ICCConfig.from_file(config_file)

        output_path = Path(config.output.path)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        logger = setup_logger(
            name='habit.icc',
            output_dir=str(output_dir),
            log_filename='icc_analysis.log',
            level=logging.DEBUG if config.debug else logging.INFO,
        )

        logger.info(f"Successfully loaded config: {config_file}")
        click.echo(f"Starting ICC analysis with config: {config_file}")

        run_icc_analysis_from_config(config)

        click.secho("OK ICC analysis completed successfully!", fg='green')
        logger.info("ICC analysis process finished.")

    except FileNotFoundError:
        click.echo(f"Error: Configuration file not found at {config_file}", err=True)
        sys.exit(1)
    except Exception as e:
        # The logger may not be initialised if config loading fails.
        logging.basicConfig()
        logging.getLogger().error(f"An unexpected error occurred during ICC analysis: {e}", exc_info=True)
        click.echo(f"An unexpected error occurred: {e}", err=True)
        sys.exit(1)
