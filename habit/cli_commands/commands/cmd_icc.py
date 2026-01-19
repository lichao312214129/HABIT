"""
ICC analysis command implementation
Performs Intraclass Correlation Coefficient analysis
"""

import sys
import logging
import click
from pathlib import Path

def run_icc(config_file: str) -> None:
    """
    Runs the ICC (Intraclass Correlation Coefficient) analysis using a config file.

    This function acts as a thin wrapper, delegating the core logic to the
    `icc` module.
    
    Args:
        config_file (str): Path to the configuration YAML file.
    """
    from habit.utils.log_utils import setup_logger
    from habit.utils.config_utils import load_config
    from habit.core.machine_learning.feature_selectors.icc.icc import run_icc_analysis_from_config

    if not config_file:
        click.echo("Error: Configuration file is required. Use --config option.", err=True)
        sys.exit(1)
    
    try:
        config = load_config(config_file)
        output_path = Path(config.get('output', {}).get('path', 'icc_analysis.json'))
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging at the CLI entry point
        logger = setup_logger(
            name='habit.icc',
            output_dir=output_dir,
            log_filename='icc_analysis.log',
            level=logging.DEBUG if config.get("debug") else logging.INFO
        )
        
        logger.info(f"Successfully loaded config: {config_file}")
        click.echo(f"Starting ICC analysis with config: {config_file}")

        # Delegate all logic to the downstream handler
        run_icc_analysis_from_config(config)
        
        click.secho("✓ ICC analysis completed successfully!", fg='green')
        logger.info("ICC analysis process finished.")

    except FileNotFoundError:
        click.echo(f"Error: Configuration file not found at {config_file}", err=True)
        sys.exit(1)
    except Exception as e:
        # The logger might not be initialized if config loading fails
        logging.basicConfig()
        logging.getLogger().error(f"An unexpected error occurred during ICC analysis: {e}", exc_info=True)
        click.echo(f"An unexpected error occurred: {e}", err=True)
        sys.exit(1)

