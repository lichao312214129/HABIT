"""
Habitat analysis command implementation
Generates habitat maps through clustering analysis
"""

import sys
import logging
import click
from datetime import datetime
from pathlib import Path


def run_habitat(config_file: str, debug_mode: bool) -> None:
    """
    Run habitat analysis pipeline
    
    Args:
        config_file (str): Path to configuration YAML file
        debug_mode (bool): Whether to enable debug mode
    """
    from habit.core.habitat_analysis import HabitatAnalysis
    from habit.utils.io_utils import load_config
    from habit.utils.log_utils import setup_logger
    
    # Check if config file is provided
    if not config_file:
        click.echo("Error: Configuration file is required. Use --config option.", err=True)
        sys.exit(1)
    
    # Load configuration
    try:
        config = load_config(config_file)
        click.echo(f"Loaded configuration from: {config_file}")
    except Exception as e:
        click.echo(f"Error: Failed to load configuration: {e}", err=True)
        sys.exit(1)

    # --- Simplified Logging and Initialization ---

    # Override debug setting if specified on the command line
    config['debug'] = debug_mode or config.get('debug', False)
    config['config_file'] = config_file

    out_dir = config.get('out_dir')
    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    log_level = logging.DEBUG if config['debug'] else logging.INFO
    logger = setup_logger(
        name='cli.habitat',
        output_dir=output_path,
        log_filename='habitat_analysis.log',
        level=log_level
    )
    
    logger.info("==== Starting Habitat Analysis ====")
    logger.info("Config file: %s", config_file)
    logger.info("Full configuration being used: %s", config)
    logger.info("=====================================")
    
    click.echo(f"Starting habitat analysis...")
    click.echo(f"  Output directory: {out_dir}")
    click.echo(f"  Log file at: {output_path / 'habitat_analysis.log'}")

    try:
        # Create and run HabitatAnalysis with the unified config dictionary
        habitat_analysis = HabitatAnalysis(config=config)
        habitat_analysis.run()
        logger.info("Habitat analysis completed successfully")
        click.secho("✓ Habitat analysis completed successfully!", fg='green')
    except Exception as e:
        logger.error("Error during habitat analysis: %s", str(e), exc_info=True)
        click.echo(f"An error occurred. See log file for details: {e}", err=True)
        sys.exit(1)

