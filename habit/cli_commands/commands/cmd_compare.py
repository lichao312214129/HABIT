"""
Model comparison command implementation
Generates comparison plots and statistics for different models
"""

import sys
import os
import logging
import click
import yaml
from pathlib import Path


def run_compare(config_file: str) -> None:
    """
    Run model comparison analysis
    
    Args:
        config_file (str): Path to configuration YAML file
    """
    import sys
    import os
    import logging
    import click
    import yaml
    from pathlib import Path
    
    from habit.utils.log_utils import setup_logger
    from habit.core.machine_learning.config_schemas import ModelComparisonConfig
    from habit.core.common.service_configurator import ServiceConfigurator
    
    if not config_file:
        click.echo("Error: Configuration file is required. Use --config option.", err=True)
        sys.exit(1)
    
    try:
        # Use the static factory method to load, resolve paths, and validate schema
        config = ModelComparisonConfig.from_file(config_file)
        click.echo(f"Loaded configuration from: {config_file}")
    except Exception as e:
        click.echo(f"Error: Failed to load configuration: {e}", err=True)
        sys.exit(1)
    
    try:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        click.echo(f"Error: Failed to create output directory: {e}", err=True)
        sys.exit(1)
    
    logger = setup_logger(
        name='cli.compare',
        output_dir=output_dir,
        log_filename='processing.log',
        level=logging.INFO
    )
    
    logger.info(f"Starting model comparison with config: {config_file}")
    click.echo(f"Starting model comparison with config: {config_file}")
    
    try:
        configurator = ServiceConfigurator(config=config, logger=logger, output_dir=str(output_dir))
        model_obj = configurator.create_model_comparison()
        model_obj.run()
        logger.info("Model comparison completed successfully")
        click.secho("Model comparison completed successfully!", fg='green')
    except Exception as e:
        logger.error(f"Error during model comparison: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

