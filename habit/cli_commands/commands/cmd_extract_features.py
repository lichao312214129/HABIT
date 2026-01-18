"""
Feature extraction command implementation
Extracts habitat features from clustered images
"""

import sys
import os
import logging
import click
from pathlib import Path


def run_extract_features(config_file: str) -> None:
    """
    Run habitat feature extraction pipeline
    
    Args:
        config_file (str): Path to configuration YAML file
    """
    from habit.utils.log_utils import setup_logger
    from habit.utils.config_utils import load_config
    
    if not config_file:
        click.echo("Error: Configuration file is required. Use --config option.", err=True)
        sys.exit(1)
    
    # Check if config file exists
    if not os.path.exists(config_file):
        click.echo(f"Error: Configuration file not found: {config_file}", err=True)
        sys.exit(1)
    
    # Load config to get output directory
    try:
        config = load_config(config_file)
        output_dir = Path(config.get('out_dir', '.'))
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        click.echo(f"Error: Failed to load configuration: {e}", err=True)
        sys.exit(1)
    
    # Setup logging at CLI entry point
    logger = setup_logger(
        name='cli.extract_features',
        output_dir=output_dir,
        log_filename='processing.log',
        level=logging.INFO
    )
    
    logger.info(f"Starting habitat feature extraction with config: {config_file}")
    click.echo(f"Starting habitat feature extraction with config: {config_file}")
    
    try:
        # Import and run the feature extraction script
        old_argv = sys.argv
        sys.argv = ['app_extracting_habitat_features.py', '--config', config_file]
        
        from scripts import app_extracting_habitat_features
        app_extracting_habitat_features.main()
        
        sys.argv = old_argv
        logger.info("Feature extraction completed successfully")
        click.secho("✓ Feature extraction completed successfully!", fg='green')
    except Exception as e:
        logger.error(f"Error during feature extraction: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

