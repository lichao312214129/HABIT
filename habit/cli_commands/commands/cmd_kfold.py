"""
K-fold cross-validation command implementation
Performs k-fold cross-validation for model evaluation
"""

import sys
import os
import logging
import click
import yaml
from pathlib import Path


def run_kfold(config_file: str) -> None:
    """
    Run K-fold cross-validation pipeline
    
    Args:
        config_file (str): Path to configuration YAML file
    """
    from habit.utils.log_utils import setup_logger
    
    if not config_file:
        click.echo("Error: Configuration file is required. Use --config option.", err=True)
        sys.exit(1)
    
    # Check if config file exists
    if not os.path.exists(config_file):
        click.echo(f"Error: Configuration file not found: {config_file}", err=True)
        sys.exit(1)
    
    # Load config to get output directory
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        output_dir = Path(config.get('output', '.'))
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        click.echo(f"Error: Failed to load configuration: {e}", err=True)
        sys.exit(1)
    
    # Setup logging at CLI entry point
    logger = setup_logger(
        name='cli.kfold',
        output_dir=output_dir,
        log_filename='kfold_cv.log',
        level=logging.INFO
    )
    
    logger.info(f"Starting K-fold cross-validation with config: {config_file}")
    click.echo(f"Starting K-fold cross-validation with config: {config_file}")
    
    try:
        # Import and run the k-fold script
        old_argv = sys.argv
        sys.argv = ['app_kfold_cv.py', '--config', config_file]
        
        from scripts import app_kfold_cv
        app_kfold_cv.main()
        
        sys.argv = old_argv
        logger.info("K-fold cross-validation completed successfully")
        click.secho("✓ K-fold cross-validation completed successfully!", fg='green')
    except Exception as e:
        logger.error(f"Error during K-fold cross-validation: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

