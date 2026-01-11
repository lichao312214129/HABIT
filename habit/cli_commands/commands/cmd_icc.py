"""
ICC analysis command implementation
Performs Intraclass Correlation Coefficient analysis
"""

import sys
import os
import logging
import click
import yaml
from pathlib import Path


def run_icc(config_file: str) -> None:
    """
    Run ICC (Intraclass Correlation Coefficient) analysis
    
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
        output = Path(config.get('output').get('path'))
        output_dir = output.parent
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        click.echo(f"Error: Failed to load configuration: {e}", err=True)
        sys.exit(1)
    
    # Setup logging at CLI entry point
    logger = setup_logger(
        name='cli.icc',
        output_dir=output_dir,
        log_filename='icc_analysis.log',
        level=logging.INFO
    )
    
    logger.info(f"Starting ICC analysis with config: {config_file}")
    click.echo(f"Starting ICC analysis with config: {config_file}")
    
    try:
        # Import and run the ICC analysis script
        old_argv = sys.argv
        sys.argv = ['app_icc_analysis.py', '--config', config_file]
        
        from scripts import app_icc_analysis
        app_icc_analysis.main()
        
        sys.argv = old_argv
        logger.info("ICC analysis completed successfully")
        click.secho("✓ ICC analysis completed successfully!", fg='green')
    except Exception as e:
        logger.error(f"Error during ICC analysis: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

