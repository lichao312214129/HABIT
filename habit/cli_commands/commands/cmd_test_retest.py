"""
Test-retest reproducibility analysis command implementation
Analyzes test-retest reproducibility of habitat features
"""

import sys
import os
import logging
import click
import yaml
from pathlib import Path


def run_test_retest(config_file: str) -> None:
    """
    Run test-retest reproducibility analysis
    
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
        output_dir = Path(config.get('out_dir', config.get('output', '.')))
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        click.echo(f"Error: Failed to load configuration: {e}", err=True)
        sys.exit(1)
    
    # Setup logging at CLI entry point
    logger = setup_logger(
        name='cli.test_retest',
        output_dir=output_dir,
        log_filename='test_retest.log',
        level=logging.INFO
    )
    
    logger.info(f"Starting test-retest analysis with config: {config_file}")
    click.echo(f"Starting test-retest analysis with config: {config_file}")
    
    try:
        # Import and run the test-retest script
        old_argv = sys.argv
        sys.argv = ['app_habitat_test_retest_mapper.py', '--config', config_file]
        
        from scripts import app_habitat_test_retest_mapper
        app_habitat_test_retest_mapper.main()
        
        sys.argv = old_argv
        logger.info("Test-retest analysis completed successfully")
        click.secho("✓ Test-retest analysis completed successfully!", fg='green')
    except Exception as e:
        logger.error(f"Error during test-retest analysis: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

