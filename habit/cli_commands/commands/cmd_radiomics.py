"""
Traditional radiomics extraction command implementation
Extracts traditional radiomics features from medical images
"""

import sys
import os
import logging
import click
import yaml
from pathlib import Path


def run_radiomics(config_file: str) -> None:
    """
    Run traditional radiomics feature extraction
    
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
        name='cli.radiomics',
        output_dir=output_dir,
        log_filename='radiomics_extraction.log',
        level=logging.INFO
    )
    
    logger.info(f"Starting traditional radiomics extraction with config: {config_file}")
    click.echo(f"Starting traditional radiomics extraction with config: {config_file}")
    
    try:
        # Import and run the radiomics extraction script
        old_argv = sys.argv
        sys.argv = ['app_traditional_radiomics_extractor.py', '--config', config_file]
        
        from scripts import app_traditional_radiomics_extractor
        app_traditional_radiomics_extractor.main()
        
        sys.argv = old_argv
        logger.info("Radiomics extraction completed successfully")
        click.secho("✓ Radiomics extraction completed successfully!", fg='green')
    except Exception as e:
        logger.error(f"Error during radiomics extraction: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

