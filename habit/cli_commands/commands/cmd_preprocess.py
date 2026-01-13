"""
Image preprocessing command implementation
Handles resampling, registration, and normalization of medical images
"""

import sys
import os
import platform
import traceback
import logging
import multiprocessing
import click
from pathlib import Path


def run_preprocess(config_path: str) -> None:
    """
    Run image preprocessing pipeline
    
    Args:
        config_path (str): Path to configuration YAML file
    """
    from habit.core.preprocessing.image_processor_pipeline import BatchProcessor
    from habit.utils.log_utils import setup_logger
    from habit.utils.config_utils import load_config
    
    # Check if config file exists first (before any logging setup)
    if not os.path.exists(config_path):
        click.echo(f"Error: Configuration file not found: {config_path}", err=True)
        sys.exit(1)
    
    # Load config to get output directory for logging
    try:
        config = load_config(config_path)
        output_dir = Path(config.get("out_dir", "."))
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        click.echo(f"Error: Failed to load configuration: {e}", err=True)
        sys.exit(1)
    
    # Setup logging at CLI entry point - all subsequent module logs go to this file
    logger = setup_logger(
        name='cli.preprocessing',
        output_dir=output_dir,
        log_filename='processing.log',
        level=logging.INFO
    )
    
    try:
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Platform: {platform.platform()}")
        logger.info(f"Using configuration file: {config_path}")
        
        msg = f"Starting image preprocessing with config: {config_path}"
        logger.info(msg)
        click.echo(msg)
        
        # Initialize processor
        try:
            # Force spawn method on Windows for multiprocessing
            if platform.system() == 'Windows':
                multiprocessing.set_start_method('spawn', force=True)
                logger.info("Set multiprocessing start method to 'spawn' on Windows")
            
            processor = BatchProcessor(config_path=config_path)
            logger.info("Successfully initialized BatchProcessor")
        except Exception as e:
            error_msg = f"Failed to initialize BatchProcessor: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            click.echo(f"Error: {error_msg}", err=True)
            sys.exit(1)
        
        # Process data
        try:
            logger.info("Starting batch processing")
            processor.process_batch()
            
            success_msg = "Image preprocessing completed successfully!"
            logger.info(success_msg)
            click.secho(f"✓ {success_msg}", fg='green')
        except Exception as e:
            error_msg = f"Error during batch processing: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            click.echo(error_msg, err=True)
            sys.exit(1)
            
    except Exception as e:
        error_msg = f"Uncaught error during execution: {e}"
        # Use global logger to avoid UnboundLocalError
        if logger is not None:
            logger.error(error_msg)
            logger.error(traceback.format_exc())
        else:
            # If logger hasn't been initialized yet
            print(error_msg)
            print(traceback.format_exc())
        click.echo(f"Fatal error: {e}", err=True)
        sys.exit(1)

