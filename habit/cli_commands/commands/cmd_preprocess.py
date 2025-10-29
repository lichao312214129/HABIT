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


def run_preprocess(config_path: str) -> None:
    """
    Run image preprocessing pipeline
    
    Args:
        config_path (str): Path to configuration YAML file
    """
    from habit.core.preprocessing.image_processor_pipeline import BatchProcessor
    from habit.utils.log_utils import get_module_logger
    
    # Get module logger for CLI command logging
    logger = get_module_logger('cli.preprocess')
    
    try:
        # Check if config file exists
        if not os.path.exists(config_path):
            logger.error(f"Configuration file does not exist: {config_path}")
            click.echo(f"Error: Configuration file not found: {config_path}", err=True)
            sys.exit(1)
        
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Platform: {platform.platform()}")
        logger.info(f"Using configuration file: {config_path}")
        click.echo(f"Starting image preprocessing with config: {config_path}")
        
        # Initialize processor
        try:
            # Force spawn method on Windows for multiprocessing
            if platform.system() == 'Windows':
                multiprocessing.set_start_method('spawn', force=True)
                logger.info("Set multiprocessing start method to 'spawn' on Windows")
            
            processor = BatchProcessor(config_path=config_path)
            logger.info("Successfully initialized BatchProcessor")
        except Exception as e:
            logger.error(f"Failed to initialize BatchProcessor: {e}")
            logger.error(traceback.format_exc())
            click.echo(f"Error: Failed to initialize processor: {e}", err=True)
            sys.exit(1)
        
        # Process data
        try:
            logger.info("Starting batch processing")
            processor.process_batch()
            logger.info("Batch processing completed")
            click.secho("✓ Image preprocessing completed successfully!", fg='green')
        except Exception as e:
            logger.error(f"Error during batch processing: {e}")
            logger.error(traceback.format_exc())
            click.echo(f"Error during processing: {e}", err=True)
            sys.exit(1)
            
    except Exception as e:
        # Use global logger to avoid UnboundLocalError
        if logger is not None:
            logger.error(f"Uncaught error during execution: {e}")
            logger.error(traceback.format_exc())
        else:
            # If logger hasn't been initialized yet
            print(f"Uncaught error during execution: {e}")
            print(traceback.format_exc())
        click.echo(f"Fatal error: {e}", err=True)
        sys.exit(1)

