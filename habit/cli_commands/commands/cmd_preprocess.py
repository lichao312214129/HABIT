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

from habit.core.preprocessing.config_schemas import PreprocessingConfig
from habit.utils.log_utils import setup_logger

def run_preprocess(config_path: str) -> None:
    """
    Run image preprocessing pipeline
    
    Args:
        config_path (str): Path to configuration YAML file
    """
    from habit.core.common.service_configurator import ServiceConfigurator
    
    # Check if config file exists first
    if not os.path.exists(config_path):
        click.echo(f"Error: Configuration file not found: {config_path}", err=True)
        sys.exit(1)
    
    try:
        # 1. Load Config (Typed & Path Resolved)
        config = PreprocessingConfig.from_file(config_path)
        
        # 2. Setup Logging
        # We set it up manually here to ensure we capture early logs, 
        # or we let ServiceConfigurator handle it. 
        # For consistency with cmd_habitat, let's setup logger explicitly so we can pass it to Configurator.
        output_dir = Path(config.out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger = setup_logger(
            name='cli.preprocessing',
            output_dir=output_dir,
            log_filename='processing.log',
            level=logging.INFO
        )
        
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Platform: {platform.platform()}")
        logger.info(f"Using configuration file: {config_path}")
        
        msg = f"Starting image preprocessing with config: {config_path}"
        logger.info(msg)
        click.echo(msg)
        
        # 3. Initialize Configurator & Service
        try:
            # Force spawn method on Windows for multiprocessing
            if platform.system() == 'Windows':
                multiprocessing.set_start_method('spawn', force=True)
                logger.info("Set multiprocessing start method to 'spawn' on Windows")
            
            configurator = ServiceConfigurator(config=config, logger=logger)
            processor = configurator.create_batch_processor()
            logger.info("Successfully initialized BatchProcessor")
        except Exception as e:
            error_msg = f"Failed to initialize BatchProcessor: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            click.echo(f"Error: {error_msg}", err=True)
            sys.exit(1)
        
        # 4. Process
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
        # Fallback error handling if config load failed
        click.echo(f"Fatal error: {e}", err=True)
        click.echo(traceback.format_exc(), err=True)
        sys.exit(1)

