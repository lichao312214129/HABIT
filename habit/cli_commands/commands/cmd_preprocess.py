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
    
    # Create global logger object
    logger = logging.getLogger("habit_preprocess")
    
    def setup_logging(config_path: str):
        """
        Setup logging configuration
        
        Args:
            config_path (str): Path to the configuration file
            
        Returns:
            logger: Configured logger object
        """
        import yaml
        
        try:
            # Read config file to get output directory
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            out_dir = config.get('out_dir', os.getcwd())
            
            # Create output directory if it doesn't exist
            os.makedirs(out_dir, exist_ok=True)
            
            # Setup logging
            log_file = os.path.join(out_dir, "preprocessing_debug.log")
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_file, mode='w', encoding='utf-8'),
                    logging.StreamHandler()
                ]
            )
            logger = logging.getLogger("habit_preprocess")
            logger.info(f"Log file saved to: {log_file}")
            return logger
        except Exception as e:
            # If error occurs while setting up logging, create basic logging configuration
            print(f"Error setting up logging: {str(e)}")
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[logging.StreamHandler()]
            )
            logger = logging.getLogger("habit_preprocess")
            logger.error(f"Error setting up logging: {str(e)}")
            logger.error(traceback.format_exc())
            return logger
    
    try:
        # Setup logging
        logger = setup_logging(config_path)
        
        # Log system information
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Platform: {platform.platform()}")
        logger.info(f"Current working directory: {os.getcwd()}")
        
        # Check if config file exists
        if not os.path.exists(config_path):
            logger.error(f"Configuration file does not exist: {config_path}")
            click.echo(f"Error: Configuration file not found: {config_path}", err=True)
            sys.exit(1)
        
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

