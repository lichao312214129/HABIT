"""
Machine learning command implementation
Handles model training, prediction, and k-fold cross-validation
"""

import sys
import os
import logging
import click
from pathlib import Path

from habit.core.machine_learning.config_schemas import MLConfig, PredictionConfig
from habit.utils.log_utils import setup_logger

def run_ml(config_path: str, mode: str) -> None:
    """
    Run machine learning pipeline (training or prediction)
    
    Args:
        config_path (str): Path to configuration YAML file
        mode (str): Operation mode ('train' or 'predict')
    """
    from habit.core.common.service_configurator import ServiceConfigurator
    
    # Training mode
    if mode == 'train':
        # Load and validate configuration
        try:
            config = MLConfig.from_file(config_path)
            click.echo(f"Loaded configuration from: {config_path}")
        except Exception as e:
            click.echo(f"Error: Failed to load configuration: {e}", err=True)
            sys.exit(1)
        
        # Create output directory
        output_dir = Path(config.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging at CLI entry point
        logger = setup_logger(
            name='cli.ml',
            output_dir=output_dir,
            log_filename='processing.log',
            level=logging.INFO
        )
        logger.info(f"Starting machine learning training with config: {config_path}")
        # Pydantic v2/v1 compat
        cfg_dump = config.model_dump() if hasattr(config, 'model_dump') else config.dict()
        logger.info(f"Full configuration: {cfg_dump}")
        
        # Initialize service using ServiceConfigurator
        click.echo("Initializing machine learning pipeline...")
        try:
            configurator = ServiceConfigurator(config=config, logger=logger, output_dir=str(output_dir))
            model_obj = configurator.create_ml_workflow()
        except Exception as e:
            logger.error(f"Failed to initialize service: {e}", exc_info=True)
            click.echo(f"Error: Failed to initialize service: {e}", err=True)
            sys.exit(1)
        
        # Run modeling pipeline
        click.echo("Starting training pipeline...")
        model_obj.run_pipeline()

        click.secho("✓ Training completed successfully!", fg='green')
    
    # Prediction mode
    elif mode == 'predict':
        # Load Prediction Config from file (Consistent with Train mode)
        try:
            config = PredictionConfig.from_file(config_path)
            click.echo(f"Loaded prediction configuration from: {config_path}")
        except Exception as e:
            click.echo(f"Error: Failed to load configuration: {e}", err=True)
            sys.exit(1)
        
        # Setup logging
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger = setup_logger(
            name='cli.ml.predict',
            output_dir=output_dir,
            log_filename='prediction.log',
            level=logging.INFO
        )
        logger.info(f"Starting prediction with config: {config_path}")
        
        try:
            # Use ServiceConfigurator for unified workflow
            configurator = ServiceConfigurator(config=config, logger=logger, output_dir=str(output_dir))
            workflow = configurator.create_prediction_workflow()
            workflow.run_pipeline()
            
            click.secho("✓ Prediction completed successfully!", fg='green')
            
        except Exception as e:
            click.echo(f"Error during prediction: {e}", err=True)
            logger.error(f"Prediction failed: {e}", exc_info=True)
            sys.exit(1)


def run_kfold(config_file: str) -> None:
    """
    Run K-fold cross-validation pipeline
    
    Args:
        config_file (str): Path to configuration YAML file
    """
    from habit.core.machine_learning.config_schemas import MLConfig
    from habit.core.common.service_configurator import ServiceConfigurator
    from habit.utils.log_utils import setup_logger
    
    if not config_file:
        click.echo("Error: Configuration file is required. Use --config option.", err=True)
        sys.exit(1)
    
    # Check if config file exists
    if not os.path.exists(config_file):
        click.echo(f"Error: Configuration file not found: {config_file}", err=True)
        sys.exit(1)
    
    # Load and validate configuration
    try:
        config = MLConfig.from_file(config_file)
        click.echo(f"Loaded configuration from: {config_file}")
    except Exception as e:
        click.echo(f"Error: Failed to load configuration: {e}", err=True)
        sys.exit(1)
    
    # Create output directory
    try:
        output_dir = Path(config.output)
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        click.echo(f"Error: Failed to create output directory: {e}", err=True)
        sys.exit(1)
    
    # Setup logging at CLI entry point
    logger = setup_logger(
        name='cli.kfold',
        output_dir=output_dir,
        log_filename='kfold_cv.log',
        level=logging.INFO
    )
    
    logger.info(f"Starting K-fold cross-validation with config: {config_file}")
    logger.info(f"Full configuration: {config.model_dump() if hasattr(config, 'model_dump') else config.dict()}")
    click.echo(f"Starting K-fold cross-validation with config: {config_file}")
    
    try:
        # Initialize service using ServiceConfigurator
        click.echo("Initializing machine learning pipeline...")
        configurator = ServiceConfigurator(config=config, logger=logger, output_dir=str(output_dir))
        model_obj = configurator.create_kfold_workflow()
        model_obj.run_pipeline()
        logger.info("K-fold cross-validation completed successfully")
        click.secho("✓ K-fold cross-validation completed successfully!", fg='green')
    except Exception as e:
        logger.error(f"Error during K-fold cross-validation: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)