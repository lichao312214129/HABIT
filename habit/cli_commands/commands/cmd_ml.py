"""
Machine learning command implementation
Handles model training, prediction, and k-fold cross-validation
"""

import sys
import os
import logging
import click
import pandas as pd
from pathlib import Path


def run_ml(config_path: str, mode: str, model: str, data: str, 
           output: str, model_name: str, evaluate: bool) -> None:
    """
    Run machine learning pipeline (training or prediction)
    
    Args:
        config_path (str): Path to configuration YAML file
        mode (str): Operation mode ('train' or 'predict')
        model (str): Path to model package file for prediction
        data (str): Path to data file for prediction
        output (str): Path to save prediction results
        model_name (str): Name of specific model to use
        evaluate (bool): Whether to evaluate model performance
    """
    from habit.core.machine_learning.workflows.holdout_workflow import MachineLearningWorkflow
    from habit.utils.config_utils import load_config
    from habit.utils.log_utils import setup_logger
    
    # Training mode
    if mode == 'train':
        # Load YAML config file
        try:
            config = load_config(config_path)
            click.echo(f"Loaded config file: {config_path}")
        except Exception as e:
            click.echo(f"Error: Failed to load config file: {e}", err=True)
            sys.exit(1)
        
        # Create output directory
        output_dir = Path(config['output'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging at CLI entry point
        logger = setup_logger(
            name='cli.ml',
            output_dir=output_dir,
            log_filename='processing.log',
            level=logging.INFO
        )
        logger.info(f"Starting machine learning training with config: {config_path}")
        
        # Initialize modeling class with config
        click.echo("Initializing machine learning pipeline...")
        model_obj = MachineLearningWorkflow(config)
        
        # Run modeling pipeline
        click.echo("Starting training pipeline...")
        model_obj.run_pipeline()

        click.secho("✓ Training completed successfully!", fg='green')
    
    # Prediction mode
    elif mode == 'predict':
        # Check required arguments
        if not model:
            click.echo("Error: --model argument is required for prediction mode", err=True)
            sys.exit(1)
        if not data:
            click.echo("Error: --data argument is required for prediction mode", err=True)
            sys.exit(1)
        
        # Setup logging for prediction mode
        output_dir = Path(output) if output else Path('.')
        output_dir.mkdir(parents=True, exist_ok=True)
        logger = setup_logger(
            name='cli.ml.predict',
            output_dir=output_dir,
            log_filename='prediction.log',
            level=logging.INFO
        )
        
        try:
            import joblib
            # Load data
            new_data = pd.read_csv(data)
            logger.info(f"Loaded data from {data}: {new_data.shape[0]} rows, {new_data.shape[1]} columns")
            click.echo(f"Loaded data from {data}: {new_data.shape[0]} rows, {new_data.shape[1]} columns")
            
            # Make predictions
            click.echo(f"Loading pipeline from {model}...")
            pipeline = joblib.load(model)
            
            click.echo("Making predictions...")
            # Predict probabilities
            probs = pipeline.predict_proba(new_data)
            if probs.ndim > 1:
                probs = probs[:, 1]
            
            # Predict labels
            preds = pipeline.predict(new_data)
            
            # Combine results
            results = new_data.copy()
            results['predicted_label'] = preds
            results['predicted_probability'] = probs
            
            # Save results
            output_path = os.path.join(output, 'prediction_results.csv') if output else 'prediction_results.csv'
            results.to_csv(output_path, index=False)
            click.echo(f"Saved prediction results to {output_path}")
            click.secho("✓ Prediction completed successfully!", fg='green')
            
        except Exception as e:
            click.echo(f"Error during prediction: {e}", err=True)
            sys.exit(1)


def run_kfold(config_file: str) -> None:
    """
    Run K-fold cross-validation pipeline
    
    Args:
        config_file (str): Path to configuration YAML file
    """
    from habit.utils.config_utils import load_config
    from habit.utils.log_utils import setup_logger
    from habit.core.machine_learning.workflows.kfold_workflow import MachineLearningKFoldWorkflow
    
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
        click.echo("Initializing machine learning pipeline...")
        model_obj = MachineLearningKFoldWorkflow(config)
        model_obj.run_pipeline()
        logger.info("K-fold cross-validation completed successfully")
        click.secho("✓ K-fold cross-validation completed successfully!", fg='green')
    except Exception as e:
        logger.error(f"Error during K-fold cross-validation: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)