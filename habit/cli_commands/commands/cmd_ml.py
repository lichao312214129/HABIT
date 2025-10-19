"""
Machine learning command implementation
Handles model training and prediction
"""

import sys
import os
import click
import yaml
import pandas as pd


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
    from habit.core.machine_learning.machine_learning import Modeling
    
    # Training mode
    if mode == 'train':
        # Load YAML config file
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            click.echo(f"Loaded config file: {config_path}")
        except Exception as e:
            click.echo(f"Error: Failed to load config file: {e}", err=True)
            sys.exit(1)
        
        # Create output directory
        os.makedirs(config['output'], exist_ok=True)
        
        # Initialize modeling class with config
        click.echo("Initializing machine learning pipeline...")
        model_obj = Modeling(config)
        
        # Run modeling pipeline
        click.echo("Starting training pipeline...")
        model_obj.read_data()\
                .preprocess_data()\
                ._split_data()\
                .feature_selection_before_normalization()\
                .normalization()\
                .feature_selection()\
                .modeling()\
                .evaluate_models()

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
        
        try:
            # Load data
            new_data = pd.read_csv(data)
            click.echo(f"Loaded data from {data}: {new_data.shape[0]} rows, {new_data.shape[1]} columns")
            
            # Make predictions with optional performance evaluation
            click.echo("Making predictions...")
            results = Modeling.load_and_predict(
                model, 
                new_data, 
                model_name,
                output,
                evaluate
            )
            
            # Save results
            output_path = os.path.join(output, 'prediction_results_of_new_data.csv') if output else 'prediction_results.csv'
            results.to_csv(output_path, index=False)
            click.echo(f"Saved prediction results to {output_path}")
            click.secho("✓ Prediction completed successfully!", fg='green')
            
        except Exception as e:
            click.echo(f"Error during prediction: {e}", err=True)
            sys.exit(1)

