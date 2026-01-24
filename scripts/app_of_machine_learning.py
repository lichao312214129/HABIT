"""
Command-line interface for running radiomics modeling pipeline
This module provides functionality for training and predicting using machine learning models
on radiomics features. It supports both training new models and making predictions with
existing models.
"""

import sys
import os
import yaml
import pandas as pd
import joblib
from habit.core.machine_learning.machine_learning import MachineLearningWorkflow
import argparse


def main() -> None:
    """
    Main function to run the radiomics modeling pipeline
    """
    parser = argparse.ArgumentParser(description='Radiomics Modeling Tool')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], default='train',
                        help='Operation mode: train (default) or predict')
    parser.add_argument('--model', type=str, help='Path to model pipeline file (.pkl) for prediction')
    parser.add_argument('--data', type=str, help='Path to data file (.csv) for prediction')
    parser.add_argument('--output', type=str, help='Path to save prediction results')
    parser.add_argument('--model_name', type=str, help='Name of specific model to use for prediction')
    parser.add_argument('--evaluate', action='store_true', default=True, help='Whether to evaluate model performance')
    
    args = parser.parse_args()
    
    # If in training mode, run the full modeling pipeline
    if args.mode == 'train':
        # Load YAML config file
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"Loaded config file: {args.config}")
        except Exception as e:
            print(f"Error: Failed to load config file: {e}")
            sys.exit(1)
        
        # Initialize workflow class with config
        workflow = MachineLearningWorkflow(config)
        
        # Run modeling pipeline
        try:
            workflow.run_pipeline()
            print("Training completed successfully")
        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            print(traceback.format_exc())
            sys.exit(1)
    
    # If in prediction mode, load model and make predictions
    elif args.mode == 'predict':
        # Check required arguments
        if not args.model:
            print("Error: --model argument is required for prediction mode")
            sys.exit(1)
        if not args.data:
            print("Error: --data argument is required for prediction mode")
            sys.exit(1)
        
        try:
            # Load data
            new_data = pd.read_csv(args.data)
            print(f"Loaded data from {args.data}: {new_data.shape[0]} rows, {new_data.shape[1]} columns")
            
            # Make predictions
            print(f"Loading pipeline from {args.model}...")
            pipeline = joblib.load(args.model)
            
            print("Making predictions...")
            probs = pipeline.predict_proba(new_data)
            if probs.ndim > 1:
                probs = probs[:, 1]
            preds = pipeline.predict(new_data)
            
            # Combine results
            results = new_data.copy()
            results['predicted_label'] = preds
            results['predicted_probability'] = probs
            
            # Save results
            os.makedirs(args.output, exist_ok=True) if args.output else None
            output_path = os.path.join(args.output, 'prediction_results.csv') if args.output else 'prediction_results.csv'
            results.to_csv(output_path, index=False)
            print(f"Saved prediction results to {output_path}")
            print("Prediction completed successfully")
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            sys.exit(1)


if __name__ == "__main__":
    # Use default config file if no arguments provided
    if len(sys.argv) == 1:
        sys.argv.extend(['--config', './config/config_machine_learning.yaml',
                         '--mode', 'train'])
        
    # 预测新数据
    # sys.argv.extend(['--config', './config/config_machine_learning.yaml',
    #                     '--mode', 'predict',
    #                     '--model', './ml_data/ml/rad/model_package.pkl',
    #                     '--data', './ml_data/breast_cancer_dataset.csv',
    #                     '--output', './ml_data/ml/rad_new/',
    #                     '--evaluate'
    #                     ])

    main()
    # Example command:
    # python scripts/app_of_machine_learning.py --config ./config/config_machine_learning.yaml

