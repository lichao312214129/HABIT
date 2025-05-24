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
from habit.core.machine_learning import Modeling
import argparse


def main() -> None:
    """
    Main function to run the radiomics modeling pipeline
    
    This function:
    1. Parses command line arguments
    2. Loads configuration from YAML file
    3. Creates output directory
    4. Initializes modeling class
    5. Runs the modeling pipeline (training or prediction)
    
    Command line arguments:
        --config: Path to YAML config file (required)
        --mode: Operation mode ('train' or 'predict', default: 'train')
        --model: Path to model package file (.pkl) for prediction
        --data: Path to data file (.csv) for prediction
        --output: Path to save prediction results
        --model_name: Name of specific model to use for prediction
        --evaluate: Whether to evaluate model performance and generate plots
    """
    parser = argparse.ArgumentParser(description='Radiomics Modeling Tool')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], default='train',
                        help='Operation mode: train (default) or predict')
    parser.add_argument('--model', type=str, help='Path to model package file (.pkl) for prediction')
    parser.add_argument('--data', type=str, help='Path to data file (.csv) for prediction')
    parser.add_argument('--output', type=str, help='Path to save prediction results')
    parser.add_argument('--model_name', type=str, help='Name of specific model to use for prediction')
    parser.add_argument('--evaluate', action='store_true', default=True, help='Whether to evaluate model performance and generate plots')
    
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
        
        # Create output directory
        os.makedirs(config['output'], exist_ok=True)
        
        # Initialize modeling class with config
        model = Modeling(config)
        
        # Run modeling pipeline
        model.read_data()\
             .preprocess_data()\
             ._split_data()\
             .feature_selection_before_normalization()\
             .normalization()\
             .feature_selection()\
             .modeling()\
             .evaluate_models()

        print("Training completed successfully")
    
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
            
            # Make predictions with optional performance evaluation
            results = Modeling.load_and_predict(
                args.model, 
                new_data, 
                args.model_name,
                args.output,
                args.evaluate
            )
            
            # Save results
            output_path = os.path.join(args.output, 'prediction_results_of_new_data.csv') if args.output else 'prediction_results.csv'
            results.to_csv(output_path, index=False)
            print(f"Saved prediction results to {output_path}")
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            sys.exit(1)
            
        print("Prediction completed successfully")


if __name__ == "__main__":
    # Use default config file if no arguments provided
    if len(sys.argv) == 1:
        sys.argv.extend(['--config', './config/config_machine_learning.yaml',
                         '--mode', 'train'])
        
    # # 预测新数据
    # sys.argv.extend(['--config', './config/config_machine_learning.yaml',
    #                     '--mode', 'predict',
    #                     '--model', './results/model_package.pkl',
    #                     '--data', 'F:/work/workstation_b/dingHuYingXiang/_the_third_training_202504/demo_data/breast_cancer_dataset.csv',
    #                     '--output', './results/'
    #                     ])

    main()
    # Example command:
    # python scripts/app_of_machine_learning.py --config ./config/config_machine_learning.yaml

