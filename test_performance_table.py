#!/usr/bin/env python3
"""
Test script for performance table saving functionality
"""

import numpy as np
import pandas as pd
import os
import sys

# Add the habit project to the path
sys.path.append('.')

from habit.core.machine_learning.evaluation.metrics import calculate_metrics
from habit.core.machine_learning.evaluation.model_evaluation import ModelEvaluator

def test_performance_table_saving():
    """Test the performance table saving functionality"""
    
    # Create test output directory
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create synthetic test data
    np.random.seed(42)
    n_samples = 100
    
    # Generate test data for two models
    y_true = np.random.binomial(1, 0.3, n_samples)
    
    # Model 1: Well-calibrated model
    y_pred_proba_1 = np.random.beta(2, 5, n_samples)  # Reasonable probabilities
    y_pred_1 = (y_pred_proba_1 >= 0.5).astype(int)
    
    # Model 2: Less calibrated model
    y_pred_proba_2 = np.random.beta(5, 2, n_samples)  # Biased towards higher probabilities
    y_pred_2 = (y_pred_proba_2 >= 0.5).astype(int)
    
    # Calculate metrics for both models
    print("Calculating metrics for Model 1...")
    metrics_1 = calculate_metrics(y_true, y_pred_1, y_pred_proba_1)
    print(f"Model 1 metrics: {metrics_1}")
    
    print("\nCalculating metrics for Model 2...")
    metrics_2 = calculate_metrics(y_true, y_pred_2, y_pred_proba_2)
    print(f"Model 2 metrics: {metrics_2}")
    
    # Create results dictionary in the format expected by ModelEvaluator
    results = {
        'train': {
            'Model1': {
                'metrics': metrics_1,
                'y_true': y_true.tolist(),
                'y_pred': y_pred_1.tolist(),
                'y_pred_proba': y_pred_proba_1.tolist()
            },
            'Model2': {
                'metrics': metrics_2,
                'y_true': y_true.tolist(),
                'y_pred': y_pred_2.tolist(),
                'y_pred_proba': y_pred_proba_2.tolist()
            }
        },
        'test': {
            'Model1': {
                'metrics': metrics_1,  # Using same data for test as demo
                'y_true': y_true.tolist(),
                'y_pred': y_pred_1.tolist(),
                'y_pred_proba': y_pred_proba_1.tolist()
            },
            'Model2': {
                'metrics': metrics_2,  # Using same data for test as demo
                'y_true': y_true.tolist(),
                'y_pred': y_pred_2.tolist(),
                'y_pred_proba': y_pred_proba_2.tolist()
            }
        }
    }
    
    # Create ModelEvaluator and test saving functionality
    evaluator = ModelEvaluator(output_dir)
    
    print("\n" + "="*60)
    print("Testing performance table printing and saving...")
    print("="*60)
    
    # Print performance table
    evaluator._print_performance_table(results)
    
    # Save performance table
    evaluator._save_performance_table(results)
    
    print("\n" + "="*60)
    print("Test completed successfully!")
    print(f"Check the '{output_dir}' directory for output files:")
    print("- performance_table.csv")
    print("- performance_table_detailed.csv")
    print("="*60)

if __name__ == "__main__":
    test_performance_table_saving() 