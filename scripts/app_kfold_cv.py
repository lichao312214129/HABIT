"""
K-Fold Cross-Validation Application Script

This script runs k-fold cross-validation for machine learning models.

Usage:
    python scripts/app_kfold_cv.py --config config/config_machine_learning_kfold.yaml

Author: Li Chao
Email: lich356@mail.sysu.edu.cn
"""

import argparse
import yaml
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from habit.core.machine_learning.machine_learning_kfold import MachineLearningKFoldWorkflow

def main():
    """
    Main function to run k-fold cross-validation
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run k-fold cross-validation for machine learning models'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Check if config file exists
    if args.config is None or not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Run k-fold cross-validation
    print("\n" + "="*80)
    print("Starting K-Fold Cross-Validation")
    print("="*80)
    
    try:
        workflow = MachineLearningKFoldWorkflow(config)
        workflow.run_pipeline()
        
        print("\n" + "="*80)
        print("K-Fold Cross-Validation Completed Successfully!")
        print("="*80)
        
        # Print summary
        print("\nPerformance Summary:")
        for model_name, res in workflow.results['aggregated'].items():
            overall = res['overall_metrics']
            print(f"\n{model_name}:")
            print(f"  AUC:       {overall['auc']:.4f} (mean: {res['auc_mean']:.4f} ± {res['auc_std']:.4f})")
            print(f"  Accuracy:  {overall['accuracy']:.4f}")
            print(f"  Sensitivity: {overall['sensitivity']:.4f}")
            print(f"  Specificity: {overall['specificity']:.4f}")
        
        print(f"\nResults saved to: {workflow.output_dir}")
        
    except Exception as e:
        print(f"\nError during k-fold cross-validation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

