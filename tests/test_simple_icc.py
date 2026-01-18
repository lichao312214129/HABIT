"""
Test script for simplified ICC analysis
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from habit.core.machine_learning.feature_selectors.icc.simple_icc_analyzer import (
    analyze_features,
    save_results,
    print_summary
)
from habit.utils.log_utils import setup_logger

# Setup logger
output_dir = Path('demo_data/ml_data')
output_dir.mkdir(parents=True, exist_ok=True)

logger = setup_logger(
    name='test_icc',
    output_dir=output_dir,
    log_filename='test_icc.log',
    level=20  # INFO level
)

# Test files
file_paths = [
    'demo_data/ml_data/breast_cancer_dataset.csv',
    'demo_data/ml_data/breast_cancer_dataset_retest_simulated.csv'
]

# Metrics to calculate
metrics = ['icc2', 'icc3', 'cohen', 'fleiss', 'krippendorff']

print("Starting simplified ICC analysis test...")
print(f"Files: {file_paths}")
print(f"Metrics: {metrics}")

try:
    # Analyze features
    results = analyze_features(
        file_paths=file_paths,
        metrics=metrics,
        logger_instance=logger
    )
    
    # Print summary
    print_summary(results, logger)
    
    # Save results
    output_path = 'demo_data/ml_data/icc_results_simple.json'
    save_results(results, output_path, logger)
    
    print(f"\nTest completed successfully!")
    print(f"Results saved to: {output_path}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)