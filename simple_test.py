#!/usr/bin/env python3
"""
Simple test for imports
"""

import sys
sys.path.append('.')

try:
    from habit.core.machine_learning.evaluation.metrics import calculate_metrics
    print("Successfully imported calculate_metrics")
except Exception as e:
    print(f"Failed to import calculate_metrics: {e}")

try:
    from habit.core.machine_learning.evaluation.model_evaluation import ModelEvaluator
    print("Successfully imported ModelEvaluator")
except Exception as e:
    print(f"Failed to import ModelEvaluator: {e}")

print("Test completed") 