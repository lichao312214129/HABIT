"""
ThresholdManager Module
Handles the calculation, storage, and application of decision thresholds.
"""

from typing import Dict, Any
from .metrics import calculate_metrics_youden
from .prediction_container import PredictionContainer

class ThresholdManager:
    """
    Manages optimal decision thresholds across datasets.
    
    This class is central to clinical research workflows where a threshold
    (e.g., from Youden's Index) is determined on a training/validation set
    and then applied to a separate test set.
    """
    def __init__(self):
        self.store: Dict[str, Dict[str, float]] = {} # {model_name: {threshold_type: value}}

    def find_and_store(self, model_name: str, container: PredictionContainer, method: str = 'youden'):
        """
        Calculates and stores the optimal threshold from a given dataset (usually training).
        """
        if method == 'youden':
            result = calculate_metrics_youden(container)
            threshold = result.get('threshold')
            if threshold is not None:
                if model_name not in self.store:
                    self.store[model_name] = {}
                self.store[model_name]['youden'] = threshold

    def get_threshold(self, model_name: str, method: str = 'youden') -> float:
        """Retrieves a stored threshold."""
        return self.store.get(model_name, {}).get(method, 0.5) # Default to 0.5 if not found
