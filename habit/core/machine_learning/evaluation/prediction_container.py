"""
PredictionContainer Module
Encapsulates prediction results and handles binary/multiclass abstraction.
"""

import numpy as np
from typing import Union, Optional

class PredictionContainer:
    """
    A unified container for model predictions.
    Ensures consistency between probabilities and predicted labels.
    """
    def __init__(self, y_true: np.ndarray, y_prob: np.ndarray, y_pred: Optional[np.ndarray] = None):
        self.y_true = np.array(y_true)
        self.y_prob = np.array(y_prob)
        
        # Detect classes
        self.classes = np.unique(self.y_true)
        self.num_classes = len(self.classes)
        
        # Priority: 
        # 1. Provided y_pred (from model.predict())
        # 2. Argmax/Threshold fallback
        if y_pred is not None:
            self.y_pred = np.array(y_pred)
        else:
            self.y_pred = self._generate_default_preds()

    def _generate_default_preds(self) -> np.ndarray:
        """Fallback prediction logic if y_pred is not provided."""
        if self.num_classes == 2:
            return (self.get_binary_probs() >= 0.5).astype(int)
        else:
            return np.argmax(self.y_prob, axis=1)

    def get_binary_probs(self) -> np.ndarray:
        """
        Returns a 1D array of probabilities for the positive class.
        """
        if self.y_prob.ndim == 1:
            return self.y_prob
        if self.y_prob.ndim == 2:
            # For multiclass, index 1 is only meaningful if it's actually binary data
            return self.y_prob[:, 1]
        return self.y_prob

    def get_eval_probs(self) -> np.ndarray:
        """
        Returns probabilities optimized for evaluation metrics.
        1D for binary, 2D for multiclass.
        """
        if self.num_classes == 2:
            return self.get_binary_probs()
        return self.y_prob

    def to_dict(self):
        """Converts results to a serializable dictionary."""
        return {
            'y_true': self.y_true.tolist(),
            'y_prob': self.get_eval_probs().tolist(),
            'y_pred': self.y_pred.tolist()
        }