"""
Ensemble Model Wrapper for K-Fold Cross Validation.
Allows treating a collection of K-fold models as a single scikit-learn estimator.
"""

from typing import List, Any
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted

class HabitEnsembleModel(BaseEstimator, ClassifierMixin):
    """
    An ensemble wrapper that aggregates predictions from multiple fitted pipelines.
    Used primarily to wrap K-Fold cross-validation results into a single predict-ready object.
    
    Attributes:
        estimators (List[Any]): List of fitted scikit-learn pipelines/models.
        voting (str): 'soft' (average probabilities) or 'hard' (majority vote). Default 'soft'.
    """
    
    def __init__(self, estimators: List[Any], voting: str = 'soft'):
        self.estimators = estimators
        self.voting = voting
        
    def fit(self, X, y=None):
        """
        No-op fit method. 
        We assume the estimators passed in __init__ are ALREADY fitted.
        This allows us to wrap the K-Fold results directly.
        """
        # We could implement a refit logic here if needed, but for K-Fold wrapper
        # the intention is to reuse the already trained folds.
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        
        Returns:
            array-like of shape (n_samples, n_classes)
        """
        if not self.estimators:
            raise ValueError("No estimators provided to HabitEnsembleModel")
            
        # Collect probabilities from all estimators
        all_probs = []
        for est in self.estimators:
            if hasattr(est, "predict_proba"):
                all_probs.append(est.predict_proba(X))
            else:
                raise ValueError(f"Estimator {type(est)} does not support predict_proba")
        
        # Stack and average (Soft Voting)
        # Shape of all_probs items: (n_samples, n_classes)
        # Stacked shape: (n_estimators, n_samples, n_classes)
        avg_probs = np.mean(all_probs, axis=0)
        return avg_probs

    def predict(self, X):
        """
        Predict class labels for X.
        """
        if self.voting == 'soft':
            # For soft voting, average probabilities then take argmax
            probs = self.predict_proba(X)
            # Assuming binary classification or standard sklearn behavior where columns are sorted classes
            # We need to map argmax index back to classes if classes_ attribute exists
            # But pipelines usually return indices or we can assume standard 0/1 for binary
            if hasattr(self.estimators[0], 'classes_'):
                indices = np.argmax(probs, axis=1)
                return self.estimators[0].classes_[indices]
            else:
                return np.argmax(probs, axis=1)
        
        elif self.voting == 'hard':
            # For hard voting, collect predictions and take mode
            all_preds = []
            for est in self.estimators:
                all_preds.append(est.predict(X))
            
            all_preds = np.array(all_preds).T # (n_samples, n_estimators)
            
            # Majority vote
            final_preds = []
            for sample_preds in all_preds:
                vals, counts = np.unique(sample_preds, return_counts=True)
                final_preds.append(vals[np.argmax(counts)])
            
            return np.array(final_preds)
        else:
            raise ValueError(f"Unknown voting method: {self.voting}")
    
    @property
    def classes_(self):
        """Delegate classes_ attribute to the first estimator."""
        if self.estimators and hasattr(self.estimators[0], 'classes_'):
            return self.estimators[0].classes_
        return None
