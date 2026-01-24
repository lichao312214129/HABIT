from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from .selector_registry import run_selector
import logging

class FeatureSelectorTransformer(BaseEstimator, TransformerMixin):
    """
    A wrapper to make HABIT feature selectors compatible with sklearn Pipeline.
    
    It runs the specific feature selector logic during `fit`, stores the 
    selected feature names, and filters the dataset during `transform`.
    """
    
    def __init__(self, method_name: str, params: Dict[str, Any] = None, logger: logging.Logger = None):
        """
        Args:
            method_name: Name of the selector (e.g., 'anova', 'icc', 'variance')
            params: Parameters for the selector
            logger: Logger instance
        """
        self.method_name = method_name
        self.params = params or {}
        self.selected_features_ = None
        self.logger = logger or logging.getLogger(__name__)

    def fit(self, X, y=None):
        """
        Run the feature selection logic.
        """
        # If X is numpy array, convert to DataFrame for our selectors
        if isinstance(X, np.ndarray):
            # Try to recover feature names if possible, otherwise generic
            # In a pipeline, previous steps might return numpy arrays (e.g. Scaler)
            # This is a critical point: We need feature names for our selectors to work properly
            # if they rely on column names (like ICC reading from a file with names).
            # However, for statistical methods (ANOVA, Variance), names are metadata.
            # If input is numpy (from Scaler), we lose column names.
            
            # TODO: This is a limitation of sklearn pipelines with pandas.
            # Ideally, we rely on set_output(transform="pandas") in sklearn v1.2+,
            # or we accept that intermediate steps might lose names.
            
            # For now, generate generic names if needed, but this might break
            # selectors that depend on specific external file matching (like ICC).
            X = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])

        # Ensure y is Series
        if y is not None and not isinstance(y, pd.Series):
            y = pd.Series(y)

        # Get current feature names
        current_features = X.columns.tolist()
        
        self.logger.info(f"Running {self.method_name} selector on {len(current_features)} features...")
        
        try:
            # Run the existing selector logic
            selected = run_selector(
                self.method_name,
                X,
                y,
                selected_features=current_features,
                **self.params
            )
            
            # Intersection to ensure validity and preserve order
            current_features_set = set(current_features)
            self.selected_features_ = [f for f in selected if f in current_features_set]
            
            self.logger.info(f"{self.method_name}: Selected {len(self.selected_features_)}/{len(current_features)} features.")
            
        except Exception as e:
            self.logger.warning(f"Selector {self.method_name} failed: {e}. Keeping all features.")
            self.selected_features_ = current_features

        return self

    def transform(self, X):
        """
        Select the features.
        """
        if self.selected_features_ is None:
            # If not fitted or failed, return X as is (or raise error)
            return X
            
        # Handle numpy/pandas conversion
        if isinstance(X, pd.DataFrame):
            # Check if features exist (column names might have changed if not managed carefully)
            # If X matches the columns we fitted on:
            missing_cols = [c for c in self.selected_features_ if c not in X.columns]
            if missing_cols:
                # Fallback for numpy array wrapped in generic DF or similar issues
                # This happens if pipeline steps strip metadata.
                # For robustness, if we can't find columns by name, we might need to rely on indices 
                # if we assume structure hasn't changed. But that's risky.
                # For this refactor, we assume data flows as DataFrame or compatible structure.
                raise ValueError(f"Features missing in transform: {missing_cols}")
            return X[self.selected_features_]
        
        elif isinstance(X, np.ndarray):
            # If we received numpy array, we assume column order matches what we saw in fit
            # We need to map selected_features_ names back to indices
            # This requires we saved the feature list passed to fit
            # Implementation detail: Use sklearn's feature_names_in_ if available
            pass
            
        return X[self.selected_features_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.selected_features_)
