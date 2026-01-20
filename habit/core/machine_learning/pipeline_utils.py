from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from .feature_selectors import run_selector
from habit.utils.log_utils import get_module_logger

class FeatureSelectTransformer(BaseEstimator, TransformerMixin):
    """
    A wrapper to make HABIT feature selectors compatible with sklearn Pipeline.
    Supports input as both pd.DataFrame and np.ndarray.
    """
    def __init__(self, methods_config: List[Dict[str, Any]], feature_names: List[str] = None, 
                 before_z_score_only: bool = False, after_z_score_only: bool = False,
                 outdir: str = None):
        self.methods_config = methods_config
        self.feature_names = feature_names
        self.before_z_score_only = before_z_score_only
        self.after_z_score_only = after_z_score_only
        self.outdir = outdir
        self.selected_features_ = None
        self.fitted_feature_names_ = None  # Store actual feature names from fit
        self.logger = get_module_logger('ml.feature_selection')  # Logger for detailed feature selection tracking

    def _ensure_dataframe(self, X: Any, use_fitted_names: bool = False) -> pd.DataFrame:
        """
        Ensures the input is a pandas DataFrame, reconstructing it from numpy if necessary.
        
        Args:
            X: Input data (DataFrame or ndarray)
            use_fitted_names: If True, use fitted_feature_names_ instead of feature_names
        """
        if isinstance(X, pd.DataFrame):
            return X
        if isinstance(X, np.ndarray):
            # Use fitted feature names if available and requested, otherwise use initial feature names
            names_to_use = self.fitted_feature_names_ if (use_fitted_names and self.fitted_feature_names_ is not None) else self.feature_names
            if names_to_use is None:
                raise ValueError(
                    "FeatureSelectTransformer received a numpy array but 'feature_names' is not available. "
                    "This usually happens when using sklearn<1.2 where scalers don't preserve DataFrame format. "
                    "Please upgrade to sklearn>=1.2 or ensure feature_names is provided at initialization."
                )
            # Validate shape consistency
            if X.shape[1] != len(names_to_use):
                raise ValueError(
                    f"Shape mismatch: received array with {X.shape[1]} features, "
                    f"but expected {len(names_to_use)} features based on stored names."
                )
            # Reconstruct DataFrame assuming columns match the feature names
            return pd.DataFrame(X, columns=names_to_use)
        return pd.DataFrame(X)

    def fit(self, X: Any, y: pd.Series = None):
        X_df = self._ensure_dataframe(X, use_fitted_names=False)
        # Store the actual feature names from the input data for later use in transform
        self.fitted_feature_names_ = list(X_df.columns)
        current_features = list(X_df.columns)
        
        # Determine the stage for logging
        stage = "Before Normalization" if self.before_z_score_only else "After Normalization" if self.after_z_score_only else "Full Pipeline"
        
        # Log initial feature information
        self.logger.info("=" * 80)
        self.logger.info(f"Feature Selection Stage: {stage}")
        self.logger.info("=" * 80)
        self.logger.info(f"Initial number of features: {len(current_features)}")
        self.logger.info(f"Initial features: {current_features}")
        self.logger.info("-" * 80)
        
        step_count = 0
        for conf in self.methods_config:
            method = conf['method']
            params = conf.get('params', {}).copy()
            
            # Check if we should execute this method based on before_z_score flag
            # 具体解释：before_z_score_only和after_z_score_only是两个标志位，用于控制是否在标准化之前或之后进行特征选择。
            # 如果before_z_score_only为True，则只会在标准化之前进行特征选择。
            # 如果after_z_score_only为True，则只会在标准化之后进行特征选择。
            # 如果before_z_score_only和after_z_score_only都为False，则会在标准化之前和之后都进行特征选择。
            before_z_score = params.get('before_z_score', False)
            if self.before_z_score_only and not before_z_score:
                self.logger.debug(f"Skipping method '{method}' (before_z_score={before_z_score}, but before_z_score_only=True)")
                continue
            if self.after_z_score_only and before_z_score:
                self.logger.debug(f"Skipping method '{method}' (before_z_score={before_z_score}, but after_z_score_only=True)")
                continue

            step_count += 1
            features_before = current_features.copy()
            
            # Log the step being executed
            self.logger.info(f"\nStep {step_count}: Applying '{method}' feature selection")
            self.logger.info(f"  Parameters: {params}")
            self.logger.info(f"  Features before this step: {len(features_before)}")

            # Pass output directory for plotting
            if self.outdir:
                params['outdir'] = self.outdir

            # Execute existing HABIT selector logic
            selected = run_selector(method, X_df, y, current_features, **params)
            
            # Maintain intersection
            current_features = [f for f in current_features if f in selected]
            
            # Calculate removed features
            removed_features = [f for f in features_before if f not in current_features]
            
            # Log detailed results
            self.logger.info(f"  Features after this step: {len(current_features)}")
            self.logger.info(f"  Number of features removed: {len(removed_features)}")
            
            if removed_features:
                self.logger.info(f"  Removed features: {removed_features}")
            else:
                self.logger.info(f"  No features removed in this step")
                
            self.logger.info(f"  Retained features: {current_features}")
            self.logger.info("-" * 80)
        
        # Log final summary
        self.logger.info(f"\nFeature Selection Summary ({stage}):")
        self.logger.info(f"  Total steps executed: {step_count}")
        self.logger.info(f"  Initial features: {len(self.fitted_feature_names_)}")
        self.logger.info(f"  Final features: {len(current_features)}")
        self.logger.info(f"  Total features removed: {len(self.fitted_feature_names_) - len(current_features)}")
        self.logger.info(f"  Retention rate: {len(current_features) / len(self.fitted_feature_names_) * 100:.2f}%")
        self.logger.info(f"  Final selected features: {current_features}")
        self.logger.info("=" * 80)
        
        self.selected_features_ = current_features
        return self

    def transform(self, X: Any):
        # Use fitted feature names to reconstruct DataFrame from numpy array
        X_df = self._ensure_dataframe(X, use_fitted_names=True)
        if self.selected_features_ is None:
            return X_df
        # Subset to selected features
        return X_df[self.selected_features_]
