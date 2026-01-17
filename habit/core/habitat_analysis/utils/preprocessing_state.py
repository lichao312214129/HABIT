"""
Preprocessing State Management and Utilities for Habitat Analysis.

This module combines stateful preprocessing (for group-level operations with train/test separation)
and stateless utility functions (for subject-level operations).
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Literal
from sklearn.preprocessing import KBinsDiscretizer

# =============================================================================
# Utility Functions (Stateless)
# =============================================================================

def handle_extreme_values(features: np.ndarray, strategy: str = 'mean_replacement') -> np.ndarray:
    """
    Handle extreme values in feature matrix
    
    Args:
        features (np.ndarray): Feature matrix
        strategy (str): Strategy for handling extreme values
            - 'mean_replacement': Replace inf/nan with column mean
            - 'median_replacement': Replace inf/nan with column median
            - 'constant': Replace inf/nan with specified constant
            
    Returns:
        np.ndarray: Feature matrix with extreme values handled
    """
    # Create a copy to avoid modifying the original
    features_clean = features.copy()
    
    # Handle infinity
    if np.any(np.isinf(features_clean)):
        for col in range(features_clean.shape[1]):
            col_data = features_clean[:, col]
            mask_inf = np.isinf(col_data)
            
            if np.all(mask_inf):  # If all values are inf, replace with 0
                features_clean[:, col] = 0
            elif np.any(mask_inf):
                valid_data = col_data[~mask_inf]
                if strategy == 'mean_replacement':
                    replace_value = np.mean(valid_data)
                elif strategy == 'median_replacement':
                    replace_value = np.median(valid_data)
                else:  # Default to 0
                    replace_value = 0
                    
                features_clean[mask_inf, col] = replace_value
    
    # Handle NaN values
    if np.any(np.isnan(features_clean)):
        for col in range(features_clean.shape[1]):
            col_data = features_clean[:, col]
            mask_nan = np.isnan(col_data)
            
            if np.all(mask_nan):  # If all values are NaN, replace with 0
                features_clean[:, col] = 0
            elif np.any(mask_nan):
                valid_data = col_data[~mask_nan]
                if strategy == 'mean_replacement':
                    replace_value = np.mean(valid_data)
                elif strategy == 'median_replacement':
                    replace_value = np.median(valid_data)
                else:  # Default to 0
                    replace_value = 0
                    
                features_clean[mask_nan, col] = replace_value
    
    return features_clean

def create_discretizer(n_bins: int = 10, bin_strategy: str = 'uniform') -> KBinsDiscretizer:
    """
    Create a KBinsDiscretizer for feature binning
    
    Args:
        n_bins (int): Number of bins
        bin_strategy (str): Binning strategy ('uniform', 'quantile', 'kmeans')
        
    Returns:
        KBinsDiscretizer: Configured discretizer
    """
    return KBinsDiscretizer(
        n_bins=n_bins,
        encode='ordinal',  # Return bin indices
        strategy=bin_strategy
    )

def process_features_pipeline(
    features: np.ndarray,
    methods: List[Dict[str, Any]]
) -> np.ndarray:
    """
    Apply a pipeline of preprocessing methods to features (Stateless)
    
    Args:
        features (np.ndarray): Feature matrix to preprocess
        methods (List[Dict[str, Any]]): List of preprocessing methods configs
    
    Returns:
        np.ndarray: Preprocessed feature matrix
    """
    processed_features = features.copy()
    
    for method_config in methods:
        method_name = method_config.get('method', 'minmax')
        config = method_config.copy()
        if 'method' in config:
            del config['method']
        processed_features = preprocess_features(processed_features, method=method_name, **config)
    
    return processed_features

def preprocess_features(
    features: np.ndarray,
    method: Literal['minmax', 'zscore', 'robust', 'binning', 'global_minmax', 'global_zscore', 'winsorize', 'log'] = 'minmax',
    n_bins: int = 10,
    bin_strategy: str = 'uniform',
    global_normalize: bool = False,
    winsor_limits: tuple = (0.05, 0.05),
    discretizer: Optional[KBinsDiscretizer] = None,
    methods: Optional[List[Dict[str, Any]]] = None
) -> np.ndarray:
    """
    Preprocess features using specified method or method pipeline (Stateless)
    """
    if methods is not None:
        return process_features_pipeline(features, methods)
    
    features = handle_extreme_values(features)
    
    if method == 'minmax':
        if global_normalize:
            min_val = np.min(features)
            max_val = np.max(features)
            return (features - min_val) / (max_val - min_val + 1e-6)
        else:
            min_vals = np.min(features, axis=0)
            max_vals = np.max(features, axis=0)
            return (features - min_vals) / (max_vals - min_vals + 1e-6)
        
    elif method == 'zscore':
        if global_normalize:
            mean = np.mean(features)
            std = np.std(features)
            return (features - mean) / (std + 1e-6)
        else:
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0)
            return (features - mean) / (std + 1e-6)
        
    elif method == 'robust':
        if global_normalize:
            q1 = np.percentile(features.flatten(), 25)
            q3 = np.percentile(features.flatten(), 75)
            iqr = q3 - q1
            median = np.median(features.flatten())
            return (features - median) / (iqr + 1e-6)
        else:
            q1 = np.percentile(features, 25, axis=0)
            q3 = np.percentile(features, 75, axis=0)
            iqr = q3 - q1
            median = np.median(features, axis=0)
            return (features - median) / (iqr + 1e-6)
        
    elif method == 'binning':
        if global_normalize:
            original_shape = features.shape
            flattened = features.flatten().reshape(-1, 1)
            if discretizer is None:
                discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=bin_strategy)
                binned_flattened = discretizer.fit_transform(flattened)
            else:
                binned_flattened = discretizer.transform(flattened)
            binned_features = binned_flattened.reshape(original_shape)
        else:
            if discretizer is None:
                discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=bin_strategy)
                binned_features = discretizer.fit_transform(features)
            else:
                binned_features = discretizer.transform(features)
        return binned_features
    
    elif method == 'winsorize':
        if global_normalize:
            flattened = features.flatten()
            lower = np.percentile(flattened, winsor_limits[0] * 100)
            upper = np.percentile(flattened, (1 - winsor_limits[1]) * 100)
            return np.clip(features, lower, upper)
        else:
            lower = np.percentile(features, winsor_limits[0] * 100, axis=0)
            upper = np.percentile(features, (1 - winsor_limits[1]) * 100, axis=0)
            return np.clip(features, lower, upper)
    
    elif method == 'log':
        if global_normalize:
            min_val = np.min(features)
            shifted_features = features - min_val + 1.0
            return np.log(shifted_features)
        else:
            min_vals = np.min(features, axis=0)
            shifted_features = features - min_vals + 1.0
            return np.log(shifted_features)
    
    else:
        raise ValueError(f"Unknown preprocessing method: {method}")

# =============================================================================
# State Management Class (Stateful)
# =============================================================================

class PreprocessingState:
    """
    Manages state for group-level preprocessing operations.
    
    Supports capturing parameters during training and applying them during testing.
    Leverages shared utility functions.
    """
    
    def __init__(self):
        """Initialize empty state."""
        # Basic statistics
        self.means: Optional[pd.Series] = None
        self.stds: Optional[pd.Series] = None
        self.mins: Optional[pd.Series] = None
        self.maxs: Optional[pd.Series] = None
        
        # Robust statistics
        self.medians: Optional[pd.Series] = None
        self.q1s: Optional[pd.Series] = None
        self.q3s: Optional[pd.Series] = None
        
        # Binning/discretizer
        self.discretizers: Dict[str, KBinsDiscretizer] = {}
        
        # Winsorization limits
        self.winsor_lowers: Optional[pd.Series] = None
        self.winsor_uppers: Optional[pd.Series] = None
        
        # Log transformation offsets
        self.log_offsets: Optional[pd.Series] = None
        
        # Global normalization parameters
        self.global_params: Dict[str, Any] = {}
        
        # Methods configuration
        self.methods_config: List[Dict] = []
        
    def fit(self, df: pd.DataFrame, methods: List[Dict[str, Any]]) -> None:
        """
        Calculate and store parameters from training data.
        """
        self.methods_config = methods
        
        # Always compute basic statistics for imputation and potential use
        self.means = df.mean()
        self.stds = df.std().replace(0, 1.0)
        self.mins = df.min()
        self.maxs = df.max()
        
        # Impute NaN first to get clean data for subsequent parameter estimation
        temp_df = df.fillna(self.means)
        
        # Compute parameters for each method
        for method_config in methods:
            method_name = method_config.get('method', 'minmax')
            global_normalize = method_config.get('global_normalize', False)
            
            if method_name in ['z_score', 'zscore', 'standardization']:
                if global_normalize:
                    self.global_params['zscore'] = {
                        'mean': temp_df.values.mean(),
                        'std': temp_df.values.std() if temp_df.values.std() != 0 else 1.0
                    }
                # Per-feature params already computed
                
            elif method_name in ['min_max', 'minmax', 'normalize']:
                if global_normalize:
                    self.global_params['minmax'] = {
                        'min': temp_df.values.min(),
                        'max': temp_df.values.max()
                    }
                # Per-feature params already computed
                
            elif method_name == 'robust':
                if global_normalize:
                    flat_values = temp_df.values.flatten()
                    self.global_params['robust'] = {
                        'median': np.median(flat_values),
                        'q1': np.percentile(flat_values, 25),
                        'q3': np.percentile(flat_values, 75)
                    }
                else:
                    self.medians = temp_df.median()
                    self.q1s = temp_df.quantile(0.25)
                    self.q3s = temp_df.quantile(0.75)
                    
            elif method_name == 'binning':
                n_bins = method_config.get('n_bins', 10)
                bin_strategy = method_config.get('bin_strategy', 'uniform')
                
                if global_normalize:
                    # Global binning
                    discretizer = create_discretizer(n_bins, bin_strategy)
                    flat_values = temp_df.values.flatten().reshape(-1, 1)
                    discretizer.fit(flat_values)
                    self.discretizers['global'] = discretizer
                else:
                    # Per-feature binning
                    discretizer = create_discretizer(n_bins, bin_strategy)
                    discretizer.fit(temp_df.values)
                    self.discretizers['per_feature'] = discretizer
                    
            elif method_name == 'winsorize':
                winsor_limits = method_config.get('winsor_limits', (0.05, 0.05))
                
                if global_normalize:
                    flat_values = temp_df.values.flatten()
                    self.global_params['winsorize'] = {
                        'lower': np.percentile(flat_values, winsor_limits[0] * 100),
                        'upper': np.percentile(flat_values, (1 - winsor_limits[1]) * 100)
                    }
                else:
                    self.winsor_lowers = temp_df.quantile(winsor_limits[0])
                    self.winsor_uppers = temp_df.quantile(1 - winsor_limits[1])
                    
            elif method_name == 'log':
                if global_normalize:
                    self.global_params['log_offset'] = temp_df.values.min()
                else:
                    self.log_offsets = temp_df.min()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply stored parameters to transform data (Training or Testing).
        """
        if self.means is None:
            raise ValueError("PreprocessingState has not been fitted. Call fit() first.")
            
        df_transformed = df.copy()
        
        # 1. Imputation (Always apply first to handle NaN)
        df_transformed = df_transformed.fillna(self.means)
        
        # 2. Handle extreme values (inf) using the shared utility
        df_values = handle_extreme_values(df_transformed.values, strategy='mean_replacement')
        df_transformed = pd.DataFrame(df_values, columns=df_transformed.columns, index=df_transformed.index)
        
        # 3. Apply methods in order
        for method_config in self.methods_config:
            method_name = method_config.get('method', 'minmax')
            global_normalize = method_config.get('global_normalize', False)
            
            df_transformed = self._apply_method(
                df_transformed, 
                method_name, 
                global_normalize,
                method_config
            )
                
        return df_transformed
    
    def _apply_method(
        self, 
        df: pd.DataFrame, 
        method_name: str,
        global_normalize: bool,
        method_config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Apply a single preprocessing method using stored parameters."""
        
        if method_name in ['z_score', 'zscore', 'standardization']:
            if global_normalize and 'zscore' in self.global_params:
                params = self.global_params['zscore']
                return (df - params['mean']) / params['std']
            else:
                return (df - self.means) / self.stds
                
        elif method_name in ['min_max', 'minmax', 'normalize']:
            if global_normalize and 'minmax' in self.global_params:
                params = self.global_params['minmax']
                denom = (params['max'] - params['min']) if params['max'] != params['min'] else 1.0
                return (df - params['min']) / denom
            else:
                denom = (self.maxs - self.mins).replace(0, 1.0)
                return (df - self.mins) / denom
                
        elif method_name == 'robust':
            if global_normalize and 'robust' in self.global_params:
                params = self.global_params['robust']
                iqr = params['q3'] - params['q1']
                iqr = iqr if iqr != 0 else 1.0
                return (df - params['median']) / iqr
            else:
                iqr = (self.q3s - self.q1s).replace(0, 1.0)
                return (df - self.medians) / iqr
                
        elif method_name == 'binning':
            if global_normalize and 'global' in self.discretizers:
                discretizer = self.discretizers['global']
                original_shape = df.shape
                flat_values = df.values.flatten().reshape(-1, 1)
                binned = discretizer.transform(flat_values)
                return pd.DataFrame(
                    binned.reshape(original_shape),
                    columns=df.columns,
                    index=df.index
                )
            elif 'per_feature' in self.discretizers:
                discretizer = self.discretizers['per_feature']
                binned = discretizer.transform(df.values)
                return pd.DataFrame(binned, columns=df.columns, index=df.index)
            else:
                return df
                
        elif method_name == 'winsorize':
            if global_normalize and 'winsorize' in self.global_params:
                params = self.global_params['winsorize']
                return df.clip(lower=params['lower'], upper=params['upper'])
            else:
                return df.clip(lower=self.winsor_lowers, upper=self.winsor_uppers, axis=1)
                
        elif method_name == 'log':
            if global_normalize and 'log_offset' in self.global_params:
                offset = self.global_params['log_offset']
                return np.log(df - offset + 1.0)
            else:
                return np.log(df - self.log_offsets + 1.0)
        
        else:
            # Unknown method, return unchanged
            return df

    def save(self, output_dir: str, filename: str = 'preprocessing_state.pkl') -> None:
        """Save state to disk."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            
    @classmethod
    def load(cls, output_dir: str, filename: str = 'preprocessing_state.pkl') -> 'PreprocessingState':
        """Load state from disk."""
        path = os.path.join(output_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Preprocessing state file not found at {path}")
            
        with open(path, 'rb') as f:
            return pickle.load(f)
