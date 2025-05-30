"""
Feature preprocessing utilities for habitat clustering
"""

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from typing import Optional, Union, Literal, List, Dict, Any

def process_features_pipeline(
    features: np.ndarray,
    methods: List[Dict[str, Any]]
) -> np.ndarray:
    """
    Apply a pipeline of preprocessing methods to features
    
    Args:
        features (np.ndarray): Feature matrix to preprocess
        methods (List[Dict[str, Any]]): List of preprocessing methods configs
            Each dict should contain a 'method' key and other parameters for that method
    
    Returns:
        np.ndarray: Preprocessed feature matrix after applying all methods in sequence
    """
    processed_features = features.copy()
    
    for method_config in methods:
        # Get the method name and create a copy of the config
        method_name = method_config.get('method', 'minmax')
        config = method_config.copy()
        
        # Remove 'method' key as it's passed as a separate param
        if 'method' in config:
            del config['method']
            
        # Apply the preprocessing method
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
    Preprocess features using specified method or method pipeline
    
    Args:
        features (np.ndarray): Feature matrix to preprocess
        method (str): Preprocessing method
            - 'minmax': Min-Max scaling to [0, 1]
            - 'zscore': Z-score standardization
            - 'robust': Robust scaling using IQR
            - 'binning': Discretize features into bins
            - 'global_minmax': Global Min-Max scaling across all features
            - 'global_zscore': Global Z-score standardization across all features
            - 'winsorize': Limit extreme values to specified percentiles
            - 'log': Apply log transformation (log(x + 1))
        n_bins (int): Number of bins for binning method
        bin_strategy (str): Strategy for binning ('uniform', 'quantile', 'kmeans')
        global_normalize (bool): Whether to normalize across all features
        winsor_limits (tuple): Lower and upper percentiles for winsorizing
        discretizer (KBinsDiscretizer, optional): Pre-initialized discretizer for binning
        methods (List[Dict[str, Any]], optional): List of preprocessing methods to apply in sequence
            If provided, this overrides the 'method' parameter and applies multiple methods
        
    Returns:
        np.ndarray: Preprocessed feature matrix
    """
    # If methods list is provided, use pipeline processing
    if methods is not None:
        return process_features_pipeline(features, methods)
    
    # Handle NaN and infinite values before preprocessing
    features = handle_extreme_values(features)
    
    if method == 'minmax':
        if global_normalize:
            # Global Min-Max scaling across all features
            min_val = np.min(features)
            max_val = np.max(features)
            return (features - min_val) / (max_val - min_val + 1e-6)
        else:
            # Per-feature Min-Max scaling
            min_vals = np.min(features, axis=0)
            max_vals = np.max(features, axis=0)
            return (features - min_vals) / (max_vals - min_vals + 1e-6)
        
    elif method == 'zscore':
        if global_normalize:
            # Global Z-score standardization
            mean = np.mean(features)
            std = np.std(features)
            return (features - mean) / (std + 1e-6)
        else:
            # Per-feature Z-score standardization
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0)
            return (features - mean) / (std + 1e-6)
        
    elif method == 'robust':
        if global_normalize:
            # Global robust scaling using IQR
            q1 = np.percentile(features.flatten(), 25)
            q3 = np.percentile(features.flatten(), 75)
            iqr = q3 - q1
            median = np.median(features.flatten())
            return (features - median) / (iqr + 1e-6)
        else:
            # Per-feature robust scaling using IQR
            q1 = np.percentile(features, 25, axis=0)
            q3 = np.percentile(features, 75, axis=0)
            iqr = q3 - q1
            median = np.median(features, axis=0)
            return (features - median) / (iqr + 1e-6)
        
    elif method == 'binning':
        if global_normalize:
            # Global binning - flatten, bin, and reshape
            original_shape = features.shape
            flattened = features.flatten().reshape(-1, 1)
            
            if discretizer is None:
                discretizer = KBinsDiscretizer(
                    n_bins=n_bins,
                    encode='ordinal',  # Return bin indices
                    strategy=bin_strategy
                )
                binned_flattened = discretizer.fit_transform(flattened)
            else:
                binned_flattened = discretizer.transform(flattened)
            
            # Reshape back to original dimensions
            binned_features = binned_flattened.reshape(original_shape)
        else:
            # Per-feature binning
            if discretizer is None:
                discretizer = KBinsDiscretizer(
                    n_bins=n_bins,
                    encode='ordinal',  # Return bin indices
                    strategy=bin_strategy
                )
                binned_features = discretizer.fit_transform(features)
            else:
                binned_features = discretizer.transform(features)
            
        return binned_features
    
    elif method == 'winsorize':
        if global_normalize:
            # Global winsorization
            flattened = features.flatten()
            lower = np.percentile(flattened, winsor_limits[0] * 100)
            upper = np.percentile(flattened, (1 - winsor_limits[1]) * 100)
            return np.clip(features, lower, upper)
        else:
            # Per-feature winsorization
            lower = np.percentile(features, winsor_limits[0] * 100, axis=0)
            upper = np.percentile(features, (1 - winsor_limits[1]) * 100, axis=0)
            return np.clip(features, lower, upper)
    
    elif method == 'log':
        if global_normalize:
            # Global log transformation
            min_val = np.min(features)
            shifted_features = features - min_val + 1.0  # Ensure all values > 0
            return np.log(shifted_features)
        else:
            # Per-feature log transformation
            min_vals = np.min(features, axis=0)
            shifted_features = features - min_vals + 1.0  # Ensure all values > 0
            return np.log(shifted_features)
    
    else:
        raise ValueError(f"Unknown preprocessing method: {method}")

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