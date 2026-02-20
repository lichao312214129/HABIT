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
from pydantic import BaseModel

# Import ResultColumns to identify metadata columns
from ..config_schemas import ResultColumns

# =============================================================================
# Utility Functions (Stateless)
# =============================================================================

def _get_method_attr(method_config: Union[Dict[str, Any], BaseModel], attr: str, default: Any = None) -> Any:
    """
    Get attribute from method_config, supporting both dict and Pydantic model.
    
    Priority: Try attribute access first (for Pydantic models), then fall back to .get() (for dicts).
    This maintains backward compatibility with dict-based configs.
    
    Args:
        method_config: Either a dict or a Pydantic model (PreprocessingMethod)
        attr: Attribute name to get
        default: Default value if attribute is missing or None
        
    Returns:
        Attribute value or default
    """
    # Try attribute access first (for Pydantic models and objects with attributes)
    if hasattr(method_config, attr):
        try:
            value = getattr(method_config, attr)
            return value if value is not None else default
        except (AttributeError, TypeError):
            pass
    
    # Fall back to dict .get() method (for dicts and backward compatibility)
    if isinstance(method_config, dict):
        return method_config.get(attr, default)
    elif hasattr(method_config, 'get') and callable(getattr(method_config, 'get')):
        # Support objects that have a .get() method (like dict-like objects)
        return method_config.get(attr, default)
    
    return default

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
    methods: List[Union[Dict[str, Any], BaseModel]]
) -> np.ndarray:
    """
    Apply a pipeline of preprocessing methods to features (Stateless)
    
    Args:
        features (np.ndarray): Feature matrix to preprocess
        methods (List[Union[Dict[str, Any], BaseModel]]): List of preprocessing methods configs
            Can be either dict or PreprocessingMethod objects
    
    Returns:
        np.ndarray: Preprocessed feature matrix
    """
    processed_features = features.copy()
    
    for method_config in methods:
        method_name = _get_method_attr(method_config, 'method', 'minmax')
        
        # Extract config parameters
        if isinstance(method_config, dict):
            config = method_config.copy()
            if 'method' in config:
                del config['method']
        else:
            # Pydantic model - convert to dict for **kwargs
            config = method_config.model_dump(exclude={'method'})
            # Remove None values to use defaults
            config = {k: v for k, v in config.items() if v is not None}
        
        processed_features = preprocess_features(processed_features, method=method_name, **config)
    
    return processed_features

def preprocess_features(
    features: np.ndarray,
    method: Literal[
        'minmax',
        'zscore',
        'robust',
        'binning',
        'global_minmax',
        'global_zscore',
        'winsorize',
        'log',
        'variance_filter',
        'correlation_filter'
    ] = 'minmax',
    n_bins: int = 10,
    bin_strategy: str = 'uniform',
    global_normalize: bool = False,
    winsor_limits: tuple = (0.05, 0.05),
    variance_threshold: float = 0.0,
    corr_threshold: float = 0.95,
    corr_method: str = 'spearman',
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

    elif method == 'variance_filter':
        variances = np.var(features, axis=0)
        selected_indices = np.where(variances > variance_threshold)[0]
        if selected_indices.size == 0:
            # Keep at least one feature to avoid empty matrix.
            selected_indices = np.array([int(np.argmax(variances))])
        return features[:, selected_indices]

    elif method == 'correlation_filter':
        if features.shape[1] <= 1:
            return features
        corr_matrix = np.corrcoef(features, rowvar=False)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        keep_indices = list(range(features.shape[1]))
        i = 0
        while i < len(keep_indices):
            current_idx = keep_indices[i]
            to_remove = []
            for j in range(i + 1, len(keep_indices)):
                compare_idx = keep_indices[j]
                if abs(corr_matrix[current_idx, compare_idx]) > corr_threshold:
                    to_remove.append(compare_idx)
            keep_indices = [idx for idx in keep_indices if idx not in to_remove]
            i += 1
        if not keep_indices:
            keep_indices = [0]
        return features[:, keep_indices]
    
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
        self.methods_config: List[Union[Dict[str, Any], BaseModel]] = []
        # Feature selection columns per preprocessing step index.
        # This guarantees predict mode uses exactly the same selected columns as train.
        self.selected_columns_by_step: Dict[int, List[str]] = {}

    @staticmethod
    def _compute_variance_selected_columns(df: pd.DataFrame, threshold: float) -> List[str]:
        """Select columns whose variance is strictly greater than threshold."""
        variances = df.var()
        selected = variances[variances > threshold].index.tolist()
        if not selected:
            selected = [variances.sort_values(ascending=False).index[0]]
        return selected

    @staticmethod
    def _compute_correlation_selected_columns(
        df: pd.DataFrame,
        threshold: float,
        method: str
    ) -> List[str]:
        """Remove highly correlated columns while preserving a deterministic order."""
        if df.shape[1] <= 1:
            return list(df.columns)
        corr = df.corr(method=method).abs().fillna(0.0)
        features = list(df.columns)
        i = 0
        while i < len(features):
            current = features[i]
            to_remove: List[str] = []
            for j in range(i + 1, len(features)):
                candidate = features[j]
                if corr.loc[current, candidate] > threshold:
                    to_remove.append(candidate)
            features = [col for col in features if col not in to_remove]
            i += 1
        if not features:
            return [df.columns[0]]
        return features
        
    def fit(self, df: pd.DataFrame, methods: List[Union[Dict[str, Any], BaseModel]]) -> None:
        """
        Calculate and store parameters from training data.
        
        Args:
            df: DataFrame with features. Non-numeric columns (like Subject ID) will be automatically excluded.
            methods: List of preprocessing method configurations
        """
        self.methods_config = methods
        self.selected_columns_by_step = {}
        
        # Debug: Check DataFrame info
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"PreprocessingState.fit() received DataFrame with shape: {df.shape}")
        logger.info(f"DataFrame columns: {list(df.columns)[:10]}...")  # First 10 columns
        logger.info(f"DataFrame dtypes sample: {df.dtypes.value_counts().to_dict()}")
        
        # Filter out non-numeric columns (metadata columns like Subject ID)
        numeric_df = self._get_numeric_columns(df)
        
        logger.info(f"After filtering, numeric columns count: {len(numeric_df.columns)}")
        if len(numeric_df.columns) > 0:
            logger.info(f"Sample numeric columns: {list(numeric_df.columns)[:5]}")
        
        if numeric_df.empty:
            logger.error(f"All DataFrame columns: {list(df.columns)}")
            logger.error(f"DataFrame dtypes:\n{df.dtypes}")
            raise ValueError("No numeric columns found in DataFrame. Cannot perform preprocessing.")
        
        # Always compute basic statistics for imputation and potential use
        self.means = numeric_df.mean()
        self.stds = numeric_df.std().replace(0, 1.0)  # 解释：如果标准差为0，则替换为1.0 为什么这样做？因为如果标准差为0，则所有数据都相同，这样会导致除数为0，所以替换为1.0
        self.mins = numeric_df.min()
        self.maxs = numeric_df.max()
        
        # Impute NaN first to get clean data for subsequent parameter estimation
        temp_df = numeric_df.fillna(self.means)
        
        # Compute parameters for each method
        for step_index, method_config in enumerate(methods):
            method_name = _get_method_attr(method_config, 'method', 'minmax')
            global_normalize = _get_method_attr(method_config, 'global_normalize', False)
            
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
                n_bins = _get_method_attr(method_config, 'n_bins', 10)
                bin_strategy = _get_method_attr(method_config, 'bin_strategy', 'uniform')
                
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
                winsor_limits_raw = _get_method_attr(method_config, 'winsor_limits', None)
                if winsor_limits_raw is None:
                    winsor_limits = (0.05, 0.05)
                elif isinstance(winsor_limits_raw, list):
                    winsor_limits = tuple(winsor_limits_raw)
                else:
                    winsor_limits = winsor_limits_raw
                
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

            elif method_name == 'variance_filter':
                variance_threshold = _get_method_attr(method_config, 'variance_threshold', 0.0)
                selected_cols = self._compute_variance_selected_columns(temp_df, variance_threshold)
                self.selected_columns_by_step[step_index] = selected_cols
                temp_df = temp_df[selected_cols]

            elif method_name == 'correlation_filter':
                corr_threshold = _get_method_attr(method_config, 'corr_threshold', 0.95)
                corr_method = _get_method_attr(method_config, 'corr_method', 'spearman')
                selected_cols = self._compute_correlation_selected_columns(
                    temp_df, corr_threshold, corr_method
                )
                self.selected_columns_by_step[step_index] = selected_cols
                temp_df = temp_df[selected_cols]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply stored parameters to transform data (Training or Testing).
        
        Args:
            df: DataFrame with features. Non-numeric columns will be preserved but not transformed.
            
        Returns:
            DataFrame with transformed numeric columns and preserved metadata columns
        """
        if self.means is None:
            raise ValueError("PreprocessingState has not been fitted. Call fit() first.")
        
        # Separate numeric and non-numeric columns
        numeric_df = self._get_numeric_columns(df)
        metadata_df = df[[col for col in df.columns if not ResultColumns.is_feature_column(col)]]
        
        if numeric_df.empty:
            # No numeric columns to transform, return original
            return df.copy()
        
        df_transformed = numeric_df.copy()
        
        # 1. Imputation (Always apply first to handle NaN)
        df_transformed = df_transformed.fillna(self.means)
        
        # 2. Handle extreme values (inf) using the shared utility
        df_values = handle_extreme_values(df_transformed.values, strategy='mean_replacement')
        df_transformed = pd.DataFrame(df_values, columns=df_transformed.columns, index=df_transformed.index)
        
        # 3. Apply methods in order
        for step_index, method_config in enumerate(self.methods_config):
            method_name = _get_method_attr(method_config, 'method', 'minmax')
            global_normalize = _get_method_attr(method_config, 'global_normalize', False)
            
            df_transformed = self._apply_method(
                df_transformed, 
                method_name, 
                global_normalize,
                method_config,
                step_index
            )
        
        # 4. Merge transformed numeric columns with metadata columns
        if not metadata_df.empty:
            # Ensure index alignment
            df_transformed = pd.concat([metadata_df, df_transformed], axis=1)
            # Reorder columns to match original order
            original_order = [col for col in df.columns if col in df_transformed.columns]
            df_transformed = df_transformed[original_order]
        
        return df_transformed
    
    def _get_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract only numeric columns from DataFrame, excluding metadata columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame containing only numeric feature columns
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        logger.info(f"select_dtypes found {len(numeric_cols)} numeric columns")
        if len(numeric_cols) > 0:
            logger.info(f"Sample numeric columns: {numeric_cols[:5]}")
        
        # Filter out metadata columns
        feature_cols = [col for col in numeric_cols if ResultColumns.is_feature_column(col)]
        logger.info(f"After filtering metadata, {len(feature_cols)} feature columns remain")
        
        if not feature_cols:
            logger.warning(f"Metadata columns found: {[col for col in df.columns if col in ResultColumns.metadata_columns()]}")
            logger.warning(f"Non-numeric columns: {df.select_dtypes(exclude=[np.number]).columns.tolist()[:10]}")
            return pd.DataFrame()
        
        return df[feature_cols]
    
    def _apply_method(
        self, 
        df: pd.DataFrame, 
        method_name: str,
        global_normalize: bool,
        method_config: Union[Dict[str, Any], BaseModel],
        step_index: int
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

        elif method_name in {'variance_filter', 'correlation_filter'}:
            selected_cols = self.selected_columns_by_step.get(step_index, list(df.columns))
            valid_cols = [col for col in selected_cols if col in df.columns]
            if not valid_cols:
                return df
            return df[valid_cols]
        
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
