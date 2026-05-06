"""
DataFrame Utilities

Common utility functions for DataFrame operations.
Eliminates code duplication across the codebase.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Union


def remove_nan_arrays(*arrays: np.ndarray) -> List[np.ndarray]:
    """
    Remove NaN values from multiple arrays simultaneously.
    
    Args:
        *arrays: Variable number of numpy arrays
        
    Returns:
        List of arrays with NaN rows removed
        
    Example:
        >>> y_true = np.array([0, 1, np.nan, 1])
        >>> y_pred = np.array([0.2, 0.8, 0.5, np.nan])
        >>> clean_true, clean_pred = remove_nan_arrays(y_true, y_pred)
        >>> len(clean_true)
        2
    """
    if not arrays:
        return []
    
    mask = ~np.isnan(arrays[0])
    for arr in arrays[1:]:
        if arr.ndim > 1:
            mask &= ~np.any(np.isnan(arr), axis=1)
        else:
            mask &= ~np.isnan(arr)
    
    return [arr[mask] for arr in arrays]


def create_prediction_dataframe(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    y_pred: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Create a DataFrame for prediction data.
    
    Args:
        y_true: True labels array
        y_pred_proba: Predicted probabilities array
        y_pred: Optional predicted labels array
        
    Returns:
        DataFrame with columns: y_true, y_pred_proba, [y_pred]
        
    Example:
        >>> df = create_prediction_dataframe(
        ...     y_true=np.array([0, 1, 0]),
        ...     y_pred_proba=np.array([0.2, 0.8, 0.3]),
        ...     y_pred=np.array([0, 1, 0])
        ... )
        >>> df.columns.tolist()
        ['y_true', 'y_pred_proba', 'y_pred']
    """
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred_proba': y_pred_proba
    })
    
    if y_pred is not None:
        df['y_pred'] = y_pred
    
    return df


def clean_prediction_data(
    y_true: Union[np.ndarray, List],
    y_pred_proba: Union[np.ndarray, List],
    y_pred: Optional[Union[np.ndarray, List]] = None
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Clean prediction data by removing NaN values.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        y_pred: Optional predicted labels
        
    Returns:
        Tuple of (y_true_clean, y_pred_proba_clean, y_pred_clean)
        
    Example:
        >>> y_true = np.array([0, 1, np.nan, 1])
        >>> y_pred_proba = np.array([0.2, 0.8, 0.5, np.nan])
        >>> clean_true, clean_prob, clean_pred = clean_prediction_data(y_true, y_pred_proba)
        >>> len(clean_true)
        2
    """
    y_true_arr = np.array(y_true)
    y_pred_proba_arr = np.array(y_pred_proba)
    
    if y_pred is not None:
        y_pred_arr = np.array(y_pred)
        clean_arrays = remove_nan_arrays(y_true_arr, y_pred_proba_arr, y_pred_arr)
        return clean_arrays[0], clean_arrays[1], clean_arrays[2]
    else:
        clean_arrays = remove_nan_arrays(y_true_arr, y_pred_proba_arr)
        return clean_arrays[0], clean_arrays[1], None


def ensure_dataframe(
    data: Union[pd.DataFrame, np.ndarray],
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Ensure input is a DataFrame, converting from numpy if necessary.
    
    Args:
        data: Input data (DataFrame or numpy array)
        columns: Optional column names for numpy arrays
        
    Returns:
        DataFrame representation of the data
        
    Example:
        >>> arr = np.array([[1, 2], [3, 4]])
        >>> df = ensure_dataframe(arr, columns=['a', 'b'])
        >>> isinstance(df, pd.DataFrame)
        True
    """
    if isinstance(data, pd.DataFrame):
        return data
    
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            return pd.DataFrame(data.reshape(-1, 1), columns=columns or ['value'])
        else:
            return pd.DataFrame(data, columns=columns)
    
    raise TypeError(f"Cannot convert {type(data)} to DataFrame")


def validate_binary_labels(y: np.ndarray) -> None:
    """
    Validate that labels are binary (0 or 1).
    
    Args:
        y: Label array to validate
        
    Raises:
        ValueError: If labels are not binary
        
    Example:
        >>> validate_binary_labels(np.array([0, 1, 0, 1]))
        >>> validate_binary_labels(np.array([0, 1, 2]))
        ValueError: Labels must be binary (0 or 1)
    """
    unique_labels = np.unique(y[~np.isnan(y)])
    if not np.array_equal(unique_labels, [0, 1]):
        raise ValueError(f"Labels must be binary (0 or 1), got {unique_labels}")


def validate_probabilities(y_pred_proba: np.ndarray) -> None:
    """
    Validate that predicted probabilities are in valid range [0, 1].
    
    Args:
        y_pred_proba: Predicted probabilities array
        
    Raises:
        ValueError: If probabilities are outside [0, 1] range
        
    Example:
        >>> validate_probabilities(np.array([0.2, 0.8, 0.5]))
        >>> validate_probabilities(np.array([0.2, 1.5, 0.5]))
        ValueError: Probabilities must be in range [0, 1]
    """
    if np.any(y_pred_proba < 0) or np.any(y_pred_proba > 1):
        raise ValueError(f"Probabilities must be in range [0, 1], got min={np.min(y_pred_proba):.4f}, max={np.max(y_pred_proba):.4f}")


def normalize_probabilities(y_pred_proba: np.ndarray) -> np.ndarray:
    """
    Normalize probabilities to [0, 1] range using min-max scaling.
    
    Args:
        y_pred_proba: Predicted probabilities array
        
    Returns:
        Normalized probabilities in [0, 1] range
        
    Example:
        >>> probs = np.array([0.1, 0.2, 0.3])
        >>> norm_probs = normalize_probabilities(probs)
        >>> np.all((norm_probs >= 0) & (norm_probs <= 1))
        True
    """
    min_val = np.min(y_pred_proba)
    max_val = np.max(y_pred_proba)
    
    if max_val == min_val:
        return np.full_like(y_pred_proba, 0.5)
    
    return (y_pred_proba - min_val) / (max_val - min_val)
