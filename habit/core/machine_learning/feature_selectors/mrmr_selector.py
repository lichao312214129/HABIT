"""
MRMR (Minimum Redundancy Maximum Relevance) Feature Selector

Uses the mrmr package to select features based on mutual information criteria.
This implementation is optimized for both classification and regression tasks.
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Union
from mrmr import mrmr_classif, mrmr_regression
from .selector_registry import register_selector

@register_selector('mrmr')
def mrmr_selector(data: pd.DataFrame,
                 target: Union[str, pd.Series],
                 n_features: int = 10,
                 task_type: str = 'classification',
                 selected_features: Optional[List[str]] = None) -> List[str]:
    """
    Select features using MRMR (Minimum Redundancy Maximum Relevance)
    
    Args:
        data: Feature data as pandas DataFrame
        target: Target variable, can be either a column name (str) or a pandas Series
        n_features: Number of features to select
        task_type: Type of task, either 'classification' or 'regression'
        selected_features: List of already selected features, if None use all columns of data
        
    Returns:
        List[str]: List of selected features
        
    Raises:
        ValueError: If task_type is not 'classification' or 'regression'
        TypeError: If input data is not a pandas DataFrame
        ValueError: If target column is not found in data (when target is str)
        TypeError: If target is not a string or pandas Series
    """
    # Input validation
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame")
    
    if not isinstance(target, (str, pd.Series)):
        raise TypeError("target must be either a string (column name) or a pandas Series")
    
    if isinstance(target, str):
        if target not in data.columns:
            raise ValueError(f"Target column '{target}' not found in data")
        y = data[target]
    else:
        y = target
        if len(y) != len(data):
            raise ValueError("Length of target Series must match the number of rows in data")
    
    if task_type not in ['classification', 'regression']:
        raise ValueError("task_type must be either 'classification' or 'regression'")
    
    if n_features <= 0:
        raise ValueError("n_features must be positive")
    
    # Prepare data
    if selected_features is None:
        if isinstance(target, str):
            selected_features = [col for col in data.columns if col != target]
        else:
            selected_features = data.columns.tolist()
    
    X = data[selected_features]
    
    try:
        # Select features based on task type
        if task_type == 'classification':
            selected = mrmr_classif(X=X, y=y, K=n_features)
        else:
            selected = mrmr_regression(X=X, y=y, K=n_features)
            
        print(f"MRMR selection: Selected {len(selected)} features from {len(selected_features)} features")
        return selected
        
    except Exception as e:
        print(f"Error in MRMR selection: {e}")
        return []