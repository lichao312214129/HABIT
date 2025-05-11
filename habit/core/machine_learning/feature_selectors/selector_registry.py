"""
Feature Selector Registry

Provides mechanisms for registering and retrieving feature selection methods.
"""

import inspect
import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Any, Optional, Union, Tuple

# Feature selector registry
_FEATURE_SELECTORS: Dict[str, Callable] = {}

def register_selector(name: str) -> Callable:
    """
    Decorator for registering feature selectors
    
    Args:
        name: Selector name
        
    Returns:
        Callable: Decorator function
    """
    def decorator(func: Callable) -> Callable:
        _FEATURE_SELECTORS[name] = func
        return func
    return decorator

def get_selector(name: str) -> Callable:
    """
    Get feature selector by name
    
    Args:
        name: Selector name
        
    Returns:
        Callable: Feature selection function
        
    Raises:
        ValueError: If selector not found
    """
    if name not in _FEATURE_SELECTORS:
        raise ValueError(f"Feature selector {name} not found")
    return _FEATURE_SELECTORS[name]

def get_available_selectors() -> List[str]:
    """
    Get list of all available feature selectors
    
    Returns:
        List[str]: List of selector names
    """
    return list(_FEATURE_SELECTORS.keys())

def run_selector(name: str, 
                X: pd.DataFrame, 
                y: pd.Series, 
                selected_features: Optional[List[str]] = None,
                **kwargs) -> List[str]:
    """
    Run a feature selector
    
    Args:
        name: Selector name
        X: Feature data
        y: Target variable
        selected_features: List of already selected features (if None, use all columns of X)
        **kwargs: Additional arguments passed to the selector
        
    Returns:
        List[str]: List of selected features
    """
    if selected_features is None:
        selected_features = X.columns.tolist()
    
    selector = get_selector(name)
    
    # Check function signature
    sig = inspect.signature(selector)
    
    # Prepare parameters
    params = {}
    param_mappings = {
        'X': ['X', 'x', 'data', 'features'],
        'y': ['y', 'Y', 'target', 'labels'],
        'selected_features': ['selected_features', 'feature_names']
    }
    
    for param_name in sig.parameters:
        matched = False
        for key, aliases in param_mappings.items():
            if param_name in aliases:
                if key == 'X':
                    params[param_name] = X[selected_features]
                elif key == 'y':
                    params[param_name] = y
                elif key == 'selected_features':
                    params[param_name] = selected_features
                matched = True
                break
                
        if not matched and param_name in kwargs:
            params[param_name] = kwargs[param_name]
    
    # Call selector
    result = selector(**params)
    
    # Process return value
    if isinstance(result, tuple):
        # If tuple returned, first element is selected features
        return result[0]
    else:
        # Otherwise entire result is selected features
        return result 