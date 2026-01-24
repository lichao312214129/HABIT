"""
Feature Selector Registry

Provides a centralized registry for feature selection methods with metadata support.
"""

import inspect
import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Any, Optional, Union, Tuple
from habit.utils.log_utils import get_module_logger

logger = get_module_logger('ml.selector_registry')

# Feature selector registry with metadata
# Format: { 'name': { 'func': callable, 'default_before_z_score': bool, 'display_name': str } }
_SELECTOR_REGISTRY: Dict[str, Dict[str, Any]] = {}

def register_selector(name: str, display_name: str = None, default_before_z_score: bool = False):
    """
    Decorator for registering feature selectors with metadata.
    
    Args:
        name: Internal unique identifier for the selector.
        display_name: Pretty name for logging and reports.
        default_before_z_score: If True, this selector is recommended to run before scaling (e.g., Variance).
    """
    def decorator(func: Callable) -> Callable:
        _SELECTOR_REGISTRY[name] = {
            'func': func,
            'display_name': display_name or name.replace('_', ' ').title(),
            'default_before_z_score': default_before_z_score
        }
        return func
    return decorator

def get_selector_info(name: str) -> Dict[str, Any]:
    """Get metadata for a specific selector."""
    if name not in _SELECTOR_REGISTRY:
        raise ValueError(f"Feature selector '{name}' not found in registry.")
    return _SELECTOR_REGISTRY[name]

def get_selector(name: str) -> Callable:
    """Backward compatible function to get the selector callable."""
    return get_selector_info(name)['func']

def get_available_selectors() -> List[str]:
    """Get list of all registered selector names."""
    return list(_SELECTOR_REGISTRY.keys())

def run_selector(name: str, 
                X: pd.DataFrame, 
                y: pd.Series, 
                selected_features: Optional[List[str]] = None,
                **kwargs) -> List[str]:
    """
    Orchestrate the execution of a feature selector with smart parameter injection.
    """
    if selected_features is None:
        selected_features = X.columns.tolist()
    
    info = get_selector_info(name)
    selector_func = info['func']
    
    # Introspect function signature to inject only required parameters
    sig = inspect.signature(selector_func)
    bound_args = {}
    
    # Mapping of semantic roles to actual data with alias support
    # This allows different selector functions to use different names for the same core inputs
    roles = {
        'X': X[selected_features],
        'y': y,
        'selected_features': selected_features,
        'outdir': kwargs.get('outdir')
    }
    
    # Aliases for mapping role data to function parameter names
    alias_map = {
        'X': ['X', 'x', 'data', 'features'],
        'y': ['y', 'Y', 'target', 'labels'],
        'selected_features': ['selected_features', 'feature_names'],
        'outdir': ['outdir', 'output_dir', 'save_path']
    }

    for param_name, param in sig.parameters.items():
        # 1. Try to find a role for this parameter name via aliases
        matched_role = None
        for role, aliases in alias_map.items():
            if param_name in aliases:
                matched_role = role
                break
        
        if matched_role:
            bound_args[param_name] = roles[matched_role]
        # 2. Try to inject from kwargs (user config)
        elif param_name in kwargs:
            bound_args[param_name] = kwargs[param_name]
        # 3. Use default if available
        elif param.default is not inspect.Parameter.empty:
            continue 
        # 4. Warn if a required parameter is missing
        elif param.kind not in [inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL]:
            logger.warning(f"Required parameter '{param_name}' for selector '{name}' not found in inputs or config.")

    # Execute and handle return value
    try:
        logger.info(f"Running feature selector: {info['display_name']} on {len(selected_features)} features...")
        result = selector_func(**bound_args)
        
        # Ensure we return a list of feature names
        if isinstance(result, tuple):
            selected_list = list(result[0])
        else:
            selected_list = list(result)
            
        logger.info(f"Selector {name} completed: {len(selected_list)} features retained.")
        return selected_list
    except Exception as e:
        logger.error(f"Error executing selector '{name}': {e}")
        raise