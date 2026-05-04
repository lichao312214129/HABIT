"""
Feature Selector Registry

Provides a centralized registry for feature selection methods with metadata support.
"""

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from habit.utils.log_utils import get_module_logger

logger = get_module_logger('ml.selector_registry')

# Feature selector registry with metadata
# Format: { 'name': { 'func': callable, 'default_before_z_score': bool, 'display_name': str } }
_SELECTOR_REGISTRY: Dict[str, Dict[str, Any]] = {}


@dataclass(frozen=True)
class SelectorContext:
    """
    Explicit runtime context passed into selector functions.

    Standardizing this context makes selector contracts explicit and reduces
    runtime surprises from implicit parameter-name matching.
    """

    X: pd.DataFrame
    y: pd.Series
    selected_features: List[str]
    outdir: Optional[str] = None
    logger: Optional[logging.Logger] = None


@dataclass(frozen=True)
class SelectorResult:
    """Normalized selector output payload."""

    selected_features: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

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
    sig = inspect.signature(selector_func)
    bound_args = {}

    context = SelectorContext(
        X=X[selected_features],
        y=y,
        selected_features=selected_features,
        outdir=kwargs.get("outdir"),
        logger=kwargs.get("logger", logger),
    )

    if "context" in sig.parameters:
        # New explicit contract.
        bound_args["context"] = context
        for param_name, param in sig.parameters.items():
            if param_name == "context":
                continue
            if param_name in kwargs:
                bound_args[param_name] = kwargs[param_name]
            elif param.default is not inspect.Parameter.empty:
                continue
            elif param.kind not in [inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL]:
                logger.warning(
                    "Required parameter '%s' for selector '%s' not found in config kwargs.",
                    param_name,
                    name,
                )
    else:
        # Backward-compatible contract with alias-based injection.
        roles = {
            "X": context.X,
            "y": context.y,
            "selected_features": context.selected_features,
            "outdir": context.outdir,
            "logger": context.logger,
        }
        alias_map = {
            "X": ["X", "x", "data", "features"],
            "y": ["y", "Y", "target", "labels"],
            "selected_features": ["selected_features", "feature_names"],
            "outdir": ["outdir", "output_dir", "save_path"],
            "logger": ["logger", "log"],
        }

        for param_name, param in sig.parameters.items():
            matched_role = None
            for role, aliases in alias_map.items():
                if param_name in aliases:
                    matched_role = role
                    break

            if matched_role:
                bound_args[param_name] = roles[matched_role]
            elif param_name in kwargs:
                bound_args[param_name] = kwargs[param_name]
            elif param.default is not inspect.Parameter.empty:
                continue
            elif param.kind not in [inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL]:
                logger.warning(
                    "Required parameter '%s' for selector '%s' not found in inputs or config.",
                    param_name,
                    name,
                )

    # Execute and handle return value
    try:
        logger.info(f"Running feature selector: {info['display_name']} on {len(selected_features)} features...")
        result = selector_func(**bound_args)
        selected_list = _normalize_selected_features(result=result, selector_name=name)
        logger.info(f"Selector {name} completed: {len(selected_list)} features retained.")
        return selected_list
    except Exception as e:
        logger.error(f"Error executing selector '{name}': {e}")
        raise


def _normalize_selected_features(result: Any, selector_name: str) -> List[str]:
    """
    Normalize diverse selector outputs into a list of feature names.

    Supported return formats:
    - List[str]
    - Tuple[List[str], ...]
    - Dict with key 'selected_features'
    - SelectorResult
    """
    if isinstance(result, SelectorResult):
        return list(result.selected_features)

    if isinstance(result, dict):
        if "selected_features" not in result:
            raise ValueError(
                f"Selector '{selector_name}' returned dict without 'selected_features' key."
            )
        return list(result["selected_features"])

    if isinstance(result, tuple):
        if not result:
            return []
        return list(result[0])

    return list(result)