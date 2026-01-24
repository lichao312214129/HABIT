"""
Core modules for HABIT package.
"""

from typing import Dict, Any

_import_errors: Dict[str, str] = {}
_available_classes: Dict[str, Any] = {}

# Import Habitat Analysis API with safe fallback if dependencies are missing.
try:
    from .habitat_analysis import HabitatAnalysis, HabitatFeatureExtractor
    _available_classes["HabitatAnalysis"] = HabitatAnalysis
    _available_classes["HabitatFeatureExtractor"] = HabitatFeatureExtractor
except ImportError as e:
    _import_errors["habitat_analysis"] = str(e)
    HabitatAnalysis = None
    HabitatFeatureExtractor = None

# Import standard ML workflow entry point with safe fallback.
try:
    from .machine_learning.workflows.holdout_workflow import MachineLearningWorkflow as Modeling
    _available_classes["Modeling"] = Modeling
except ImportError as e:
    _import_errors["machine_learning"] = str(e)
    Modeling = None

__all__ = ["HabitatAnalysis", "HabitatFeatureExtractor", "Modeling"]

def get_import_errors() -> Dict[str, str]:
    """
    Return import errors collected during module initialization.

    Returns:
        Dict[str, str]: Mapping from module key to error message.
    """
    return _import_errors.copy()

def get_available_classes() -> Dict[str, Any]:
    """
    Return classes that were successfully imported.

    Returns:
        Dict[str, Any]: Mapping from public name to class object.
    """
    return _available_classes.copy()

def is_class_available(class_name: str) -> bool:
    """
    Check if a public class is available in this module.

    Args:
        class_name (str): Public class name to check.

    Returns:
        bool: True if the class was imported successfully.
    """
    return class_name in _available_classes

__all__.extend(["get_import_errors", "get_available_classes", "is_class_available"])

if _import_errors:
    import warnings
    warning_msg = "Some core modules failed to import:\n"
    for module_name, error in _import_errors.items():
        warning_msg += f"  - {module_name}: {error}\n"
    warnings.warn(warning_msg, ImportWarning)