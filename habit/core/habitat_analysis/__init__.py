"""
Habitat Analysis module for HABIT package.
"""

# Import error handling for robust module loading
_import_errors = {}
_available_classes = {}

# Try to import HabitatAnalysis
try:
    from .habitat_analysis import HabitatAnalysis
    _available_classes['HabitatAnalysis'] = HabitatAnalysis
except ImportError as e:
    _import_errors['HabitatAnalysis'] = str(e)
    HabitatAnalysis = None

# Try to import HabitatFeatureExtractor
try:
    from .feature_extraction import HabitatFeatureExtractor
    _available_classes['HabitatFeatureExtractor'] = HabitatFeatureExtractor
except ImportError as e:
    _import_errors['HabitatFeatureExtractor'] = str(e)
    HabitatFeatureExtractor = None

__all__ = ["HabitatAnalysis", "HabitatFeatureExtractor"]

# Add utility functions for checking import status
def get_import_errors():
    """
    Get dictionary of import errors that occurred during module loading.
    
    Returns:
        dict: Dictionary mapping class names to error messages
    """
    return _import_errors.copy()

def get_available_classes():
    """
    Get dictionary of successfully imported classes.
    
    Returns:
        dict: Dictionary mapping class names to their classes
    """
    return _available_classes.copy()

def is_class_available(class_name: str) -> bool:
    """
    Check if a specific class is available.
    
    Args:
        class_name (str): Name of the class to check
        
    Returns:
        bool: True if class is available, False otherwise
    """
    return class_name in _available_classes

# Add these utility functions to __all__
__all__.extend(["get_import_errors", "get_available_classes", "is_class_available"])

# Print warning if any imports failed
if _import_errors:
    import warnings
    warning_msg = "Some classes failed to import in habitat_analysis module:\n"
    for class_name, error in _import_errors.items():
        warning_msg += f"  - {class_name}: {error}\n"
    warnings.warn(warning_msg, ImportWarning)
