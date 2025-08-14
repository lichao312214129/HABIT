"""
HABIT - Habitat Analysis Tool for Medical Images

A comprehensive tool for analyzing tumor habitats using radiomics features
and machine learning techniques.
"""

__version__ = "0.1.0"

# Import error handling for robust package loading
# This ensures the package can be imported even if some modules fail to load

# Initialize variables to track import status
_import_errors = {}
_available_modules = {}

# Try to import HabitatAnalysis
try:
    from .core.habitat_analysis import HabitatAnalysis
    _available_modules['HabitatAnalysis'] = HabitatAnalysis
except ImportError as e:
    _import_errors['HabitatAnalysis'] = str(e)
    HabitatAnalysis = None

# Try to import HabitatFeatureExtractor
try:
    from .core.habitat_analysis import HabitatFeatureExtractor
    _available_modules['HabitatFeatureExtractor'] = HabitatFeatureExtractor
except ImportError as e:
    _import_errors['HabitatFeatureExtractor'] = str(e)
    HabitatFeatureExtractor = None

# Try to import Modeling
try:
    from .core.machine_learning import Modeling
    _available_modules['Modeling'] = Modeling
except ImportError as e:
    _import_errors['Modeling'] = str(e)
    Modeling = None

# Define available modules for export
__all__ = ["HabitatAnalysis", "HabitatFeatureExtractor", "Modeling"]

# Add utility functions for checking import status
def get_import_errors():
    """
    Get dictionary of import errors that occurred during package loading.
    
    Returns:
        dict: Dictionary mapping module names to error messages
    """
    return _import_errors.copy()

def get_available_modules():
    """
    Get dictionary of successfully imported modules.
    
    Returns:
        dict: Dictionary mapping module names to their classes
    """
    return _available_modules.copy()

def is_module_available(module_name: str) -> bool:
    """
    Check if a specific module is available.
    
    Args:
        module_name (str): Name of the module to check
        
    Returns:
        bool: True if module is available, False otherwise
    """
    return module_name in _available_modules

# Add these utility functions to __all__
__all__.extend(["get_import_errors", "get_available_modules", "is_module_available"])

# Print warning if any imports failed
if _import_errors:
    import warnings
    warning_msg = "Some modules failed to import:\n"
    for module, error in _import_errors.items():
        warning_msg += f"  - {module}: {error}\n"
    warnings.warn(warning_msg, ImportWarning) 