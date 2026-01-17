"""
Habitat Analysis module for HABIT package.

This module provides:
- HabitatAnalysis: Main class for habitat clustering analysis
- Configuration classes: HabitatConfig, ClusteringConfig, IOConfig, RuntimeConfig
- Mode classes: TrainingMode, TestingMode (formerly Pipeline)
- Analyzer classes: HabitatMapAnalyzer (formerly HabitatFeatureExtractor)
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

# Try to import Configuration classes
try:
    from .config import (
        HabitatConfig,
        ClusteringConfig,
        IOConfig,
        RuntimeConfig,
        OneStepConfig,
        ResultColumns
    )
    _available_classes['HabitatConfig'] = HabitatConfig
    _available_classes['ClusteringConfig'] = ClusteringConfig
    _available_classes['IOConfig'] = IOConfig
    _available_classes['RuntimeConfig'] = RuntimeConfig
    _available_classes['OneStepConfig'] = OneStepConfig
    _available_classes['ResultColumns'] = ResultColumns
except ImportError as e:
    _import_errors['Config'] = str(e)
    HabitatConfig = ClusteringConfig = IOConfig = RuntimeConfig = OneStepConfig = ResultColumns = None

# Try to import Mode classes
try:
    from .modes import (
        BaseMode,
        TrainingMode,
        TestingMode,
        create_mode
    )
    _available_classes['BaseMode'] = BaseMode
    _available_classes['TrainingMode'] = TrainingMode
    _available_classes['TestingMode'] = TestingMode
    
    # Aliases for backward compatibility
    BasePipeline = BaseMode
    TrainingPipeline = TrainingMode
    TestingPipeline = TestingMode
    create_pipeline = create_mode
    
except ImportError as e:
    _import_errors['Mode'] = str(e)
    BaseMode = TrainingMode = TestingMode = create_mode = None
    BasePipeline = TrainingPipeline = TestingPipeline = create_pipeline = None

# Try to import HabitatMapAnalyzer
try:
    from .analyzers.habitat_analyzer import HabitatMapAnalyzer
    _available_classes['HabitatMapAnalyzer'] = HabitatMapAnalyzer
    
    # Alias for backward compatibility
    HabitatFeatureExtractor = HabitatMapAnalyzer
except ImportError as e:
    _import_errors['HabitatMapAnalyzer'] = str(e)
    HabitatMapAnalyzer = None
    HabitatFeatureExtractor = None

__all__ = [
    # Main class
    "HabitatAnalysis",
    # Configuration classes
    "HabitatConfig",
    "ClusteringConfig",
    "IOConfig",
    "RuntimeConfig",
    "OneStepConfig",
    "ResultColumns",
    # Mode classes
    "BaseMode",
    "TrainingMode",
    "TestingMode",
    "create_mode",
    # Legacy Pipeline aliases
    "BasePipeline",
    "TrainingPipeline",
    "TestingPipeline",
    "create_pipeline",
    # Analyzer classes
    "HabitatMapAnalyzer",
    # Legacy Analyzer aliases
    "HabitatFeatureExtractor"
]

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
