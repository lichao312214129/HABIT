"""
Habitat Analysis module for HABIT package.

This module provides:
- HabitatAnalysis: Main class for habitat clustering analysis
- Configuration classes: HabitatConfig, ClusteringConfig, IOConfig, RuntimeConfig
- Pipeline classes: TrainingPipeline, TestingPipeline
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

# Try to import Pipeline classes
try:
    from .strategies.clustering_pipeline import (
        BasePipeline,
        TrainingPipeline,
        TestingPipeline,
        create_pipeline
    )
    _available_classes['BasePipeline'] = BasePipeline
    _available_classes['TrainingPipeline'] = TrainingPipeline
    _available_classes['TestingPipeline'] = TestingPipeline
except ImportError as e:
    _import_errors['Pipeline'] = str(e)
    BasePipeline = TrainingPipeline = TestingPipeline = create_pipeline = None

# Try to import HabitatFeatureExtractor
try:
    from .habitat_feature_extraction import HabitatFeatureExtractor
    _available_classes['HabitatFeatureExtractor'] = HabitatFeatureExtractor
except ImportError as e:
    _import_errors['HabitatFeatureExtractor'] = str(e)
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
    # Pipeline classes
    "BasePipeline",
    "TrainingPipeline",
    "TestingPipeline",
    "create_pipeline",
    # Feature extraction
    "HabitatFeatureExtractor",
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
