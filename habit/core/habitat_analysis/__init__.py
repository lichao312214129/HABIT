"""
Habitat Analysis module for HABIT package.

This module provides:
- HabitatAnalysis: Main class for habitat clustering analysis
- Configuration schemas: HabitatAnalysisConfig, ResultColumns
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

# Try to import configuration schemas
try:
    from .config_schemas import HabitatAnalysisConfig, ResultColumns
    _available_classes['HabitatAnalysisConfig'] = HabitatAnalysisConfig
    _available_classes['ResultColumns'] = ResultColumns
except ImportError as e:
    _import_errors['Config'] = str(e)
    HabitatAnalysisConfig = ResultColumns = None

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

# Try to import Pipeline classes
try:
    from .pipelines import (
        BasePipelineStep,
        HabitatPipeline,
        build_habitat_pipeline
    )
    from .pipelines.steps import (
        GroupPreprocessingStep,
        PopulationClusteringStep
    )
    _available_classes['BasePipelineStep'] = BasePipelineStep
    _available_classes['HabitatPipeline'] = HabitatPipeline
    _available_classes['build_habitat_pipeline'] = build_habitat_pipeline
    _available_classes['GroupPreprocessingStep'] = GroupPreprocessingStep
    _available_classes['PopulationClusteringStep'] = PopulationClusteringStep
except ImportError as e:
    _import_errors['Pipeline'] = str(e)
    BasePipelineStep = HabitatPipeline = build_habitat_pipeline = None
    GroupPreprocessingStep = PopulationClusteringStep = None

__all__ = [
    # Main class
    "HabitatAnalysis",
    # Configuration schemas
    "HabitatAnalysisConfig",
    "ResultColumns",
    # Analyzer classes
    "HabitatMapAnalyzer",
    # Legacy Analyzer aliases
    "HabitatFeatureExtractor",
    # Pipeline classes
    "BasePipelineStep",
    "HabitatPipeline",
    "build_habitat_pipeline",
    "GroupPreprocessingStep",
    "PopulationClusteringStep",
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
