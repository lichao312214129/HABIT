"""
Base class for clustering feature extraction
"""

from abc import ABC, abstractmethod
import numpy as np
import importlib
import inspect
import os
import pkgutil
from typing import Dict, List, Any, Type, Optional

# Feature extractor registry
_EXTRACTOR_REGISTRY = {}

def register_feature_extractor(name: str):
    """
    Decorator for registering feature extractor classes
    
    Args:
        name (str): Name of the feature extractor
    
    Returns:
        callable: Decorator function
    """
    def decorator(cls):
        _EXTRACTOR_REGISTRY[name.lower()] = cls
        return cls
    return decorator

def get_feature_extractor(name: str, **kwargs) -> 'BaseClusteringExtractor':
    """
    Get feature extractor class by name
    
    Args:
        name (str): Name of the feature extractor
        **kwargs: Parameters to pass to the feature extractor constructor
    
    Returns:
        BaseClusteringExtractor: Feature extractor instance
    
    Raises:
        ValueError: If the feature extractor is not found
    """
    # Lazy discovery of feature extractors to avoid circular imports
    if not _EXTRACTOR_REGISTRY:
        discover_feature_extractors()
        
    # First try to find registered feature extractors
    if name.lower() in _EXTRACTOR_REGISTRY:
        return _EXTRACTOR_REGISTRY[name.lower()](**kwargs)
    
    # If not found, try dynamic import
    try:
        # Try to import module with specified name using relative import
        module_name = f"{name}_feature_extractor"
        module = importlib.import_module(f".{module_name}", package=__package__)
        
        # Find feature extractor class in the module
        for attr_name, attr_value in inspect.getmembers(module, inspect.isclass):
            if attr_name != 'BaseClusteringExtractor' and 'BaseClusteringExtractor' in [base.__name__ for base in attr_value.__mro__ if base.__name__ != 'object']:
                # Automatically register the found class
                _EXTRACTOR_REGISTRY[name.lower()] = attr_value
                return attr_value(**kwargs)
    except (ImportError, ModuleNotFoundError):
        pass
    
    # If still not found, raise error
    raise ValueError(f"Unknown feature extractor: {name}, available extractors: {list(_EXTRACTOR_REGISTRY.keys())}")

def get_available_feature_extractors() -> List[str]:
    """
    Get all available feature extractor names
    
    Returns:
        List[str]: List of feature extractor names
    """
    # Lazy discovery of feature extractors to avoid circular imports
    if not _EXTRACTOR_REGISTRY:
        discover_feature_extractors()
        
    return list(_EXTRACTOR_REGISTRY.keys())

def discover_feature_extractors() -> None:
    """
    Automatically discover all feature extractors defined in the extractors directory
    """
    # Use the directory of the current file to avoid circular imports
    package_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Iterate through all modules in the package
    for _, module_name, _ in pkgutil.iter_modules([package_dir]):
        # Skip base_extractor and non-extractor files
        if module_name == 'base_extractor' or module_name == 'feature_extractor_factory':
            continue
            
        if module_name.endswith('_feature_extractor') or module_name.endswith('_extractor'):
            try:
                # Dynamically import the module using relative import
                module = importlib.import_module(f".{module_name}", package=__package__)
                
                # Find and register feature extractors defined in the module
                for attr_name, attr_value in inspect.getmembers(module, inspect.isclass):
                    is_subclass = False
                    try:
                        is_subclass = issubclass(attr_value, BaseClusteringExtractor)
                    except TypeError:
                        pass
                        
                    if is_subclass and attr_value is not BaseClusteringExtractor:
                        # Extract feature extractor name from module name
                        if module_name.endswith('_feature_extractor'):
                            extractor_name = module_name.replace('_feature_extractor', '')
                        elif module_name.endswith('_extractor'):
                            extractor_name = module_name.replace('_extractor', '')
                        else:
                            extractor_name = module_name
                            
                        _EXTRACTOR_REGISTRY[extractor_name.lower()] = attr_value
            except ImportError:
                pass


class BaseClusteringExtractor(ABC):
    """
    Base class for feature extraction used in clustering.
    
    Subclasses must implement the following methods:
    - extract_features: Extract features
    - get_feature_names: Get feature names
    """
    
    # Class attribute indicating whether timestamps are required, can be overridden by subclasses
    requires_timestamp = False
    
    def __init__(self, **kwargs):
        """
        Initialize the feature extractor
        
        Args:
            **kwargs: Additional parameters to be handled by subclasses
        """
        # Subclasses should initialize this attribute in their __init__ method
        self.feature_names = None
        
        # Warn if the feature extractor requires timestamps but no timestamp-related parameters are provided
        if self.requires_timestamp and not any(k for k in kwargs if 'time' in k.lower()):
            import warnings
            warnings.warn(f"Feature extractor {self.__class__.__name__} requires timestamps, but no timestamp-related parameters were provided.")
    
    @abstractmethod
    def extract_features(self, **kwargs) -> np.ndarray:
        """
        Extract features from data
        
        Args:
            **kwargs: Parameters needed for feature extraction, such as image_data, timestamps, etc.
            
        Returns:
            np.ndarray: Extracted features
        """
        pass
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names
        
        Returns:
            List[str]: List of feature names
        
        Raises:
            ValueError: If feature names are not set
        """
        if self.feature_names is None:
            raise ValueError("Feature names are not set, please set feature_names in the __init__ method of your feature extractor")
        
        return self.feature_names

# Backward compatibility alias
BaseFeatureExtractor = BaseClusteringExtractor
