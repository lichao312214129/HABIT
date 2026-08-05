# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
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

from habit.registry.base import ClassRegistry


class FeatureExtractorRegistry(ClassRegistry["BaseClusteringExtractor"]):
    """
    Registry for clustering-stage feature extractors.

    Uses the shared :class:`~habit.core.common.registry.ClassRegistry` contract
    (``register`` / ``create`` / ``get`` / ``available`` / ...). Keys are
    case-insensitive and implementations are discovered lazily by importing
    every ``*_extractor`` module in this package.
    """

    kind = "feature extractor"

    @classmethod
    def _normalize(cls, name: str) -> str:
        """Feature extractor names are case-insensitive."""
        return name.lower()

    @classmethod
    def _discover(cls) -> None:
        """Import all ``*_extractor`` modules so decorated extractors register."""
        package_dir = os.path.dirname(os.path.abspath(__file__))
        for _, module_name, _ in pkgutil.iter_modules([package_dir]):
            if module_name in ('base_extractor', 'feature_extractor_factory'):
                continue
            if module_name.endswith('_feature_extractor') or module_name.endswith('_extractor'):
                try:
                    module = importlib.import_module(f".{module_name}", package=__package__)
                    for attr_name, attr_value in inspect.getmembers(module, inspect.isclass):
                        is_subclass = False
                        try:
                            is_subclass = issubclass(attr_value, BaseClusteringExtractor)
                        except TypeError:
                            pass
                        if is_subclass and attr_value is not BaseClusteringExtractor:
                            if module_name.endswith('_feature_extractor'):
                                extractor_name = module_name.replace('_feature_extractor', '')
                            elif module_name.endswith('_extractor'):
                                extractor_name = module_name.replace('_extractor', '')
                            else:
                                extractor_name = module_name
                            cls._registry[extractor_name.lower()] = attr_value
                except ImportError:
                    pass


def get_feature_extractor(name: str, **kwargs) -> 'BaseClusteringExtractor':
    """
    Create a feature extractor instance by name.

    Convenience wrapper around ``FeatureExtractorRegistry.create``.

    Args:
        name (str): Name of the feature extractor.
        **kwargs: Parameters forwarded to the extractor constructor.

    Returns:
        BaseClusteringExtractor: Feature extractor instance.

    Raises:
        ValueError: If the feature extractor is not found.
    """
    return FeatureExtractorRegistry.create(name, **kwargs)


def get_available_feature_extractors() -> List[str]:
    """
    Get all available feature extractor names.

    Convenience wrapper around ``FeatureExtractorRegistry.available``.

    Returns:
        List[str]: List of feature extractor names.
    """
    return FeatureExtractorRegistry.available()


def discover_feature_extractors() -> None:
    """
    Discover all feature extractors defined in this package.

    Convenience wrapper around ``FeatureExtractorRegistry._discover``.
    """
    FeatureExtractorRegistry._discover()


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
