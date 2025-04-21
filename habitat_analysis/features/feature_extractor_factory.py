"""
Feature extractor factory for creating and configuring feature extractors
"""

import importlib
from typing import Dict, Any, List, Type
from habitat_clustering.features.base_feature_extractor import (
    BaseFeatureExtractor, 
    get_feature_extractor,
    get_available_feature_extractors
)

class FeatureExtractorConfig:
    """Configuration class for feature extractors"""
    
    def __init__(self, name: str, params: Dict[str, Any] = None):
        """
        Initialize feature extractor configuration
        
        Args:
            name: Name of the feature extractor
            params: Parameters for the feature extractor
        """
        self.name = name
        self.params = params or {}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FeatureExtractorConfig':
        """
        Create feature extractor configuration from dictionary
        
        Args:
            config_dict: Configuration dictionary containing name and params fields
        
        Returns:
            FeatureExtractorConfig: Feature extractor configuration instance
        """
        if not isinstance(config_dict, dict):
            raise ValueError(f"Configuration must be a dictionary, got: {type(config_dict)}")
        
        name = config_dict.get('method')  # Use 'method' instead of 'name' to match YAML config
        if not name:
            raise ValueError("Configuration must contain 'method' field")
        
        # Copy all parameters except 'method'
        params = {k: v for k, v in config_dict.items() if k != 'method'}
        
        return cls(name=name, params=params)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        return {
            'method': self.name,  # Use 'method' as the key name
            **self.params  # Expand all parameters to top level
        }
    
    def create_extractor(self) -> BaseFeatureExtractor:
        """
        Create feature extractor instance
        
        Returns:
            BaseFeatureExtractor: Feature extractor instance
        """
        return get_feature_extractor(self.name, **self.params)


class FeatureExtractorFactory:
    """Feature extractor factory class"""
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> BaseFeatureExtractor:
        """
        Create feature extractor from configuration
        
        Args:
            config: Configuration dictionary in format:
                {
                    "method": "feature extractor name",
                    "timestamps": "timestamp file path",
                    "other_params": value,
                    ...
                }
                
        Returns:
            BaseFeatureExtractor: Feature extractor instance
        """
        feature_config = FeatureExtractorConfig.from_dict(config)
        return feature_config.create_extractor()
    
    @staticmethod
    def create_from_name(name: str, **kwargs) -> BaseFeatureExtractor:
        """
        Create feature extractor from name
        
        Args:
            name: Feature extractor name
            **kwargs: Feature extractor parameters
            
        Returns:
            BaseFeatureExtractor: Feature extractor instance
        """
        return get_feature_extractor(name, **kwargs)
    
    @staticmethod
    def get_available_extractors() -> List[str]:
        """
        Get all available feature extractor names
        
        Returns:
            List[str]: List of feature extractor names
        """
        return get_available_feature_extractors()


def create_feature_extractor(config_or_name, **kwargs) -> BaseFeatureExtractor:
    """
    Convenience function for creating feature extractors
    
    Args:
        config_or_name: Feature extractor configuration or name
            - If string, treated as feature extractor name
            - If dictionary, treated as feature extractor configuration
        **kwargs: Parameters passed to feature extractor when config_or_name is a string
        
    Returns:
        BaseFeatureExtractor: Feature extractor instance
    """
    if isinstance(config_or_name, dict):
        # If dictionary, extract name and parameters
        config_dict = config_or_name.copy()
        # Use 'method' field first (compatible with YAML config), then 'name' field
        name = config_dict.pop('method', None) or config_dict.pop('name', None)
        if not name:
            raise ValueError(f"Feature extractor configuration must contain 'method' or 'name' field")
        
        # Merge remaining parameters
        params = {}
        # If params field exists, merge it
        if 'params' in config_dict:
            params.update(config_dict.pop('params'))
        # Add remaining top-level parameters to params
        params.update(config_dict)
        # Add parameters from kwargs to params
        params.update(kwargs)
        
        return FeatureExtractorFactory.create_from_name(name, **params)
    elif isinstance(config_or_name, str):
        return FeatureExtractorFactory.create_from_name(config_or_name, **kwargs)
    else:
        raise ValueError(f"Invalid feature extractor configuration or name: {config_or_name}") 