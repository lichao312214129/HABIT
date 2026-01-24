"""
Configuration Accessor

Provides unified and type-safe access to nested configuration dictionaries.
Eliminates repetitive config.get() calls and provides dot notation access.
"""

from typing import Dict, Any, Optional, Union


class ConfigAccessor:
    """
    Unified configuration accessor with dot notation support.
    
    Provides type-safe access to nested configuration structures.
    Eliminates repetitive config.get('section', {}).get('field', default) patterns.
    """
    
    def __init__(self, config: Union[Dict[str, Any], Any]):
        """
        Initialize accessor with configuration.
        
        Args:
            config: Configuration dictionary or Pydantic model
        """
        if hasattr(config, 'model_dump'):
            self._config = config.model_dump()
        elif isinstance(config, dict):
            self._config = config
        else:
            raise TypeError(f"Config must be dict or Pydantic model, got {type(config)}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key with dot notation support.
        
        Args:
            key: Configuration key (supports dot notation like 'metrics.basic_metrics.enabled')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Examples:
            >>> accessor = ConfigAccessor({'metrics': {'basic_metrics': {'enabled': True}}})
            >>> accessor.get('metrics.basic_metrics.enabled')
            True
            >>> accessor.get('metrics.basic_metrics.enabled', False)
            True
            >>> accessor.get('nonexistent.key', 'default')
            'default'
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get an entire configuration section.
        
        Args:
            section: Section name
            
        Returns:
            Dictionary containing the section, or empty dict if not found
        """
        section_value = self._config.get(section)
        return section_value if isinstance(section_value, dict) else {}
    
    def has(self, key: str) -> bool:
        """
        Check if a configuration key exists.
        
        Args:
            key: Configuration key (supports dot notation)
            
        Returns:
            True if key exists, False otherwise
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return False
        
        return True
    
    def set(self, key: str, value: Any):
        """
        Set a configuration value by key with dot notation support.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config or not isinstance(config[k], dict):
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get the underlying configuration dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self._config
    
    def __getitem__(self, key: str) -> Any:
        """
        Allow dictionary-style access with square brackets.
        
        Args:
            key: Configuration key
            
        Returns:
            Configuration value or None if not found
        """
        return self.get(key)
    
    def __contains__(self, key: str) -> bool:
        """
        Allow 'in' operator for checking key existence.
        
        Args:
            key: Configuration key
            
        Returns:
            True if key exists, False otherwise
        """
        return self.has(key)
