"""
Base configuration classes for unified configuration management.

This module provides:
1. BaseConfig: Abstract base class for all configuration schemas
2. ConfigValidationError: Custom exception for configuration validation errors
3. ConfigAccessor: Unified interface for accessing configuration values
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type, TypeVar, Union
from pathlib import Path
from pydantic import BaseModel, ValidationError
import logging

# Type variable for configuration classes
ConfigType = TypeVar('ConfigType', bound='BaseConfig')


class ConfigValidationError(Exception):
    """
    Custom exception for configuration validation errors.
    
    Provides detailed information about validation failures.
    """
    
    def __init__(self, message: str, errors: Optional[Dict[str, Any]] = None, config_path: Optional[str] = None):
        """
        Initialize configuration validation error.
        
        Args:
            message: Error message
            errors: Detailed validation errors from Pydantic
            config_path: Path to the configuration file that failed validation
        """
        super().__init__(message)
        self.message = message
        self.errors = errors or {}
        self.config_path = config_path
    
    def __str__(self) -> str:
        """Format error message with details."""
        msg = self.message
        if self.config_path:
            msg += f" (config file: {self.config_path})"
        if self.errors:
            msg += f"\nValidation errors: {self.errors}"
        return msg


class BaseConfig(BaseModel, ABC):
    """
    Abstract base class for all configuration schemas in HABIT.
    
    Provides common functionality:
    - Version tracking
    - Configuration file path tracking
    - Validation hooks
    - Accessor methods
    
    All configuration classes should inherit from this base class.
    """
    
    # Configuration metadata
    config_file: Optional[str] = None
    config_version: Optional[str] = None
    
    class Config:
        """Pydantic configuration."""
        extra = 'forbid'  # Forbid extra fields by default
        validate_assignment = True  # Validate on assignment
        use_enum_values = True  # Use enum values
    
    def __init__(self, **data: Any):
        """
        Initialize configuration with validation.
        
        Args:
            **data: Configuration data
            
        Raises:
            ConfigValidationError: If validation fails
        """
        try:
            super().__init__(**data)
        except ValidationError as e:
            raise ConfigValidationError(
                message=f"Configuration validation failed for {self.__class__.__name__}",
                errors=e.errors(),
                config_path=data.get('config_file')
            ) from e
    
    @classmethod
    def from_dict(cls: Type[ConfigType], config_dict: Dict[str, Any], config_path: Optional[str] = None) -> ConfigType:
        """
        Create configuration instance from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            config_path: Optional path to configuration file (for error reporting)
            
        Returns:
            Configuration instance
            
        Raises:
            ConfigValidationError: If validation fails
        """
        if config_path:
            config_dict['config_file'] = config_path
        
        try:
            return cls(**config_dict)
        except ValidationError as e:
            raise ConfigValidationError(
                message=f"Failed to create {cls.__name__} from dictionary",
                errors=e.errors(),
                config_path=config_path
            ) from e
    
    @classmethod
    def from_file(cls: Type[ConfigType], config_path: Union[str, Path]) -> ConfigType:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
            
        Returns:
            Configuration instance
            
        Raises:
            FileNotFoundError: If configuration file not found
            ConfigValidationError: If validation fails
        """
        from habit.utils.config_utils import load_config
        
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        config_dict = load_config(str(config_path), resolve_paths=True)
        return cls.from_dict(config_dict, config_path=str(config_path))
    
    def to_dict(self, exclude_none: bool = False, exclude_unset: bool = False) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Args:
            exclude_none: Whether to exclude None values
            exclude_unset: Whether to exclude unset values
            
        Returns:
            Configuration dictionary
        """
        if hasattr(self, 'model_dump'):
            # Pydantic v2
            return self.model_dump(exclude_none=exclude_none, exclude_unset=exclude_unset)
        else:
            # Pydantic v1
            return self.dict(exclude_none=exclude_none, exclude_unset=exclude_unset)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (dictionary-like access).
        
        This method provides backward compatibility with dictionary access patterns.
        However, direct attribute access (config.field_name) is preferred.
        
        Args:
            key: Configuration key (supports dot notation for nested keys)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        try:
            # Support dot notation for nested access
            if '.' in key:
                parts = key.split('.')
                value = self
                for part in parts:
                    if hasattr(value, part):
                        value = getattr(value, part)
                    elif isinstance(value, dict):
                        value = value.get(part, default)
                    else:
                        return default
                return value
            else:
                # Direct attribute access
                if hasattr(self, key):
                    return getattr(self, key)
                elif hasattr(self, 'model_dump'):
                    # Pydantic v2
                    return self.model_dump().get(key, default)
                else:
                    # Pydantic v1
                    return self.dict().get(key, default)
        except (AttributeError, KeyError, TypeError):
            return default
    
    def validate(self) -> bool:
        """
        Validate configuration (re-validate after modifications).
        
        Returns:
            True if valid
            
        Raises:
            ConfigValidationError: If validation fails
        """
        try:
            # Trigger validation by creating a new instance
            self.__class__(**self.to_dict())
            return True
        except ValidationError as e:
            raise ConfigValidationError(
                message=f"Configuration validation failed for {self.__class__.__name__}",
                errors=e.errors(),
                config_path=self.config_file
            ) from e
    
    def __getitem__(self, key: str) -> Any:
        """
        Dictionary-like access for backward compatibility.
        
        Prefer direct attribute access: config.field_name
        """
        return self.get(key)
    
    def __contains__(self, key: str) -> bool:
        """Check if configuration contains a key."""
        return hasattr(self, key) or key in self.to_dict()


class ConfigAccessor:
    """
    Unified interface for accessing configuration values.
    
    Provides a consistent API for accessing configuration regardless of
    whether it's a Pydantic model or a dictionary.
    
    This class helps transition from dictionary-based config access
    to strongly-typed Pydantic model access.
    """
    
    def __init__(self, config: Union[BaseConfig, Dict[str, Any]]):
        """
        Initialize config accessor.
        
        Args:
            config: Configuration object (BaseConfig instance or dict)
        """
        self._config = config
        self._is_pydantic = isinstance(config, BaseConfig)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Supports:
        - Direct attribute access for Pydantic models: config.field_name
        - Dot notation for nested access: config.section.subsection.field
        - Dictionary access for backward compatibility
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        if self._is_pydantic:
            return self._config.get(key, default)
        else:
            # Dictionary access with dot notation support
            if '.' in key:
                parts = key.split('.')
                value = self._config
                for part in parts:
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        return default
                return value
            else:
                return self._config.get(key, default)
    
    def has(self, key: str) -> bool:
        """
        Check if configuration contains a key.
        
        Args:
            key: Configuration key (supports dot notation)
            
        Returns:
            True if key exists
        """
        if self._is_pydantic:
            return key in self._config
        else:
            if '.' in key:
                parts = key.split('.')
                value = self._config
                for part in parts:
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        return False
                return True
            else:
                return key in self._config
    
    def get_section(self, section_name: str) -> Optional[Union[BaseConfig, Dict[str, Any]]]:
        """
        Get a configuration section.
        
        Args:
            section_name: Section name (supports dot notation)
            
        Returns:
            Configuration section or None
        """
        value = self.get(section_name)
        if value is None:
            return None
        
        # If it's a Pydantic model or dict, return as ConfigAccessor
        if isinstance(value, (BaseConfig, dict)):
            return ConfigAccessor(value)
        
        return value
    
    @property
    def raw_config(self) -> Union[BaseConfig, Dict[str, Any]]:
        """Get raw configuration object."""
        return self._config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        if self._is_pydantic:
            return self._config.to_dict()
        else:
            return self._config.copy() if isinstance(self._config, dict) else dict(self._config)
