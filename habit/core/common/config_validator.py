"""
Configuration validation middleware and utilities.

Provides unified configuration validation and loading across all HABIT modules.
"""

from typing import Dict, Any, Optional, Type, TypeVar, Union
from pathlib import Path
import logging
from pydantic import ValidationError

from .config_base import BaseConfig, ConfigValidationError, ConfigAccessor
from habit.utils.config_utils import load_config, resolve_config_paths

logger = logging.getLogger(__name__)

ConfigType = TypeVar('ConfigType', bound=BaseConfig)


class ConfigValidator:
    """
    Unified configuration validator and loader.
    
    Provides a single entry point for loading and validating configurations
    across all HABIT modules.
    """
    
    @staticmethod
    def validate_and_load(
        config_path: Union[str, Path],
        config_class: Type[ConfigType],
        resolve_paths: bool = True,
        strict: bool = True
    ) -> ConfigType:
        """
        Load and validate configuration from file.
        
        This is the recommended way to load configurations in HABIT.
        It provides:
        - Automatic path resolution
        - Unified error handling
        - Type-safe configuration objects
        
        Args:
            config_path: Path to configuration file
            config_class: Configuration class (must inherit from BaseConfig)
            resolve_paths: Whether to resolve relative paths (default: True)
            strict: Whether to raise exceptions on validation errors (default: True)
            
        Returns:
            Validated configuration instance
            
        Raises:
            FileNotFoundError: If configuration file not found
            ConfigValidationError: If validation fails and strict=True
            
        Example:
            >>> from habit.core.habitat_analysis.config_schemas import HabitatAnalysisConfig
            >>> config = ConfigValidator.validate_and_load(
            ...     'config.yaml',
            ...     HabitatAnalysisConfig
            ... )
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            error_msg = f"Configuration file not found: {config_path}"
            if strict:
                raise FileNotFoundError(error_msg)
            else:
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
        
        try:
            # Load raw configuration
            config_dict = load_config(str(config_path), resolve_paths=resolve_paths)
            
            # Resolve paths if requested
            if resolve_paths:
                config_dict = resolve_config_paths(config_dict, config_path)
            
            # Create and validate configuration instance
            config = config_class.from_dict(config_dict, config_path=str(config_path))
            
            logger.info(f"Successfully loaded and validated configuration: {config_path}")
            return config
            
        except ValidationError as e:
            error = ConfigValidationError(
                message=f"Configuration validation failed: {config_path}",
                errors=e.errors(),
                config_path=str(config_path)
            )
            if strict:
                raise error
            else:
                logger.error(str(error))
                raise error
        except Exception as e:
            error = ConfigValidationError(
                message=f"Failed to load configuration: {str(e)}",
                config_path=str(config_path)
            )
            if strict:
                raise error
            else:
                logger.error(str(error))
                raise error
    
    @staticmethod
    def validate_dict(
        config_dict: Dict[str, Any],
        config_class: Type[ConfigType],
        config_path: Optional[str] = None,
        strict: bool = True
    ) -> ConfigType:
        """
        Validate configuration dictionary.
        
        Args:
            config_dict: Configuration dictionary
            config_class: Configuration class
            config_path: Optional path for error reporting
            strict: Whether to raise exceptions on validation errors
            
        Returns:
            Validated configuration instance
            
        Raises:
            ConfigValidationError: If validation fails and strict=True
        """
        try:
            config = config_class.from_dict(config_dict, config_path=config_path)
            return config
        except ValidationError as e:
            error = ConfigValidationError(
                message=f"Configuration validation failed",
                errors=e.errors(),
                config_path=config_path
            )
            if strict:
                raise error
            else:
                logger.error(str(error))
                raise error
    
    @staticmethod
    def safe_validate(
        config_dict: Dict[str, Any],
        config_class: Type[ConfigType],
        default: Optional[ConfigType] = None
    ) -> Optional[ConfigType]:
        """
        Safely validate configuration (returns None on failure instead of raising).
        
        Useful for optional configurations or when you want to handle
        validation errors gracefully.
        
        Args:
            config_dict: Configuration dictionary
            config_class: Configuration class
            default: Default value to return on validation failure
            
        Returns:
            Validated configuration instance or default
        """
        try:
            return config_class.from_dict(config_dict)
        except (ConfigValidationError, ValidationError) as e:
            logger.warning(f"Configuration validation failed: {e}")
            return default


def load_and_validate_config(
    config_path: Union[str, Path],
    config_class: Type[ConfigType],
    resolve_paths: bool = True
) -> ConfigType:
    """
    Convenience function for loading and validating configurations.
    
    This is a shorthand for ConfigValidator.validate_and_load().
    
    Args:
        config_path: Path to configuration file
        config_class: Configuration class
        resolve_paths: Whether to resolve relative paths
        
    Returns:
        Validated configuration instance
        
    Example:
        >>> from habit.core.habitat_analysis.config_schemas import HabitatAnalysisConfig
        >>> config = load_and_validate_config('config.yaml', HabitatAnalysisConfig)
    """
    return ConfigValidator.validate_and_load(
        config_path=config_path,
        config_class=config_class,
        resolve_paths=resolve_paths,
        strict=True
    )
