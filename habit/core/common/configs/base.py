"""
Base configuration classes for unified configuration management.

This module provides:
1. BaseConfig: Abstract base class for all configuration schemas
2. ConfigValidationError: Custom exception for configuration validation errors
3. ConfigAccessor: Unified interface for accessing configuration values
"""

from abc import ABC
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union

from pydantic import BaseModel, ValidationError

ConfigType = TypeVar('ConfigType', bound='BaseConfig')


class ConfigValidationError(Exception):
    def __init__(self, message: str, errors: Optional[Dict[str, Any]] = None, config_path: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.errors = errors or {}
        self.config_path = config_path

    def __str__(self) -> str:
        msg = self.message
        if self.config_path:
            msg += f" (config file: {self.config_path})"
        if self.errors:
            msg += f"\nValidation errors: {self.errors}"
        return msg


class BaseConfig(BaseModel, ABC):
    config_file: Optional[str] = None
    config_version: Optional[str] = None

    class Config:
        extra = 'forbid'
        validate_assignment = True
        use_enum_values = True

    def __init__(self, **data: Any):
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
        from .loader import load_config

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        config_dict = load_config(str(config_path), resolve_paths=True)
        return cls.from_dict(config_dict, config_path=str(config_path))

    def to_dict(self, exclude_none: bool = False, exclude_unset: bool = False) -> Dict[str, Any]:
        if hasattr(self, 'model_dump'):
            return self.model_dump(exclude_none=exclude_none, exclude_unset=exclude_unset)
        return self.dict(exclude_none=exclude_none, exclude_unset=exclude_unset)

    def get(self, key: str, default: Any = None) -> Any:
        try:
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
            if hasattr(self, key):
                return getattr(self, key)
            if hasattr(self, 'model_dump'):
                return self.model_dump().get(key, default)
            return self.dict().get(key, default)
        except (AttributeError, KeyError, TypeError):
            return default

    def validate(self) -> bool:
        try:
            self.__class__(**self.to_dict())
            return True
        except ValidationError as e:
            raise ConfigValidationError(
                message=f"Configuration validation failed for {self.__class__.__name__}",
                errors=e.errors(),
                config_path=self.config_file
            ) from e

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) or key in self.to_dict()


class ConfigAccessor:
    def __init__(self, config: Union[BaseConfig, Dict[str, Any]]):
        self._config = config
        self._is_pydantic = isinstance(config, BaseConfig)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            if self._is_pydantic:
                return self._config.get(key, default)
            if '.' in key:
                parts = key.split('.')
                value = self._config
                for part in parts:
                    if isinstance(value, dict):
                        value = value.get(part, default)
                    else:
                        return default
                return value
            return self._config.get(key, default)
        except (AttributeError, KeyError, TypeError):
            return default

    def get_required(self, key: str) -> Any:
        value = self.get(key)
        if value is None:
            raise KeyError(f"Required configuration key not found: {key}")
        return value

    def has(self, key: str) -> bool:
        return self.get(key) is not None

    def get_section(self, section_name: str) -> Optional[Union[BaseConfig, Dict[str, Any]]]:
        value = self.get(section_name)
        if isinstance(value, (BaseConfig, dict)):
            return value
        return None

    @property
    def raw_config(self) -> Union[BaseConfig, Dict[str, Any]]:
        return self._config
