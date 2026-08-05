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
"""Base configuration classes for unified YAML/Pydantic configuration management.

This module is the v1 canonical home of :class:`BaseConfig` and related helpers.
``habit.core.common.configs.base`` re-exports these symbols during the transition
so v0.1 internal imports continue to resolve.
"""

from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union

from pydantic import BaseModel, ConfigDict, ValidationError

from habit.exceptions import ConfigurationError

ConfigType = TypeVar("ConfigType", bound="BaseConfig")

__all__ = [
    "BaseConfig",
    "ConfigAccessor",
    "ConfigValidationError",
]


class ConfigValidationError(ConfigurationError):
    """Raised when a Pydantic workflow config fails validation."""

    def __init__(
        self,
        message: str,
        errors: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
    ) -> None:
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
    """Abstract base for all HABIT workflow configuration schemas."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True,
    )

    config_file: Optional[str] = None
    config_version: Optional[str] = None

    def __init__(self, **data: Any) -> None:
        try:
            super().__init__(**data)
        except ValidationError as e:
            raise ConfigValidationError(
                message=f"Configuration validation failed for {self.__class__.__name__}",
                errors=e.errors(),
                config_path=data.get("config_file"),
            ) from e

    @classmethod
    def from_dict(
        cls: Type[ConfigType],
        config_dict: Dict[str, Any],
        config_path: Optional[str] = None,
    ) -> ConfigType:
        """Build a config instance from a plain mapping."""
        if config_path:
            config_dict["config_file"] = config_path

        try:
            return cls(**config_dict)
        except ValidationError as e:
            raise ConfigValidationError(
                message=f"Failed to create {cls.__name__} from dictionary",
                errors=e.errors(),
                config_path=config_path,
            ) from e

    @classmethod
    def from_file(cls: Type[ConfigType], config_path: Union[str, Path]) -> ConfigType:
        """Load and validate a YAML/JSON config file."""
        from habit.utils.config_loader import load_config

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        config_dict = load_config(str(config_path), resolve_paths=True)
        return cls.from_dict(config_dict, config_path=str(config_path))

    def to_dict(
        self,
        exclude_none: bool = False,
        exclude_unset: bool = False,
    ) -> Dict[str, Any]:
        """Serialise the config to a plain dictionary."""
        if hasattr(self, "model_dump"):
            return self.model_dump(
                exclude_none=exclude_none,
                exclude_unset=exclude_unset,
            )
        return self.dict(exclude_none=exclude_none, exclude_unset=exclude_unset)

    def get(self, key: str, default: Any = None) -> Any:
        """Return a top-level or dotted config value."""
        try:
            if "." in key:
                parts = key.split(".")
                value: Any = self
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
            if hasattr(self, "model_dump"):
                return self.model_dump().get(key, default)
            return self.dict().get(key, default)
        except (AttributeError, KeyError, TypeError):
            return default

    def validate(self) -> bool:
        """Re-validate the current config state."""
        try:
            self.__class__(**self.to_dict())
            return True
        except ValidationError as e:
            raise ConfigValidationError(
                message=f"Configuration validation failed for {self.__class__.__name__}",
                errors=e.errors(),
                config_path=self.config_file,
            ) from e

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) or key in self.to_dict()


class ConfigAccessor:
    """Dot-path accessor over a :class:`BaseConfig` or plain mapping."""

    def __init__(self, config: Union[BaseConfig, Dict[str, Any]]) -> None:
        self._config = config
        self._is_pydantic = isinstance(config, BaseConfig)

    def get(self, key: str, default: Any = None) -> Any:
        """Return a config value, supporting dotted keys."""
        try:
            if self._is_pydantic:
                return self._config.get(key, default)
            if "." in key:
                parts = key.split(".")
                value: Any = self._config
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
        """Return a config value or raise if absent."""
        value = self.get(key)
        if value is None:
            raise KeyError(f"Required configuration key not found: {key}")
        return value

    def has(self, key: str) -> bool:
        """Return whether a key resolves to a non-None value."""
        return self.get(key) is not None

    def get_section(
        self,
        section_name: str,
    ) -> Optional[Union[BaseConfig, Dict[str, Any]]]:
        """Return a nested config section when present."""
        value = self.get(section_name)
        if isinstance(value, (BaseConfig, dict)):
            return value
        return None

    @property
    def raw_config(self) -> Union[BaseConfig, Dict[str, Any]]:
        """Return the underlying config object."""
        return self._config
