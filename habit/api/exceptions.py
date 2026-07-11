"""Stable exception contract for the public HABIT API.

Public callers should catch exceptions from this module rather than importing
``habit.core`` implementation modules.  The hierarchy intentionally preserves
the established core exceptions so existing application-level error handling
continues to work.
"""

from __future__ import annotations

from sklearn.exceptions import NotFittedError

from habit.core.common.exceptions import (
    CompatibilityError,
    ComponentNotFoundError,
    ConfigurationError,
    DataFormatError,
    HabitError,
    ProcessingError,
)

__all__ = [
    "HABITAPIError",
    "HabitError",
    "ConfigurationError",
    "DataFormatError",
    "GeometryError",
    "OptionalDependencyError",
    "ComponentNotFoundError",
    "CompatibilityError",
    "ProcessingError",
    "NotFittedError",
]


class HABITAPIError(DataFormatError):
    """Raised when a value violates a documented public API data contract."""


class GeometryError(DataFormatError):
    """Raised when image and mask physical-space geometry is incompatible."""


class OptionalDependencyError(HabitError, ImportError):
    """Raised when a requested optional HABIT backend is not installed."""
