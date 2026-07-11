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
    "ComponentNotFoundError",
    "CompatibilityError",
    "ProcessingError",
    "NotFittedError",
]


class HABITAPIError(DataFormatError):
    """Raised when a value violates a documented public API data contract."""
