"""Public exception module for stable HABIT error handling.

Prefer importing exceptions from this module instead of ``habit.core`` so
application code remains compatible across internal refactors.
"""

from habit.api.exceptions import (
    HABITAPIError,
    CompatibilityError,
    ComponentNotFoundError,
    ConfigurationError,
    DataFormatError,
    HabitError,
    NotFittedError,
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
