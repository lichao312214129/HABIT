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
