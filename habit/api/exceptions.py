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
"""Public API facade for HABIT's exception hierarchy.

The canonical definitions live in :mod:`habit.exceptions` (foundation layer).
This module re-exports them so the documented ``habit.api.exceptions`` import
path keeps working for external callers; internal code must import from
``habit.exceptions`` directly.
"""

from habit.exceptions import (
    CompatibilityError,
    ComponentNotFoundError,
    ConfigurationError,
    DataFormatError,
    GeometryError,
    HABITAPIError,
    HabitError,
    NotFittedError,
    OptionalDependencyError,
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
