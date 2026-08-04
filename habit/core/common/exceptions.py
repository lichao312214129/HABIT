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
"""Backward-compatible re-exports of HABIT's exception hierarchy.

The canonical definitions moved to :mod:`habit.exceptions` in v1.0 so that
foundation layers no longer depend on ``habit.core``. This module is kept so
v0.1 internal imports continue to resolve; new code must import from
``habit.exceptions`` directly.
"""

from habit.exceptions import (
    CompatibilityError,
    ComponentNotFoundError,
    ConfigurationError,
    DataFormatError,
    HabitError,
    NotFittedError,
    ProcessingError,
)

__all__ = [
    "HabitError",
    "ConfigurationError",
    "DataFormatError",
    "NotFittedError",
    "ComponentNotFoundError",
    "ProcessingError",
    "CompatibilityError",
]
