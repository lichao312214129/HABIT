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
"""
HABIT Exception Hierarchy

Defines the core exception classes used throughout the HABIT package.
Following top-tier open-source practices, all custom exceptions inherit
from a base HabitError.
"""


class HabitError(Exception):
    """Base exception class for all HABIT errors."""

    pass


class ConfigurationError(HabitError):
    """Raised when there is an error in the configuration (YAML or dict)."""

    pass


class DataFormatError(HabitError):
    """Raised when input data format is invalid or unsupported."""

    pass


class NotFittedError(HabitError, ValueError):
    """Raised when a model or transformer is used before being fitted.
    Inherits from ValueError for scikit-learn compatibility.
    """

    pass


class ComponentNotFoundError(HabitError):
    """Raised when a requested component (model, selector, etc.) is not found in the registry."""

    pass


class ProcessingError(HabitError):
    """Raised when an error occurs during data processing or pipeline execution."""

    pass


class CompatibilityError(HabitError):
    """Raised when a saved HABIT artifact cannot be safely loaded."""

    pass
