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
"""Stable exception contract for HABIT.

This module is the canonical home of HABIT's exception hierarchy. It sits at
the foundation of the layering rules: it must never import other ``habit``
modules, so every layer (kernels -> contracts -> domain -> api -> interfaces)
can depend on it without creating import cycles.

``habit.api.exceptions`` (public API facade) and
``habit.core.common.exceptions`` (v0.1 internal module) re-export these
classes for backward compatibility; new code should import from here.

``NotFittedError`` is constructed lazily via PEP 562 ``__getattr__``: it must
subclass :class:`sklearn.exceptions.NotFittedError` for sklearn interop, but
importing sklearn at module scope would drag the entire scientific-Python
stack into every bare ``import habit`` (this module sits on the foundation
import path). The sklearn import therefore happens only on first access of
the class; ``import habit`` itself stays sklearn-free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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

_NOT_FITTED_DOC = """Raised when a model or transformer is used before being fitted.

Unifies the v0.1 ``core.common`` class (``HabitError`` + ``ValueError``)
with :class:`sklearn.exceptions.NotFittedError` so a single ``except``
clause catches HABIT estimators and sklearn pipelines alike.
"""


class HabitError(Exception):
    """Base exception class for all HABIT errors."""


class ConfigurationError(HabitError):
    """Raised when there is an error in the configuration (YAML or dict)."""


class DataFormatError(HabitError):
    """Raised when input data format is invalid or unsupported."""


if TYPE_CHECKING:
    # Static view for type checkers: ``NotFittedError`` is a real class with
    # the documented bases. At runtime the identical class is built lazily by
    # ``__getattr__`` below so that importing this module stays sklearn-free.
    from sklearn.exceptions import NotFittedError as _SklearnNotFittedError

    class NotFittedError(HabitError, _SklearnNotFittedError):
        """Raised when a model or transformer is used before being fitted."""


def _build_not_fitted_error() -> type:
    """
    Construct ``NotFittedError`` on first access, importing sklearn lazily.

    Returns:
        A class equivalent to ``class NotFittedError(HabitError,
        sklearn.exceptions.NotFittedError)`` defined in this module, so
        ``__module__`` is ``habit.exceptions`` and pickle-by-reference keeps
        working once the class is cached in ``globals()``.
    """
    from sklearn.exceptions import NotFittedError as sklearn_not_fitted_error

    return type(
        "NotFittedError",
        (HabitError, sklearn_not_fitted_error),
        {"__doc__": _NOT_FITTED_DOC, "__module__": __name__},
    )


def __getattr__(name: str) -> Any:
    """Resolve lazily constructed members (PEP 562) on first access."""
    if name == "NotFittedError":
        cls = _build_not_fitted_error()
        # Cache in the module namespace: subsequent lookups bypass
        # ``__getattr__`` (stable class identity) and pickle resolves the
        # class by reference through this module's globals.
        globals()["NotFittedError"] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class ComponentNotFoundError(HabitError):
    """Raised when a requested component (model, selector, etc.) is not found in the registry."""


class ProcessingError(HabitError):
    """Raised when an error occurs during data processing or pipeline execution."""


class CompatibilityError(HabitError):
    """Raised when a saved HABIT artifact cannot be safely loaded."""


class HABITAPIError(DataFormatError):
    """Raised when a value violates a documented public API data contract."""


class GeometryError(DataFormatError):
    """Raised when image and mask physical-space geometry is incompatible."""


class OptionalDependencyError(HabitError, ImportError):
    """Raised when a requested optional HABIT backend is not installed."""
