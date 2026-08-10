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
Stable programmatic API facade for HABIT.

Import pipeline runners, config helpers, and v1 layered symbols from here or
from the top-level ``habit`` package. This subpackage re-exports the registry
in :mod:`habit._public_api` without changing behaviour.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from habit.api.registry import PUBLIC_API_SYMBOLS, build_lazy_exports
from habit.utils.lazy_exports import lazy_getattr

# ``build_lazy_exports`` targets the top-level ``habit`` package.  Symbols
# from the v0.1 facade modules (relative path ``.api.<module>``) are re-based
# onto ``habit.api``; symbols from the v1.0 layered packages (e.g.
# ``.contracts``) resolve through their absolute ``habit.<package>`` path so
# the same registry drives ``from habit.api import ...`` without eagerly
# importing optional backends.
_TOP_LEVEL_EXPORTS: Dict[str, Tuple[str, str]] = build_lazy_exports()
_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {}
for _name, (_relative_module, _attribute) in _TOP_LEVEL_EXPORTS.items():
    if _relative_module.startswith(".api"):
        _LAZY_EXPORTS[_name] = (_relative_module.removeprefix(".api"), _attribute)
    else:
        _LAZY_EXPORTS[_name] = (f"habit{_relative_module}", _attribute)
del _name, _relative_module, _attribute

__all__ = list(PUBLIC_API_SYMBOLS)


def __getattr__(name: str) -> Any:
    """Resolve a stable API symbol lazily from its domain submodule."""
    return lazy_getattr(name, globals(), _LAZY_EXPORTS)
