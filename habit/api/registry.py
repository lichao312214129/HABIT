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
Canonical registry of stable symbols exposed from ``import habit``.

The registry data lives in :mod:`habit._public_api`, a zero-dependency module
that ``habit/__init__.py`` can read without importing the ``habit.api``
package (that import chain used to drag the v0.1 core stack and sklearn into
every bare ``import habit``). This facade re-exports the same names so
existing callers -- including the public-API contract tests -- keep working.

Tests import this module to verify the public contract without duplicating lists.
"""

from __future__ import annotations

from habit._public_api import (
    _PUBLIC_API_MODULES,
    PUBLIC_API_SYMBOLS,
    build_lazy_exports,
)

__all__ = ["PUBLIC_API_SYMBOLS", "build_lazy_exports"]
