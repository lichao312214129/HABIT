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
Helpers for lazy public exports in package ``__init__`` modules.

Heavy subsystems (ML plotting, habitat radiomics, etc.) should not load when
a caller only imports an unrelated submodule under the same package tree.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Tuple


def lazy_getattr(
    name: str,
    module_globals: Dict[str, Any],
    lazy_exports: Dict[str, Tuple[str, str]],
) -> Any:
    """
    Resolve a lazily exported attribute on first access.

    Args:
        name: Attribute name requested on the package module.
        module_globals: ``globals()`` of the package module (for caching).
        lazy_exports: Mapping of export name -> ``(relative_module, attribute)``.

    Returns:
        The resolved export value.

    Raises:
        AttributeError: When ``name`` is not a known lazy export.
    """
    if name not in lazy_exports:
        raise AttributeError(
            f"module {module_globals['__name__']!r} has no attribute {name!r}"
        )

    relative_module, attribute = lazy_exports[name]
    package_name = module_globals["__name__"]
    module = importlib.import_module(relative_module, package_name)
    value = getattr(module, attribute)
    module_globals[name] = value
    return value
