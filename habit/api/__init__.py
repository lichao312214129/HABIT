# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""
Stable programmatic API facade for HABIT.

Import pipeline runners and config classes from here or from the top-level
``habit`` package. Implementation remains in ``habit.core.*``; this subpackage
only re-exports symbols without changing behaviour.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from habit.api.registry import PUBLIC_API_SYMBOLS, build_lazy_exports
from habit.utils.lazy_exports import lazy_getattr

# ``build_lazy_exports`` targets the top-level ``habit`` package.  Remove that
# prefix so the same registry also drives ``from habit.api import ...`` without
# eagerly importing optional backends.
_TOP_LEVEL_EXPORTS: Dict[str, Tuple[str, str]] = build_lazy_exports()
_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    name: (relative_module.removeprefix(".api"), attribute)
    for name, (relative_module, attribute) in _TOP_LEVEL_EXPORTS.items()
}

__all__ = list(PUBLIC_API_SYMBOLS)


def __getattr__(name: str) -> Any:
    """Resolve a stable API symbol lazily from its domain submodule."""
    return lazy_getattr(name, globals(), _LAZY_EXPORTS)
