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
HABIT — Habitat Analysis: Biomedical Imaging Toolkit.

Stable programmatic entry points are exposed on this package namespace. Heavy
subsystems load lazily on first attribute access so ``import habit`` stays
lightweight.

Example::

    from habit import PreprocessingConfig, run_preprocess

    config = PreprocessingConfig.from_file("config/preprocessing/config_preprocessing_demo.yaml")
    run_preprocess(config)

Internal implementation lives under ``habit.compat.engines`` and is not part of the
public API contract.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Tuple

from habit._version import __version__

# The export table comes from ``habit._public_api`` (pure data, zero internal
# imports) rather than ``habit.api.registry``: importing the ``habit.api``
# package here would chain into the v0.1 core stack and defeat the lazy
# loading documented above. ``habit.api.registry`` re-exports the same table.
from habit._public_api import PUBLIC_API_SYMBOLS, build_lazy_exports
from habit.utils.lazy_exports import lazy_getattr

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = build_lazy_exports()

#: v1.0 layered packages resolvable as attributes (``habit.compat`` etc.)
#: after a bare ``import habit``, loaded only on first access.
_LAZY_SUBPACKAGES = frozenset(
    {
        "adapters",
        "compat",
        "contracts",
        "domain",
        "execution",
        "kernels",
        "registry",
        "spec",
        "viz",
    }
)

__all__ = ["__version__", *PUBLIC_API_SYMBOLS]


def __getattr__(name: str) -> Any:
    """Resolve a stable public symbol or layered subpackage on first access."""
    if name in _LAZY_SUBPACKAGES:
        module = importlib.import_module(f"habit.{name}")
        globals()[name] = module
        return module
    return lazy_getattr(name, globals(), _LAZY_EXPORTS)
