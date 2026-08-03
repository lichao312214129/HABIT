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

Internal implementation modules under ``habit.core`` are not part of the public
API contract.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from habit._version import __version__
from habit.api.registry import PUBLIC_API_SYMBOLS, build_lazy_exports
from habit.utils.lazy_exports import lazy_getattr

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = build_lazy_exports()

__all__ = ["__version__", *PUBLIC_API_SYMBOLS]


def __getattr__(name: str) -> Any:
    """Resolve a stable public symbol on first access."""
    return lazy_getattr(name, globals(), _LAZY_EXPORTS)
