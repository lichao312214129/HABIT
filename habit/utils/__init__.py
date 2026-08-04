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
Utilities module for HABIT package.

Provides common utilities including:
- parallel_utils: Parallel processing with unified interface
- log_utils: Centralized logging management
- progress_utils: Progress bar utilities
- config_utils: Configuration loading and validation
- io_utils: Input/output operations

The parallel-processing helpers are exported LAZILY: ``parallel_utils``
chains into the v0.1 core stack (``isolated_runner`` -> ``job_cancel`` ->
``habit.core.common``), so an eager re-export here would pull sklearn/pandas
into every bare ``import habit`` (``habit/__init__`` itself only needs the
lightweight :mod:`habit.utils.lazy_exports` helper from this package).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from habit.utils.lazy_exports import lazy_getattr

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "parallel_map": (".parallel_utils", "parallel_map"),
    "parallel_map_simple": (".parallel_utils", "parallel_map_simple"),
    "ParallelProcessor": (".parallel_utils", "ParallelProcessor"),
    "ProcessingResult": (".parallel_utils", "ProcessingResult"),
}

__all__ = [
    "parallel_map",
    "parallel_map_simple",
    "ParallelProcessor",
    "ProcessingResult",
]


def __getattr__(name: str) -> Any:
    """Resolve parallel-processing helpers on first access (PEP 562)."""
    return lazy_getattr(name, globals(), _LAZY_EXPORTS)
