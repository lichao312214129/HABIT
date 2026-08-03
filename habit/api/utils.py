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
"""Lightweight utilities re-exported from the top-level ``habit`` namespace."""

from __future__ import annotations

import importlib.util

from habit.utils.log_utils import setup_logger

__all__ = ["setup_logger", "is_available"]


def is_available(module_name: str) -> bool:
    """
    Return whether an optional third-party module can be imported.

    Args:
        module_name: Top-level package name (e.g. ``"radiomics"``, ``"torch"``).

    Returns:
        True when ``importlib`` finds a spec for the module.
    """
    return importlib.util.find_spec(module_name) is not None
