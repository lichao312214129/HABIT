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
"""Visualization components for machine-learning workflows (lazy exports)."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from habit.utils.lazy_exports import lazy_getattr

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "Plotter": (".plotting", "Plotter"),
    "KMSurvivalPlotter": (".km_survival", "KMSurvivalPlotter"),
}

__all__ = ["Plotter", "KMSurvivalPlotter"]


def __getattr__(name: str) -> Any:
    """Resolve plotting classes on first access (avoids eager ``shap`` import)."""
    return lazy_getattr(name, globals(), _LAZY_EXPORTS)
