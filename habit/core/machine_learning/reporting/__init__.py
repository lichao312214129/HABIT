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
Reporting components for machine-learning workflows.

Heavy plot/report wiring is lazy so importing lightweight report writers does
not pull visualization / ``shap`` dependencies.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from habit.utils.lazy_exports import lazy_getattr

from .model_store import ModelStore
from .report_exporter import MetricsStore, ReportExporter
from .report_writer import ReportWriter

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "PlotComposer": (".plot_composer", "PlotComposer"),
}

__all__ = [
    "ModelStore",
    "PlotComposer",
    "ReportWriter",
    "ReportExporter",
    "MetricsStore",
]


def __getattr__(name: str) -> Any:
    """Resolve reporting components that depend on visualization on first access."""
    return lazy_getattr(name, globals(), _LAZY_EXPORTS)
