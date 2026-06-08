# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text. Summary:
#
#   - Non-commercial use (academic, research, education, personal) is permitted
#     provided that copyright notices are retained and HABIT usage is
#     acknowledged in publications, reports, or documentation.
#   - Commercial use requires prior written consent from the copyright holder
#     (lichao19870617@163.com) and public acknowledgment of HABIT usage in
#     product documentation or user-facing materials.
#   - Unauthorized commercial use or removal of attribution is prohibited.
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
