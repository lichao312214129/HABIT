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
