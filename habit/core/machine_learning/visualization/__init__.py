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
