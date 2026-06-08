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
