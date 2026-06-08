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
Feature extraction module for habitat analysis.

Exports are lazy so importing sibling habitat modules does not require
PyRadiomics until post-clustering feature extraction is actually used.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from habit.utils.lazy_exports import lazy_getattr

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "HabitatMapAnalyzer": (".habitat_analyzer", "HabitatMapAnalyzer"),
}

__all__ = ["HabitatMapAnalyzer", "HabitatFeatureExtractor"]


def __getattr__(name: str) -> Any:
    """Resolve habitat feature extractors on first access."""
    if name == "HabitatFeatureExtractor":
        return lazy_getattr("HabitatMapAnalyzer", globals(), _LAZY_EXPORTS)
    return lazy_getattr(name, globals(), _LAZY_EXPORTS)
