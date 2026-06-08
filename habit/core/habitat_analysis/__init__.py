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
Habitat Analysis module for HABIT package.

V1: imports are fail-fast. If a habitat-analysis dependency is missing the
package will raise ``ImportError`` at first import, not silently set
attributes to ``None``. For genuinely optional dependencies (e.g. optional
extractor backends), prefer ``habit.is_available(name)``.

Public exports are lazy so lightweight imports such as
``habit.core.habitat_analysis.config_schemas`` do not pull PyRadiomics or
pipeline orchestration until the symbol is actually used.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from habit.utils.lazy_exports import lazy_getattr

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "HabitatAnalysis": (".habitat_analysis", "HabitatAnalysis"),
    "HabitatConfigurator": (".configurator", "HabitatConfigurator"),
    "HabitatAnalysisConfig": (".config_schemas", "HabitatAnalysisConfig"),
    "ResultColumns": (".config_schemas", "ResultColumns"),
    "HabitatMapAnalyzer": (".habitat_features.habitat_analyzer", "HabitatMapAnalyzer"),
    "BasePipelineStep": (".pipelines", "BasePipelineStep"),
    "HabitatPipeline": (".pipelines", "HabitatPipeline"),
    "GroupClusteringStep": (".pipelines.steps", "GroupClusteringStep"),
    "GroupPreprocessingStep": (".pipelines.steps", "GroupPreprocessingStep"),
}

__all__ = [
    "HabitatAnalysis",
    "HabitatConfigurator",
    "HabitatAnalysisConfig",
    "ResultColumns",
    "HabitatMapAnalyzer",
    "HabitatFeatureExtractor",
    "BasePipelineStep",
    "HabitatPipeline",
    "GroupClusteringStep",
    "GroupPreprocessingStep",
]


def __getattr__(name: str) -> Any:
    """Resolve habitat-analysis exports on first access."""
    if name == "HabitatFeatureExtractor":
        return lazy_getattr("HabitatMapAnalyzer", globals(), _LAZY_EXPORTS)
    return lazy_getattr(name, globals(), _LAZY_EXPORTS)
