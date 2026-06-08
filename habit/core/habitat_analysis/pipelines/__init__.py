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
Pipeline module for habitat analysis.

This module provides a sklearn-style pipeline interface for habitat
analysis workflows. Mode dispatch (``two_step`` / ``one_step`` /
``direct_pooling``) lives inside ``habitat_analysis.HabitatAnalysis``
via the ``_PIPELINE_RECIPES`` dictionary; there is no longer a separate
``pipeline_builder`` module or ``build_habitat_pipeline`` factory in V1.

Exports are lazy to avoid circular imports with the checkpoint package.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from habit.utils.lazy_exports import lazy_getattr

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "BasePipelineStep": (".base_pipeline", "BasePipelineStep"),
    "HabitatPipeline": (".base_pipeline", "HabitatPipeline"),
    "VoxelFeatureExtractor": (".steps", "VoxelFeatureExtractor"),
    "IndividualPreprocessingStep": (".steps", "IndividualPreprocessingStep"),
    "IndividualClusteringStep": (".steps", "IndividualClusteringStep"),
    "SupervoxelFeatureExtractionStep": (".steps", "SupervoxelFeatureExtractionStep"),
    "CalculateMeanVoxelFeaturesStep": (".steps", "CalculateMeanVoxelFeaturesStep"),
    "MergeSupervoxelFeaturesStep": (".steps", "MergeSupervoxelFeaturesStep"),
    "ConcatenateVoxelsStep": (".steps", "ConcatenateVoxelsStep"),
    "GroupPreprocessingStep": (".steps", "GroupPreprocessingStep"),
    "GroupClusteringStep": (".steps", "GroupClusteringStep"),
}

__all__ = [
    "BasePipelineStep",
    "HabitatPipeline",
    "VoxelFeatureExtractor",
    "IndividualPreprocessingStep",
    "IndividualClusteringStep",
    "SupervoxelFeatureExtractionStep",
    "CalculateMeanVoxelFeaturesStep",
    "MergeSupervoxelFeaturesStep",
    "ConcatenateVoxelsStep",
    "GroupPreprocessingStep",
    "GroupClusteringStep",
]


def __getattr__(name: str) -> Any:
    """Resolve pipeline exports on first access."""
    return lazy_getattr(name, globals(), _LAZY_EXPORTS)
