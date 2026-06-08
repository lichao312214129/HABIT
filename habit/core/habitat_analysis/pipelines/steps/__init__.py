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
Pipeline steps for habitat analysis.

This module contains all concrete pipeline step implementations.
"""

from .voxel_feature_extraction import VoxelFeatureExtractor
from .individual_preprocessing import IndividualPreprocessingStep
from .individual_clustering import IndividualClusteringStep
from .supervoxel_feature_extraction import SupervoxelFeatureExtractionStep
from .mean_voxel_features import CalculateMeanVoxelFeaturesStep
from .supervoxel_feature_merge import MergeSupervoxelFeaturesStep
from .supervoxel_combination import CombineSupervoxelsStep
from .voxel_concatenation import ConcatenateVoxelsStep
from .group_preprocessing import GroupPreprocessingStep
from .group_clustering import GroupClusteringStep
from habit.core.habitat_analysis.checkpoint.step import CheckpointSaveStep

__all__ = [
    'VoxelFeatureExtractor',
    'IndividualPreprocessingStep',
    'IndividualClusteringStep',
    'SupervoxelFeatureExtractionStep',
    'CalculateMeanVoxelFeaturesStep',
    'MergeSupervoxelFeaturesStep',
    'CheckpointSaveStep',
    'CombineSupervoxelsStep',
    'ConcatenateVoxelsStep',
    'GroupPreprocessingStep',
    'GroupClusteringStep',
]
