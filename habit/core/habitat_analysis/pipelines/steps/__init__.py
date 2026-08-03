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
