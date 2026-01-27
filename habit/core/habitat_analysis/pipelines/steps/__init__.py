"""
Pipeline steps for habitat analysis.

This module contains all concrete pipeline step implementations.
"""

from .voxel_feature_extractor import VoxelFeatureExtractor
from .subject_preprocessing import SubjectPreprocessingStep
from .individual_clustering import IndividualClusteringStep
from .supervoxel_feature_extraction import SupervoxelFeatureExtractionStep
from .calculate_mean_voxel_features import CalculateMeanVoxelFeaturesStep
from .merge_supervoxel_features import MergeSupervoxelFeaturesStep
from .supervoxel_aggregation import SupervoxelAggregationStep  # DEPRECATED: kept for backward compatibility
from .combine_supervoxels import CombineSupervoxelsStep
from .concatenate_voxels import ConcatenateVoxelsStep
from .group_preprocessing import GroupPreprocessingStep
from .population_clustering import PopulationClusteringStep

__all__ = [
    'VoxelFeatureExtractor',
    'SubjectPreprocessingStep',
    'IndividualClusteringStep',
    'SupervoxelFeatureExtractionStep',
    'CalculateMeanVoxelFeaturesStep',
    'MergeSupervoxelFeaturesStep',
    'SupervoxelAggregationStep',  # DEPRECATED
    'CombineSupervoxelsStep',
    'ConcatenateVoxelsStep',
    'GroupPreprocessingStep',
    'PopulationClusteringStep',
]
