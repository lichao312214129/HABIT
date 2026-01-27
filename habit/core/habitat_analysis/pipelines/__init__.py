"""
Pipeline module for habitat analysis.

This module provides a sklearn-like pipeline interface for habitat analysis workflows.
"""

from .base_pipeline import BasePipelineStep, HabitatPipeline
from .pipeline_builder import build_habitat_pipeline

# Import steps for convenience
from .steps import (
    VoxelFeatureExtractor,
    SubjectPreprocessingStep,
    IndividualClusteringStep,
    SupervoxelFeatureExtractionStep,
    CalculateMeanVoxelFeaturesStep,
    MergeSupervoxelFeaturesStep,
    SupervoxelAggregationStep,  # DEPRECATED
    ConcatenateVoxelsStep,
    GroupPreprocessingStep,
    PopulationClusteringStep,
)

__all__ = [
    'BasePipelineStep',
    'HabitatPipeline',
    'build_habitat_pipeline',
    # Steps
    'VoxelFeatureExtractor',
    'SubjectPreprocessingStep',
    'IndividualClusteringStep',
    'SupervoxelFeatureExtractionStep',
    'CalculateMeanVoxelFeaturesStep',
    'MergeSupervoxelFeaturesStep',
    'SupervoxelAggregationStep',  # DEPRECATED
    'ConcatenateVoxelsStep',
    'GroupPreprocessingStep',
    'PopulationClusteringStep',
]
