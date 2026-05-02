"""
Pipeline module for habitat analysis.

This module provides a sklearn-style pipeline interface for habitat
analysis workflows. Mode dispatch (``two_step`` / ``one_step`` /
``direct_pooling``) lives inside ``habitat_analysis.HabitatAnalysis``
via the ``_PIPELINE_RECIPES`` dictionary; there is no longer a separate
``pipeline_builder`` module or ``build_habitat_pipeline`` factory in V1.
"""

from .base_pipeline import BasePipelineStep, HabitatPipeline
from .steps import (
    VoxelFeatureExtractor,
    SubjectPreprocessingStep,
    IndividualClusteringStep,
    SupervoxelFeatureExtractionStep,
    CalculateMeanVoxelFeaturesStep,
    MergeSupervoxelFeaturesStep,
    ConcatenateVoxelsStep,
    GroupPreprocessingStep,
    PopulationClusteringStep,
)

__all__ = [
    'BasePipelineStep',
    'HabitatPipeline',
    # Steps
    'VoxelFeatureExtractor',
    'SubjectPreprocessingStep',
    'IndividualClusteringStep',
    'SupervoxelFeatureExtractionStep',
    'CalculateMeanVoxelFeaturesStep',
    'MergeSupervoxelFeaturesStep',
    'ConcatenateVoxelsStep',
    'GroupPreprocessingStep',
    'PopulationClusteringStep',
]
