"""
Direct pooling strategy: concatenate all voxel features across subjects and cluster once.
Refactored to use HabitatPipeline with template method pattern.
"""

from typing import TYPE_CHECKING, Optional
import pandas as pd

from .base_strategy import BaseClusteringStrategy
from ..pipelines.base_pipeline import HabitatPipeline

if TYPE_CHECKING:
    from habit.core.habitat_analysis.habitat_analysis import HabitatAnalysis


class DirectPoolingStrategy(BaseClusteringStrategy):
    """
    Direct pooling strategy using HabitatPipeline.

    Flow:
    1) Voxel feature extraction (Pipeline Step 1)
    2) Subject-level preprocessing (Pipeline Step 2)
    3) Concatenate all voxels (Pipeline Step 3) - merge all subjects' voxels
    4) Group-level preprocessing (Pipeline Step 4)
    5) Population clustering (all voxels -> habitat) (Pipeline Step 5)
    
    Note: This strategy supports parallel processing through HabitatPipeline.
    Use config.processes to control the number of parallel workers for 
    individual-level steps (Steps 1-2). Group-level steps (3-5) process 
    all subjects together.
    """

    def __init__(self, analysis: "HabitatAnalysis"):
        """
        Initialize direct pooling strategy.

        Args:
            analysis: HabitatAnalysis instance with shared utilities
        """
        super().__init__(analysis)
        self.pipeline: Optional[HabitatPipeline] = None