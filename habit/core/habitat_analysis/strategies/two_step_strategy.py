"""
Two-step strategy: voxel -> supervoxel -> habitat clustering.
Refactored to use HabitatPipeline with template method pattern.
"""

from typing import TYPE_CHECKING, Optional
import pandas as pd

from .base_strategy import BaseClusteringStrategy
from ..pipelines.base_pipeline import HabitatPipeline

if TYPE_CHECKING:
    from habit.core.habitat_analysis.habitat_analysis import HabitatAnalysis


class TwoStepStrategy(BaseClusteringStrategy):
    """
    Two-step clustering strategy using HabitatPipeline.

    Flow:
    1) Voxel feature extraction (Pipeline Step 1)
    2) Subject-level preprocessing (Pipeline Step 2)
    3) Individual clustering (voxel -> supervoxel) (Pipeline Step 3)
    4) Supervoxel feature extraction (conditional) (Pipeline Step 4)
    5) Supervoxel feature aggregation (Pipeline Step 5)
    6) Combine supervoxels (Pipeline Step 6) - merge all subjects' supervoxels
    7) Group-level preprocessing (Pipeline Step 7)
    8) Population clustering (supervoxel -> habitat) (Pipeline Step 8)
    
    Note: This strategy supports parallel processing through HabitatPipeline.
    Use config.processes to control the number of parallel workers for 
    individual-level steps (Steps 1-5). Group-level steps (6-8) process 
    all subjects together.
    """

    def __init__(self, analysis: "HabitatAnalysis"):
        """
        Initialize two-step strategy.

        Args:
            analysis: HabitatAnalysis instance with shared utilities
        """
        super().__init__(analysis)
        self.pipeline: Optional[HabitatPipeline] = None