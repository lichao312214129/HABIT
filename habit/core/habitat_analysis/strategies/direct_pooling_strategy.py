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
    
    ## Overview
    
    This strategy pools (concatenates) voxel features from ALL subjects into a single 
    feature matrix before clustering. This enables the discovery of population-level 
    tissue patterns that are representative across the entire cohort.
    
    ## Workflow
    
    1) Voxel feature extraction (Pipeline Step 1) - extract features for each subject
    2) Subject-level preprocessing (Pipeline Step 2) - normalize within each subject
    3) Concatenate all voxels (Pipeline Step 3) - merge all subjects' voxels into one matrix
    4) Group-level preprocessing (Pipeline Step 4) - apply population-level transformations
    5) Population clustering (Pipeline Step 5) - cluster all voxels -> discover habitats
    
    ## Why Pool All Voxels?
    
    **Rationale**: By pooling voxels from all subjects, the clustering algorithm can discover 
    tissue patterns that are **consistent and reproducible** across the entire population. 
    This approach is particularly effective for:
    - Discovering common biological phenotypes (e.g., "highly perfused tissue" vs "necrotic tissue")
    - Identifying dominant habitat patterns shared by multiple subjects
    - Quickly prototyping and exploring population-level tissue heterogeneity
    
    ## About Data Leakage
    
    **Important**: This strategy is **NOT equivalent to label leakage** in the traditional 
    machine learning sense. Here's why:
    
    - **Unsupervised Learning**: Habitat discovery is an UNSUPERVISED process (no labels involved)
    - **Feature Space Only**: Pooling occurs in the FEATURE space (imaging intensities), 
      not the label space (clinical outcomes)
    - **Pre-modeling Step**: Habitat segmentation is performed BEFORE building predictive models
    - **Pipeline Isolation**: When used in predictive workflows, the clustering model is fitted 
      on training data only and applied to test data via the saved Pipeline
    
    **Analogy**: It's similar to performing k-means clustering on pooled MRI intensities to 
    discover tissue types—the clustering doesn't "know" which subjects are diseased vs healthy.
    
    ## Use Cases
    
    **Recommended for**:
    - Exploratory analysis to discover dominant tissue patterns
    - Fast prototyping and hypothesis generation
    - Cohorts with moderate inter-subject variability
    - Studies focusing on population-level habitat characterization
    
    **Not recommended for**:
    - Extremely heterogeneous cohorts where individual differences dominate
    - Small sample sizes (prefer Two-Step or One-Step strategies)
    - Studies requiring subject-specific habitat definitions
    
    ## Parallel Processing
    
    This strategy supports parallel processing through HabitatPipeline:
    - **config.processes**: Controls parallel workers for individual-level steps (Steps 1-2)
    - **Group-level steps (3-5)**: Process all subjects together (not parallelized)
    """

    def __init__(self, analysis: "HabitatAnalysis"):
        """
        Initialize direct pooling strategy.

        Args:
            analysis: HabitatAnalysis instance with shared utilities
        """
        super().__init__(analysis)
        self.pipeline: Optional[HabitatPipeline] = None