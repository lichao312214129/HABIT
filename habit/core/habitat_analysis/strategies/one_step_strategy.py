"""
One-step strategy: voxel -> habitat clustering per subject.
Refactored to use HabitatPipeline with template method pattern.
"""

from typing import TYPE_CHECKING, Optional
import pandas as pd

from .base_strategy import BaseClusteringStrategy
from ..pipelines.base_pipeline import HabitatPipeline
from ..config_schemas import ResultColumns

if TYPE_CHECKING:
    from habit.core.habitat_analysis.habitat_analysis import HabitatAnalysis


class OneStepStrategy(BaseClusteringStrategy):
    """
    One-step clustering strategy using HabitatPipeline.

    Flow:
    1) Voxel feature extraction (Pipeline Step 1)
    2) Subject-level preprocessing (Pipeline Step 2)
    3) Individual clustering (voxel -> habitat per subject) (Pipeline Step 3)
    4) Supervoxel aggregation (Pipeline Step 4) - calculates means per habitat
    5) Combine supervoxels (Pipeline Step 5) - merge all subjects' results
    
    Note: This strategy supports parallel processing through HabitatPipeline.
    Use config.processes to control the number of parallel workers.
    """

    def __init__(self, analysis: "HabitatAnalysis"):
        """
        Initialize one-step strategy.

        Args:
            analysis: HabitatAnalysis instance with shared utilities
        """
        super().__init__(analysis)
        self.pipeline: Optional[HabitatPipeline] = None
    
    def _post_process_results(self) -> None:
        """
        Post-process results specific to One-Step strategy.
        
        In One-Step, the Aggregation step calculates means per habitat.
        Habitat column is usually same as Supervoxel column in this case.
        """
        if ResultColumns.HABITATS not in self.analysis.results_df.columns:
            if ResultColumns.SUPERVOXEL in self.analysis.results_df.columns:
                self.analysis.results_df[ResultColumns.HABITATS] = \
                    self.analysis.results_df[ResultColumns.SUPERVOXEL]
    
    def _save_results(self) -> None:
        """
        Save results for One-Step strategy.
        
        Overrides base implementation because One-Step saves images differently.
        """
        from pathlib import Path
        
        if self.config.verbose:
            self.logger.info("Saving results...")
        
        # Save results CSV with consistent column order (metadata first, then features)
        csv_path = Path(self.config.out_dir) / "habitats.csv"
        from habit.core.habitat_analysis.strategies.base_strategy import _canonical_csv_column_order
        df = self.analysis.results_df
        canonical_order = _canonical_csv_column_order(df)
        df[canonical_order].to_csv(str(csv_path), index=False)
        if self.config.verbose:
            self.logger.info(f"Results saved to {csv_path}")
        
        # Note: In One-Step, IndividualClusteringStep already saved habitat maps directly
        # No need to call save_all_habitat_images which expects supervoxel files
        if self.config.verbose:
            self.logger.info("One-Step mode: habitat maps have been saved during clustering.")