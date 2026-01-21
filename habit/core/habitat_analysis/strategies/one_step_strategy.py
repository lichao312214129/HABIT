"""
One-step strategy: voxel -> habitat clustering per subject.
"""

import os
import logging
from functools import partial
from typing import TYPE_CHECKING, List, Optional, Tuple, Union, Any

import pandas as pd

from habit.utils.parallel_utils import parallel_map
from .base_strategy import BaseClusteringStrategy
from ..managers import FeatureManager, ClusteringManager, ResultManager
from ..config import ResultColumns

if TYPE_CHECKING:
    from habit.core.habitat_analysis.habitat_analysis import HabitatAnalysis


# -----------------------------------------------------------------------------
# Module-level worker function (Pure function, Picklable)
# -----------------------------------------------------------------------------
def _process_subject_one_step(
    subject: str,
    feature_manager: FeatureManager,
    clustering_manager: ClusteringManager,
    result_manager: ResultManager,
    min_clusters: int,
    max_clusters: int,
    selection_method: str,
    best_n_clusters: Optional[int],
    plot_validation: bool,
    logger: Optional[logging.Logger] = None
) -> Tuple[str, Union[pd.DataFrame, Exception]]:
    """
    Process a single subject for One-Step strategy.
    
    This is a module-level function to ensure picklability.
    
    Args:
        subject: Subject ID
        feature_manager: FeatureManager instance
        clustering_manager: ClusteringManager instance
        result_manager: ResultManager instance
        min_clusters: Minimum clusters to try
        max_clusters: Maximum clusters to try
        selection_method: Validation method to use
        best_n_clusters: Fixed number of clusters (if specified)
        plot_validation: Whether to plot validation curves
        logger: Logger instance (optional)
        
    Returns:
        Tuple of (subject, mean_features_df or Exception)
    """
    # Ensure logging in subprocess
    feature_manager._ensure_logging_in_subprocess()
    
    if logger:
        logger.info(f"Processing subject (One-Step): {subject}")
    
    try:
        # 1. Extract features
        _, feature_df, raw_df, mask_info = feature_manager.extract_voxel_features(subject)
        
        # 2. Apply preprocessing
        feature_df = feature_manager.apply_preprocessing(feature_df, level='subject')
        
        # 3. Find optimal clusters for this subject (One-Step specific)
        if best_n_clusters is not None:
            # Use fixed number of clusters
            optimal_n = best_n_clusters
            if logger:
                logger.info(f"Subject {subject}: Using fixed cluster number {optimal_n}")
        else:
            # Find optimal using validation methods
            optimal_n = clustering_manager.find_optimal_clusters_for_subject(
                subject, feature_df, min_clusters, max_clusters, 
                selection_method, plot_validation
            )
        
        # 4. Perform clustering with optimal number
        habitat_labels = clustering_manager.cluster_subject_voxels(
            subject, feature_df, n_clusters=optimal_n
        )
        
        # 5. Calculate habitat means
        mean_features_df = feature_manager.calculate_supervoxel_means(
            subject, feature_df, raw_df, habitat_labels, 
            optimal_n
        )
        
        # In One-Step, supervoxels ARE habitats
        mean_features_df[ResultColumns.HABITATS] = mean_features_df[ResultColumns.SUPERVOXEL]
        
        # 6. Save supervoxel image (which represents habitats in One-Step)
        result_manager.save_supervoxel_image(subject, habitat_labels, mask_info)
        
        # Cleanup
        del feature_df, raw_df, habitat_labels
        
        return subject, mean_features_df
        
    except Exception as e:
        if logger:
            logger.error(f"Error in one_step process for subject {subject}: {e}")
        return subject, Exception(str(e))


class OneStepStrategy(BaseClusteringStrategy):
    """
    One-step clustering strategy.

    Flow:
    1) Extract voxel features for each subject
    2) Cluster voxels directly to form habitats (per-subject, no population-level step)
    3) Each subject gets its own optimal number of habitats

    Decoupled from TwoStepStrategy - directly inherits from BaseHabitatStrategy.
    """

    def __init__(self, analysis: "HabitatAnalysis"):
        """
        Initialize one-step strategy.

        Args:
            analysis: HabitatAnalysis instance with shared utilities
        """
        super().__init__(analysis)

    def run(
        self,
        subjects: Optional[List[str]] = None,
        save_results_csv: bool = True
    ) -> pd.DataFrame:
        """
        Execute one-step clustering.

        Args:
            subjects: List of subjects to process (None means all subjects)
            save_results_csv: Whether to save results to CSV

        Returns:
            Results DataFrame
        """
        if subjects is None:
            subjects = list(self.analysis.images_paths.keys())

        # Process all subjects
        results_df, failed_subjects = self._batch_process_one_step(subjects)

        if len(results_df) == 0:
            raise ValueError("No valid features for analysis")

        # In One-Step, the "supervoxel" means ARE the final habitat results
        self.analysis.results_df = results_df

        # Save results (CSV files)
        if save_results_csv:
            self._save_results(subjects, failed_subjects)

        return self.analysis.results_df

    def _batch_process_one_step(self, subjects: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """
        Execute parallel processing for One-Step clustering.
        """
        if self.config.verbose:
            self.logger.info("Executing One-Step clustering (per-subject optimal)...")
        
        one_step_config = self.config.HabitatsSegmention.supervoxel.one_step_settings

        # Create a partial function with all the fixed arguments
        worker_func = partial(
            _process_subject_one_step,
            feature_manager=self.analysis.feature_manager,
            clustering_manager=self.analysis.clustering_manager,
            result_manager=self.analysis.result_manager,
            min_clusters=one_step_config.min_clusters,
            max_clusters=one_step_config.max_clusters,
            selection_method=one_step_config.selection_method,
            best_n_clusters=one_step_config.best_n_clusters,
            plot_validation=one_step_config.plot_validation_curves,
            logger=None  # Worker uses local logger config
        )

        results, failed_subjects = parallel_map(
            func=worker_func,
            items=subjects,
            n_processes=self.config.processes,
            desc="Processing One-Step",
            logger=self.logger,
            show_progress=True,
            log_file_path=self.analysis._log_file_path,
            log_level=self.analysis._log_level,
        )
        
        # Combine results
        all_results = pd.DataFrame()
        for result in results:
            if result.success and result.result is not None:
                # result.result is already the DataFrame (unpacked by _worker_wrapper)
                # result.item_id contains the subject ID
                df = result.result
                all_results = pd.concat([all_results, df], ignore_index=True)
        
        if self.config.verbose:
            if failed_subjects:
                self.logger.warning(f"Failed to process {len(failed_subjects)} subject(s)")
            self.logger.info(f"One-Step processing complete.")
        
        return all_results, failed_subjects
    
    def _save_results(
        self, 
        subjects: List[str], 
        failed_subjects: List[str]
    ) -> None:
        """
        Save results for One-Step strategy.
        """
        if self.config.verbose:
            self.logger.info("Saving results...")
        
        os.makedirs(self.config.out_dir, exist_ok=True)
        
        # Save configuration (no global optimal_n_clusters for One-Step)
        if self.analysis.mode_handler:
            self.analysis.mode_handler.save_config(optimal_n_clusters=None)
        
        # Save results CSV
        csv_path = os.path.join(self.config.out_dir, 'habitats.csv')
        self.analysis.results_df.to_csv(csv_path, index=False)
        if self.config.verbose:
            self.logger.info(f"Results saved to {csv_path}")
        
        # Note: In One-Step, supervoxel images already represent habitats
        # No need to save separate habitat images
        if self.config.verbose:
            self.logger.info(
                "One-Step mode: supervoxel images are the final habitat maps "
                "(no separate habitat images needed)"
            )
