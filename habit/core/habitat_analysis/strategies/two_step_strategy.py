"""
Two-step strategy: voxel -> supervoxel -> habitat clustering.
"""

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
def _process_subject_supervoxels(
    subject: str,
    feature_manager: FeatureManager,
    clustering_manager: ClusteringManager,
    result_manager: ResultManager,
    n_clusters_supervoxel: int,
    plot_curves: bool = False,
    logger: Optional[logging.Logger] = None
) -> Tuple[str, Union[pd.DataFrame, Exception]]:
    """
    Process a single subject: extract features and cluster to supervoxels.
    
    This is a module-level function to ensure picklability for multiprocessing.
    It contains logic specific to the Two-Step strategy (supervoxel generation).
    
    Args:
        subject: Subject ID
        feature_manager: FeatureManager instance
        clustering_manager: ClusteringManager instance
        result_manager: ResultManager instance
        n_clusters_supervoxel: Number of supervoxel clusters
        plot_curves: Whether to create visualization plots
        logger: Logger instance (optional)
        
    Returns:
        Tuple of (subject, mean_features_df or Exception)
    """
    # Ensure logging in subprocess
    feature_manager._ensure_logging_in_subprocess()
    
    if logger:
        logger.info(f"Processing subject: {subject}")
    
    try:
        # 1. Extract features
        _, feature_df, raw_df, mask_info = feature_manager.extract_voxel_features(subject)
        
        # 2. Apply preprocessing
        feature_df = feature_manager.apply_preprocessing(feature_df, level='subject')
        
        # 3. Perform clustering
        supervoxel_labels = clustering_manager.cluster_subject_voxels(subject, feature_df)
        
        # 4. Calculate supervoxel means
        mean_features_df = feature_manager.calculate_supervoxel_means(
            subject, feature_df, raw_df, supervoxel_labels, 
            n_clusters_supervoxel
        )
        
        # 5. Save supervoxel image
        result_manager.save_supervoxel_image(subject, supervoxel_labels, mask_info)
        
        # 6. Visualize
        if plot_curves:
            clustering_manager.visualize_supervoxel_clustering(
                subject, feature_df, supervoxel_labels
            )
        
        # Cleanup
        del feature_df, raw_df, supervoxel_labels
        
        return subject, mean_features_df
        
    except Exception as e:
        if logger:
            logger.error(f"Error in process_single_subject for subject {subject}: {e}")
        return subject, Exception(str(e))


class TwoStepStrategy(BaseClusteringStrategy):
    """
    Two-step clustering strategy (default).

    Flow:
    1) Extract voxel features for each subject.
    2) Cluster voxels to form supervoxels (per-subject).
    3) Calculate supervoxel-level feature means (Baseline for CSV output).
    4) Prepare features for population clustering (Step 4):
       - If configured, extract advanced supervoxel features (e.g., radiomics).
       - Otherwise, use the mean features from Step 3.
    5) Cluster supervoxels to identify habitats (population-level).
    """

    def __init__(self, analysis: "HabitatAnalysis"):
        """
        Initialize two-step strategy.

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
        Execute two-step clustering.

        Args:
            subjects: List of subjects to process (None means all subjects)
            save_results_csv: Whether to save results to CSV

        Returns:
            Results DataFrame
        """
        if subjects is None:
            subjects = list(self.analysis.images_paths.keys())

        # Step 1-3: Process all subjects (Extract -> Supervoxel Cluster -> Mean Features)
        mean_features_all, failed_subjects = self._batch_process_supervoxels(subjects)

        if len(mean_features_all) == 0:
            raise ValueError("No valid features for analysis")

        # Step 4: Preprocess population-level supervoxel features
        features_for_clustering = self._prepare_population_features(
            mean_features_all, subjects, failed_subjects, self.analysis.mode_handler
        )

        # Step 5: Cluster supervoxel features to identify habitats (population-level)
        self.analysis.results_df = self._perform_population_clustering(
            mean_features_all, features_for_clustering, self.analysis.mode_handler
        )

        # Step 6: Save results (CSV files and habitat images)
        if save_results_csv:
            optimal_n_clusters = None
            if hasattr(self.analysis.clustering_manager.supervoxel2habitat_clustering, 'n_clusters'):
                optimal_n_clusters = self.analysis.clustering_manager.supervoxel2habitat_clustering.n_clusters
            
            self._save_results(
                subjects, failed_subjects, self.analysis.mode_handler, optimal_n_clusters
            )

        return self.analysis.results_df

    def _batch_process_supervoxels(self, subjects: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """
        Execute parallel processing for supervoxel generation.
        """
        if self.config.runtime.verbose:
            self.logger.info("Extracting features and performing supervoxel clustering...")
        
        # Create a partial function with all the fixed arguments
        worker_func = partial(
            _process_subject_supervoxels,
            feature_manager=self.analysis.feature_manager,
            clustering_manager=self.analysis.clustering_manager,
            result_manager=self.analysis.result_manager,
            n_clusters_supervoxel=self.config.clustering.n_clusters_supervoxel,
            plot_curves=self.config.runtime.plot_curves,
            logger=None  # Worker uses local logger config
        )
        
        # Call the module-level worker function
        results, failed_subjects = parallel_map(
            func=worker_func,
            items=subjects,
            n_processes=self.config.runtime.n_processes,
            desc="Processing subjects",
            logger=self.logger,
            show_progress=True,
            log_file_path=self.analysis._log_file_path,
            log_level=self.analysis._log_level,
        )
        
        # Combine results
        mean_features_all = pd.DataFrame()
        for result in results:
            if result.success and result.result is not None:
                # result.result is already the DataFrame (unpacked by _worker_wrapper)
                # result.item_id contains the subject ID
                df = result.result
                mean_features_all = pd.concat(
                    [mean_features_all, df], 
                    ignore_index=True
                )
        
        if self.config.runtime.verbose:
            if failed_subjects:
                self.logger.warning(f"Failed to process {len(failed_subjects)} subject(s)")
            self.logger.info(
                f"All {len(subjects)} subjects have been processed. "
                "Proceeding to clustering..."
            )
        
        return mean_features_all, failed_subjects

    def _prepare_population_features(
        self,
        mean_features_all: pd.DataFrame,
        subjects: List[str],
        failed_subjects: List[str],
        mode_handler: Any = None
    ) -> pd.DataFrame:
        """
        Prepare features for population-level clustering.
        Strategy-specific logic for Two-Step approach.
        """
        # Get feature columns (exclude metadata columns)
        feature_columns = [
            col for col in mean_features_all.columns 
            if ResultColumns.is_feature_column(col)
        ]
        features = mean_features_all[feature_columns]
        
        # Setup supervoxel file dictionary (file discovery)
        self.analysis.feature_manager.setup_supervoxel_files(
            subjects, failed_subjects, self.config.io.out_folder
        )
        
        # Check if we need to extract supervoxel-level features (Two-Step specific)
        method = self.config.feature_config['supervoxel_level']['method']
        should_extract = 'mean_voxel_features' not in method
        
        if should_extract:
            # Extract supervoxel-level features in parallel
            features = self._extract_all_supervoxel_features(subjects, failed_subjects)
        
        # Clean features (handle inf, types)
        features = self.analysis.feature_manager.clean_features(features)
        
        # Apply group-level preprocessing (delegates to mode_handler for stateful processing)
        if mode_handler:
            features = self.analysis.feature_manager.apply_preprocessing(
                features, level='group', mode_handler=mode_handler
            )
        else:
            # Fallback: just apply without state management
            features = self.analysis.feature_manager.apply_preprocessing(features, level='subject')
        
        return features
    
    def _extract_all_supervoxel_features(
        self,
        subjects: List[str],
        failed_subjects: List[str]
    ) -> pd.DataFrame:
        """
        Extract supervoxel-level features for all subjects (batch operation).
        Strategy-level orchestration of parallel extraction.
        """
        if self.config.runtime.verbose:
            self.logger.info("Extracting supervoxel-level features...")
        
        from habit.utils.parallel_utils import parallel_map
        
        # Create a partial function for supervoxel feature extraction
        extract_func = partial(
            self.analysis.feature_manager.extract_supervoxel_features
        )
        
        results, new_failed = parallel_map(
            func=extract_func,
            items=subjects,
            n_processes=self.config.runtime.n_processes,
            desc="Extracting supervoxel features",
            logger=self.logger,
            show_progress=True,
            log_file_path=self.analysis._log_file_path,
            log_level=self.analysis._log_level,
        )
        
        failed_subjects.extend(new_failed)
        
        if self.config.runtime.verbose and new_failed:
            self.logger.warning(
                f"Failed to extract supervoxel features for {len(new_failed)} subject(s)"
            )
        
        # Combine results
        features_list = [r.result for r in results if r.success and r.result is not None]
        if features_list:
            return pd.concat(features_list, ignore_index=True)
        else:
            raise ValueError("No valid supervoxel features extracted")
    
    def _perform_population_clustering(
        self,
        mean_features_all: pd.DataFrame,
        features: pd.DataFrame,
        mode_handler: Any
    ) -> pd.DataFrame:
        """
        Perform population-level clustering to determine habitats.
        Strategy-specific logic for Two-Step approach.
        
        Args:
            mean_features_all: Original combined features with metadata
            features: Cleaned features for clustering
            mode_handler: Mode handler instance for clustering logic
            
        Returns:
            Results DataFrame with habitat labels
        """
        # Perform population-level clustering
        habitat_labels, optimal_n_clusters, scores = mode_handler.cluster_habitats(
            features, self.analysis.clustering_manager.supervoxel2habitat_clustering
        )
        
        # Plot scores if available
        if scores and self.config.runtime.plot_curves:
            self.analysis.clustering_manager.plot_habitat_scores(scores, optimal_n_clusters)
        
        # Visualize clustering results
        if self.config.runtime.plot_curves:
            self.analysis.clustering_manager.visualize_habitat_clustering(
                features, habitat_labels, optimal_n_clusters
            )
        
        # Save model for training mode
        if self.config.runtime.mode == 'training':
            mode_handler.save_model(
                self.analysis.clustering_manager.supervoxel2habitat_clustering,
                'supervoxel2habitat_clustering_strategy'
            )
        
        # Add habitat labels to results
        mean_features_all[ResultColumns.HABITATS] = habitat_labels
        
        return mean_features_all.copy()
    
    def _save_results(
        self, 
        subjects: List[str], 
        failed_subjects: List[str],
        mode_handler: Any,
        optimal_n_clusters: int
    ) -> None:
        """
        Save all results including config, CSV, and habitat images.
        Strategy-specific save logic.
        """
        if self.config.runtime.verbose:
            self.logger.info("Saving results...")
        
        import os
        os.makedirs(self.config.io.out_folder, exist_ok=True)
        
        # Save configuration
        if mode_handler:
            mode_handler.save_config(optimal_n_clusters)
        
        # Save results CSV
        csv_path = os.path.join(self.config.io.out_folder, 'habitats.csv')
        self.analysis.results_df.to_csv(csv_path, index=False)
        if self.config.runtime.verbose:
            self.logger.info(f"Results saved to {csv_path}")
        
        # Save habitat images for each subject
        # Ensure ResultManager has the latest results_df
        self.analysis.result_manager.results_df = self.analysis.results_df
        self.analysis.result_manager.save_all_habitat_images(failed_subjects)
