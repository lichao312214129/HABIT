"""
Individual-level clustering step for habitat analysis pipeline.

This step clusters voxels to supervoxels (or habitats) for each subject independently.
"""

from typing import Dict, Any, Optional, Literal, Tuple
import pandas as pd
import numpy as np
import logging

from ..base_pipeline import BasePipelineStep
from ...managers.clustering_manager import ClusteringManager
from ...managers.result_manager import ResultManager
from ...config_schemas import HabitatAnalysisConfig
from habit.utils.parallel_utils import parallel_map


class IndividualClusteringStep(BasePipelineStep):
    """
    Individual-level clustering (voxel → supervoxel or voxel → habitat).
    
    Stateless: clustering parameters are fixed by configuration or computed per subject.
    
    Attributes:
        feature_manager: FeatureManager instance (for accessing data)
        clustering_manager: ClusteringManager instance
        result_manager: ResultManager instance (for saving supervoxel maps)
        config: Configuration object
        target: 'supervoxel' for two-step strategy, 'habitat' for one-step strategy
        find_optimal: Whether to find optimal cluster number (for one-step strategy)
        fitted_: bool indicating whether the step has been fitted
    """
    
    def __init__(
        self,
        feature_manager: Any,  # FeatureManager
        clustering_manager: ClusteringManager,
        result_manager: ResultManager,
        config: HabitatAnalysisConfig,
        target: Literal['supervoxel', 'habitat'] = 'supervoxel',
        find_optimal: bool = False
    ):
        """
        Initialize individual clustering step.
        
        Args:
            feature_manager: FeatureManager instance
            clustering_manager: ClusteringManager instance
            result_manager: ResultManager instance (for saving supervoxel maps)
            config: Configuration object
            target: 'supervoxel' for two-step, 'habitat' for one-step
            find_optimal: Whether to find optimal cluster number (one-step only)
        """
        super().__init__()
        self.feature_manager = feature_manager
        self.clustering_manager = clustering_manager
        self.result_manager = result_manager
        self.config = config
        self.target = target
        self.find_optimal = find_optimal
        self.logger = logging.getLogger(__name__)
    
    def fit(self, X: Dict[str, Dict], y: Optional[Any] = None, **fit_params) -> 'IndividualClusteringStep':
        """
        Fit the step (stateless operation, just mark as fitted).
        
        Args:
            X: Dict of subject_id -> {
                'features': pd.DataFrame,
                'raw': pd.DataFrame,
                'mask_info': dict
            }
            y: Optional target data (not used)
            **fit_params: Additional fitting parameters (not used)
            
        Returns:
            self
        """
        # Stateless step - no parameters to learn
        # Each subject will be clustered independently in transform()
        self.fitted_ = True
        return self
    
    def _cluster_single_subject(
        self, 
        item: Tuple[str, Dict]
    ) -> Tuple[str, Dict]:
        """
        Cluster a single subject (wrapper for parallel processing).
        
        Args:
            item: Tuple of (subject_id, data dict)
            
        Returns:
            Tuple of (subject_id, result dict or Exception)
        """
        subject_id, data = item
        
        try:
            feature_df = data['features']
            raw_df = data['raw']
            mask_info = data['mask_info']
            
            # Determine number of clusters
            n_clusters = None
            if self.target == 'supervoxel':
                n_clusters = self.config.HabitatsSegmention.supervoxel.n_clusters
            elif self.target == 'habitat':
                if self.find_optimal:
                    one_step_cfg = self.config.HabitatsSegmention.supervoxel.one_step_settings
                    n_clusters = self.clustering_manager.find_optimal_clusters_for_subject(
                        subject_id,
                        feature_df,
                        min_clusters=one_step_cfg.min_clusters,
                        max_clusters=one_step_cfg.max_clusters,
                        selection_method=one_step_cfg.selection_method,
                        plot_validation=self.config.plot_curves
                    )
                else:
                    n_clusters = self.config.HabitatsSegmention.supervoxel.n_clusters
            
            # Perform clustering
            labels = self.clustering_manager.cluster_subject_voxels(
                subject_id,
                feature_df,
                n_clusters=n_clusters
            )
            
            # Save images
            if self.config.save_images:
                if self.target == 'supervoxel':
                    self.result_manager.save_supervoxel_image(
                        subject_id, labels, mask_info
                    )
                elif self.target == 'habitat':
                    self.result_manager.save_habitat_image_from_voxels(
                        subject_id, labels, mask_info
                    )
            
            # Visualize
            if self.config.plot_curves:
                if self.target == 'supervoxel':
                    self.clustering_manager.visualize_supervoxel_clustering(
                        subject_id, feature_df, labels
                    )
                elif self.target == 'habitat':
                    self.clustering_manager.visualize_habitat_clustering(
                        feature_df.values, labels, n_clusters,
                        subject=subject_id, output_dir=self.config.out_dir
                    )
            
            result = {
                'features': feature_df,
                'raw': raw_df,
                'mask_info': mask_info,
                'supervoxel_labels': labels
            }
            
            return subject_id, result
            
        except Exception as e:
            self.logger.error(f"Error clustering subject {subject_id}: {e}")
            return subject_id, e
    
    def transform(self, X: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Cluster voxels to supervoxels (or habitats) for each subject with parallel processing.
        
        Args:
            X: Dict of subject_id -> {
                'features': pd.DataFrame,
                'raw': pd.DataFrame,
                'mask_info': dict
            }
            
        Returns:
            Dict of subject_id -> {
                'features': pd.DataFrame,
                'raw': pd.DataFrame,
                'mask_info': dict,
                'supervoxel_labels': np.ndarray
            }
        """
        # Get number of processes from config
        n_processes = getattr(self.config, 'processes', 1)
        
        # Prepare items for parallel processing
        items = [(subject_id, data) for subject_id, data in X.items()]
        
        # Process subjects in parallel
        desc = f"Clustering to {self.target}s"
        successful_results, failed_subjects = parallel_map(
            func=self._cluster_single_subject,
            items=items,
            n_processes=n_processes,
            desc=desc,
            logger=self.logger,
            show_progress=True,
        )
        
        # Convert results to dict
        results = {}
        for proc_result in successful_results:
            # proc_result.item_id contains subject_id
            # proc_result.result contains the result dict
            results[proc_result.item_id] = proc_result.result
        
        # Log failed subjects
        if failed_subjects:
            self.logger.error(
                f"Failed to cluster {len(failed_subjects)} subject(s): "
                f"{', '.join(str(s[0]) if isinstance(s, tuple) else str(s) for s in failed_subjects)}"
            )
        
        return results
