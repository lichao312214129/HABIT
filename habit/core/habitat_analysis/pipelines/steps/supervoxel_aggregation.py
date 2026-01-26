"""
Supervoxel aggregation step for habitat analysis pipeline.

This step aggregates voxel features to supervoxel level and optionally merges
with advanced features from SupervoxelFeatureExtractionStep.
"""

from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import logging

from ..base_pipeline import BasePipelineStep
from ...managers.feature_manager import FeatureManager
from ...config_schemas import HabitatAnalysisConfig, ResultColumns
from habit.utils.parallel_utils import parallel_map


class SupervoxelAggregationStep(BasePipelineStep):
    """
    Aggregate voxel features to supervoxel level.
    
    For two-step strategy:
    1. Calculate mean voxel features per supervoxel (always done)
    2. Optionally merge with advanced features from Step 4 (if Step 4 was executed)
    
    **Important**: This step always calculates mean features. If Step 4 was executed,
    it merges the advanced features from Step 4's output.
    
    Attributes:
        feature_manager: FeatureManager instance
        config: Configuration object
        fitted_: bool indicating whether the step has been fitted
    """
    
    def __init__(
        self,
        feature_manager: FeatureManager,
        config: HabitatAnalysisConfig
    ):
        """
        Initialize supervoxel aggregation step.
        
        Args:
            feature_manager: FeatureManager instance
            config: Configuration object
        """
        super().__init__()
        self.feature_manager = feature_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def fit(self, X: Dict[str, Dict], y: Optional[Any] = None, **fit_params) -> 'SupervoxelAggregationStep':
        """
        Fit the step (stateless operation, just mark as fitted).
        
        Args:
            X: Dict of subject_id -> {
                'features': pd.DataFrame,
                'raw': pd.DataFrame,
                'mask_info': dict,
                'supervoxel_labels': np.ndarray
            }
            y: Optional target data (not used)
            **fit_params: Additional fitting parameters (not used)
            
        Returns:
            self
        """
        # Stateless step - no parameters to learn
        self.fitted_ = True
        return self
    
    def _aggregate_single_subject(
        self, 
        item: Tuple[str, Dict]
    ) -> Tuple[str, pd.DataFrame]:
        """
        Aggregate features for a single subject (wrapper for parallel processing).
        
        Args:
            item: Tuple of (subject_id, data dict)
            
        Returns:
            Tuple of (subject_id, aggregated DataFrame or Exception)
        """
        subject_id, data = item
        
        try:
            feature_df = data['features']
            raw_df = data['raw']
            supervoxel_labels = data['supervoxel_labels']
            
            # Get number of clusters
            unique_labels = np.unique(supervoxel_labels)
            n_clusters = len(unique_labels)
            
            # Calculate mean voxel features per supervoxel
            mean_features_df = self.feature_manager.calculate_supervoxel_means(
                subject_id, 
                feature_df, 
                raw_df, 
                supervoxel_labels, 
                n_clusters
            )
            
            # If Step 4 was executed, merge advanced features
            if 'supervoxel_features' in data:
                advanced_features_df = data['supervoxel_features'].copy()
                
                # Standardize column names
                if 'SupervoxelID' in advanced_features_df.columns and ResultColumns.SUPERVOXEL not in advanced_features_df.columns:
                    advanced_features_df[ResultColumns.SUPERVOXEL] = advanced_features_df['SupervoxelID']
                
                if ResultColumns.SUBJECT not in advanced_features_df.columns:
                    advanced_features_df[ResultColumns.SUBJECT] = subject_id
                
                # Merge
                merge_keys = [ResultColumns.SUBJECT, ResultColumns.SUPERVOXEL]
                if all(key in mean_features_df.columns for key in merge_keys):
                    if all(key in advanced_features_df.columns for key in merge_keys):
                        if 'SupervoxelID' in advanced_features_df.columns:
                            advanced_features_df = advanced_features_df.drop(columns=['SupervoxelID'])
                        
                        mean_features_df = mean_features_df.merge(
                            advanced_features_df,
                            on=merge_keys,
                            how='left',
                            suffixes=('', '_advanced')
                        )
                    else:
                        if self.config.verbose:
                            self.logger.warning(
                                f"Advanced features for {subject_id} don't have merge keys, "
                                "attempting index-based merge"
                            )
                        if len(mean_features_df) == len(advanced_features_df):
                            mean_features_df = pd.concat(
                                [mean_features_df.reset_index(drop=True), 
                                 advanced_features_df.reset_index(drop=True)], 
                                axis=1
                            )
            
            return subject_id, mean_features_df
            
        except Exception as e:
            self.logger.error(f"Error aggregating subject {subject_id}: {e}")
            return subject_id, e
    
    def transform(self, X: Dict[str, Dict]) -> pd.DataFrame:
        """
        Aggregate features to supervoxel level with parallel processing.
        
        Args:
            X: Dict of subject_id -> {
                'features': pd.DataFrame,
                'raw': pd.DataFrame,
                'mask_info': dict,
                'supervoxel_labels': np.ndarray,
                'supervoxel_features': pd.DataFrame (optional)
            }
            
        Returns:
            Combined DataFrame with supervoxel-level features for all subjects
        """
        # Get number of processes from config
        n_processes = getattr(self.config, 'processes', 1)
        
        # Prepare items for parallel processing
        items = [(subject_id, data) for subject_id, data in X.items()]
        
        # Process subjects in parallel
        successful_results, failed_subjects = parallel_map(
            func=self._aggregate_single_subject,
            items=items,
            n_processes=n_processes,
            desc="Aggregating supervoxel features",
            logger=self.logger,
            show_progress=True,
        )
        
        # Collect results
        all_supervoxel_features = []
        for proc_result in successful_results:
            # proc_result.item_id contains subject_id
            # proc_result.result contains result_df
            all_supervoxel_features.append(proc_result.result)
        
        # Log failed subjects
        if failed_subjects:
            self.logger.error(
                f"Failed to aggregate {len(failed_subjects)} subject(s): "
                f"{', '.join(str(s[0]) if isinstance(s, tuple) else str(s) for s in failed_subjects)}"
            )
        
        # Combine all subjects
        if not all_supervoxel_features:
            raise ValueError("No supervoxel features to aggregate")
        
        combined_df = pd.concat(all_supervoxel_features, ignore_index=True)
        
        return combined_df
