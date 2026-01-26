"""
Subject-level preprocessing step for habitat analysis pipeline.

This step applies preprocessing at the individual subject level (stateless).
"""

from typing import Dict, Any, Optional, Tuple
import pandas as pd
import logging

from ..base_pipeline import BasePipelineStep
from ...managers.feature_manager import FeatureManager
from habit.utils.parallel_utils import parallel_map


class SubjectPreprocessingStep(BasePipelineStep):
    """
    Subject-level preprocessing (stateless).
    
    Each subject is preprocessed independently using its own statistics.
    No state needs to be saved between training and testing.
    
    Attributes:
        feature_manager: FeatureManager instance for preprocessing
        fitted_: bool indicating whether the step has been fitted (always True after fit)
    """
    
    def __init__(self, feature_manager: FeatureManager):
        """
        Initialize subject preprocessing step.
        
        Args:
            feature_manager: FeatureManager instance
        """
        super().__init__()
        self.feature_manager = feature_manager
        self.logger = logging.getLogger(__name__)
    
    def fit(self, X: Dict[str, Dict], y: Optional[Any] = None, **fit_params) -> 'SubjectPreprocessingStep':
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
        # Each subject will be preprocessed independently in transform()
        self.fitted_ = True
        return self
    
    def _preprocess_single_subject(
        self, 
        item: Tuple[str, Dict]
    ) -> Tuple[str, Dict]:
        """
        Preprocess a single subject (wrapper for parallel processing).
        
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
            
            # Apply subject-level preprocessing
            processed_features = self.feature_manager.apply_preprocessing(
                feature_df, 
                level='subject'
            )
            
            # Clean features (handle inf, nan)
            processed_features = self.feature_manager.clean_features(processed_features)
            
            result = {
                'features': processed_features,
                'raw': raw_df,
                'mask_info': mask_info
            }
            
            return subject_id, result
            
        except Exception as e:
            self.logger.error(f"Error preprocessing subject {subject_id}: {e}")
            return subject_id, e
    
    def transform(self, X: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Apply subject-level preprocessing to each subject with parallel processing.
        
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
                'mask_info': dict
            }
        """
        # Get number of processes from config
        n_processes = getattr(self.feature_manager.config, 'processes', 1)
        
        # Prepare items for parallel processing
        items = [(subject_id, data) for subject_id, data in X.items()]
        
        # Process subjects in parallel
        successful_results, failed_subjects = parallel_map(
            func=self._preprocess_single_subject,
            items=items,
            n_processes=n_processes,
            desc="Preprocessing subjects",
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
                f"Failed to preprocess {len(failed_subjects)} subject(s): "
                f"{', '.join(str(s[0]) if isinstance(s, tuple) else str(s) for s in failed_subjects)}"
            )
        
        return results
