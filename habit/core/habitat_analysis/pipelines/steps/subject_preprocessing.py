"""
Subject-level preprocessing step for habitat analysis pipeline.

This step applies preprocessing at the individual subject level (stateless).
"""

from typing import Dict, Any, Optional
import pandas as pd
import logging

from ..base_pipeline import IndividualLevelStep
from ...managers.feature_manager import FeatureManager


class SubjectPreprocessingStep(IndividualLevelStep):
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
    
    def transform(self, X: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Apply subject-level preprocessing to each subject sequentially.
        
        Pipeline handles parallelization at subject level, so this method
        processes subjects sequentially without parallel logic.
        
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
        results = {}
        
        # Process each subject sequentially (pipeline handles parallelization)
        for subject_id, data in X.items():
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
                
                results[subject_id] = {
                    'features': processed_features,
                    'raw': raw_df,
                    'mask_info': mask_info
                }
                
            except Exception as e:
                self.logger.error(f"Error preprocessing subject {subject_id}: {e}")
                raise
        
        return results
