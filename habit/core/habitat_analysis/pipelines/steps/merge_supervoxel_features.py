"""
Merge supervoxel features step for habitat analysis pipeline.

This step selects which supervoxel features to use for group-level clustering:
- Mean voxel features (averaged within each supervoxel)
- OR advanced supervoxel features (shape, texture, radiomics)
- NOT both (mutually exclusive)
"""

from typing import Dict, Any, Optional
import pandas as pd
import logging

from ..base_pipeline import IndividualLevelStep
from ...config_schemas import HabitatAnalysisConfig, ResultColumns


class MergeSupervoxelFeaturesStep(IndividualLevelStep):
    """
    Select supervoxel features for group-level clustering.
    
    This step decides which type of supervoxel features to use based on configuration:
    
    **Mode 1: Use mean voxel features** (default)
    - Uses averaged voxel features within each supervoxel
    - Fast and simple
    - Always available (from CalculateMeanVoxelFeaturesStep)
    
    **Mode 2: Use advanced supervoxel features**
    - Uses shape, texture, or radiomics features
    - More informative but slower
    - Only available if SupervoxelFeatureExtractionStep was executed
    
    **Important**: The two modes are MUTUALLY EXCLUSIVE. You can only use one type.
    
    **Configuration**:
    ```yaml
    FeatureConstruction:
      supervoxel_level:
        method: mean_voxel_features()  # Mode 1
        # OR
        method: supervoxel_radiomics()  # Mode 2
    ```
    
    **Individual-level step**: Processes each subject independently.
    
    Attributes:
        config: Configuration object
        fitted_: bool indicating whether the step has been fitted
    """
    
    def __init__(self, config: HabitatAnalysisConfig):
        """
        Initialize merge supervoxel features step.
        
        Args:
            config: Configuration object
        """
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Determine which feature type to use
        supervoxel_config = config.FeatureConstruction.supervoxel_level
        method = supervoxel_config.method if supervoxel_config else None
        
        # Check if advanced features are requested
        self.use_advanced_features = (
            method is not None and 
            'mean_voxel_features' not in method
        )
        
        if self.use_advanced_features:
            self.logger.info("Will use ADVANCED supervoxel features for clustering")
        else:
            self.logger.info("Will use MEAN VOXEL features for clustering")
    
    def fit(self, X: Dict[str, Dict], y: Optional[Any] = None, **fit_params) -> 'MergeSupervoxelFeaturesStep':
        """
        Fit the step (stateless operation, just mark as fitted).
        
        Args:
            X: Dict of subject_id -> {
                'mean_voxel_features': pd.DataFrame (always present),
                'supervoxel_features': pd.DataFrame (optional)
            }
            y: Optional target data (not used)
            **fit_params: Additional fitting parameters (not used)
            
        Returns:
            self
        """
        # Validate that required features are available
        if self.use_advanced_features:
            # Check if at least one subject has advanced features
            has_advanced = any('supervoxel_features' in data for data in X.values())
            if not has_advanced:
                raise ValueError(
                    "Configuration requests advanced supervoxel features, but "
                    "SupervoxelFeatureExtractionStep was not executed or failed. "
                    "Make sure supervoxel_level.method is configured correctly."
                )
        
        self.fitted_ = True
        return self
    
    def transform(self, X: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Select appropriate supervoxel features for each subject.
        
        **Decision logic**:
        - If use_advanced_features=True: Use 'supervoxel_features'
        - If use_advanced_features=False: Use 'mean_voxel_features'
        
        Args:
            X: Dict of subject_id -> {
                'mean_voxel_features': pd.DataFrame,
                'supervoxel_features': pd.DataFrame (optional),
                ... (other fields)
            }
            
        Returns:
            Dict of subject_id -> {
                'supervoxel_df': pd.DataFrame  # Selected features
            }
        """
        results = {}
        
        for subject_id, data in X.items():
            try:
                if self.use_advanced_features:
                    # Mode 2: Use advanced features
                    if 'supervoxel_features' not in data:
                        raise ValueError(
                            f"Subject {subject_id} missing 'supervoxel_features'. "
                            "SupervoxelFeatureExtractionStep must be executed first."
                        )
                    
                    supervoxel_df = data['supervoxel_features'].copy()
                    
                    # Ensure standard column names
                    if 'SupervoxelID' in supervoxel_df.columns and ResultColumns.SUPERVOXEL not in supervoxel_df.columns:
                        supervoxel_df[ResultColumns.SUPERVOXEL] = supervoxel_df['SupervoxelID']
                        supervoxel_df = supervoxel_df.drop(columns=['SupervoxelID'])
                    
                    if ResultColumns.SUBJECT not in supervoxel_df.columns:
                        supervoxel_df[ResultColumns.SUBJECT] = subject_id
                    
                    if self.config.verbose:
                        self.logger.info(
                            f"Subject {subject_id}: Using ADVANCED features "
                            f"({len(supervoxel_df)} supervoxels, {len(supervoxel_df.columns)} features)"
                        )
                
                else:
                    # Mode 1: Use mean voxel features
                    if 'mean_voxel_features' not in data:
                        raise ValueError(
                            f"Subject {subject_id} missing 'mean_voxel_features'. "
                            "CalculateMeanVoxelFeaturesStep must be executed first."
                        )
                    
                    supervoxel_df = data['mean_voxel_features'].copy()
                    
                    if self.config.verbose:
                        self.logger.info(
                            f"Subject {subject_id}: Using MEAN VOXEL features "
                            f"({len(supervoxel_df)} supervoxels, {len(supervoxel_df.columns)} features)"
                        )
                
                # Store result with unified key name
                results[subject_id] = {
                    'supervoxel_df': supervoxel_df
                }
                
            except Exception as e:
                self.logger.error(f"Error selecting supervoxel features for subject {subject_id}: {e}")
                raise
        
        return results
