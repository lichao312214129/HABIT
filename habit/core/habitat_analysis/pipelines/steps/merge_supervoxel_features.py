"""
Merge supervoxel features step for habitat analysis pipeline.

This step selects which supervoxel features to use for group-level clustering:
- Mean voxel features (averaged within each supervoxel)
- OR advanced supervoxel features (shape, texture, radiomics)
- NOT both (mutually exclusive)
"""

from typing import Dict, Any, Optional
import numpy as np
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
    
    def fit(
        self,
        X: Dict[str, Any],
        y: Optional[Any] = None,
        **fit_params,
    ) -> 'MergeSupervoxelFeaturesStep':
        """
        Validate that the upstream advanced-feature step actually produced
        results when the configuration asks for advanced features.

        This is the only individual-level step that needs an explicit fit
        beyond the base class default — it is the right place to fail fast
        before the parallel transform fans out across workers.
        """
        if self.use_advanced_features:
            has_advanced = any('supervoxel_features' in data for data in X.values())
            if not has_advanced:
                raise ValueError(
                    "Configuration requests advanced supervoxel features, but "
                    "SupervoxelFeatureExtractionStep was not executed or failed. "
                    "Make sure supervoxel_level.method is configured correctly."
                )
        self.fitted_ = True
        return self

    def transform_one(self, subject_id: str, subject_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pick advanced or mean-voxel supervoxel features for one subject and
        normalise them into the ``supervoxel_df`` contract consumed by
        :class:`CombineSupervoxelsStep`.
        """
        if self.use_advanced_features:
            supervoxel_df = self._build_advanced_supervoxel_df(subject_id, subject_data)
            label = "ADVANCED"
        else:
            supervoxel_df = self._build_mean_supervoxel_df(subject_id, subject_data)
            label = "MEAN VOXEL"

        # In one_step mode the per-subject clustering already produced habitat
        # labels (the Supervoxel column IS the habitat label). Mirror it into
        # Habitats here so downstream code does not have to special-case mode.
        clustering_mode = (
            self.config.HabitatsSegmention.clustering_mode
            if self.config.HabitatsSegmention is not None
            else None
        )
        if (
            clustering_mode == 'one_step'
            and ResultColumns.SUPERVOXEL in supervoxel_df.columns
            and ResultColumns.HABITATS not in supervoxel_df.columns
        ):
            supervoxel_df[ResultColumns.HABITATS] = supervoxel_df[ResultColumns.SUPERVOXEL]

        if self.config.verbose:
            true_feature_columns = sum(
                ResultColumns.is_feature_column(col) for col in supervoxel_df.columns
            )
            self.logger.info(
                f"Subject {subject_id}: Using {label} features "
                f"({len(supervoxel_df)} supervoxels, "
                f"feature columns: {true_feature_columns})"
            )

        return {'supervoxel_df': supervoxel_df}

    def _build_advanced_supervoxel_df(
        self,
        subject_id: str,
        subject_data: Dict[str, Any],
    ) -> pd.DataFrame:
        if 'supervoxel_features' not in subject_data:
            raise ValueError(
                f"Subject {subject_id} missing 'supervoxel_features'. "
                "SupervoxelFeatureExtractionStep must be executed first."
            )

        supervoxel_df = subject_data['supervoxel_features'].copy()

        if (
            'SupervoxelID' in supervoxel_df.columns
            and ResultColumns.SUPERVOXEL not in supervoxel_df.columns
        ):
            supervoxel_df[ResultColumns.SUPERVOXEL] = supervoxel_df['SupervoxelID']
            supervoxel_df = supervoxel_df.drop(columns=['SupervoxelID'])

        if ResultColumns.SUBJECT not in supervoxel_df.columns:
            supervoxel_df[ResultColumns.SUBJECT] = subject_id

        # Voxel-count column for parity with the mean_voxel_features path.
        if (
            ResultColumns.COUNT not in supervoxel_df.columns
            and 'supervoxel_labels' in subject_data
        ):
            labels = np.asarray(subject_data['supervoxel_labels']).ravel()
            labels_in_mask = labels[labels > 0]
            unique, counts = np.unique(labels_in_mask, return_counts=True)
            count_series = pd.Series(counts, index=unique)
            supervoxel_df[ResultColumns.COUNT] = (
                supervoxel_df[ResultColumns.SUPERVOXEL].map(count_series).values
            )

        # Coerce any object-typed feature columns to numeric so downstream
        # statistics / clustering do not blow up on stray strings.
        non_metadata_cols = [
            col for col in supervoxel_df.columns
            if col not in (ResultColumns.SUBJECT, ResultColumns.SUPERVOXEL, ResultColumns.COUNT)
        ]
        for col in non_metadata_cols:
            supervoxel_df[col] = pd.to_numeric(supervoxel_df[col], errors='coerce')

        return supervoxel_df

    def _build_mean_supervoxel_df(
        self,
        subject_id: str,
        subject_data: Dict[str, Any],
    ) -> pd.DataFrame:
        if 'mean_voxel_features' not in subject_data:
            raise ValueError(
                f"Subject {subject_id} missing 'mean_voxel_features'. "
                "CalculateMeanVoxelFeaturesStep must be executed first."
            )
        return subject_data['mean_voxel_features'].copy()
