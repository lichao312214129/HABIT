"""
Configuration Schemas for Habitat Analysis Workflows
Uses Pydantic for robust validation and type safety.
"""

from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field, model_validator

from habit.core.common.config_base import BaseConfig

# -----------------------------------------------------------------------------
# General/Root Configuration
# -----------------------------------------------------------------------------

class HabitatAnalysisConfig(BaseConfig):
    """Root model for the entire habitat analysis configuration."""
    data_dir: str = Field(..., description="Path to the input data directory or a file list YAML.")
    out_dir: str = Field(..., description="Path to the output directory for results.")
    config_file: Optional[str] = Field(None, description="Path to original config file.")
    run_mode: Literal['train', 'predict'] = Field(
        'train',
        description="Run mode for habitat analysis: train or predict."
    )
    pipeline_path: Optional[str] = Field(
        None,
        description="Path to a trained pipeline file used in predict mode."
    )
    
    FeatureConstruction: Optional['FeatureConstructionConfig'] = Field(
        None,
        description="Feature construction configuration (required for train mode, optional for predict mode)."
    )
    HabitatsSegmention: Optional['HabitatsSegmentionConfig'] = Field(
        None,
        description="Habitat segmentation configuration (required for train mode, optional for predict mode but clustering_mode is needed)."
    )
    
    processes: int = Field(2, description="Number of parallel processes to use.", gt=0)
    use_streaming_pipeline: bool = Field(
        True, 
        description="Use streaming pipeline for memory-efficient processing. "
                    "Processes subjects in batches to reduce memory usage. "
                    "Set to False to use standard pipeline (all subjects in memory)."
    )
    streaming_batch_size: int = Field(
        10, 
        description="Batch size for streaming pipeline (number of subjects per batch). "
                    "Higher values = faster but more memory. "
                    "Lower values = slower but less memory. "
                    "Set to 0 to disable batching (process all subjects at once).",
        ge=0
    )
    plot_curves: bool = Field(True, description="Whether to generate and save plots.")
    save_images: bool = Field(True, description="Whether to save any output images during runs.")
    save_results_csv: bool = Field(True, description="Whether to save results as CSV files.")
    random_state: int = Field(42, description="Global random seed for reproducibility.")
    verbose: bool = Field(True, description="Whether to output detailed logs.")
    debug: bool = Field(False, description="Enable debug mode for verbose logging.")
    
    @model_validator(mode='after')
    def validate_mode_dependent_fields(self):
        """
        Validate that required fields are present based on run_mode.
        
        - In train mode: FeatureConstruction and HabitatsSegmention are required
        - In predict mode: FeatureConstruction is optional, but HabitatsSegmention.clustering_mode is needed
        """
        if self.run_mode == 'train':
            if self.FeatureConstruction is None:
                raise ValueError("FeatureConstruction is required in train mode")
            if self.HabitatsSegmention is None:
                raise ValueError("HabitatsSegmention is required in train mode")
        elif self.run_mode == 'predict':
            # In predict mode, FeatureConstruction is optional (not used)
            # But HabitatsSegmention.clustering_mode is needed to select the strategy class
            if self.HabitatsSegmention is None or self.HabitatsSegmention.clustering_mode is None:
                raise ValueError(
                    "HabitatsSegmention.clustering_mode is required in predict mode "
                    "to select the correct strategy class. "
                    "You can provide a minimal config with only clustering_mode, e.g.:\n"
                    "HabitatsSegmention:\n"
                    "  clustering_mode: one_step  # or two_step, direct_pooling"
                )
        return self

# -----------------------------------------------------------------------------
# Feature Construction Schemas
# -----------------------------------------------------------------------------

class VoxelLevelConfig(BaseModel):
    method: str = Field(..., description="Feature extraction method expression for voxels.")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the voxel-level feature extractor.")

class SupervoxelLevelConfig(BaseModel):
    supervoxel_file_keyword: str = Field("*_supervoxel.nrrd", description="Glob pattern to find supervoxel files.")
    method: str = Field("mean_voxel_features()", description="Aggregation method for supervoxel features.")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the supervoxel-level feature aggregator.")

class PreprocessingMethod(BaseModel):
    method: Literal['winsorize', 'minmax', 'zscore', 'robust', 'log', 'binning']
    global_normalize: bool = False
    winsor_limits: Optional[List[float]] = None
    n_bins: Optional[int] = None
    bin_strategy: Optional[Literal['uniform', 'quantile', 'kmeans']] = None

class PreprocessingConfig(BaseModel):
    methods: List[PreprocessingMethod] = Field(default_factory=list)

class FeatureConstructionConfig(BaseModel):
    voxel_level: VoxelLevelConfig
    supervoxel_level: Optional[SupervoxelLevelConfig] = None
    preprocessing_for_subject_level: Optional[PreprocessingConfig] = None
    preprocessing_for_group_level: Optional[PreprocessingConfig] = None
    

# -----------------------------------------------------------------------------
# Habitat Segmentation Schemas
# -----------------------------------------------------------------------------

class OneStepSettings(BaseModel):
    min_clusters: int = 2
    max_clusters: int = 10
    selection_method: Literal['silhouette', 'calinski_harabasz', 'davies_bouldin', 'inertia'] = 'silhouette'
    plot_validation_curves: bool = True

class SupervoxelClusteringConfig(BaseModel):
    algorithm: Literal['kmeans', 'gmm'] = 'kmeans'
    n_clusters: int = 50
    random_state: int = 42
    max_iter: int = 300
    n_init: int = 10
    one_step_settings: OneStepSettings = Field(default_factory=OneStepSettings)

class HabitatClusteringConfig(BaseModel):
    algorithm: Literal['kmeans', 'gmm'] = 'kmeans'
    max_clusters: int = 10
    min_clusters: Optional[int] = 2
    habitat_cluster_selection_method: Union[str, List[str]] = 'inertia'
    best_n_clusters: Optional[int] = None
    random_state: int = 42
    max_iter: int = 300
    n_init: int = 10

class HabitatsSegmentionConfig(BaseModel):
    clustering_mode: Literal['one_step', 'two_step', 'direct_pooling'] = 'two_step'
    supervoxel: SupervoxelClusteringConfig = Field(default_factory=SupervoxelClusteringConfig)
    habitat: HabitatClusteringConfig = Field(default_factory=HabitatClusteringConfig)

# -----------------------------------------------------------------------------
# Result Column Names
# -----------------------------------------------------------------------------

class ResultColumns:
    """
    Centralized column name definitions for pipeline outputs.
    
    This avoids magic strings across the codebase and keeps feature/metadata
    column handling consistent in all pipeline steps and managers.
    """
    SUBJECT = "Subject"
    SUPERVOXEL = "Supervoxel"
    COUNT = "Count"
    HABITATS = "Habitats"
    
    # Suffix for original (non-processed) feature columns
    ORIGINAL_SUFFIX = "-original"
    
    @classmethod
    def metadata_columns(cls) -> List[str]:
        """
        Return list of metadata column names (non-feature columns).
        
        Returns:
            List[str]: Columns that are metadata and should not be treated as features
        """
        return [cls.SUBJECT, cls.SUPERVOXEL, cls.COUNT]
    
    @classmethod
    def is_feature_column(cls, col_name: str) -> bool:
        """
        Check if a column name represents a feature (not metadata).
        
        Args:
            col_name: Column name to check
        
        Returns:
            bool: True if the column is a feature column
        """
        return (
            col_name not in cls.metadata_columns() and 
            not col_name.endswith(cls.ORIGINAL_SUFFIX)
        )

# -----------------------------------------------------------------------------
# Habitat Feature Extraction Schemas
# -----------------------------------------------------------------------------

class FeatureExtractionConfig(BaseConfig):
    """Configuration for habitat feature extraction workflow."""
    
    params_file_of_non_habitat: str = Field(..., description="Path to radiomics params file for original images")
    params_file_of_habitat: str = Field(..., description="Path to radiomics params file for habitat maps")
    
    raw_img_folder: str = Field(..., description="Directory containing raw images")
    habitats_map_folder: str = Field(..., description="Directory containing habitat maps")
    out_dir: str = Field(..., description="Output directory for extracted features")
    
    n_processes: int = Field(4, description="Number of parallel processes")
    habitat_pattern: str = Field("*_habitats.nrrd", description="Glob pattern for habitat files")
    
    feature_types: List[str] = Field(..., description="List of feature types to extract")
    n_habitats: Optional[int] = Field(None, description="Number of habitats (auto-detected if None)")
    
    debug: bool = Field(False, description="Enable debug mode")

# Update forward references
HabitatAnalysisConfig.model_rebuild()
FeatureConstructionConfig.model_rebuild()
FeatureExtractionConfig.model_rebuild()
