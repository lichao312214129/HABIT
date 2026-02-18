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
    
    processes: int = Field(
        2, 
        description="Number of parallel processes for individual-level steps. "
                    "Controls memory usage and processing speed. "
                    "Recommended: processes=2 (default, 1-2GB), processes=4 (2-4GB), "
                    "processes=8 (4-8GB). Reduce if memory is limited.", 
        gt=0
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
    """
    Settings for one-step clustering mode (voxel -> habitat directly).
    
    In one-step mode, each subject is clustered independently. You can either:
    1. Specify a fixed number of clusters (fixed_n_clusters)
    2. Let the algorithm automatically select optimal clusters (min/max_clusters + selection_method)
    """
    min_clusters: int = 2
    max_clusters: int = 10
    fixed_n_clusters: Optional[int] = Field(
        None,
        description="Fixed number of clusters for all subjects. If specified, automatic selection is disabled."
    )
    selection_method: Literal[
        'silhouette',
        'calinski_harabasz',
        'davies_bouldin',
        'inertia',
        'kneedle'
    ] = 'silhouette'
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
    fixed_n_clusters: Optional[int] = Field(
        None,
        description="Fixed number of habitat clusters. If specified, automatic selection is disabled."
    )
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

# -----------------------------------------------------------------------------
# Traditional Radiomics Extraction Schemas
# -----------------------------------------------------------------------------

class PathsConfig(BaseModel):
    """Paths configuration for radiomics extraction."""
    params_file: str = Field(..., description="Path to pyradiomics parameter file")
    images_folder: str = Field(..., description="Root directory containing images/ and masks/ subdirectories")
    out_dir: str = Field(..., description="Output directory for extracted features")

class ProcessingConfig(BaseModel):
    """Processing configuration for radiomics extraction."""
    n_processes: int = Field(2, description="Number of parallel processes", gt=0)
    save_every_n_files: int = Field(5, description="Save intermediate results every N files", gt=0)
    process_image_types: Optional[List[str]] = Field(None, description="List of image types to process (None = all)")
    target_labels: List[int] = Field(
        default_factory=lambda: [1],
        description="Mask labels to extract. Selected labels are merged into binary foreground."
    )

class ExportConfig(BaseModel):
    """Export configuration for radiomics extraction."""
    export_by_image_type: bool = Field(True, description="Export features by image type")
    export_combined: bool = Field(True, description="Export combined features")
    export_format: Literal['csv', 'json', 'pickle'] = Field('csv', description="Export format")
    add_timestamp: bool = Field(True, description="Add timestamp to output files")

class LoggingConfig(BaseModel):
    """Logging configuration for radiomics extraction."""
    level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = Field('INFO', description="Log level")
    console_output: bool = Field(True, description="Enable console output")
    file_output: bool = Field(True, description="Enable file output")

class RadiomicsConfig(BaseConfig):
    """Configuration for traditional radiomics feature extraction."""
    
    paths: PathsConfig = Field(..., description="Paths configuration")
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig, description="Processing configuration")
    export: ExportConfig = Field(default_factory=ExportConfig, description="Export configuration")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging configuration")
    
    # For backward compatibility, allow top-level params
    params_file: Optional[str] = Field(None, description="DEPRECATED: Use paths.params_file instead")
    images_folder: Optional[str] = Field(None, description="DEPRECATED: Use paths.images_folder instead")
    out_dir: Optional[str] = Field(None, description="DEPRECATED: Use paths.out_dir instead")
    n_processes: Optional[int] = Field(None, description="DEPRECATED: Use processing.n_processes instead")

# Update forward references
HabitatAnalysisConfig.model_rebuild()
FeatureConstructionConfig.model_rebuild()
FeatureExtractionConfig.model_rebuild()
RadiomicsConfig.model_rebuild()
