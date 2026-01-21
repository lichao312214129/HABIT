"""
Configuration Schemas for Habitat Analysis Workflows
Uses Pydantic for robust validation and type safety.
"""

from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# General/Root Configuration
# -----------------------------------------------------------------------------

class HabitatAnalysisConfig(BaseModel):
    """Root model for the entire habitat analysis configuration."""
    data_dir: str = Field(..., description="Path to the input data directory or a file list YAML.")
    out_dir: str = Field(..., description="Path to the output directory for results.")
    config_file: Optional[str] = Field(None, description="Path to original config file.")
    
    FeatureConstruction: 'FeatureConstructionConfig'
    HabitatsSegmention: 'HabitatsSegmentionConfig'
    
    processes: int = Field(2, description="Number of parallel processes to use.", gt=0)
    plot_curves: bool = Field(True, description="Whether to generate and save plots.")
    random_state: int = Field(42, description="Global random seed for reproducibility.")
    verbose: bool = Field(True, description="Whether to output detailed logs.")
    debug: bool = Field(False, description="Enable debug mode for verbose logging.")

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
    supervoxel_level: SupervoxelLevelConfig
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
    mode: Literal['training', 'testing'] = 'training'
    algorithm: Literal['kmeans', 'gmm'] = 'kmeans'
    max_clusters: int = 10
    min_clusters: Optional[int] = 2
    habitat_cluster_selection_method: Union[str, List[str]] = 'inertia'
    best_n_clusters: Optional[int] = None
    random_state: int = 42
    max_iter: int = 300
    n_init: int = 10

class HabitatsSegmentionConfig(BaseModel):
    clustering_mode: Literal['one_step', 'two_step'] = 'two_step'
    supervoxel: SupervoxelClusteringConfig = Field(default_factory=SupervoxelClusteringConfig)
    habitat: HabitatClusteringConfig = Field(default_factory=HabitatClusteringConfig)

# Update forward references
HabitatAnalysisConfig.model_rebuild()
FeatureConstructionConfig.model_rebuild()
