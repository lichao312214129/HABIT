# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text. Summary:
#
#   - Non-commercial use (academic, research, education, personal) is permitted
#     provided that copyright notices are retained and HABIT usage is
#     acknowledged in publications, reports, or documentation.
#   - Commercial use requires prior written consent from the copyright holder
#     (lichao19870617@163.com) and public acknowledgment of HABIT usage in
#     product documentation or user-facing materials.
#   - Unauthorized commercial use or removal of attribution is prohibited.
#
"""
Configuration Schemas for Habitat Analysis Workflows
Uses Pydantic for robust validation and type safety.
"""

from typing import List, Dict, Any, Optional, Union, Literal, FrozenSet
from pydantic import BaseModel, Field, model_validator, field_validator

from habit.core.common.configs.base import BaseConfig

# Preprocessing methods that DROP feature columns (variance / correlation filtering).
# Keep in sync with handlers that set ``changes_columns=True`` in builtin_methods.
DROPPING_PREPROCESSING_METHODS: FrozenSet[str] = frozenset({
    "variance_filter",
    "correlation_filter",
})

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
        description=(
            "Run mode for habitat analysis: 'train' or 'predict'. "
            "DEPRECATED for new code: prefer calling HabitatAnalysis.fit() / "
            ".predict() explicitly instead of relying on run_mode dispatch via "
            "HabitatAnalysis.run(). Kept for backward compatibility with the "
            "CLI and existing YAML configs."
        ),
    )
    pipeline_path: Optional[str] = Field(
        None,
        description=(
            "Path to a trained pipeline file used in predict mode. "
            "DEPRECATED for new code: prefer passing the path explicitly to "
            "HabitatAnalysis.predict(pipeline_path=...). Kept for backward "
            "compatibility with the CLI."
        ),
    )
    
    FeatureConstruction: Optional['FeatureConstructionConfig'] = Field(
        None,
        description="Feature construction configuration (required for train mode, optional for predict mode)."
    )
    HabitatSegmentation: Optional['HabitatSegmentationConfig'] = Field(
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
    cap_processes_to_gpu_pool: bool = Field(
        False,
        description=(
            "When True and Torch GPU radiomics is active, cap Stage-1 parallel workers "
            "to the configured torchGpus pool size (one worker slot per GPU). When False, "
            "keep the full processes count; multiple workers share GPUs via "
            "gpuSlotIndex modulo mapping so CPU-heavy steps can run in parallel on "
            "machines with fewer GPUs than CPU cores. May increase GPU memory contention."
        ),
    )
    individual_subject_timeout_sec: Optional[float] = Field(
        900.0,
        description=(
            "Wall-clock seconds allowed for each subject during the individual-level "
            "parallel stage before marking that subject as failed and continuing. "
            "Default 900 (15 minutes). Set to null in YAML to disable (no per-subject "
            "timeout). Must be positive when not null."
        ),
    )
    individual_subject_graceful_shutdown_sec: float = Field(
        15.0,
        description=(
            "Seconds to wait after terminate() before kill() when a subject exceeds "
            "individual_subject_timeout_sec. Applies to isolated per-subject child processes."
        ),
        gt=0,
    )
    individual_subject_spawn_timeout_sec: Optional[float] = Field(
        120.0,
        description=(
            "Wall-clock seconds allowed for a spawn child process to start before "
            "marking that subject as failed. Prevents the parent poll loop from "
            "blocking when startup imports hang under memory pressure. Set to null "
            "to disable spawn startup timeout."
        ),
    )
    on_subject_failure: Literal["continue", "fail_fast"] = Field(
        "continue",
        description=(
            "Individual-level parallel failure policy. 'continue': log failures and "
            "proceed with successful subjects when possible. 'fail_fast': abort the run "
            "if any subject fails or times out."
        ),
    )
    oom_backoff: bool = Field(
        True,
        description=(
            "When True, reduce individual-level parallel workers after a subject hits "
            "a fatal memory error (MemoryError) so pending subjects can still run."
        ),
    )
    oom_reduce_workers_by: int = Field(
        1,
        description=(
            "Number of parallel workers to subtract after each fatal memory error when "
            "oom_backoff is enabled. Minimum effective workers is always 1."
        ),
        ge=1,
    )
    resume: bool = Field(
        True,
        description=(
            "When True, skip individual-level processing for subjects already present "
            "in the checkpoint directory. Failed checkpoint subjects are skipped unless "
            "retry_failed_subjects is True or they appear in force_rerun_subjects. "
            "Applies to both train and predict runs."
        ),
    )
    strict_checkpoint_hash: bool = Field(
        False,
        description=(
            "When True with resume=True, raise an error instead of discarding the "
            "checkpoint when the manifest config hash or run_mode is incompatible "
            "with the current YAML. Legacy Stage-1-compatible manifests that only "
            "differ in group-stage settings still resume with a hash migration warning."
        ),
    )
    checkpoint_dir: Optional[str] = Field(
        None,
        description=(
            "Directory for train/predict checkpoints. Defaults to "
            "`<out_dir>/.habitat_checkpoint` for train and "
            "`<out_dir>/.habitat_predict_checkpoint` for predict when null."
        ),
    )
    force_rerun_subjects: List[str] = Field(
        default_factory=list,
        description=(
            "Subject IDs to reprocess even when resume=True and a checkpoint exists."
        ),
    )
    retry_failed_subjects: bool = Field(
        False,
        description=(
            "When True with resume=True, automatically re-queue every subject listed "
            "in the checkpoint manifest failed_subjects for individual-level processing. "
            "Successful subjects remain skipped unless also listed in force_rerun_subjects. "
            "Applies to both train and predict runs."
        ),
    )
    individual_subject_auto_retry_rounds: int = Field(
        2,
        description=(
            "After the initial individual-level parallel pass in a single train or "
            "predict run, automatically re-dispatch checkpoint failed subjects up to "
            "this many additional rounds (0 disables). Distinct from "
            "retry_failed_subjects, which only affects the next CLI invocation."
        ),
        ge=0,
    )
    individual_subject_parallel_mode: Literal["isolated", "persistent"] = Field(
        "persistent",
        description=(
            "Individual-level parallel execution strategy. 'persistent': one long-lived "
            "worker process per worker slot (default); reduces repeated import/spawn "
            "overhead. 'isolated': one spawn child process per subject."
        ),
    )
    persistent_worker_max_consecutive_failures: int = Field(
        1,
        description=(
            "When individual_subject_parallel_mode is 'persistent', reserved for "
            "fatal-class worker restarts. Recoverable subject failures (for example "
            "NaN validation errors) no longer restart the worker slot."
        ),
        ge=1,
    )
    persistent_worker_recycle_after_tasks: int = Field(
        0,
        description=(
            "When individual_subject_parallel_mode is 'persistent', restart a worker "
            "after this many consecutive successful tasks (0 disables periodic recycle)."
        ),
        ge=0,
    )
    clear_checkpoint_on_success: bool = Field(
        False,
        description=(
            "Remove the train/predict checkpoint directory after a successful run."
        ),
    )
    plot_curves: bool = Field(True, description="Whether to generate and save plots.")
    save_images: bool = Field(True, description="Whether to save any output images during runs.")
    save_results_csv: bool = Field(
        True,
        description="Whether to save the habitats results table to disk.",
    )
    habitats_results_format: Literal["parquet", "csv"] = Field(
        "parquet",
        description=(
            "On-disk format for the habitats results table when "
            "save_results_csv is true. Writes habitats.parquet or habitats.csv."
        ),
    )
    random_state: int = Field(42, description="Global random seed for reproducibility.")
    verbose: bool = Field(True, description="Whether to output detailed logs.")
    debug: bool = Field(False, description="Enable debug mode for verbose logging.")
    
    @field_validator('individual_subject_timeout_sec')
    @classmethod
    def validate_individual_subject_timeout(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v <= 0:
            raise ValueError(
                "individual_subject_timeout_sec must be positive when set; "
                "use null in YAML to disable per-subject timeout."
            )
        return v

    @field_validator('individual_subject_spawn_timeout_sec')
    @classmethod
    def validate_individual_subject_spawn_timeout(
        cls,
        v: Optional[float],
    ) -> Optional[float]:
        if v is not None and v <= 0:
            raise ValueError(
                "individual_subject_spawn_timeout_sec must be positive when set; "
                "use null in YAML to disable spawn startup timeout."
            )
        return v

    @model_validator(mode='after')
    def validate_mode_dependent_fields(self):
        """
        Validate that required fields are present based on run_mode.
        
        - In train mode: FeatureConstruction and HabitatSegmentation are required
        - In predict mode: FeatureConstruction is optional, but HabitatSegmentation.clustering_mode is needed
        """
        if self.run_mode == 'train':
            if self.FeatureConstruction is None:
                raise ValueError("FeatureConstruction is required in train mode")
            if self.HabitatSegmentation is None:
                raise ValueError("HabitatSegmentation is required in train mode")
        elif self.run_mode == 'predict':
            # In predict mode, FeatureConstruction is optional (not used)
            # But HabitatSegmentation.clustering_mode is needed to select the strategy class
            if self.HabitatSegmentation is None or self.HabitatSegmentation.clustering_mode is None:
                raise ValueError(
                    "HabitatSegmentation.clustering_mode is required in predict mode "
                    "to select the correct strategy class. "
                    "You can provide a minimal config with only clustering_mode, e.g.:\n"
                    "HabitatSegmentation:\n"
                    "  clustering_mode: one_step  # or two_step, direct_pooling"
                )

        # Guardrail: in two-step mode, subject-level feature-dropping filters
        # can produce inconsistent columns across subjects, which may introduce
        # heavy NaN after cross-subject concatenation.
        if (
            self.HabitatSegmentation is not None
            and self.HabitatSegmentation.clustering_mode == 'two_step'
            and self.FeatureConstruction is not None
            and self.FeatureConstruction.preprocessing_for_subject_level is not None
        ):
            subject_methods = self.FeatureConstruction.preprocessing_for_subject_level.methods
            dropping_methods = {
                method.method
                for method in subject_methods
                if method.method in DROPPING_PREPROCESSING_METHODS
            }
            if dropping_methods:
                methods_text = ", ".join(sorted(dropping_methods))
                raise ValueError(
                    "Subject-level feature-dropping methods are not allowed in two_step mode: "
                    f"{methods_text}. "
                    "Please move these methods to preprocessing_for_group_level."
                )
        return self

    def effective_supervoxel_random_state(self) -> int:
        """
        Resolve the ``supervoxel`` block seed (two_step supervoxel clustering).

        For per-subject clustering in any mode, prefer
        :meth:`effective_individual_clustering_random_state`.

        Returns:
            int: Effective random seed from ``HabitatSegmentation.supervoxel``.
        """
        from habit.utils.random_utils import resolve_random_state

        explicit: Optional[int] = None
        if self.HabitatSegmentation is not None:
            explicit = self.HabitatSegmentation.supervoxel.random_state
        return resolve_random_state(explicit, self.random_state)

    def effective_habitat_random_state(self) -> int:
        """
        Resolve group-level habitat clustering seed (two_step / direct_pooling).

        Returns:
            int: Effective random seed for population / group habitat clustering.
        """
        from habit.utils.random_utils import resolve_random_state

        explicit: Optional[int] = None
        if self.HabitatSegmentation is not None:
            explicit = self.HabitatSegmentation.habitat.random_state
        return resolve_random_state(explicit, self.random_state)

    def effective_individual_clustering_random_state(self) -> int:
        """
        Resolve the seed for per-subject voxel-level clustering.

        Mode-specific priority:
        - ``one_step`` (voxel -> habitat per subject):
          ``habitat.random_state`` > ``supervoxel.random_state`` > top-level
        - ``two_step`` (voxel -> supervoxel per subject):
          ``supervoxel.random_state`` > top-level

        ``direct_pooling`` does not run individual clustering; this method is
        unused there but still resolves consistently if called.

        Returns:
            int: Effective random seed for individual-level clustering steps.
        """
        from habit.utils.random_utils import resolve_random_state_chain

        if self.HabitatSegmentation is None:
            return resolve_random_state_chain(global_seed=self.random_state)

        seg = self.HabitatSegmentation
        if seg.clustering_mode == "one_step":
            return resolve_random_state_chain(
                seg.habitat.random_state,
                seg.supervoxel.random_state,
                global_seed=self.random_state,
            )
        return resolve_random_state_chain(
            seg.supervoxel.random_state,
            global_seed=self.random_state,
        )

    def effective_clustering_plot_random_state(
        self,
        scope: Literal["individual", "group"],
    ) -> int:
        """
        Resolve the random seed used for clustering scatter / t-SNE plots.

        Plot seeds follow the same clustering scope so figures stay aligned
        with the clustering step that produced the labels.

        Args:
            scope: ``individual`` for per-subject plots; ``group`` for
                population-level habitat plots.

        Returns:
            int: Effective plot random seed.
        """
        if scope == "individual":
            return self.effective_individual_clustering_random_state()
        return self.effective_habitat_random_state()

# -----------------------------------------------------------------------------
# Feature Construction Schemas
# -----------------------------------------------------------------------------

class VoxelLevelConfig(BaseModel):
    method: str = Field(..., description="Feature extraction method expression for voxels.")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the voxel-level feature extractor.")

def _validate_use_supervoxel_cext_value(value: object) -> None:
    """
    Validate ``useSupervoxelCext`` when present under ``supervoxel_level.params``.

    Args:
        value: Raw YAML value (bool or str).

    Raises:
        ValueError: When the value is not ``auto`` / ``true`` / ``false`` (case-insensitive).
    """
    if value is True or value is False:
        return
    if isinstance(value, str) and value.lower() in ("auto", "true", "false"):
        return
    raise ValueError(
        "FeatureConstruction.supervoxel_level.params.useSupervoxelCext must be "
        "auto, true, or false (bool or str)."
    )


class SupervoxelLevelConfig(BaseModel):
    supervoxel_file_keyword: str = Field("*_supervoxel.nrrd", description="Glob pattern to find supervoxel files.")
    method: str = Field("mean_voxel_features()", description="Aggregation method for supervoxel features.")
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Parameters for the supervoxel-level feature aggregator. "
            "For supervoxel_radiomics, habit keys include params_file, supervoxelBatch "
            "(default 64), supervoxelUnionBboxCrop (default true), supervoxelPadDistance, "
            "useSupervoxelCext (default auto: native C extension when built; false forces "
            "Torch/PyRadiomics stacked-matrix path), useTorchRadiomics, torchGpus, "
            "torchGpuCount, torchDevice, and torchDtype (torch keys may inherit from "
            "voxel_level.params)."
        ),
    )

    @field_validator("params")
    @classmethod
    def validate_supervoxel_params(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        """Reject invalid useSupervoxelCext values early at config load time."""
        if not value:
            return value
        flag = value.get("useSupervoxelCext")
        if flag is not None:
            _validate_use_supervoxel_cext_value(flag)
        return value

class PreprocessingMethod(BaseModel):
    method: Literal[
        'winsorize',
        'minmax',
        'zscore',
        'robust',
        'log',
        'binning',
        'variance_filter',
        'correlation_filter'
    ]
    global_normalize: bool = False
    winsor_limits: Optional[List[float]] = None
    n_bins: Optional[int] = None
    bin_strategy: Optional[Literal['uniform', 'quantile', 'kmeans']] = None
    variance_threshold: Optional[float] = None
    corr_threshold: Optional[float] = None
    corr_method: Optional[Literal['pearson', 'spearman', 'kendall']] = None

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
        'kneedle',
        'elbow',
        'gap',
        'aic',
        'bic',
    ] = 'elbow'
    plot_validation_curves: bool = True

class ConnectedComponentPostprocessConfig(BaseModel):
    """
    Connected-component post-processing settings for label-map cleanup.
    """
    enabled: bool = False
    min_component_size: int = Field(
        30,
        ge=1,
        description="Minimum connected-component size in voxels. Smaller components are reassigned."
    )
    connectivity: Literal[1, 2, 3] = Field(
        1,
        description="Neighborhood connectivity: 1(6-neighbor), 2(18-neighbor), 3(26-neighbor)."
    )
    reassign_method: Literal['neighbor_vote'] = Field(
        'neighbor_vote',
        description="Strategy to reassign tiny components."
    )
    max_iterations: int = Field(
        3,
        ge=1,
        description="Maximum cleanup iterations."
    )

class SupervoxelClusteringConfig(BaseModel):
    algorithm: Literal['kmeans', 'gmm', 'slic'] = 'kmeans'
    n_clusters: int = 50
    random_state: Optional[int] = Field(
        None,
        description=(
            "Random seed for two_step supervoxel clustering and one_step fallback "
            "when habitat.random_state is omitted. Inherits "
            "HabitatAnalysisConfig.random_state when null."
        ),
    )
    max_iter: int = 300
    n_init: int = 10
    compactness: float = Field(
        0.1,
        description="SLIC compactness factor balancing feature similarity and spatial proximity."
    )
    sigma: float = Field(
        0.0,
        description="Gaussian smoothing width used by SLIC before segmentation."
    )
    enforce_connectivity: bool = Field(
        True,
        description="Whether SLIC should enforce connected components."
    )
    one_step_settings: OneStepSettings = Field(default_factory=OneStepSettings)

class HabitatClusteringConfig(BaseModel):
    algorithm: Literal['kmeans', 'gmm'] = 'kmeans'
    max_clusters: int = 10
    min_clusters: Optional[int] = 2
    habitat_cluster_selection_method: Union[str, List[str]] = 'elbow'
    fixed_n_clusters: Optional[int] = Field(
        None,
        description="Fixed number of habitat clusters. If specified, automatic selection is disabled."
    )
    random_state: Optional[int] = Field(
        None,
        description=(
            "Random seed for group-level habitat clustering (two_step / "
            "direct_pooling) and one_step per-subject voxel->habitat clustering. "
            "Inherits HabitatAnalysisConfig.random_state when null."
        ),
    )
    max_iter: int = 300
    n_init: int = 10
    parallel_cluster_search: bool = Field(
        True,
        description=(
            "When True, evaluate candidate habitat cluster counts in parallel "
            "for direct_pooling and two_step group-level clustering."
        ),
    )
    cluster_search_workers: Optional[int] = Field(
        None,
        ge=1,
        description=(
            "Worker processes for parallel cluster-count search. "
            "None uses 2."
        ),
    )

class HabitatSegmentationConfig(BaseModel):
    clustering_mode: Literal['one_step', 'two_step', 'direct_pooling'] = 'two_step'
    supervoxel: SupervoxelClusteringConfig = Field(default_factory=SupervoxelClusteringConfig)
    habitat: HabitatClusteringConfig = Field(default_factory=HabitatClusteringConfig)
    postprocess_supervoxel: ConnectedComponentPostprocessConfig = Field(
        default_factory=ConnectedComponentPostprocessConfig
    )
    postprocess_habitat: ConnectedComponentPostprocessConfig = Field(
        default_factory=ConnectedComponentPostprocessConfig
    )

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
