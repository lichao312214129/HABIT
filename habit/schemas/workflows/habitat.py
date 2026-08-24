# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Configuration Schemas for Habitat Analysis Workflows
Uses Pydantic for robust validation and type safety.
"""

from typing import List, Dict, Any, Optional, Union, Literal, FrozenSet
from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator

from habit.schemas.base import BaseConfig

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
    
    feature_construction: Optional['FeatureConstructionConfig'] = Field(
        None,
        description="Feature construction configuration (required for train mode, optional for predict mode)."
    )
    habitat_segmentation: Optional['HabitatSegmentationConfig'] = Field(
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
            "to the configured torch_gpus pool size (one worker slot per GPU). When False, "
            "keep the full processes count; multiple workers share GPUs via "
            "gpu_slot_index modulo mapping so CPU-heavy steps can run in parallel on "
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
        
        - In train mode: feature_construction and habitat_segmentation are required
        - In predict mode: feature_construction is optional, but habitat_segmentation.clustering_mode is needed
        """
        if self.run_mode == 'train':
            if self.feature_construction is None:
                raise ValueError("feature_construction is required in train mode")
            if self.habitat_segmentation is None:
                raise ValueError("habitat_segmentation is required in train mode")
        elif self.run_mode == 'predict':
            # In predict mode, feature_construction is optional (not used)
            # But habitat_segmentation.clustering_mode is needed to select the strategy class
            if self.habitat_segmentation is None or self.habitat_segmentation.clustering_mode is None:
                raise ValueError(
                    "habitat_segmentation.clustering_mode is required in predict mode "
                    "to select the correct strategy class. "
                    "You can provide a minimal config with only clustering_mode, e.g.:\n"
                    "habitat_segmentation:\n"
                    "  clustering_mode: one_step  # or two_step, direct_pooling"
                )

        # Guardrail: in two-step mode, subject-level feature-dropping filters
        # can produce inconsistent columns across subjects, which may introduce
        # heavy NaN after cross-subject concatenation.
        if (
            self.habitat_segmentation is not None
            and self.habitat_segmentation.clustering_mode == 'two_step'
            and self.feature_construction is not None
            and self.feature_construction.preprocessing_for_subject_level is not None
        ):
            subject_methods = self.feature_construction.preprocessing_for_subject_level.methods
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
        ``effective_individual_clustering_random_state`` instead.

        Returns:
            int: Effective random seed from ``habitat_segmentation.supervoxel``.
        """
        from habit.utils.random_utils import resolve_random_state

        explicit: Optional[int] = None
        if self.habitat_segmentation is not None:
            explicit = self.habitat_segmentation.supervoxel.random_state
        return resolve_random_state(explicit, self.random_state)

    def effective_habitat_random_state(self) -> int:
        """
        Resolve group-level habitat clustering seed (two_step / direct_pooling).

        Returns:
            int: Effective random seed for population / group habitat clustering.
        """
        from habit.utils.random_utils import resolve_random_state

        explicit: Optional[int] = None
        if self.habitat_segmentation is not None:
            explicit = self.habitat_segmentation.habitat.random_state
        return resolve_random_state(explicit, self.random_state)

    def effective_individual_clustering_random_state(self) -> int:
        """Resolve the seed for per-subject voxel-level clustering.

        In ``one_step`` mode the priority is ``habitat.random_state``, then
        ``supervoxel.random_state``, then the top-level ``random_state``. In
        ``two_step`` mode the priority is ``supervoxel.random_state``, then the
        top-level seed. ``direct_pooling`` does not run individual clustering;
        this method is unused there but still resolves consistently if called.

        Returns:
            int: Effective random seed for individual-level clustering steps.
        """
        from habit.utils.random_utils import resolve_random_state_chain

        if self.habitat_segmentation is None:
            return resolve_random_state_chain(global_seed=self.random_state)

        seg = self.habitat_segmentation
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
    Validate ``use_supervoxel_cext`` when present under ``supervoxel_level.params``.

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
        "feature_construction.supervoxel_level.params.use_supervoxel_cext must be "
        "auto, true, or false (bool or str)."
    )


class SupervoxelLevelConfig(BaseModel):
    supervoxel_file_keyword: str = Field("*_supervoxel.nrrd", description="Glob pattern to find supervoxel files.")
    method: str = Field("mean_voxel_features()", description="Aggregation method for supervoxel features.")
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Parameters for the supervoxel-level feature aggregator. "
            "For supervoxel_radiomics, habit keys include params_file, supervoxel_batch "
            "(default 64), supervoxel_union_bbox_crop (default true), supervoxel_pad_distance, "
            "union_bin (default false: per-label binWidth matching execute(); true = one "
            "shared union-mask bin), use_supervoxel_cext (default auto: native C extension "
            "when built; false forces Torch/PyRadiomics stacked-matrix path), "
            "use_torch_radiomics, torch_gpus, torch_gpu_count, torch_device, and "
            "torch_dtype (torch keys may inherit from voxel_level.params)."
        ),
    )

    @field_validator("params")
    @classmethod
    def validate_supervoxel_params(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        """Reject invalid use_supervoxel_cext values early at config load time."""
        if not value:
            return value
        flag = value.get("use_supervoxel_cext")
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
            "None uses max(1, cpu_count - 4)."
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
    SUBJECT = "subject"
    SUPERVOXEL = "supervoxel"
    COUNT = "count"
    HABITATS = "habitats"
    
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

class GraphFeatureBlock(BaseModel):
    """Settings of the optional top-level ``graph:`` block in the
    feature-extraction YAML.

    The extraction fields mirror
    :class:`habit.domain.habitat_features.GraphHabitatFeaturesParams`
    field-for-field (a regression test guards the correspondence); the schema
    lives here rather than in the domain layer because it is part of the YAML
    configuration surface, not of the extractor contract. The
    ``visualization_*`` fields are consumed by the L4 recipe's figure hook and
    never reach the domain extractor, so the extractor's spec fingerprint
    stays extraction-only. ``enabled`` / ``n_workers`` are tolerated legacy
    keys from the v0.1 plugin block: family activation is governed by
    ``feature_types`` and figure rendering runs in the main process.
    """

    model_config = ConfigDict(extra="forbid")

    # --- Extraction parameters (mirror GraphHabitatFeaturesParams) ---------
    include_single_habitat_graph: bool = Field(
        True, description="Compute within-habitat region graphs per habitat label."
    )
    include_pairwise_habitat_graph: bool = Field(
        True, description="Compute pairwise inter-habitat region graphs."
    )
    edge_method: Literal["centroid_distance", "adjacency", "min_distance"] = Field(
        "min_distance",
        description=(
            "Rule used to identify graph edges. Default 'min_distance': "
            "connect regions whose closest voxels are within "
            "distance_threshold. 'adjacency' uses contact voxels."
        ),
    )
    distance_threshold: float = Field(
        5.0,
        ge=0.0,
        description=(
            "Distance threshold in voxel-index units. Used by "
            "centroid_distance (centroid-to-centroid) and min_distance "
            "(closest-voxel)."
        ),
    )
    adjacency_connectivity: Literal["face", "edge", "corner"] = Field(
        "corner",
        description=(
            "Neighbor definition for the 'adjacency' edge method. Default "
            "'corner' = 8-connectivity in 2D / 26-connectivity in 3D. "
            "'face' = 4/6, 'edge' = 8/18."
        ),
    )
    adjacency_min_voxels: int = Field(
        10,
        ge=1,
        description=(
            "Minimum number of adjacent voxel pairs required to create an "
            "edge when edge_method is 'adjacency'. Default 10: an edge exists "
            "only when two regions are adjacent and the contact voxel count "
            "is >= 10."
        ),
    )
    edge_weight: Literal["none", "distance", "inverse_distance", "contact_voxels"] = Field(
        "none", description="Optional edge weight source."
    )
    min_region_voxels: int = Field(
        1, ge=1, description="Minimum connected-region size retained as a graph node."
    )
    connectivity: Literal["face", "full"] = Field(
        "full",
        description=(
            "Connected-component neighborhood rule. Default 'full' = "
            "8-connectivity in 2D / 26-connectivity in 3D. Pass 'face' "
            "for 4/6-connectivity."
        ),
    )
    erosion_radius: int = Field(
        0,
        ge=0,
        description=(
            "Binary erosion iterations applied before component labeling. "
            "Default 0 (off): adjacency and contact are measured on the "
            "habitat labels as drawn. Set a positive value to shrink each "
            "habitat before edges."
        ),
    )
    node_method: Literal["uniform_grid", "component"] = Field(
        "uniform_grid",
        description=(
            "How voxels become graph nodes. Default 'uniform_grid': "
            "equal-volume cubes on a global VOI lattice. 'component' uses "
            "connected components, optionally split when larger than "
            "subdivide_region_voxels."
        ),
    )
    subdivide_region_voxels: int = Field(
        1000,
        ge=0,
        description=(
            "In component mode, split connected components larger than this "
            "voxel count into grid blocks. Set 0 to disable. Ignored by "
            "uniform_grid."
        ),
    )
    block_size: int = Field(
        8,
        ge=1,
        description=(
            "Cube edge length in voxels (default 8), not millimetres. "
            "Paired with distance_threshold=5: face-adjacent 8-cubes "
            "connect; one empty lattice cell (distance about 8) stays "
            "disconnected."
        ),
    )
    block_min_coverage: float = Field(
        0.2,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum covered fraction of a block volume to keep it as a "
            "node (strictly greater than this value; default 0.2)."
        ),
    )
    pairwise_include_intra_edges: bool = Field(
        True,
        description=(
            "Add same-habitat proximity edges to pairwise graphs so whole-graph "
            "metrics reflect real tissue organization; interface metrics still "
            "use inter-class edges only."
        ),
    )
    include_extended_metrics: bool = Field(
        True,
        description=(
            "Compute extended graph metrics: global/local efficiency, "
            "small-world sigma, rich-club coefficient, and node-level "
            "distribution summaries. Default true (analytic Humphries "
            "ER sigma). Set false to omit the extra columns."
        ),
    )
    extended_min_nodes: int = Field(
        10,
        ge=3,
        description=(
            "Minimum node count in the analysis subgraph required to compute "
            "either small-world sigma; smaller graphs return 0 for that metric."
        ),
    )
    small_world_nrand: int = Field(
        100,
        ge=1,
        description=(
            "Number of degree-preserving null graphs when "
            "graph_null_sampler is config or rewire (default 100). "
            "Ignored by the default analytic Humphries S."
        ),
    )
    small_world_niter: int = Field(
        100,
        ge=1,
        description=(
            "Rewires per edge when graph_null_sampler is rewire "
            "(NetworkX / Milo default 100). Ignored by config and by ER."
        ),
    )
    rich_club_q: int = Field(
        100,
        ge=1,
        description=(
            "Mixing floor for graph_null_sampler=rewire. Rich-club "
            "phi_rand is the mean over the same nrand ensemble, not "
            "the number of null graphs."
        ),
    )
    graph_null_sampler: Literal["analytic", "config", "rewire"] = Field(
        "analytic",
        description=(
            "Small-world null. analytic (default) is Humphries ER S. "
            "config / rewire replace that one column with a "
            "degree-preserving ensemble."
        ),
    )
    graph_null_device: str = Field(
        "auto",
        description=(
            "Batched C/L device for the null ensemble: auto, cpu, cuda, "
            "or cuda:N. auto uses CUDA only when the Floyd-Warshall "
            "work is large enough."
        ),
    )
    # --- Figure rendering (consumed by the L4 recipe, not the extractor) ---
    visualize: bool = Field(
        False,
        description=(
            "Render per-subject habitat graph topology figures under "
            "<out_dir>/visualizations/graph/ after the CSV export."
        ),
    )
    visualization_format: Literal["png", "pdf", "both"] = Field(
        "both",
        description=(
            "File format for the 2D figures; 'both' writes PNG and PDF. "
            "3D renders are raster-only and always written as PNG."
        ),
    )
    visualization_dpi: int = Field(
        600, gt=0, description="Raster resolution (DPI) for saved figures."
    )
    visualization_show_background: bool = Field(
        True,
        description="Draw the faint habitat partitions behind 2D network graphs.",
    )
    visualization_show_grid: bool = Field(
        True,
        description=(
            "Draw the same uniform-grid lattice as dashed lines on 2D "
            "habitat / network figures (default on)."
        ),
    )
    visualization_block_size: Optional[int] = Field(
        default=None,
        description=(
            "Cube edge length drawn on 2D lattice figures. None uses the "
            "extraction block_size (library default 8 voxels) so the overlay "
            "matches the nodes."
        ),
    )
    visualization_grid_linestyle: str = Field(
        "--",
        description=(
            "Matplotlib linestyle for the display lattice (default dashed)."
        ),
    )
    visualization_save_3d: bool = Field(
        True,
        description=(
            "Also render 3D surface / network views for 3D habitat maps. "
            "Requires the optional pyvista stack; missing dependencies skip "
            "3D rendering with a warning instead of failing the run."
        ),
    )

    # --- Tolerated legacy keys from the v0.1 plugin block ------------------
    enabled: bool = Field(
        True,
        description=(
            "Legacy v0.1 plugin switch, accepted for config compatibility. "
            "Has no effect: the graph family runs when 'graph' is listed in "
            "feature_types."
        ),
    )
    n_workers: int = Field(
        1,
        ge=1,
        description=(
            "Legacy v0.1 rendering parallelism knob, accepted for config "
            "compatibility. Has no effect: figure rendering runs serially in "
            "the main process after the CSV export."
        ),
    )

    @field_validator("visualization_block_size")
    @classmethod
    def _visualization_block_size_positive(
        cls, value: Optional[int]
    ) -> Optional[int]:
        """Reject a non-positive display cube size (``None`` stays allowed)."""
        if value is not None and int(value) < 1:
            raise ValueError("visualization_block_size must be >= 1.")
        return value


class FeatureExtractionConfig(BaseConfig):
    """Configuration for habitat feature extraction workflow."""
    
    params_file_of_non_habitat: Optional[str] = Field(
        None,
        description=(
            "Path to radiomics params file for original images. Optional: when "
            "omitted, HABIT falls back to the bundled 'roi' preset. Accepts an "
            "'@preset:<key>' reference too."
        ),
    )
    params_file_of_habitat: Optional[str] = Field(
        None,
        description=(
            "Path to radiomics params file for habitat maps. Optional: when "
            "omitted, HABIT falls back to the bundled 'habitat' preset. Accepts "
            "an '@preset:<key>' reference too."
        ),
    )
    
    raw_img_folder: str = Field(..., description="Directory containing raw images")
    habitats_map_folder: str = Field(..., description="Directory containing habitat maps")
    out_dir: str = Field(..., description="Output directory for extracted features")
    
    n_processes: int = Field(4, description="Number of parallel processes")
    habitat_pattern: str = Field("*_habitats.nrrd", description="Glob pattern for habitat files")
    
    feature_types: List[str] = Field(..., description="List of feature types to extract")
    n_habitats: Optional[int] = Field(None, description="Number of habitats (auto-detected if None)")

    use_torch_radiomics: Union[str, bool] = Field(
        False,
        description=(
            "TorchRadiomics backend for traditional / whole_habitat / each_habitat. "
            "false (default) keeps CPU PyRadiomics; auto uses torch when CUDA is "
            "available; true forces torch. Does not change bin width or feature "
            "classes. Supervoxel cext belongs to get-habitat, not extract."
        ),
    )
    torch_device: str = Field(
        "auto",
        description="Torch device for habitat radiomics when use_torch_radiomics is enabled.",
    )
    torch_dtype: str = Field(
        "float32",
        description="Torch dtype for habitat radiomics when use_torch_radiomics is enabled.",
    )

    debug: bool = Field(False, description="Enable debug mode")

# -----------------------------------------------------------------------------
# Traditional Radiomics Extraction Schemas
# -----------------------------------------------------------------------------

class PathsConfig(BaseModel):
    """Paths configuration for radiomics extraction."""
    params_file: Optional[str] = Field(
        None,
        description=(
            "Path to PyRadiomics parameter file. Optional: when omitted, HABIT "
            "falls back to the bundled 'roi' preset (habit/resources/radiomics/"
            "parameter.yaml). Accepts an '@preset:<key>' reference too."
        ),
    )
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
