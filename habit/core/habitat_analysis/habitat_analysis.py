"""
Habitat Analysis - Deep module unifying clustering-mode dispatch, train/predict
orchestration, and pipeline lifecycle (build / fit / save / load / transform).

Design notes (V1 architecture):

This module collapses three previously separate layers into a single deep module:

    HabitatAnalysis(controller)  -> ClusteringStrategy subclass  -> pipeline_builder

The Strategy layer and the pipeline_builder layer have been removed in V1.
All real behaviour lives here:

- ``clustering_mode`` (``one_step`` / ``two_step`` / ``direct_pooling``) is
  branched in exactly one place: ``_build_pipeline``.
- Train and predict orchestration are exposed as ``fit()`` / ``predict()``
  with ``run()`` kept as a backward-compat alias that dispatches based on
  ``load_from`` or ``config.run_mode``.
- Manager injection into a loaded pipeline is performed via an explicit
  attribute whitelist (``_PIPELINE_MANAGER_ATTRS``). The old reflection-based
  ``dir()`` discovery is gone (it would silently absorb any future ``*_manager``
  attribute and inject it into every step).
- predict-mode skips ``plot_habitat_scores`` so it can never consume a
  ``selection_methods`` value that was never initialised in predict mode.
"""

import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Suppress noisy library warnings during long-running runs.
warnings.simplefilter('ignore')

from habit.utils.io_utils import get_image_and_mask_paths
from habit.utils.log_utils import LoggerManager

from .config_schemas import HabitatAnalysisConfig, ResultColumns
from .managers import ClusteringManager, FeatureManager, ResultManager
from .pipelines.base_pipeline import BasePipelineStep, HabitatPipeline
from .pipelines.steps.calculate_mean_voxel_features import CalculateMeanVoxelFeaturesStep
from .pipelines.steps.combine_supervoxels import CombineSupervoxelsStep
from .pipelines.steps.concatenate_voxels import ConcatenateVoxelsStep
from .pipelines.steps.group_preprocessing import GroupPreprocessingStep
from .pipelines.steps.individual_clustering import IndividualClusteringStep
from .pipelines.steps.merge_supervoxel_features import MergeSupervoxelFeaturesStep
from .pipelines.steps.population_clustering import PopulationClusteringStep
from .pipelines.steps.subject_preprocessing import SubjectPreprocessingStep
from .pipelines.steps.supervoxel_feature_extraction import SupervoxelFeatureExtractionStep
from .pipelines.steps.voxel_feature_extractor import VoxelFeatureExtractor


# Module-level helper used by the result post-processing path to enforce a
# stable column order on the habitats.csv output.
def _canonical_csv_column_order(df: pd.DataFrame) -> List[str]:
    """
    Return habitats.csv column order: metadata columns first, features after.

    Args:
        df: Result DataFrame produced by the pipeline.

    Returns:
        List of column names in canonical order, preserving any unknown columns
        in their current order after the metadata block.
    """
    meta_order = [
        ResultColumns.SUBJECT,
        ResultColumns.SUPERVOXEL,
        ResultColumns.HABITATS,
        ResultColumns.COUNT,
    ]
    meta_cols = [c for c in meta_order if c in df.columns]
    other_cols = [c for c in df.columns if c not in meta_cols]
    return meta_cols + other_cols


# =============================================================================
# Pipeline recipes (mode -> step list).
#
# These are module-level helpers, looked up via ``_PIPELINE_RECIPES``. Mode
# branching is contained here (one recipe per mode) and in
# ``HabitatAnalysis._build_pipeline`` (one dictionary lookup).
# =============================================================================

def _build_two_step_steps(
    config: HabitatAnalysisConfig,
    feature_manager: FeatureManager,
    clustering_manager: ClusteringManager,
    result_manager: ResultManager,
) -> List[Tuple[str, BasePipelineStep]]:
    """
    Build the ordered step list for the ``two_step`` clustering mode.

    Two-step flow: voxel features -> subject preprocessing -> individual
    clustering (voxel -> supervoxel) -> mean voxel features -> optional advanced
    supervoxel features -> select supervoxel features -> combine subjects ->
    optional group-level preprocessing -> population clustering.

    Args:
        config: Validated habitat-analysis configuration.
        feature_manager: Provides voxel/supervoxel feature extraction.
        clustering_manager: Provides clustering algorithms and validation.
        result_manager: Required for saving supervoxel maps in two-step mode.

    Returns:
        List of (name, step) tuples to feed into ``HabitatPipeline``.
    """
    if result_manager is None:
        raise ValueError("result_manager is required for two-step pipeline")

    steps: List[Tuple[str, BasePipelineStep]] = []

    steps.append((
        'voxel_features',
        VoxelFeatureExtractor(feature_manager, result_manager),
    ))

    steps.append((
        'subject_preprocessing',
        SubjectPreprocessingStep(feature_manager),
    ))

    # Individual-level clustering: voxel -> supervoxel (one cluster set per subject).
    steps.append((
        'individual_clustering',
        IndividualClusteringStep(
            feature_manager=feature_manager,
            clustering_manager=clustering_manager,
            result_manager=result_manager,
            config=config,
            target='supervoxel',
        ),
    ))

    # Always produce mean-voxel features as a baseline aggregation per supervoxel.
    steps.append((
        'calculate_mean_voxel_features',
        CalculateMeanVoxelFeaturesStep(feature_manager, config),
    ))

    # Conditionally extract advanced supervoxel features (e.g. radiomics) when
    # the configured method is not the default ``mean_voxel_features``.
    supervoxel_config = config.FeatureConstruction.supervoxel_level
    method = supervoxel_config.method if supervoxel_config else None
    should_extract_advanced = (
        method is not None
        and 'mean_voxel_features' not in method
    )
    if should_extract_advanced:
        steps.append((
            'supervoxel_advanced_features',
            SupervoxelFeatureExtractionStep(feature_manager, config),
        ))

    # Choose between mean-voxel and advanced supervoxel features (mutually exclusive).
    steps.append((
        'merge_supervoxel_features',
        MergeSupervoxelFeaturesStep(config),
    ))

    # Group-level: stitch all subjects' supervoxels into one DataFrame.
    steps.append((
        'combine_supervoxels',
        CombineSupervoxelsStep(),
    ))

    # Optional group-level preprocessing (variance/correlation/normalisation).
    if config.FeatureConstruction.preprocessing_for_group_level:
        methods = config.FeatureConstruction.preprocessing_for_group_level.methods
        if methods:
            steps.append((
                'group_preprocessing',
                GroupPreprocessingStep(
                    methods=methods,
                    out_dir=config.out_dir,
                ),
            ))

    # Population clustering: supervoxel -> habitat.
    steps.append((
        'population_clustering',
        PopulationClusteringStep(
            clustering_manager=clustering_manager,
            config=config,
            out_dir=config.out_dir,
        ),
    ))

    return steps


def _build_one_step_steps(
    config: HabitatAnalysisConfig,
    feature_manager: FeatureManager,
    clustering_manager: ClusteringManager,
    result_manager: Optional[ResultManager],
) -> List[Tuple[str, BasePipelineStep]]:
    """
    Build the ordered step list for the ``one_step`` clustering mode.

    One-step flow: voxel features -> subject preprocessing -> individual
    clustering (voxel -> habitat, per subject) -> mean voxel features ->
    select supervoxel features -> combine subjects.

    Args:
        config: Validated habitat-analysis configuration.
        feature_manager: Provides voxel feature extraction.
        clustering_manager: Provides clustering algorithms and validation.
        result_manager: Optional; created on-the-fly if missing because the
            individual-clustering step needs it for saving habitat maps.

    Returns:
        List of (name, step) tuples to feed into ``HabitatPipeline``.
    """
    steps: List[Tuple[str, BasePipelineStep]] = []

    steps.append((
        'voxel_features',
        VoxelFeatureExtractor(feature_manager, result_manager),
    ))

    steps.append((
        'subject_preprocessing',
        SubjectPreprocessingStep(feature_manager),
    ))

    # Lazily build a minimal ResultManager when one was not injected.
    # IndividualClusteringStep with target='habitat' needs it for saving maps.
    if result_manager is None:
        result_manager = ResultManager(config, logging.getLogger(__name__))

    # In one-step we cluster voxels directly into habitats per subject and we
    # ask the step to find an optimal cluster count for each subject.
    steps.append((
        'individual_clustering',
        IndividualClusteringStep(
            feature_manager=feature_manager,
            clustering_manager=clustering_manager,
            result_manager=result_manager,
            config=config,
            target='habitat',
            find_optimal=True,
        ),
    ))

    # Even in one-step we want aggregated per-label features for downstream output.
    steps.append((
        'calculate_mean_voxel_features',
        CalculateMeanVoxelFeaturesStep(feature_manager, config),
    ))

    steps.append((
        'merge_supervoxel_features',
        MergeSupervoxelFeaturesStep(config),
    ))

    # Group-level concatenation across subjects so the downstream CSV layout
    # is consistent with the two-step mode.
    steps.append((
        'combine_supervoxels',
        CombineSupervoxelsStep(),
    ))

    return steps


def _build_pooling_steps(
    config: HabitatAnalysisConfig,
    feature_manager: FeatureManager,
    clustering_manager: ClusteringManager,
    result_manager: Optional[ResultManager],
) -> List[Tuple[str, BasePipelineStep]]:
    """
    Build the ordered step list for the ``direct_pooling`` clustering mode.

    Direct-pooling flow: voxel features -> subject preprocessing -> concatenate
    voxels across subjects -> optional group-level preprocessing -> population
    clustering on the pooled voxels.

    Args:
        config: Validated habitat-analysis configuration.
        feature_manager: Provides voxel feature extraction.
        clustering_manager: Provides clustering algorithms and validation.
        result_manager: Unused for the pooling flow but accepted for parity
            with other recipes.

    Returns:
        List of (name, step) tuples to feed into ``HabitatPipeline``.
    """
    steps: List[Tuple[str, BasePipelineStep]] = []

    steps.append((
        'voxel_features',
        VoxelFeatureExtractor(feature_manager, result_manager),
    ))

    steps.append((
        'subject_preprocessing',
        SubjectPreprocessingStep(feature_manager),
    ))

    steps.append((
        'concatenate_voxels',
        ConcatenateVoxelsStep(),
    ))

    if config.FeatureConstruction.preprocessing_for_group_level:
        methods = config.FeatureConstruction.preprocessing_for_group_level.methods
        if methods:
            steps.append((
                'group_preprocessing',
                GroupPreprocessingStep(
                    methods=methods,
                    out_dir=config.out_dir,
                ),
            ))

    steps.append((
        'population_clustering',
        PopulationClusteringStep(
            clustering_manager=clustering_manager,
            config=config,
            out_dir=config.out_dir,
        ),
    ))

    return steps


# Map clustering_mode -> recipe builder. The single source of truth for the
# mode dispatch (the rest of the module branches via ``config.HabitatsSegmention
# .clustering_mode`` only for save-side variations, which are tiny).
_PIPELINE_RECIPES = {
    'two_step': _build_two_step_steps,
    'one_step': _build_one_step_steps,
    'direct_pooling': _build_pooling_steps,
}


# =============================================================================
# HabitatAnalysis - the deep module
# =============================================================================

class HabitatAnalysis:
    """
    Run habitat clustering analysis end-to-end.

    The class owns three responsibilities that used to be spread across three
    layers (HabitatAnalysis controller, ClusteringStrategy subclass tree, and
    ``pipeline_builder``):

    1. Build a ``HabitatPipeline`` whose step list depends on
       ``config.HabitatsSegmention.clustering_mode``.
    2. Train (``fit``) -> persist as ``habitat_pipeline.pkl`` -> return results.
    3. Load a saved pipeline (``predict``) -> reconcile runtime config and
       data paths -> transform -> return results.

    Public entry points:

    - ``fit(subjects=None, save_results_csv=None)``: train and persist.
    - ``predict(pipeline_path, subjects=None, save_results_csv=None)``: load
      and transform only.
    - ``run(subjects=None, save_results_csv=None, load_from=None)``: backward
      compatible dispatcher (delegates to ``fit`` / ``predict``).

    All three return the resulting ``pd.DataFrame``.

    Notes:
        Manager injection into a loaded pipeline uses the explicit whitelist
        ``_PIPELINE_MANAGER_ATTRS``. Adding a new manager type requires a
        deliberate change here, which is precisely the goal: new managers
        should not be silently injected via reflection.
    """

    # Explicit whitelist of manager attribute names that are injected into a
    # loaded pipeline's steps. Replaces the previous ``dir(self)`` reflection
    # which would absorb any future ``*_manager`` attribute. If a new manager
    # is introduced, it must be added here on purpose.
    _PIPELINE_MANAGER_ATTRS: Tuple[str, ...] = (
        'feature_manager',
        'clustering_manager',
        'result_manager',
    )

    def __init__(
        self,
        config: Union[Dict[str, Any], HabitatAnalysisConfig],
        feature_manager: FeatureManager,
        clustering_manager: ClusteringManager,
        result_manager: ResultManager,
        logger: Any,
    ) -> None:
        """
        Initialize HabitatAnalysis with all required collaborators.

        Args:
            config: Either a dict (validated against ``HabitatAnalysisConfig``)
                or an already-validated ``HabitatAnalysisConfig`` instance.
            feature_manager: Manager for voxel/supervoxel feature extraction.
            clustering_manager: Manager for clustering algorithms.
            result_manager: Manager for result persistence.
            logger: A configured logger instance (typically from CLI entry point).

        Raises:
            TypeError: If ``config`` is neither dict nor ``HabitatAnalysisConfig``.
        """
        if isinstance(config, HabitatAnalysisConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = HabitatAnalysisConfig.model_validate(config)
        else:
            raise TypeError("config must be a dict or HabitatAnalysisConfig")

        self.feature_manager = feature_manager
        self.clustering_manager = clustering_manager
        self.result_manager = result_manager
        self.logger = logger

        # Pipeline instance is populated by either ``fit`` or ``predict``.
        self.pipeline: Optional[HabitatPipeline] = None

        self._setup_logging_info()
        self._setup_data_paths()

        # Propagate logging targets to managers that need to log from subprocesses.
        self.feature_manager.set_logging_info(self._log_file_path, self._log_level)
        self.result_manager.set_logging_info(self._log_file_path, self._log_level)

    # ------------------------------------------------------------------
    # Setup helpers (kept private; ``__init__`` calls them in order).
    # ------------------------------------------------------------------

    def _setup_logging_info(self) -> None:
        """
        Resolve the log file path and effective log level for subprocesses.

        Order of preference: LoggerManager-managed file -> logger's own
        ``log_file`` attribute -> default file inside ``config.out_dir``.
        """
        manager = LoggerManager()

        log_file = manager.get_log_file()
        if log_file:
            self._log_file_path = log_file
        elif hasattr(self.logger, 'log_file'):
            self._log_file_path = self.logger.log_file
        else:
            self._log_file_path = os.path.join(self.config.out_dir, 'habitat_analysis.log')

        if manager._root_logger:
            self._log_level = manager._root_logger.getEffectiveLevel()
        else:
            self._log_level = logging.INFO

    def _setup_data_paths(self) -> None:
        """
        Ensure ``config.out_dir`` exists and pass image/mask paths to FeatureManager.
        """
        os.makedirs(self.config.out_dir, exist_ok=True)
        images_paths, mask_paths = get_image_and_mask_paths(self.config.data_dir)
        self._validate_data_paths(images_paths, mask_paths)
        self.feature_manager.set_data_paths(images_paths, mask_paths)

    def _validate_data_paths(
        self,
        images_paths: Dict[str, Any],
        mask_paths: Dict[str, Any],
    ) -> None:
        """
        Validate discovered image and mask paths before the pipeline starts.

        Args:
            images_paths: Mapping from subject ID to image-type path mapping.
            mask_paths: Mapping from subject ID to mask-type path mapping.

        Raises:
            ValueError: If no usable image or mask paths are discovered.
        """
        if not images_paths:
            raise ValueError(
                f"No image paths found under data_dir: {self.config.data_dir}. "
                "Expected an 'images' folder or an image/mask YAML file."
            )

        if not mask_paths:
            raise ValueError(
                f"No mask paths found under data_dir: {self.config.data_dir}. "
                "Expected a non-empty 'masks' folder or a YAML file with a 'masks' section."
            )

        subjects_without_masks = sorted(
            subject for subject in images_paths
            if subject not in mask_paths or not mask_paths[subject]
        )
        if subjects_without_masks:
            missing_subjects = ", ".join(subjects_without_masks)
            raise ValueError(
                "Mask paths are missing for subject(s): "
                f"{missing_subjects}. Please check the masks folder or data YAML."
            )

    # ------------------------------------------------------------------
    # Public entry points.
    # ------------------------------------------------------------------

    def fit(
        self,
        subjects: Optional[List[str]] = None,
        save_results_csv: Optional[bool] = None,
    ) -> pd.DataFrame:
        """
        Train the habitat clustering pipeline and persist it to disk.

        Workflow: build pipeline based on ``clustering_mode`` -> ``fit_transform``
        -> save pkl to ``<out_dir>/habitat_pipeline.pkl`` -> post-process and
        save results.

        Args:
            subjects: Optional explicit subject ID list. ``None`` means use all
                subjects discovered from ``config.data_dir``.
            save_results_csv: Whether to save ``habitats.csv``. ``None`` falls
                back to ``config.save_results_csv``.

        Returns:
            DataFrame with habitat clustering results.
        """
        save_results_csv = self._resolve_save_flag(save_results_csv)
        subjects_list = self._prepare_subjects(subjects)
        X = self._build_input(subjects_list)
        Path(self.config.out_dir).mkdir(parents=True, exist_ok=True)

        if self.config.verbose:
            self.logger.info(
                f"Building and fitting {self.config.HabitatsSegmention.clustering_mode} pipeline..."
            )

        self.pipeline = self._build_pipeline()
        results_df = self.pipeline.fit_transform(X)

        pipeline_path = self._resolve_pipeline_path(None)
        if self.config.verbose:
            self.logger.info(f"Saving fitted pipeline to {pipeline_path}")
        self.pipeline.save(str(pipeline_path))

        return self._finalise_results(results_df, save_results_csv, mode='train')

    def predict(
        self,
        pipeline_path: Union[str, Path],
        subjects: Optional[List[str]] = None,
        save_results_csv: Optional[bool] = None,
    ) -> pd.DataFrame:
        """
        Load a previously trained pipeline and transform new subjects.

        ``plot_curves`` is forced off in predict mode regardless of config:
        in predict mode the population-clustering step has no validation
        scores cached, and emitting them would consume an uninitialised
        ``selection_methods`` (this used to be a real ``TypeError`` trigger).

        Args:
            pipeline_path: Path to a saved ``habitat_pipeline.pkl`` file.
            subjects: Optional explicit subject ID list.
            save_results_csv: Whether to save ``habitats.csv``.

        Returns:
            DataFrame with habitat clustering results.

        Raises:
            FileNotFoundError: If ``pipeline_path`` does not exist.
        """
        resolved_path = Path(pipeline_path)
        if not resolved_path.exists():
            raise FileNotFoundError(
                f"Saved pipeline not found at {resolved_path}. "
                "Provide a valid path or run fit() to train one."
            )

        save_results_csv = self._resolve_save_flag(save_results_csv)
        subjects_list = self._prepare_subjects(subjects)
        X = self._build_input(subjects_list)
        Path(self.config.out_dir).mkdir(parents=True, exist_ok=True)

        if self.config.verbose:
            self.logger.info(
                f"Loading and running {self.config.HabitatsSegmention.clustering_mode} pipeline..."
            )

        self.pipeline = HabitatPipeline.load(str(resolved_path))

        # Reconcile the loaded pipeline's collaborators and config with the
        # current runtime context (paths, loggers, runtime-only flags).
        self._inject_managers_into_pipeline(self.pipeline)

        # Hard guarantee: predict mode must never trigger plot_habitat_scores;
        # the validation scores are not part of a loaded pipeline.
        self.pipeline.config.plot_curves = False

        results_df = self.pipeline.transform(X)
        return self._finalise_results(results_df, save_results_csv, mode='predict')

    def run(
        self,
        subjects: Optional[List[str]] = None,
        save_results_csv: Optional[bool] = None,
        load_from: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """
        Backward-compatible dispatcher to ``fit`` / ``predict``.

        Dispatch rules (highest priority first):

        1. If ``load_from`` is provided -> ``predict(load_from, ...)``.
        2. Else if ``config.run_mode == 'predict'`` and ``config.pipeline_path``
           is set -> ``predict(config.pipeline_path, ...)``.
        3. Otherwise -> ``fit(...)``.

        Args:
            subjects: Optional explicit subject ID list.
            save_results_csv: Whether to save ``habitats.csv``.
            load_from: Optional explicit pipeline path; if given, forces predict.

        Returns:
            DataFrame with habitat clustering results.
        """
        if load_from is not None:
            return self.predict(
                pipeline_path=load_from,
                subjects=subjects,
                save_results_csv=save_results_csv,
            )

        if (
            self.config.run_mode == 'predict'
            and self.config.pipeline_path
        ):
            return self.predict(
                pipeline_path=self.config.pipeline_path,
                subjects=subjects,
                save_results_csv=save_results_csv,
            )

        return self.fit(
            subjects=subjects,
            save_results_csv=save_results_csv,
        )

    # ------------------------------------------------------------------
    # Pipeline construction.
    # ------------------------------------------------------------------

    def _build_pipeline(self) -> HabitatPipeline:
        """
        Build a fresh ``HabitatPipeline`` for the current ``clustering_mode``.

        This is the *single* place where mode dispatch lives. Adding a new
        clustering mode means: add a recipe function and register it in
        ``_PIPELINE_RECIPES``.

        Returns:
            Configured ``HabitatPipeline`` ready for ``fit_transform``.

        Raises:
            ValueError: If ``clustering_mode`` is not one of the registered modes.
        """
        mode = self.config.HabitatsSegmention.clustering_mode
        if mode not in _PIPELINE_RECIPES:
            raise ValueError(
                f"Unknown clustering_mode: {mode}. "
                f"Available modes: {list(_PIPELINE_RECIPES.keys())}"
            )

        recipe = _PIPELINE_RECIPES[mode]
        steps = recipe(
            config=self.config,
            feature_manager=self.feature_manager,
            clustering_manager=self.clustering_manager,
            result_manager=self.result_manager,
        )
        return HabitatPipeline(steps=steps, config=self.config)

    # ------------------------------------------------------------------
    # Loaded-pipeline reconciliation.
    # ------------------------------------------------------------------

    def _select_pipeline_config(self, pipeline: HabitatPipeline) -> HabitatAnalysisConfig:
        """
        Pick the config to apply to a loaded pipeline.

        For predict runs that supply a minimal YAML (no ``FeatureConstruction``
        block), we keep the trained config that came pickled with the pipeline
        and only override runtime-safe fields like output paths and verbosity.

        Args:
            pipeline: The just-loaded pipeline (already deserialised).

        Returns:
            The config object to install on the pipeline and its steps.
        """
        if (
            self.config.run_mode == 'predict'
            and self.config.FeatureConstruction is None
            and pipeline.config is not None
        ):
            pipeline_config = pipeline.config
            # Override runtime fields only; never touch trained-model parameters.
            pipeline_config.out_dir = self.config.out_dir
            pipeline_config.plot_curves = self.config.plot_curves
            pipeline_config.save_results_csv = self.config.save_results_csv
            pipeline_config.save_images = self.config.save_images
            pipeline_config.processes = self.config.processes
            pipeline_config.random_state = self.config.random_state
            pipeline_config.verbose = self.config.verbose
            pipeline_config.debug = self.config.debug
            return pipeline_config

        return self.config

    def _sync_feature_manager(
        self,
        pipeline_feature_manager: Any,
        runtime_feature_manager: Any,
    ) -> None:
        """
        Refresh the loaded pipeline's FeatureManager with current paths/logging.

        We deliberately keep the *trained* FeatureManager instance (it carries
        the fitted feature-extraction state) and only update the data paths
        and logging targets so it works against the current dataset.

        Args:
            pipeline_feature_manager: FeatureManager loaded from the pkl file.
            runtime_feature_manager: FeatureManager built from the current run config.
        """
        if (
            getattr(runtime_feature_manager, 'images_paths', None) is not None
            and getattr(runtime_feature_manager, 'mask_paths', None) is not None
        ):
            pipeline_feature_manager.set_data_paths(
                runtime_feature_manager.images_paths,
                runtime_feature_manager.mask_paths,
            )

        if hasattr(runtime_feature_manager, '_log_file_path'):
            pipeline_feature_manager.set_logging_info(
                runtime_feature_manager._log_file_path,
                runtime_feature_manager._log_level,
            )

    def _inject_managers_into_pipeline(self, pipeline: HabitatPipeline) -> None:
        """
        Push current config and managers into a freshly loaded pipeline.

        Uses an explicit whitelist (``_PIPELINE_MANAGER_ATTRS``) instead of
        scanning ``dir(self)`` for ``*_manager`` attributes. The whitelist
        prevents accidental injection of any future manager that doesn't
        belong inside pipeline steps.

        Args:
            pipeline: The loaded pipeline whose ``config`` and step references
                should be reconciled with the current runtime instance.
        """
        config_to_apply = self._select_pipeline_config(pipeline)
        pipeline.config = config_to_apply

        # Build the explicit name -> instance map. Only attributes listed in
        # the whitelist participate.
        attributes_to_update: Dict[str, Any] = {'config': config_to_apply}
        for attr_name in self._PIPELINE_MANAGER_ATTRS:
            manager = getattr(self, attr_name, None)
            if manager is not None:
                attributes_to_update[attr_name] = manager

        for _, step in pipeline.steps:
            for attr_name, attr_value in attributes_to_update.items():
                if not hasattr(step, attr_name):
                    continue
                # FeatureManager has fitted state we must keep; only sync
                # paths/logging onto the trained instance.
                if attr_name == 'feature_manager':
                    pipeline_feature_manager = getattr(step, attr_name, None)
                    if pipeline_feature_manager is not None:
                        self._sync_feature_manager(pipeline_feature_manager, attr_value)
                        continue
                setattr(step, attr_name, attr_value)

    # ------------------------------------------------------------------
    # Result handling.
    # ------------------------------------------------------------------

    def _finalise_results(
        self,
        results_df: pd.DataFrame,
        save_results_csv: bool,
        mode: str,
    ) -> pd.DataFrame:
        """
        Apply post-processing, store results on the manager, and optionally save.

        Args:
            results_df: Output produced by the pipeline.
            save_results_csv: Whether to write ``habitats.csv``.
            mode: ``'train'`` or ``'predict'`` (used by the result-saving branch
                to avoid re-running plot_curves on predict).

        Returns:
            The (possibly post-processed) results DataFrame.
        """
        results_df = self._post_process_results(results_df)
        self.result_manager.results_df = results_df

        if save_results_csv:
            self._save_results(results_df, mode=mode)

        return results_df

    def _post_process_results(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Mode-specific tweaks applied to the raw pipeline output.

        For ``one_step``, the aggregation step produces per-habitat rows but
        does not always populate ``Habitats``; we copy ``Supervoxel`` into
        ``Habitats`` when it's missing.

        Args:
            results_df: Raw output from the pipeline.

        Returns:
            A DataFrame with mode-appropriate columns ensured.
        """
        mode = self.config.HabitatsSegmention.clustering_mode
        if mode == 'one_step':
            if (
                ResultColumns.HABITATS not in results_df.columns
                and ResultColumns.SUPERVOXEL in results_df.columns
            ):
                results_df[ResultColumns.HABITATS] = results_df[ResultColumns.SUPERVOXEL]
        return results_df

    def _save_results(self, results_df: pd.DataFrame, mode: str) -> None:
        """
        Write ``habitats.csv`` and any habitat-map images.

        For ``one_step`` mode, the IndividualClusteringStep already wrote the
        per-subject habitat NRRDs; we only emit the CSV here.
        For ``two_step`` and ``direct_pooling``, we delegate to
        ``ResultManager.save_all_habitat_images`` after syncing the
        ``mask_info_cache`` from the pipeline (which lives in the parent process).

        Args:
            results_df: Final results DataFrame.
            mode: ``'train'`` or ``'predict'``; only affects logging.
        """
        if self.config.verbose:
            self.logger.info("Saving results...")

        # Always write the CSV first with a stable column ordering.
        csv_path = Path(self.config.out_dir) / "habitats.csv"
        canonical_order = _canonical_csv_column_order(results_df)
        results_df[canonical_order].to_csv(str(csv_path), index=False)
        if self.config.verbose:
            self.logger.info(f"Results saved to {csv_path}")

        clustering_mode = self.config.HabitatsSegmention.clustering_mode
        if clustering_mode == 'one_step':
            # IndividualClusteringStep saved habitat maps directly during fit/transform;
            # nothing more to do here.
            if self.config.verbose:
                self.logger.info(
                    "One-Step mode: habitat maps were saved during clustering."
                )
            return

        # two_step / direct_pooling: rely on ResultManager.save_all_habitat_images.
        if not self.config.save_images:
            return

        # Surface the mask cache from the pipeline (populated in the main
        # process) to the result manager so it can reconstruct images.
        if (
            self.pipeline is not None
            and getattr(self.pipeline, 'mask_info_cache', None)
        ):
            self.result_manager.mask_info_cache = self.pipeline.mask_info_cache
        self.result_manager.save_all_habitat_images(failed_subjects=[])

    # ------------------------------------------------------------------
    # Small private helpers.
    # ------------------------------------------------------------------

    def _resolve_save_flag(self, save_results_csv: Optional[bool]) -> bool:
        """
        Resolve the ``save_results_csv`` argument: explicit > config default.
        """
        if save_results_csv is None:
            return self.config.save_results_csv
        return save_results_csv

    def _prepare_subjects(self, subjects: Optional[List[str]]) -> List[str]:
        """
        Normalise the subjects argument and verify it's not empty.

        Args:
            subjects: Optional list of subject IDs. ``None`` means "use all
                subjects discovered from images_paths".

        Returns:
            A non-empty list of subject IDs.

        Raises:
            ValueError: If no subjects can be resolved.
        """
        if subjects is None:
            subjects = list(self.feature_manager.images_paths.keys())

        if not subjects:
            raise ValueError(
                f"No subjects provided for {self.config.HabitatsSegmention.clustering_mode} run."
            )
        return list(subjects)

    def _build_input(self, subjects: List[str]) -> Dict[str, Dict]:
        """
        Build the per-subject input dict consumed by the pipeline.

        The pipeline's first step (``VoxelFeatureExtractor``) populates these
        dicts with image/mask/feature data; we just provide an empty stub for
        each subject.

        Args:
            subjects: List of subject IDs.

        Returns:
            ``{subject_id: {} for ...}``.
        """
        return {subject: {} for subject in subjects}

    def _resolve_pipeline_path(self, load_from: Optional[Union[str, Path]]) -> Path:
        """
        Resolve the on-disk pipeline path used by ``fit`` for saving.

        Args:
            load_from: Optional path supplied by the caller; if given, returned
                as-is so external scripts can override the default location.

        Returns:
            ``Path`` to write/read the pipeline file.
        """
        if load_from:
            return Path(load_from)
        return Path(self.config.out_dir) / "habitat_pipeline.pkl"

    # ------------------------------------------------------------------
    # Property forwarding kept for backward compatibility.
    # ------------------------------------------------------------------

    @property
    def results_df(self) -> Optional[pd.DataFrame]:
        """Forward to ``ResultManager.results_df``."""
        return self.result_manager.results_df

    @results_df.setter
    def results_df(self, value: pd.DataFrame) -> None:
        self.result_manager.results_df = value

    @property
    def supervoxel2habitat_clustering(self) -> Any:
        """Forward to ``ClusteringManager.supervoxel2habitat_clustering``."""
        return self.clustering_manager.supervoxel2habitat_clustering

    @property
    def images_paths(self) -> Optional[Dict[str, Any]]:
        """Forward to ``FeatureManager.images_paths``."""
        return self.feature_manager.images_paths

    @property
    def mask_paths(self) -> Optional[Dict[str, Any]]:
        """Forward to ``FeatureManager.mask_paths``."""
        return self.feature_manager.mask_paths
