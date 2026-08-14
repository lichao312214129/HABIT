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
Declarative table of the stable symbols exposed from ``import habit``.

This module is intentionally PURE DATA plus one small builder: it must never
import another ``habit`` module (or any third-party package) so that
``habit/__init__.py`` can construct its lazy export table without pulling in
the ``habit.api`` package -- importing ``habit.api.registry`` from
``habit/__init__.py`` used to chain into ``habit.utils`` and the v0.1 core
stack, dragging sklearn/pandas/scipy into every bare ``import habit``.

``habit.api.registry`` re-exports these names for backward compatibility;
tests import the registry through that facade.
"""

from __future__ import annotations

from typing import Dict, Tuple

#: Dotted module path relative to the ``habit`` package -> export names
#: defined in that module. Keys beginning with ``api.`` are the v0.1 facade
#: modules; the remaining keys are the v1.0 layered packages. Name clashes
#: bind to the v1.0 contract symbols: top-level ``Cohort`` is the imaging
#: cohort (``habit.contracts.subject.Cohort``) since v1.0.0.
_PUBLIC_API_MODULES: Dict[str, Tuple[str, ...]] = {
    "api.contracts": ("WorkflowResult",),
    "api.provenance": (
        "RunManifest",
        "create_run_manifest",
        "write_run_manifest",
    ),
    # Exceptions resolve to the canonical foundation module ``habit.exceptions``
    # (``habit.api.exceptions`` remains as a backward-compatible facade).
    "exceptions": (
        "HABITAPIError",
        "HabitError",
        "ConfigurationError",
        "DataFormatError",
        "GeometryError",
        "OptionalDependencyError",
        "ComponentNotFoundError",
        "CompatibilityError",
        "ProcessingError",
        "NotFittedError",
    ),
    "api.image": (
        "GeometryPolicy",
        "GeometryReport",
        "ImageVolume",
        "MaskVolume",
        "ImageMaskPair",
        "read_image",
        "read_mask",
        "validate_geometry",
        "align_image_mask",
    ),
    "api.radiomics": (
        "FeatureResult",
        "FeatureTableResult",
        "extract_features",
        "extract_batch",
    ),
    "api.plugins": (
        "PluginInfo",
        "PluginParamInfo",
        "PluginCatalogEntry",
        "PluginLoadReport",
        "list_plugins",
        "get_plugin_info",
        "get_param_schema",
        "plugin_catalog",
        "format_plugin_catalog_rst",
        "load_plugins",
    ),
    "api.utils": (
        "setup_logger",
        "is_available",
        "show_versions",
        "check_component",
    ),
    # Materialize bundled demo YAML (wheel ships config/, not demo_data/).
    "api.demo_config": (
        "copy_demo_config",
        "demo_config_root",
        "iter_demo_config_files",
    ),
    "api.preprocessing": (
        "PreprocessingConfig",
        "run_preprocess",
        "preprocess_subject",
        "preprocess_image",
    ),
    "api.dicom_sort": (
        "DicomSortConfig",
        "run_dicom_sort",
    ),
    "api.habitat": (
        "HabitatAnalysisConfig",
        "FeatureExtractionConfig",
        "RadiomicsConfig",
        "apply_habitat_cli_overrides",
        "build_feature_extraction_config",
        "load_feature_extraction_config",
        "run_habitat_analysis",
        "run_feature_extraction",
        "run_radiomics",
    ),
    "api.machine_learning": (
        "MLConfig",
        "ModelComparisonConfig",
        "apply_ml_mode_override",
        "run_ml",
        "run_kfold",
        "run_model_comparison",
    ),
    "api.analysis": (
        "ICCConfig",
        "TestRetestConfig",
        "run_icc_analysis",
        "run_test_retest_analysis",
    ),
    # ------------------------------------------------------------------
    # v1.0 layered packages (additive; see developer/api_upgrade/06-08).
    # ------------------------------------------------------------------
    "datasets": (
        "make_synthetic_cohort",
        "make_synthetic_feature_table",
    ),
    "contracts": (
        "Geometry",
        "ImageRef",
        "ArrayImageRef",
        "Subject",
        "Cohort",
        "CohortFingerprint",
        "STEP_VOXEL_FEATURES_RAW",
        "STEP_VOXEL_FEATURES_PREPROCESSED",
        "STEP_SUPERVOXELS_PARTITION",
        "STEP_SUPERVOXELS_POSTPROCESSED",
        "STEP_SUPERVOXELS_DESCRIBED",
        "STEP_SUPERVOXELS_PREPROCESSED",
        "STEP_UNITS_COHORT_PREPROCESSED",
        "STEP_HABITAT_MAP",
        "STEP_HABITAT_FEATURES",
        "STEP_NAMES",
        "StepRecord",
        "StepObserver",
        "cohort_from_directory",
        "Provenance",
        "VoxelFeatureField",
        "Supervoxelization",
        "HabitatMap",
        "HabitatModel",
        "FeatureTable",
        "SubjectOperator",
        "CohortOperator",
        "SubjectResult",
        "ExecutionBackend",
        "DataSource",
        "ResultWriter",
    ),
    "execution": (
        "SerialBackend",
        "ProcessPoolBackend",
        "SubjectTimeoutError",
        "CheckpointStore",
    ),
    "adapters": (
        "DirectoryDataSource",
        "DirectoryResultWriter",
        "FileImageRef",
    ),
    # L4 recipes: Study factories plus the objects they return.
    "recipes": (
        "Study",
        "StudyResult",
        "ModelResult",
        "CVResult",
        "PredictionResult",
        "SearchResult",
        "two_step_habitat",
        "one_step_habitat",
        "direct_pooling_habitat",
        "extract_habitat_features",
        "traditional_radiomics",
        "identify_precise_voxel_features",
        "voxel_radiomics_factory",
        "train_model",
        "cross_validate",
        "search_hyperparameters",
        "predict_model",
        "compare_models",
        "pairwise_delong_test",
        "preprocess_images",
        "run_from_yaml",
        "icc_analysis",
        "test_retest_analysis",
        "sort_dicom",
        "dice",
        "dicom_info",
        "merge_tables",
    ),
    "spec": (
        "Spec",
        "Stage",
        "HabitatSpec",
        "MLSpec",
        "RunPolicy",
        "load_habitat_spec",
        "save_habitat_spec",
        "load_run_policy",
        "save_run_policy",
        "build_habitat_document",
        "save_habitat_config",
        "load_habitat_config",
        "LegacyConfigAdapter",
        "detect_yaml_version",
        "migrate_yaml",
        "validate_v1_document",
        "coerce_spec",
        "parse_feature_expression",
    ),
    "registry": ("ComponentRegistry",),
    "domain": (
        "VoxelFeatureExtractor",
        "Supervoxelizer",
        "HabitatModelFitter",
        "HabitatAssigner",
        "HabitatFeatureExtractor",
        "Combiner",
        "Seedable",
        "TablePreprocessor",
        "FeatureSelector",
        "Classifier",
        "Metric",
        "SubjectPipeline",
        "TablePipeline",
        "PooledUnits",
        "fan_in",
        "ConnectedComponentPostprocess",
        # Precision screen (Prior et al. 2024): the perturbation protocol and
        # its built-ins, the analysis functions, and the resulting artifact.
        "Preprocessor",
        "PreprocessorRegistry",
        "ImagePerturbation",
        "ImagePerturbationRegistry",
        "GaussianNoisePerturbation",
        "TranslationPerturbation",
        "RotationPerturbation",
        "RigidPerturbation",
        "BSplineDeformPerturbation",
        "PerturbationChain",
        "PreciseFeatureSet",
        "FeatureWhitelist",
        "precision_panel",
        "aggregate_panels",
        "identify_precise_features",
        "habitat_stability",
        "prior2024_retest_perturbation",
    ),
    # L0 kernels: the pure numerical core, mirrored from
    # ``habit.kernels.__all__`` so third parties can call a single metric
    # (e.g. ``habit.ith_score``) the way they would call a scipy function.
    "kernels": (
        "SCORE_DIRECTIONS",
        "MAXIMIZE",
        "MINIMIZE",
        "KNEE",
        "score_direction",
        "knee_index",
        "best_index",
        "vote_best_index",
        "gap_statistic",
        "local_entropy_map",
        "spatial_interaction_matrix",
        "msi_features_from_matrix",
        "habitat_volume_fractions",
        "habitat_region_stats",
        "ith_score",
        "HabitatGraphFeatureOptions",
        "extract_graph_features",
        "extract_graph_features_for_labels",
        "extract_habitat_nodes",
        "build_centroid_distance_graph",
        "build_adjacency_graph",
        "pair_count",
        "compute_midrank",
        "fast_delong",
        "delong_roc_variance",
        "delong_roc_test",
        "delong_roc_ci",
        "hosmer_lemeshow_test",
        "spiegelhalter_z_test",
        "two_way_mean_squares",
        "icc3_1",
        "icc2_1",
        "estimate_noise_sigma",
        "add_gaussian_noise",
        "translate_image",
        "rotate_image",
        "rigid_transform_image",
        "ICCEstimate",
        "icc3a_1",
        "icc3c_1",
        "remove_small_connected_components",
    ),
    # Table-level scikit-learn interop (L3). The factories are top-level; the
    # adapter classes stay namespaced under ``habit.domain.sklearn_interop``.
    # ``FrameToTable`` is the exception: an sklearn cross-validation driver
    # must name it to declare the column schema of the frames it slices, so
    # it is part of the public vocabulary. ``habit.compat.sklearn`` keeps
    # deprecated aliases of the two older factories for all of v1.x.
    "domain.sklearn_interop": (
        "FrameToTable",
        "as_transformer",
        "as_classifier",
        "as_regressor",
        "as_survival_model",
        "as_outcome_model",
    ),
    # Ecosystem interop adapters (``habit.compat.*``). Only the factory
    # function is top-level; the generated estimator class stays namespaced
    # under ``habit.compat.sklearn``.
    "compat.sklearn": ("as_estimator",),
    "compat.monai": (
        "to_monai_dict",
        "from_monai_dict",
        "AsMonaiDict",
        "FromMonaiDict",
        "AsDictTransform",
    ),
    "compat.nnunet": ("NnUNetDataSource",),
    # In-memory step inspection (L3). Only ``StepRecorder`` is re-exported
    # from this package; step-name constants live under ``contracts``.
    "inspection": (
        "StepRecorder",
    ),
    # Publication figures, mirrored from ``habit.viz.__all__``. matplotlib
    # itself stays function-level inside ``habit.viz``, so these registrations
    # do not make ``import habit`` pull a plotting backend.
    "viz": (
        "StyleSpec",
        "use_style",
        "get_style",
        "register_style",
        "available_styles",
        "sanitize_label",
        "plot_kaplan_meier",
        "plot_risk_triptych",
        "plot_time_dependent_auc",
        "plot_survival_calibration",
        "plot_brier_curve",
        "plot_cox_forest",
        "plot_predicted_vs_observed",
        "plot_residuals",
        "plot_residual_qq",
        "plot_bland_altman",
        "plot_coefficient_forest",
        "plot_roc",
        "plot_precision_recall",
        "plot_calibration",
        "plot_decision_curve",
        "plot_confusion_matrix",
        "plot_shap_summary",
        "plot_shap_dependence",
        "plot_shap_waterfall",
        "plot_permutation_importance",
        "rank_shap_feature_indices",
        "select_representative_sample_indices",
        "net_benefit",
        "plot_habitat_clustering_pca_2d",
        "plot_habitat_clustering_pca_3d",
        "plot_habitat_clustering_pca_3d_interactive",
        "plot_habitat_overlay",
        "plot_cluster_validation_curves",
        "plot_cluster_validation_from_report",
        "plot_habitat_volume_fractions",
        "plot_msi_matrix",
        "plot_ith_summary",
        "plot_habitat_label_compare",
        "plot_partition_triptych",
        "plot_habitat_graph_slice",
        "plot_habitat_graph_network_2d",
        "render_habitat_graph_surface_3d",
        "render_habitat_graph_network_3d",
        "dense_voxel_feature_map",
        "plot_intensity_slice",
        "plot_voxel_texture_slice",
        "view_habitat_napari",
    ),
}

#: Sorted stable public names (excluding ``__version__``).
PUBLIC_API_SYMBOLS: Tuple[str, ...] = tuple(
    sorted(name for names in _PUBLIC_API_MODULES.values() for name in names)
)


def build_lazy_exports() -> Dict[str, Tuple[str, str]]:
    """
    Build ``(relative_submodule, attribute)`` pairs for ``habit`` lazy imports.

    Returns:
        Mapping suitable for :func:`~habit.utils.lazy_exports.lazy_getattr`.
    """
    exports: Dict[str, Tuple[str, str]] = {}
    for module_path, names in _PUBLIC_API_MODULES.items():
        relative_module = f".{module_path}"
        for name in names:
            exports[name] = (relative_module, name)
    return exports
