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
Canonical registry of stable symbols exposed from ``import habit``.

Tests import this module to verify the public contract without duplicating lists.
"""

from __future__ import annotations

from typing import Dict, Tuple

#: Dotted module path relative to the ``habit`` package -> export names
#: defined in that module. Keys beginning with ``api.`` are the v0.1 facade
#: modules; the remaining keys are the v1.0 layered packages, which are
#: additive and never shadow the v0.1 names (name clashes stay bound to the
#: v0.1 symbols, e.g. top-level ``Cohort`` remains the clinical cohort).
_PUBLIC_API_MODULES: Dict[str, Tuple[str, ...]] = {
    "api.contracts": ("WorkflowResult",),
    "api.clinical": (
        "Cohort",
        "PreparedCohort",
        "HabitatResult",
        "ClinicalPreprocessor",
        "HabitatSegmenter",
    ),
    "api.provenance": (
        "RunManifest",
        "create_run_manifest",
        "write_run_manifest",
    ),
    "api.exceptions": (
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
        "PluginLoadReport",
        "list_plugins",
        "get_plugin_info",
        "get_param_schema",
        "load_plugins",
    ),
    "api.utils": (
        "setup_logger",
        "is_available",
    ),
    "api.preprocessing": (
        "PreprocessingConfig",
        "run_preprocess",
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
    "api.estimators": (
        "EstimatorPersistenceMixin",
        "HabitatClusterer",
        "HabitClassifier",
        "OutcomeClassifier",
        "SubjectFeatureAggregator",
    ),
    # ------------------------------------------------------------------
    # v1.0 layered packages (additive; see developer/api_upgrade/06-08).
    # ------------------------------------------------------------------
    "contracts": (
        "Geometry",
        "ImageRef",
        "ArrayImageRef",
        "Subject",
        "CohortFingerprint",
        "cohort_from_directory",
        "Provenance",
        "VoxelFeatureField",
        "Supervoxelization",
        "HabitatMap",
        "HabitatModel",
        "FeatureTable",
        "StudyResult",
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
    ),
    "spec": (
        "Spec",
        "HabitatSpec",
        "RunPolicy",
        "load_habitat_spec",
        "save_habitat_spec",
        "load_run_policy",
        "save_run_policy",
        "LegacyConfigAdapter",
        "detect_yaml_version",
        "migrate_yaml",
        "validate_v1_document",
    ),
    "registry": (
        "ComponentRegistry",
    ),
    "domain": (
        "VoxelFeatureExtractor",
        "Supervoxelizer",
        "HabitatModelFitter",
        "HabitatAssigner",
        "HabitatFeatureExtractor",
        "Seedable",
        "TablePreprocessor",
        "FeatureSelector",
        "Classifier",
        "Metric",
        "SubjectPipeline",
        "TablePipeline",
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
