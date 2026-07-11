# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""
Canonical registry of stable symbols exposed from ``import habit``.

Tests import this module to verify the public contract without duplicating lists.
"""

from __future__ import annotations

from typing import Dict, Tuple

#: Submodule suffix (under ``habit.api``) -> export names defined in that module.
_PUBLIC_API_MODULES: Dict[str, Tuple[str, ...]] = {
    "contracts": ("WorkflowResult",),
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
    "image": (
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
    "radiomics": (
        "FeatureResult",
        "FeatureTableResult",
        "extract_features",
        "extract_batch",
    ),
    "utils": (
        "setup_logger",
        "is_available",
    ),
    "preprocessing": (
        "PreprocessingConfig",
        "run_preprocess",
    ),
    "dicom_sort": (
        "DicomSortConfig",
        "run_dicom_sort",
    ),
    "habitat": (
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
    "machine_learning": (
        "MLConfig",
        "ModelComparisonConfig",
        "apply_ml_mode_override",
        "run_ml",
        "run_kfold",
        "run_model_comparison",
    ),
    "analysis": (
        "ICCConfig",
        "TestRetestConfig",
        "run_icc_analysis",
        "run_test_retest_analysis",
    ),
    "estimators": (
        "EstimatorPersistenceMixin",
        "HabitatClusterer",
        "HabitClassifier",
        "SubjectFeatureAggregator",
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
    for module_suffix, names in _PUBLIC_API_MODULES.items():
        relative_module = f".api.{module_suffix}"
        for name in names:
            exports[name] = (relative_module, name)
    return exports
