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
Orchestrator contract for HABIT domain workflows.

An *orchestrator* is the top-level runtime object a :class:`BaseConfigurator`
assembles from a validated config and hands back to ``core/*/run.py``. Each
business domain has one (or a few) orchestrators; they intentionally expose an
**idiomatic terminal method** rather than a single forced name:

* Single-shot pipelines expose ``run()`` — for example ``BatchProcessor`` (preprocessing),
  ``HoldoutWorkflow`` / ``KFoldWorkflow`` / ``ModelComparison`` (machine learning),
  ``HabitatMapAnalyzer`` (habitat feature extraction) and
  ``TraditionalRadiomicsExtractor`` (radiomics).
* Train / predict pipelines expose ``fit()`` **and** ``predict()`` — currently
  ``HabitatAnalysis`` (habitat segmentation), because it genuinely has two modes
  that share one serialized pipeline.

This mirrors scikit-learn: implementations keep descriptive class names, but the
*interface* is uniform and predictable. The :data:`ORCHESTRATOR_CONTRACT` table
is the single source of truth used by the contract self-check test
(``tests/.../test_orchestrator_contract.py``) so new orchestrators cannot
silently drift from this convention.
"""

from __future__ import annotations

from typing import Any, Protocol, Tuple, runtime_checkable


@runtime_checkable
class RunOrchestrator(Protocol):
    """Orchestrator for a single-shot pipeline (exposes ``run``)."""

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the full pipeline and return its result."""
        ...


@runtime_checkable
class FitPredictOrchestrator(Protocol):
    """Orchestrator for a train/predict pipeline (exposes ``fit`` + ``predict``)."""

    def fit(self, *args: Any, **kwargs: Any) -> Any:
        """Fit / train the pipeline and (typically) serialize it."""
        ...

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """Apply a previously fitted pipeline to new data."""
        ...


# Canonical orchestrators: {domain_key: (import_path, class_name, terminal_methods)}.
# ``terminal_methods`` lists the public method(s) that must exist. This table is
# consumed by the contract self-check test; keep it in sync when adding an
# orchestrator so the convention stays enforceable.
ORCHESTRATOR_CONTRACT: dict[str, Tuple[str, str, Tuple[str, ...]]] = {
    "preprocessing": (
        "habit.core.preprocessing.image_processor_pipeline",
        "BatchProcessor",
        ("run",),
    ),
    "habitat_segmentation": (
        "habit.core.habitat_analysis.habitat_analysis",
        "HabitatAnalysis",
        ("fit", "predict"),
    ),
    "habitat_feature_extraction": (
        "habit.core.habitat_analysis.habitat_features.habitat_analyzer",
        "HabitatMapAnalyzer",
        ("run",),
    ),
    "radiomics": (
        "habit.core.habitat_analysis.habitat_features.traditional_radiomics_extractor",
        "TraditionalRadiomicsExtractor",
        ("run",),
    ),
    "ml_holdout": (
        "habit.core.machine_learning.workflows.holdout_workflow",
        "HoldoutWorkflow",
        ("run",),
    ),
    "ml_kfold": (
        "habit.core.machine_learning.workflows.kfold_workflow",
        "KFoldWorkflow",
        ("run",),
    ),
    "ml_comparison": (
        "habit.core.machine_learning.workflows.comparison_workflow",
        "ModelComparison",
        ("run",),
    ),
}


def check_orchestrator_class(cls: type, terminal_methods: Tuple[str, ...]) -> None:
    """
    Assert that an orchestrator class satisfies the terminal-method contract.

    Args:
        cls: Orchestrator class to validate.
        terminal_methods: Method names that must exist and be callable
            (e.g. ``("run",)`` or ``("fit", "predict")``).

    Raises:
        AssertionError: If any required method is missing or not callable.
    """
    for method_name in terminal_methods:
        attr = getattr(cls, method_name, None)
        assert callable(attr), (
            f"Orchestrator {cls.__name__!r} must expose a callable "
            f"'{method_name}()' per the HABIT orchestrator contract."
        )
