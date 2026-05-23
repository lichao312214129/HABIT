"""Tests for lazy package exports and cross-domain import isolation."""

from __future__ import annotations

import importlib
import sys


def _unload_modules(prefixes: tuple[str, ...]) -> None:
    """Remove cached modules so import tests start from a clean slate."""
    for module_name in list(sys.modules):
        if module_name.startswith(prefixes):
            del sys.modules[module_name]


def test_habitat_config_import_does_not_load_ml_plotting_stack() -> None:
    """Importing habitat config schemas must not eagerly load shap/torch."""
    _unload_modules(("habit.core", "habit.utils.lazy_exports"))
    importlib.import_module("habit.core.habitat_analysis.config_schemas")

    assert "shap" not in sys.modules
    assert "torch" not in sys.modules
    assert "habit.core.machine_learning.visualization.plotting" not in sys.modules


def test_habitat_pipeline_import_does_not_load_shap() -> None:
    """Individual-level pipeline modules must not pull ML SHAP dependencies."""
    _unload_modules(("habit.core", "habit.utils.lazy_exports"))
    importlib.import_module("habit.core.habitat_analysis.pipelines.base_pipeline")

    assert "shap" not in sys.modules
    assert "torch" not in sys.modules


def test_core_public_exports_remain_available() -> None:
    """Lazy core exports still resolve on first access."""
    _unload_modules(("habit.core", "habit.utils.lazy_exports"))
    core = importlib.import_module("habit.core")
    assert "HabitatAnalysis" in core.__all__
    assert "Modeling" in core.__all__
