"""Tests for habit.utils.lazy_exports helper."""

from __future__ import annotations

import importlib

from habit.utils.lazy_exports import lazy_getattr


def test_lazy_getattr_caches_resolved_symbol() -> None:
    """Resolved lazy exports are cached in module globals."""
    module = importlib.import_module("habit.core.habitat_analysis")
    lazy_map = {
        "HabitatAnalysisConfig": (".config_schemas", "HabitatAnalysisConfig"),
    }
    module_globals = module.__dict__

    first = lazy_getattr("HabitatAnalysisConfig", module_globals, lazy_map)
    second = lazy_getattr("HabitatAnalysisConfig", module_globals, lazy_map)

    assert first is second
    assert module_globals["HabitatAnalysisConfig"] is first


def test_lazy_getattr_unknown_name_raises() -> None:
    """Unknown export names raise AttributeError."""
    module_globals = {"__name__": "habit.core.habitat_analysis"}
    try:
        lazy_getattr("Missing", module_globals, {})
        raise AssertionError("Expected AttributeError")
    except AttributeError:
        pass
