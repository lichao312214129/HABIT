"""
HABIT - Habitat Analysis Tool for Medical Images

A comprehensive tool for analyzing tumor habitats using radiomics features
and machine learning techniques.
"""

from typing import Any

__version__ = "0.1.0"

_CORE_EXPORTS = {
    "HabitatAnalysis",
    "HabitatFeatureExtractor",
    "Modeling",
    "get_import_errors",
    "get_available_classes",
    "is_class_available",
}

# Define the public API for the top-level package
__all__ = [
    "HabitatAnalysis",
    "HabitatFeatureExtractor",
    "Modeling",
    "get_import_errors",
    "get_available_classes",
    "is_class_available",
]


def __getattr__(name: str) -> Any:
    """
    Lazily load heavy core exports only when users access them.

    Importing the package happens before console scripts load submodules such as
    habit.cli. Keeping core imports lazy prevents simple commands like
    ``habit -h`` from importing imaging and machine-learning dependencies.
    """
    if name not in _CORE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from . import core

    value = getattr(core, name)
    globals()[name] = value
    return value