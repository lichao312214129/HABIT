"""
HABIT - Habitat Analysis Tool for Medical Images

A comprehensive tool for analyzing tumor habitats using radiomics features
and machine learning techniques.
"""

__version__ = "0.1.0"

# Import the public API from the core module, which handles robust imports.
from .core import (
    HabitatAnalysis,
    HabitatFeatureExtractor,
    Modeling,
    get_import_errors,
    get_available_classes,
    is_class_available,
)

# Define the public API for the top-level package
__all__ = [
    "HabitatAnalysis",
    "HabitatFeatureExtractor",
    "Modeling",
    "get_import_errors",
    "get_available_classes",
    "is_class_available",
]