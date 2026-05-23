"""
Feature extraction module for habitat analysis.

Exports are lazy so importing sibling habitat modules does not require
PyRadiomics until post-clustering feature extraction is actually used.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from habit.utils.lazy_exports import lazy_getattr

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "HabitatMapAnalyzer": (".habitat_analyzer", "HabitatMapAnalyzer"),
}

__all__ = ["HabitatMapAnalyzer", "HabitatFeatureExtractor"]


def __getattr__(name: str) -> Any:
    """Resolve habitat feature extractors on first access."""
    if name == "HabitatFeatureExtractor":
        return lazy_getattr("HabitatMapAnalyzer", globals(), _LAZY_EXPORTS)
    return lazy_getattr(name, globals(), _LAZY_EXPORTS)
