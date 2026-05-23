"""
Core modules for HABIT package.

V1: imports are fail-fast. If a core import is broken, you get a real
``ImportError`` immediately rather than a silent ``None`` attribute.
For genuinely optional third-party dependencies, use the
``habit.is_available`` / ``habit.import_error`` helpers exposed at the
package root.

Public exports are lazy so importing ``habit.core.habitat_analysis`` (or any
other subdomain) does not eagerly load unrelated domains such as machine
learning visualization (``shap`` / ``torch``).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from habit.utils.lazy_exports import lazy_getattr

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "HabitatAnalysis": (".habitat_analysis", "HabitatAnalysis"),
    "HabitatFeatureExtractor": (".habitat_analysis", "HabitatFeatureExtractor"),
    "Modeling": (".machine_learning.workflows.holdout_workflow", "HoldoutWorkflow"),
}

__all__ = [
    "HabitatAnalysis",
    "HabitatFeatureExtractor",
    "Modeling",
]


def __getattr__(name: str) -> Any:
    """Resolve cross-domain core exports on first access."""
    return lazy_getattr(name, globals(), _LAZY_EXPORTS)
