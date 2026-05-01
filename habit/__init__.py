"""
HABIT - Habitat Analysis Tool for Medical Images.

A comprehensive tool for analysing tumour habitats from medical images using
radiomics features and machine-learning techniques.

V1 import policy
----------------
Imports of core HABIT components are fail-fast. If a core import is broken
you get a real ``ImportError``, not a silent ``None``.

For genuinely optional third-party dependencies (e.g. AutoGluon), use:

>>> import habit
>>> habit.is_available('autogluon')
False
>>> habit.import_error('autogluon')
ModuleNotFoundError("No module named 'autogluon'")
"""

from __future__ import annotations

import importlib
from typing import Any, Optional

__version__ = "0.1.0"

# Lazy-loaded core exports. Importing ``habit`` does not pull these in;
# attribute access does. Keeps ``habit -h`` cheap.
_CORE_EXPORTS = {
    "HabitatAnalysis",
    "HabitatFeatureExtractor",
    "Modeling",
}

# Genuinely optional third-party dependencies. ``is_available`` /
# ``import_error`` only accept names from this whitelist — adding new
# optional deps is a deliberate edit here.
_OPTIONAL_DEPENDENCIES = (
    "autogluon",
)

# Cache for ``ImportError`` instances raised by optional-dep probes. Keys
# are dependency names from ``_OPTIONAL_DEPENDENCIES``.
_optional_dep_errors: dict = {}
# Sentinel set when a dep was probed and imported successfully.
_optional_dep_ok: set = set()

__all__ = [
    "HabitatAnalysis",
    "HabitatFeatureExtractor",
    "Modeling",
    "is_available",
    "import_error",
]


def is_available(name: str) -> bool:
    """
    Check whether a known optional dependency can be imported.

    Args:
        name: optional dependency name; must be in
            ``habit._OPTIONAL_DEPENDENCIES``.

    Returns:
        ``True`` if the dependency imports cleanly, ``False`` otherwise.

    Raises:
        ValueError: if ``name`` is not a recognised optional dependency.
    """
    if name not in _OPTIONAL_DEPENDENCIES:
        raise ValueError(
            f"Unknown optional dependency: {name!r}. "
            f"Recognised: {_OPTIONAL_DEPENDENCIES}"
        )
    if name in _optional_dep_ok:
        return True
    if name in _optional_dep_errors:
        return False
    try:
        importlib.import_module(name)
    except ImportError as exc:
        _optional_dep_errors[name] = exc
        return False
    _optional_dep_ok.add(name)
    return True


def import_error(name: str) -> Optional[ImportError]:
    """
    Return the cached ``ImportError`` for an optional dependency, or ``None``.

    Args:
        name: optional dependency name; must be in
            ``habit._OPTIONAL_DEPENDENCIES``.

    Returns:
        The cached ``ImportError`` instance if the dep is unavailable, else
        ``None``.

    Raises:
        ValueError: if ``name`` is not a recognised optional dependency.
    """
    if name not in _OPTIONAL_DEPENDENCIES:
        raise ValueError(
            f"Unknown optional dependency: {name!r}. "
            f"Recognised: {_OPTIONAL_DEPENDENCIES}"
        )
    # Populate cache on first call.
    is_available(name)
    return _optional_dep_errors.get(name)


def __getattr__(name: str) -> Any:
    """
    Lazily resolve heavy core exports on first access.

    Importing the package happens before console scripts load submodules
    such as ``habit.cli``. Keeping core imports lazy prevents simple
    commands like ``habit -h`` from importing imaging and ML dependencies.
    """
    if name not in _CORE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from . import core

    value = getattr(core, name)
    globals()[name] = value
    return value
