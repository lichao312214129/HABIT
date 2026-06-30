# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""
Registry adapters: bridge dynamic backend registries to SchemaForm.

Instead of hardcoding model/selector lists in tab files, tabs call these
functions to get up-to-date choices from the backend registries. When a new
model or selector is registered, the GUI picks it up automatically — zero
code changes in the GUI layer.
"""

from __future__ import annotations

from typing import List


def get_model_choices() -> List[str]:
    """Get available model names from :class:`ModelFactory`.

    Returns an empty list if the registry cannot be imported (e.g. optional
    dependencies missing).
    """
    try:
        from habit.core.machine_learning.models.factory import ModelFactory
        return ModelFactory.get_available_models()
    except Exception:
        return []


def get_selector_choices() -> List[str]:
    """Get available feature selector names from ``selector_registry``.

    Returns an empty list if the registry cannot be imported.
    """
    try:
        from habit.core.machine_learning.feature_selectors.selector_registry import (
            get_available_selectors,
        )
        return get_available_selectors()
    except Exception:
        return []


def get_prep_step_choices() -> List[str]:
    """Get available preprocessing step names.

    Returns an empty list if the registry cannot be imported.
    """
    try:
        from habit.core.preprocessing.preprocessor_factory import (
            get_available_preprocessors,
        )
        return get_available_preprocessors()
    except Exception:
        return []
