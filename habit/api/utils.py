# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""Lightweight utilities re-exported from the top-level ``habit`` namespace."""

from __future__ import annotations

import importlib.util

from habit.utils.log_utils import setup_logger

__all__ = ["setup_logger", "is_available"]


def is_available(module_name: str) -> bool:
    """
    Return whether an optional third-party module can be imported.

    Args:
        module_name: Top-level package name (e.g. ``"radiomics"``, ``"torch"``).

    Returns:
        True when ``importlib`` finds a spec for the module.
    """
    return importlib.util.find_spec(module_name) is not None
