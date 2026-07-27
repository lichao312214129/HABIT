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
Shared helpers for feature selection parameter handling.

Keeping the ``n_features_to_select`` semantics in one place guarantees that every
ranking-based selector (anova, chi2, statistical_test, ...) interprets the same
YAML value identically.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union


def resolve_n_features_to_select(
    value: Optional[Union[int, float]],
    n_candidates: int,
) -> Tuple[Optional[int], Optional[str]]:
    """
    Translate a user-supplied ``n_features_to_select`` value into a feature count.

    Two notations are supported so that a single YAML key can express both an
    absolute and a relative target:

    - ``value >= 1``: absolute number of features to keep (must be a whole
      number, e.g. ``20`` or ``20.0``). Clipped to ``n_candidates``.
    - ``0 < value < 1``: fraction of the candidate features to keep, e.g. ``0.2``
      keeps the top 20%. The count is rounded up (``ceil``) and forced to at
      least 1 so the selector never returns an empty feature set.

    Args:
        value: Raw configuration value. ``None`` means "not specified", in which
            case the caller should fall back to its p-value threshold.
        n_candidates: Number of features currently available for selection.

    Returns:
        Tuple[Optional[int], Optional[str]]: ``(n_features, description)`` where
            ``n_features`` is the resolved absolute count (``None`` when ``value``
            is ``None``) and ``description`` is a human-readable summary usable in
            logs / saved metadata (``None`` when ``value`` is ``None``).

    Raises:
        ValueError: If ``value`` is not positive, or is a fractional number
            greater than 1 (ambiguous: neither a valid count nor a ratio).
    """
    if value is None:
        return None, None

    if n_candidates <= 0:
        raise ValueError(
            "n_features_to_select cannot be resolved because no candidate features are available."
        )

    numeric_value = float(value)

    if numeric_value <= 0:
        raise ValueError(
            f"n_features_to_select must be positive, got {value!r}. "
            "Use an integer >= 1 for an absolute count, or a value in (0, 1) for a ratio."
        )

    # Ratio notation: keep the top fraction of the ranked features.
    if numeric_value < 1:
        n_features = max(1, int(math.ceil(n_candidates * numeric_value)))
        n_features = min(n_features, n_candidates)
        description = f"top {numeric_value:g} ratio ({n_features} of {n_candidates} features)"
        return n_features, description

    # Absolute notation: only whole numbers are meaningful as a feature count.
    if not float(numeric_value).is_integer():
        raise ValueError(
            f"n_features_to_select must be a whole number when >= 1, got {value!r}. "
            "Use a value in (0, 1) to select a ratio of features instead."
        )

    n_features = min(int(numeric_value), n_candidates)
    description = f"top {n_features} features"
    return n_features, description
