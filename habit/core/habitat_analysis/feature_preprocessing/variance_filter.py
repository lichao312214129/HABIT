"""
Variance-filter algorithm.

Drops columns whose variance lies at or below ``threshold``. If every column
falls below the threshold, the single column with the largest variance is
kept as a fallback so downstream code is never handed an empty frame.

The algorithm is exposed in two forms:

* :func:`select_variance_columns` returns just the surviving column names.
  Used by stateful preprocessors that need to cache the column list at fit
  time and replay it at predict time (see ``PreprocessingState``).
* :func:`apply_variance_filter` returns the filtered DataFrame directly.
  Used by stateless / per-subject paths that simply transform once.
"""
from __future__ import annotations

from typing import List

import pandas as pd


def select_variance_columns(
    feature_df: pd.DataFrame,
    threshold: float = 0.0,
) -> List[str]:
    """
    Return the names of columns whose variance exceeds ``threshold``.

    Args:
        feature_df: Input feature matrix (rows = samples, cols = features).
        threshold: Columns with ``var <= threshold`` are dropped. A threshold
            of ``0.0`` removes only constant columns.

    Returns:
        List of surviving column names. If no column survives, the column
        with the highest variance is kept as a fallback so callers never
        receive an empty list.
    """
    variances = feature_df.var()
    selected = variances[variances > threshold].index.tolist()
    if not selected:
        selected = [variances.sort_values(ascending=False).index[0]]
    return selected


def apply_variance_filter(
    feature_df: pd.DataFrame,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Drop low-variance feature columns and return the filtered DataFrame.

    Equivalent to ``feature_df[select_variance_columns(feature_df, threshold)]``.
    """
    return feature_df[select_variance_columns(feature_df, threshold)]
