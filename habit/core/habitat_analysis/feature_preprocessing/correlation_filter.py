"""
Correlation-filter algorithm.

Greedy column-wise pruning: walk the columns left-to-right, and for each
"current" column drop every later column whose absolute correlation with it
exceeds ``threshold``. Surviving columns are therefore mutually
sub-threshold-correlated.

The algorithm is order-dependent in which column it keeps when multiple
columns are mutually correlated. This matches the historical behaviour
that lived inline inside ``FeatureService._apply_preprocessing`` and inside
``PreprocessingState``.

The algorithm is exposed in two forms:

* :func:`select_correlation_columns` returns just the surviving column
  names. Used by stateful preprocessors that need to cache the column
  list at fit time and replay it at predict time.
* :func:`apply_correlation_filter` returns the filtered DataFrame directly.
"""
from __future__ import annotations

from typing import List

import pandas as pd


def select_correlation_columns(
    feature_df: pd.DataFrame,
    threshold: float = 0.95,
    corr_method: str = "spearman",
) -> List[str]:
    """
    Return the names of columns left after greedy correlation pruning.

    Args:
        feature_df: Input feature matrix (rows = samples, cols = features).
        threshold: Absolute-correlation cut-off. A column whose
            ``|corr(current, candidate)| > threshold`` is dropped.
        corr_method: Correlation method passed to ``DataFrame.corr``
            (e.g. ``"pearson"``, ``"spearman"``, ``"kendall"``).

    Returns:
        List of surviving column names. If every column gets filtered out,
        the first column is kept as a fallback.
    """
    if feature_df.shape[1] <= 1:
        return list(feature_df.columns)

    # Absolute correlation; NaNs (zero-variance columns) treated as zero corr.
    corr = feature_df.corr(method=corr_method).abs().fillna(0.0)

    kept_cols = list(feature_df.columns)
    i = 0
    while i < len(kept_cols):
        current = kept_cols[i]
        to_remove = []
        for j in range(i + 1, len(kept_cols)):
            candidate = kept_cols[j]
            if corr.loc[current, candidate] > threshold:
                to_remove.append(candidate)
        kept_cols = [col for col in kept_cols if col not in to_remove]
        i += 1

    if not kept_cols:
        kept_cols = [feature_df.columns[0]]
    return kept_cols


def apply_correlation_filter(
    feature_df: pd.DataFrame,
    threshold: float = 0.95,
    corr_method: str = "spearman",
) -> pd.DataFrame:
    """
    Drop highly-correlated columns and return the filtered DataFrame.

    Equivalent to
    ``feature_df[select_correlation_columns(feature_df, threshold, corr_method)]``.
    """
    return feature_df[select_correlation_columns(feature_df, threshold, corr_method)]
