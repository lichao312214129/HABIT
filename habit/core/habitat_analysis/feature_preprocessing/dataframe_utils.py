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
DataFrame helpers for habitat feature preprocessing.

All preprocessing handlers operate on feature columns inside a DataFrame while
preserving metadata columns (``Subject``, ``Supervoxel``, etc.) at the edges of
the pipeline.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def split_metadata_and_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a habitat feature table into metadata and numeric feature blocks.

    Args:
        df: Input table that may contain metadata and feature columns.

    Returns:
        ``(metadata_df, feature_df)`` where ``metadata_df`` may be empty.
    """
    from ..config_schemas import ResultColumns

    metadata_cols = [
        col for col in df.columns if not ResultColumns.is_feature_column(col)
    ]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if ResultColumns.is_feature_column(col)]

    metadata_df = df[metadata_cols] if metadata_cols else pd.DataFrame(index=df.index)
    feature_df = df[feature_cols] if feature_cols else pd.DataFrame(index=df.index)
    return metadata_df, feature_df


def merge_metadata_and_features(
    metadata_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    original_columns: pd.Index,
) -> pd.DataFrame:
    """
    Merge metadata and feature blocks back together preserving column order.

    Args:
        metadata_df: Non-feature columns (may be empty).
        feature_df: Transformed feature columns.
        original_columns: Column order from the input DataFrame.

    Returns:
        Combined DataFrame with columns filtered to those still present.
    """
    if metadata_df.empty:
        combined = feature_df
    else:
        combined = pd.concat([metadata_df, feature_df], axis=1)

    ordered = [col for col in original_columns if col in combined.columns]
    return combined[ordered]
