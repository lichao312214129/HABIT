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
Unified DataFrame preprocessing pipeline for habitat analysis.

Both subject-level (stateless) and group-level (stateful) paths run through
``apply_preprocessing_pipeline`` so dispatch stays consistent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from pydantic import BaseModel

from habit.utils.log_utils import get_module_logger

from .base_preprocessing import BaselineStats, PreprocessingMethodFactory
from .dataframe_utils import merge_metadata_and_features, split_metadata_and_features
from .method_config_utils import resolve_method_name
from .value_transforms import handle_extreme_values


def _prepare_feature_block(
    feature_df: pd.DataFrame,
    baseline: Optional[BaselineStats],
) -> pd.DataFrame:
    """
    Apply initial fillna and extreme-value cleaning before step handlers run.

    Args:
        feature_df: Raw numeric feature columns.
        baseline: When provided, NaNs are filled with training means.

    Returns:
        Cleaned feature DataFrame.
    """
    if feature_df.empty:
        return feature_df.copy()

    prepared = feature_df.copy()
    if baseline is not None:
        prepared = prepared.fillna(baseline.means)

    cleaned_values = handle_extreme_values(prepared.values, strategy="mean_replacement")
    return pd.DataFrame(
        cleaned_values,
        columns=prepared.columns,
        index=prepared.index,
    )


def apply_preprocessing_pipeline(
    df: pd.DataFrame,
    methods: List[Union[Dict[str, Any], BaseModel]],
    *,
    step_states: Optional[List[Any]] = None,
    baseline: Optional[BaselineStats] = None,
    fit: bool = False,
    level_label: str = "group-level",
) -> Tuple[pd.DataFrame, Optional[BaselineStats], List[Any]]:
    """
    Run a sequence of registered preprocessing methods on a DataFrame.

    Args:
        df: Input table (may include metadata columns).
        methods: Ordered preprocessing step configurations.
        step_states: Per-step state list from a previous ``fit`` (required when
            ``fit=False``).
        baseline: Baseline stats from training (required when ``fit=False``).
        fit: When ``True``, learn ``baseline`` and per-step states from ``df``.
        level_label: Log prefix distinguishing subject-level vs group-level runs
            (e.g. ``"subject-level"`` or ``"group-level"``).

    Returns:
        ``(output_df, baseline, step_states)``.

    Raises:
        ValueError: When ``fit=False`` but ``step_states`` or ``baseline`` is missing.
    """
    if not methods:
        return df.copy(), baseline, step_states or []

    metadata_df, feature_df = split_metadata_and_features(df)
    if feature_df.empty:
        return df.copy(), baseline, step_states or []

    if fit:
        baseline = BaselineStats.from_dataframe(feature_df)
        feature_df = _prepare_feature_block(feature_df, baseline)
        new_states: List[Any] = []
        logger = get_module_logger(__name__)
        n_features_in = feature_df.shape[1]
        logger.info(
            f"{level_label} preprocessing pipeline started (fit): "
            f"rows={len(feature_df)}, feature_columns={n_features_in}, "
            f"steps={len(methods)}"
        )

        for step_idx, method_config in enumerate(methods, start=1):
            method_name = resolve_method_name(method_config)
            n_before = feature_df.shape[1]
            handler = PreprocessingMethodFactory.get_handler_for_config(method_config)
            state = handler.fit(feature_df, method_config, baseline)
            feature_df = handler.transform(feature_df, method_config, state, baseline)
            new_states.append(state)
            n_after = feature_df.shape[1]
            logger.info(
                f"{level_label} preprocessing step {step_idx}/{len(methods)} "
                f"'{method_name}': feature_columns {n_before} -> {n_after} "
                f"(rows={len(feature_df)})"
            )
            if n_after != n_before:
                if handler.changes_columns:
                    logger.info(
                        f"{level_label} preprocessing '{method_name}' filtered "
                        f"{n_before - n_after} feature column(s)"
                    )
                else:
                    logger.warning(
                        f"{level_label} preprocessing '{method_name}' changed "
                        f"feature_columns {n_before} -> {n_after} unexpectedly"
                    )

        logger.info(
            f"{level_label} preprocessing pipeline finished (fit): "
            f"feature_columns {n_features_in} -> {feature_df.shape[1]}"
        )

        output = merge_metadata_and_features(metadata_df, feature_df, df.columns)
        return output, baseline, new_states

    if baseline is None or step_states is None:
        raise ValueError(
            "baseline and step_states are required when fit=False. "
            "Call fit first or pass fit=True."
        )
    if len(step_states) != len(methods):
        raise ValueError(
            f"step_states length ({len(step_states)}) must match methods ({len(methods)})."
        )

    feature_df = _prepare_feature_block(feature_df, baseline)
    logger = get_module_logger(__name__)
    n_features_in = feature_df.shape[1]
    logger.info(
        f"{level_label} preprocessing pipeline started (transform): "
        f"rows={len(feature_df)}, feature_columns={n_features_in}, "
        f"steps={len(methods)}"
    )

    for step_idx, (method_config, state) in enumerate(
        zip(methods, step_states), start=1
    ):
        method_name = resolve_method_name(method_config)
        n_before = feature_df.shape[1]
        handler = PreprocessingMethodFactory.get_handler_for_config(method_config)
        feature_df = handler.transform(feature_df, method_config, state, baseline)
        n_after = feature_df.shape[1]
        logger.info(
            f"{level_label} preprocessing step {step_idx}/{len(methods)} "
            f"'{method_name}': feature_columns {n_before} -> {n_after} "
            f"(rows={len(feature_df)})"
        )
        if n_after != n_before:
            if handler.changes_columns:
                logger.info(
                    f"{level_label} preprocessing '{method_name}' filtered "
                    f"{n_before - n_after} feature column(s)"
                )
            else:
                logger.warning(
                    f"{level_label} preprocessing '{method_name}' changed "
                    f"feature_columns {n_before} -> {n_after} unexpectedly"
                )

    logger.info(
        f"{level_label} preprocessing pipeline finished (transform): "
        f"feature_columns {n_features_in} -> {feature_df.shape[1]}"
    )

    output = merge_metadata_and_features(metadata_df, feature_df, df.columns)
    return output, baseline, step_states


def apply_stateless_preprocessing(
    df: pd.DataFrame,
    methods: List[Union[Dict[str, Any], BaseModel]],
) -> pd.DataFrame:
    """
    Fit and transform in one pass (subject-level preprocessing).

    Args:
        df: Input feature table.
        methods: Ordered preprocessing steps.

    Returns:
        Transformed DataFrame.
    """
    output_df, _, _ = apply_preprocessing_pipeline(
        df,
        methods,
        fit=True,
        level_label="subject-level",
    )
    return output_df
