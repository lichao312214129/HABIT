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
Built-in habitat feature preprocessing handlers.

Import this module (via ``PreprocessingMethodFactory._ensure_builtin_handlers_loaded``)
to register all default methods. Custom handlers can follow the same pattern in
user code:

    from habit.core.habitat_analysis.feature_preprocessing import (
        BaseFeaturePreprocessing,
        register_preprocessing,
    )

    @register_preprocessing("my_method")
    class MyMethod(BaseFeaturePreprocessing):
        ...
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.preprocessing import KBinsDiscretizer

from .base_preprocessing import BaseFeaturePreprocessing, BaselineStats, register_preprocessing
from .method_config_utils import parse_winsor_limits, read_method_field
from .value_transforms import create_discretizer


# ---------------------------------------------------------------------------
# Variance / correlation filter algorithms (column-dropping)
# ---------------------------------------------------------------------------


def select_variance_columns(
    feature_df: pd.DataFrame,
    threshold: float = 0.0,
) -> List[str]:
    """
    Return column names whose variance exceeds ``threshold``.

    Args:
        feature_df: Input feature matrix (rows = samples, cols = features).
        threshold: Columns with ``var <= threshold`` are dropped. ``0.0`` removes
            only constant columns.

    Returns:
        Surviving column names. If none survive, keeps the highest-variance column.
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
    Drop low-variance columns and return the filtered DataFrame.

    Args:
        feature_df: Input feature matrix.
        threshold: Variance cut-off passed to :func:`select_variance_columns`.

    Returns:
        Filtered DataFrame with surviving columns only.
    """
    return feature_df[select_variance_columns(feature_df, threshold)]


def select_correlation_columns(
    feature_df: pd.DataFrame,
    threshold: float = 0.95,
    corr_method: str = "spearman",
) -> List[str]:
    """
    Return column names after greedy correlation pruning.

    Walks columns left-to-right; drops later columns whose absolute correlation
    with the current column exceeds ``threshold``.

    Args:
        feature_df: Input feature matrix (rows = samples, cols = features).
        threshold: Absolute-correlation cut-off.
        corr_method: Method for ``DataFrame.corr`` (``pearson`` / ``spearman`` / ``kendall``).

    Returns:
        Surviving column names. If none survive, keeps the first column.
    """
    if feature_df.shape[1] <= 1:
        return list(feature_df.columns)

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

    Args:
        feature_df: Input feature matrix.
        threshold: Correlation cut-off passed to :func:`select_correlation_columns`.
        corr_method: Correlation method for ``DataFrame.corr``.

    Returns:
        Filtered DataFrame with surviving columns only.
    """
    return feature_df[select_correlation_columns(feature_df, threshold, corr_method)]


# ---------------------------------------------------------------------------
# Per-feature scaling helpers (sequential pipeline steps)
# ---------------------------------------------------------------------------


def _per_feature_minmax_state(feature_df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Capture per-column min/max from the current pipeline step input.

    Args:
        feature_df: Numeric feature block at the minmax step.

    Returns:
        State dict with ``mins`` and ``maxs`` series aligned to columns.
    """
    return {"mins": feature_df.min(), "maxs": feature_df.max()}


def _apply_per_feature_minmax(
    feature_df: pd.DataFrame,
    state: Dict[str, pd.Series],
) -> pd.DataFrame:
    """
    Scale columns to [0, 1] using min/max learned on the current pipeline step.

    Args:
        feature_df: Feature block to transform.
        state: Output of :func:`_per_feature_minmax_state`.

    Returns:
        Min-max scaled DataFrame.
    """
    cols = feature_df.columns
    denom = (state["maxs"][cols] - state["mins"][cols]).replace(0, 1.0)
    return (feature_df - state["mins"][cols]) / denom


def _per_feature_zscore_state(feature_df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Capture per-column mean/std from the current pipeline step input.

    Args:
        feature_df: Numeric feature block at the zscore step.

    Returns:
        State dict with ``means`` and ``stds`` series aligned to columns.
    """
    return {
        "means": feature_df.mean(),
        "stds": feature_df.std().replace(0, 1.0),
    }


def _apply_per_feature_zscore(
    feature_df: pd.DataFrame,
    state: Dict[str, pd.Series],
) -> pd.DataFrame:
    """
    Z-score columns using mean/std learned on the current pipeline step.

    Args:
        feature_df: Feature block to transform.
        state: Output of :func:`_per_feature_zscore_state`.

    Returns:
        Standardized DataFrame.
    """
    cols = feature_df.columns
    return (feature_df - state["means"][cols]) / state["stds"][cols]


# ---------------------------------------------------------------------------
# Registered preprocessing handlers
# ---------------------------------------------------------------------------


@register_preprocessing("minmax")
class MinMaxPreprocessing(BaseFeaturePreprocessing):
    """Min-max scaling to [0, 1] per feature or globally."""

    changes_columns = False

    @classmethod
    def method_name(cls) -> str:
        return "minmax"

    def fit(
        self,
        feature_df: pd.DataFrame,
        method_config: Union[Dict[str, Any], BaseModel],
        baseline: Optional[BaselineStats] = None,
    ) -> Any:
        global_normalize = read_method_field(method_config, "global_normalize", False)
        if global_normalize:
            return {
                "min": float(feature_df.values.min()),
                "max": float(feature_df.values.max()),
            }
        return _per_feature_minmax_state(feature_df)

    def transform(
        self,
        feature_df: pd.DataFrame,
        method_config: Union[Dict[str, Any], BaseModel],
        state: Any,
        baseline: Optional[BaselineStats] = None,
    ) -> pd.DataFrame:
        global_normalize = read_method_field(method_config, "global_normalize", False)
        if global_normalize and state is not None:
            denom = (state["max"] - state["min"]) if state["max"] != state["min"] else 1.0
            return (feature_df - state["min"]) / denom

        if state is None or not isinstance(state, dict) or "mins" not in state:
            raise ValueError(
                "Per-feature minmax requires fit state from the current pipeline step."
            )
        return _apply_per_feature_minmax(feature_df, state)


@register_preprocessing("zscore")
class ZScorePreprocessing(BaseFeaturePreprocessing):
    """Z-score standardization per feature or globally."""

    changes_columns = False

    @classmethod
    def method_name(cls) -> str:
        return "zscore"

    def fit(
        self,
        feature_df: pd.DataFrame,
        method_config: Union[Dict[str, Any], BaseModel],
        baseline: Optional[BaselineStats] = None,
    ) -> Any:
        global_normalize = read_method_field(method_config, "global_normalize", False)
        if global_normalize:
            values = feature_df.values
            std = float(values.std()) if values.std() != 0 else 1.0
            return {"mean": float(values.mean()), "std": std}
        return _per_feature_zscore_state(feature_df)

    def transform(
        self,
        feature_df: pd.DataFrame,
        method_config: Union[Dict[str, Any], BaseModel],
        state: Any,
        baseline: Optional[BaselineStats] = None,
    ) -> pd.DataFrame:
        global_normalize = read_method_field(method_config, "global_normalize", False)
        if global_normalize and state is not None:
            return (feature_df - state["mean"]) / state["std"]

        if state is None or not isinstance(state, dict) or "means" not in state:
            raise ValueError(
                "Per-feature zscore requires fit state from the current pipeline step."
            )
        return _apply_per_feature_zscore(feature_df, state)


@register_preprocessing("robust")
class RobustPreprocessing(BaseFeaturePreprocessing):
    """Robust scaling using median and IQR."""

    changes_columns = False

    @classmethod
    def method_name(cls) -> str:
        return "robust"

    def fit(
        self,
        feature_df: pd.DataFrame,
        method_config: Union[Dict[str, Any], BaseModel],
        baseline: Optional[BaselineStats] = None,
    ) -> Any:
        global_normalize = read_method_field(method_config, "global_normalize", False)
        if global_normalize:
            flat_values = feature_df.values.flatten()
            return {
                "median": float(np.median(flat_values)),
                "q1": float(np.percentile(flat_values, 25)),
                "q3": float(np.percentile(flat_values, 75)),
            }
        return {
            "medians": feature_df.median(),
            "q1s": feature_df.quantile(0.25),
            "q3s": feature_df.quantile(0.75),
        }

    def transform(
        self,
        feature_df: pd.DataFrame,
        method_config: Union[Dict[str, Any], BaseModel],
        state: Any,
        baseline: Optional[BaselineStats] = None,
    ) -> pd.DataFrame:
        global_normalize = read_method_field(method_config, "global_normalize", False)
        if global_normalize and isinstance(state, dict) and "median" in state:
            iqr = state["q3"] - state["q1"]
            iqr = iqr if iqr != 0 else 1.0
            return (feature_df - state["median"]) / iqr

        iqr = (state["q3s"] - state["q1s"]).replace(0, 1.0)
        return (feature_df - state["medians"]) / iqr


@register_preprocessing("binning")
class BinningPreprocessing(BaseFeaturePreprocessing):
    """KBins discretization per feature or globally."""

    changes_columns = False

    @classmethod
    def method_name(cls) -> str:
        return "binning"

    def fit(
        self,
        feature_df: pd.DataFrame,
        method_config: Union[Dict[str, Any], BaseModel],
        baseline: Optional[BaselineStats] = None,
    ) -> Any:
        n_bins = int(read_method_field(method_config, "n_bins", 10))
        bin_strategy = read_method_field(method_config, "bin_strategy", "uniform")
        global_normalize = read_method_field(method_config, "global_normalize", False)

        discretizer = create_discretizer(n_bins, bin_strategy)
        if global_normalize:
            flat_values = feature_df.values.flatten().reshape(-1, 1)
            discretizer.fit(flat_values)
            return {"mode": "global", "discretizer": discretizer}

        discretizer.fit(feature_df.values)
        return {"mode": "per_feature", "discretizer": discretizer}

    def transform(
        self,
        feature_df: pd.DataFrame,
        method_config: Union[Dict[str, Any], BaseModel],
        state: Any,
        baseline: Optional[BaselineStats] = None,
    ) -> pd.DataFrame:
        discretizer: KBinsDiscretizer = state["discretizer"]
        if state["mode"] == "global":
            original_shape = feature_df.shape
            flat_values = feature_df.values.flatten().reshape(-1, 1)
            binned = discretizer.transform(flat_values)
            return pd.DataFrame(
                binned.reshape(original_shape),
                columns=feature_df.columns,
                index=feature_df.index,
            )

        binned = discretizer.transform(feature_df.values)
        return pd.DataFrame(binned, columns=feature_df.columns, index=feature_df.index)


@register_preprocessing("winsorize")
class WinsorizePreprocessing(BaseFeaturePreprocessing):
    """Clip extreme values at configured lower/upper quantiles."""

    changes_columns = False

    @classmethod
    def method_name(cls) -> str:
        return "winsorize"

    def fit(
        self,
        feature_df: pd.DataFrame,
        method_config: Union[Dict[str, Any], BaseModel],
        baseline: Optional[BaselineStats] = None,
    ) -> Any:
        winsor_limits = parse_winsor_limits(method_config)
        global_normalize = read_method_field(method_config, "global_normalize", False)

        if global_normalize:
            flat_values = feature_df.values.flatten()
            return {
                "mode": "global",
                "lower": float(np.percentile(flat_values, winsor_limits[0] * 100)),
                "upper": float(np.percentile(flat_values, (1 - winsor_limits[1]) * 100)),
            }

        return {
            "mode": "per_feature",
            "lower": feature_df.quantile(winsor_limits[0]),
            "upper": feature_df.quantile(1 - winsor_limits[1]),
        }

    def transform(
        self,
        feature_df: pd.DataFrame,
        method_config: Union[Dict[str, Any], BaseModel],
        state: Any,
        baseline: Optional[BaselineStats] = None,
    ) -> pd.DataFrame:
        if state["mode"] == "global":
            return feature_df.clip(lower=state["lower"], upper=state["upper"])
        return feature_df.clip(lower=state["lower"], upper=state["upper"], axis=1)


@register_preprocessing("log")
class LogPreprocessing(BaseFeaturePreprocessing):
    """Log transform with per-feature or global shift to handle non-positive values."""

    changes_columns = False

    @classmethod
    def method_name(cls) -> str:
        return "log"

    def fit(
        self,
        feature_df: pd.DataFrame,
        method_config: Union[Dict[str, Any], BaseModel],
        baseline: Optional[BaselineStats] = None,
    ) -> Any:
        global_normalize = read_method_field(method_config, "global_normalize", False)
        if global_normalize:
            return {"mode": "global", "offset": float(feature_df.values.min())}
        return {"mode": "per_feature", "offsets": feature_df.min()}

    def transform(
        self,
        feature_df: pd.DataFrame,
        method_config: Union[Dict[str, Any], BaseModel],
        state: Any,
        baseline: Optional[BaselineStats] = None,
    ) -> pd.DataFrame:
        if state["mode"] == "global":
            return np.log(feature_df - state["offset"] + 1.0)
        return np.log(feature_df - state["offsets"] + 1.0)


@register_preprocessing("variance_filter")
class VarianceFilterPreprocessing(BaseFeaturePreprocessing):
    """Drop columns with variance at or below a threshold."""

    changes_columns = True

    @classmethod
    def method_name(cls) -> str:
        return "variance_filter"

    def fit(
        self,
        feature_df: pd.DataFrame,
        method_config: Union[Dict[str, Any], BaseModel],
        baseline: Optional[BaselineStats] = None,
    ) -> Any:
        raw = read_method_field(method_config, "variance_threshold", None)
        threshold = float(raw) if raw is not None else 0.0
        return select_variance_columns(feature_df, threshold)

    def transform(
        self,
        feature_df: pd.DataFrame,
        method_config: Union[Dict[str, Any], BaseModel],
        state: Any,
        baseline: Optional[BaselineStats] = None,
    ) -> pd.DataFrame:
        selected_cols = [col for col in state if col in feature_df.columns]
        if not selected_cols:
            return feature_df
        return feature_df[selected_cols]


@register_preprocessing("correlation_filter")
class CorrelationFilterPreprocessing(BaseFeaturePreprocessing):
    """Greedy removal of highly correlated redundant columns."""

    changes_columns = True

    @classmethod
    def method_name(cls) -> str:
        return "correlation_filter"

    def fit(
        self,
        feature_df: pd.DataFrame,
        method_config: Union[Dict[str, Any], BaseModel],
        baseline: Optional[BaselineStats] = None,
    ) -> Any:
        thresh_raw = read_method_field(method_config, "corr_threshold", None)
        threshold = float(thresh_raw) if thresh_raw is not None else 0.95
        corr_method = read_method_field(method_config, "corr_method", None) or "spearman"
        return select_correlation_columns(feature_df, threshold, str(corr_method))

    def transform(
        self,
        feature_df: pd.DataFrame,
        method_config: Union[Dict[str, Any], BaseModel],
        state: Any,
        baseline: Optional[BaselineStats] = None,
    ) -> pd.DataFrame:
        selected_cols = [col for col in state if col in feature_df.columns]
        if not selected_cols:
            return feature_df
        return feature_df[selected_cols]
