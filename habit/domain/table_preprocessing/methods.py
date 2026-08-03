# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Built-in table preprocessors (domain ``table_preprocessor``).

The eight methods are numerically equivalent to the v0.1 handlers in
``habit.core.habitat_analysis.feature_preprocessing.builtin_methods``, but
reshape the interface around the :class:`~habit.domain.table_protocols.TablePreprocessor`
protocol: constructor parameters are explicit typed arguments (validated by a
registered Pydantic schema), ``fit`` learns state from the TRAINING table and
stores it on the instance, and ``transform`` applies that state to any
row-aligned table -- which is exactly what a train/predict split needs to not
leak test statistics into features.

Two methods (``variance_filter``, ``correlation_filter``) change the column
set: they learn the surviving columns at fit time and restrict every later
table to that same subset.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel

from habit.api.exceptions import HABITAPIError
from habit.contracts.table import FeatureTable
from habit.domain.table_preprocessing._base import (
    fit_feature_block,
    replace_feature_values,
)
from habit.domain.table_preprocessing.registry import TablePreprocessorRegistry
from habit.spec.specs import Spec

__all__ = [
    "MinMaxPreprocessor",
    "MinMaxPreprocessorParams",
    "ZScorePreprocessor",
    "ZScorePreprocessorParams",
    "RobustPreprocessor",
    "RobustPreprocessorParams",
    "BinningPreprocessor",
    "BinningPreprocessorParams",
    "WinsorizePreprocessor",
    "WinsorizePreprocessorParams",
    "LogPreprocessor",
    "LogPreprocessorParams",
    "VarianceFilterPreprocessor",
    "VarianceFilterPreprocessorParams",
    "CorrelationFilterPreprocessor",
    "CorrelationFilterPreprocessorParams",
]


class _FittedPreprocessor:
    """
    Shared fitted-state bookkeeping for the built-in preprocessors.

    Tracks the feature columns seen at ``fit`` time and provides the
    transform-side guard that turns silent schema drift into an explicit
    error. Subclasses set ``_spec_name`` and implement ``fit``/``transform``.
    """

    _spec_name: str = ""

    def __init__(self) -> None:
        self._state: Optional[Dict[str, Any]] = None
        self._fit_columns: Tuple[str, ...] = ()

    def _remember_fit(self, table: FeatureTable, state: Dict[str, Any]) -> None:
        """Store the fitted state together with the fitted column schema."""
        self._state = state
        self._fit_columns = tuple(table.feature_columns)

    def _block_for_transform(self, table: FeatureTable) -> pd.DataFrame:
        """Return the validated feature block of a table to transform."""
        if self._state is None:
            raise HABITAPIError(
                f"table_preprocessor.{self._spec_name} must be fitted before "
                "transform."
            )
        return fit_feature_block(
            table, self._fit_columns, owner=f"table_preprocessor.{self._spec_name}"
        )

    def _finish(
        self,
        table: FeatureTable,
        values: pd.DataFrame,
        feature_columns: Tuple[str, ...],
    ) -> FeatureTable:
        """Assemble the output table with this preprocessor's provenance."""
        return replace_feature_values(table, values, feature_columns, self.spec)

    @property
    def spec(self) -> Spec:  # pragma: no cover - subclasses override
        raise NotImplementedError


# ---------------------------------------------------------------------------
# minmax / zscore / robust: per-feature (or global) affine scalers
# ---------------------------------------------------------------------------


class MinMaxPreprocessorParams(BaseModel):
    """Constructor parameters for :class:`MinMaxPreprocessor`."""

    #: When true, learn ONE (min, max) over the whole feature block instead of
    #: per-column statistics (the v0.1 ``global_normalize`` flag).
    global_normalize: bool = False


@TablePreprocessorRegistry.register("minmax")
class MinMaxPreprocessor(_FittedPreprocessor):
    """
    Min-max scaling of every feature to [0, 1].

    Per feature by default (each column scaled by its own training minimum and
    maximum); ``global_normalize=True`` scales the whole block by one global
    pair, matching the v0.1 method of the same name. A constant training
    column divides by 1.0 so it maps to 0 rather than NaN.
    """

    _spec_name = "minmax"

    def __init__(self, global_normalize: bool = False) -> None:
        super().__init__()
        self._global_normalize = bool(global_normalize)

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(
            name=self._spec_name,
            params={"global_normalize": self._global_normalize},
        )

    def fit(self, table: FeatureTable) -> "MinMaxPreprocessor":
        """Learn per-column (or global) min/max from the training table."""
        block = table.frame[list(table.feature_columns)]
        if self._global_normalize:
            self._remember_fit(
                table,
                {
                    "mode": "global",
                    "min": float(block.values.min()),
                    "max": float(block.values.max()),
                },
            )
        else:
            self._remember_fit(
                table, {"mode": "per_feature", "mins": block.min(), "maxs": block.max()}
            )
        return self

    def transform(self, table: FeatureTable) -> FeatureTable:
        """Scale the table's features with the training min/max."""
        block = self._block_for_transform(table)
        state = self._state
        assert state is not None  # guaranteed by _block_for_transform
        if state["mode"] == "global":
            denom = (state["max"] - state["min"]) if state["max"] != state["min"] else 1.0
            transformed = (block - state["min"]) / denom
        else:
            mins = state["mins"][list(block.columns)]
            denom = (state["maxs"][list(block.columns)] - mins).replace(0, 1.0)
            transformed = (block - mins) / denom
        return self._finish(table, transformed, self._fit_columns)


class ZScorePreprocessorParams(BaseModel):
    """Constructor parameters for :class:`ZScorePreprocessor`."""

    #: When true, learn ONE (mean, std) over the whole feature block.
    global_normalize: bool = False


@TablePreprocessorRegistry.register("zscore")
class ZScorePreprocessor(_FittedPreprocessor):
    """
    Z-score standardisation of every feature.

    Per feature by default (training mean/std per column);
    ``global_normalize=True`` standardises by one global pair. A zero-variance
    training column divides by 1.0 so it maps to 0 rather than NaN.
    """

    _spec_name = "zscore"

    def __init__(self, global_normalize: bool = False) -> None:
        super().__init__()
        self._global_normalize = bool(global_normalize)

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(
            name=self._spec_name,
            params={"global_normalize": self._global_normalize},
        )

    def fit(self, table: FeatureTable) -> "ZScorePreprocessor":
        """Learn per-column (or global) mean/std from the training table."""
        block = table.frame[list(table.feature_columns)]
        if self._global_normalize:
            std = float(block.values.std())
            self._remember_fit(
                table,
                {
                    "mode": "global",
                    "mean": float(block.values.mean()),
                    "std": std if std != 0 else 1.0,
                },
            )
        else:
            self._remember_fit(
                table,
                {
                    "mode": "per_feature",
                    "means": block.mean(),
                    "stds": block.std().replace(0, 1.0),
                },
            )
        return self

    def transform(self, table: FeatureTable) -> FeatureTable:
        """Standardise the table's features with the training mean/std."""
        block = self._block_for_transform(table)
        state = self._state
        assert state is not None
        if state["mode"] == "global":
            transformed = (block - state["mean"]) / state["std"]
        else:
            means = state["means"][list(block.columns)]
            stds = state["stds"][list(block.columns)]
            transformed = (block - means) / stds
        return self._finish(table, transformed, self._fit_columns)


class RobustPreprocessorParams(BaseModel):
    """Constructor parameters for :class:`RobustPreprocessor`."""

    #: When true, learn ONE (median, IQR) over the whole feature block.
    global_normalize: bool = False


@TablePreprocessorRegistry.register("robust")
class RobustPreprocessor(_FittedPreprocessor):
    """
    Robust scaling of every feature by training median and IQR.

    The outlier-resistant alternative to z-score: columns are centred on the
    training median and divided by the training interquartile range, so a few
    extreme values do not compress the bulk of the distribution. A zero-IQR
    column divides by 1.0.
    """

    _spec_name = "robust"

    def __init__(self, global_normalize: bool = False) -> None:
        super().__init__()
        self._global_normalize = bool(global_normalize)

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(
            name=self._spec_name,
            params={"global_normalize": self._global_normalize},
        )

    def fit(self, table: FeatureTable) -> "RobustPreprocessor":
        """Learn per-column (or global) median/IQR from the training table."""
        block = table.frame[list(table.feature_columns)]
        if self._global_normalize:
            flat = block.values.flatten()
            self._remember_fit(
                table,
                {
                    "mode": "global",
                    "median": float(np.median(flat)),
                    "q1": float(np.percentile(flat, 25)),
                    "q3": float(np.percentile(flat, 75)),
                },
            )
        else:
            self._remember_fit(
                table,
                {
                    "mode": "per_feature",
                    "medians": block.median(),
                    "q1s": block.quantile(0.25),
                    "q3s": block.quantile(0.75),
                },
            )
        return self

    def transform(self, table: FeatureTable) -> FeatureTable:
        """Robust-scale the table's features with the training median/IQR."""
        block = self._block_for_transform(table)
        state = self._state
        assert state is not None
        if state["mode"] == "global":
            iqr = state["q3"] - state["q1"]
            iqr = iqr if iqr != 0 else 1.0
            transformed = (block - state["median"]) / iqr
        else:
            iqr_series = (state["q3s"] - state["q1s"]).replace(0, 1.0)
            transformed = (block - state["medians"]) / iqr_series[list(block.columns)]
        return self._finish(table, transformed, self._fit_columns)


# ---------------------------------------------------------------------------
# binning / winsorize / log: value-shape transforms
# ---------------------------------------------------------------------------


class BinningPreprocessorParams(BaseModel):
    """Constructor parameters for :class:`BinningPreprocessor`."""

    #: Number of bins per feature (or globally).
    n_bins: int = 10
    #: Binning strategy passed to sklearn's ``KBinsDiscretizer``.
    bin_strategy: str = "uniform"
    #: When true, learn bin edges over the flattened feature block.
    global_normalize: bool = False


@TablePreprocessorRegistry.register("binning")
class BinningPreprocessor(_FittedPreprocessor):
    """
    K-bins discretisation of every feature to ordinal bin indices.

    Wraps sklearn's ``KBinsDiscretizer(encode="ordinal")`` exactly as the v0.1
    method did: edges are learned on the training table only, and prediction
    tables are binned with those frozen edges. The ``kmeans`` strategy is
    stochastic, so this component is :class:`~habit.domain.protocols.Seedable`.
    """

    _spec_name = "binning"

    def __init__(
        self,
        n_bins: int = 10,
        bin_strategy: str = "uniform",
        global_normalize: bool = False,
    ) -> None:
        super().__init__()
        self._n_bins = int(n_bins)
        self._bin_strategy = str(bin_strategy)
        self._global_normalize = bool(global_normalize)
        self._seed: Optional[int] = None

    def set_random_state(self, seed: int) -> None:
        """Set the random state used by stochastic bin strategies (kmeans)."""
        self._seed = int(seed)

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(
            name=self._spec_name,
            params={
                "n_bins": self._n_bins,
                "bin_strategy": self._bin_strategy,
                "global_normalize": self._global_normalize,
            },
        )

    def _make_discretizer(self) -> Any:
        """Build the sklearn discretizer (lazy import keeps sklearn out of L3)."""
        from sklearn.preprocessing import KBinsDiscretizer

        return KBinsDiscretizer(
            n_bins=self._n_bins,
            encode="ordinal",
            strategy=self._bin_strategy,
            random_state=self._seed,
        )

    def fit(self, table: FeatureTable) -> "BinningPreprocessor":
        """Learn bin edges from the training table."""
        block = table.frame[list(table.feature_columns)]
        discretizer = self._make_discretizer()
        if self._global_normalize:
            discretizer.fit(block.values.flatten().reshape(-1, 1))
            mode = "global"
        else:
            discretizer.fit(block.values)
            mode = "per_feature"
        self._remember_fit(table, {"mode": mode, "discretizer": discretizer})
        return self

    def transform(self, table: FeatureTable) -> FeatureTable:
        """Discretise the table's features with the training bin edges."""
        block = self._block_for_transform(table)
        state = self._state
        assert state is not None
        discretizer = state["discretizer"]
        if state["mode"] == "global":
            shape = block.shape
            binned = discretizer.transform(block.values.flatten().reshape(-1, 1))
            transformed = pd.DataFrame(
                binned.reshape(shape), columns=block.columns, index=block.index
            )
        else:
            transformed = pd.DataFrame(
                discretizer.transform(block.values),
                columns=block.columns,
                index=block.index,
            )
        return self._finish(table, transformed, self._fit_columns)


class WinsorizePreprocessorParams(BaseModel):
    """Constructor parameters for :class:`WinsorizePreprocessor`."""

    #: Lower/upper tail fractions clipped at the corresponding quantiles.
    winsor_limits: Tuple[float, float] = (0.05, 0.05)
    #: When true, learn clip bounds over the flattened feature block.
    global_normalize: bool = False


@TablePreprocessorRegistry.register("winsorize")
class WinsorizePreprocessor(_FittedPreprocessor):
    """
    Clip extreme feature values at training quantiles.

    Values below the ``winsor_limits[0]`` quantile and above the
    ``1 - winsor_limits[1]`` quantile of the TRAINING distribution are clipped
    to those bounds, so outliers still influence the model but cannot dominate
    it. Bounds learned at fit time are applied unchanged to prediction data.
    """

    _spec_name = "winsorize"

    def __init__(
        self,
        winsor_limits: Tuple[float, float] = (0.05, 0.05),
        global_normalize: bool = False,
    ) -> None:
        super().__init__()
        limits = tuple(float(v) for v in winsor_limits)
        if len(limits) != 2 or not all(0.0 <= v < 0.5 for v in limits):
            raise HABITAPIError(
                "winsor_limits must be two fractions in [0, 0.5); got "
                f"{winsor_limits!r}."
            )
        self._winsor_limits = limits
        self._global_normalize = bool(global_normalize)

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(
            name=self._spec_name,
            params={
                "winsor_limits": list(self._winsor_limits),
                "global_normalize": self._global_normalize,
            },
        )

    def fit(self, table: FeatureTable) -> "WinsorizePreprocessor":
        """Learn clip bounds from the training table."""
        block = table.frame[list(table.feature_columns)]
        lower_q, upper_q = self._winsor_limits
        if self._global_normalize:
            flat = block.values.flatten()
            self._remember_fit(
                table,
                {
                    "mode": "global",
                    "lower": float(np.percentile(flat, lower_q * 100)),
                    "upper": float(np.percentile(flat, (1 - upper_q) * 100)),
                },
            )
        else:
            self._remember_fit(
                table,
                {
                    "mode": "per_feature",
                    "lower": block.quantile(lower_q),
                    "upper": block.quantile(1 - upper_q),
                },
            )
        return self

    def transform(self, table: FeatureTable) -> FeatureTable:
        """Clip the table's features at the training quantile bounds."""
        block = self._block_for_transform(table)
        state = self._state
        assert state is not None
        if state["mode"] == "global":
            transformed = block.clip(lower=state["lower"], upper=state["upper"])
        else:
            transformed = block.clip(
                lower=state["lower"], upper=state["upper"], axis=1
            )
        return self._finish(table, transformed, self._fit_columns)


class LogPreprocessorParams(BaseModel):
    """Constructor parameters for :class:`LogPreprocessor`."""

    #: When true, shift by ONE global minimum over the feature block.
    global_normalize: bool = False


@TablePreprocessorRegistry.register("log")
class LogPreprocessor(_FittedPreprocessor):
    """
    Log transform ``log(x - min_train + 1)`` of every feature.

    The shift by the TRAINING minimum plus one guarantees a positive argument
    for any value seen at fit time, so right-skewed features (voxel counts,
    volumes) become approximately symmetric without hand-tuned offsets.
    """

    _spec_name = "log"

    def __init__(self, global_normalize: bool = False) -> None:
        super().__init__()
        self._global_normalize = bool(global_normalize)

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(
            name=self._spec_name,
            params={"global_normalize": self._global_normalize},
        )

    def fit(self, table: FeatureTable) -> "LogPreprocessor":
        """Learn the shift offsets from the training table."""
        block = table.frame[list(table.feature_columns)]
        if self._global_normalize:
            self._remember_fit(
                table, {"mode": "global", "offset": float(block.values.min())}
            )
        else:
            self._remember_fit(
                table, {"mode": "per_feature", "offsets": block.min()}
            )
        return self

    def transform(self, table: FeatureTable) -> FeatureTable:
        """Log-transform the table's features with the training offsets."""
        block = self._block_for_transform(table)
        state = self._state
        assert state is not None
        if state["mode"] == "global":
            transformed = pd.DataFrame(
                np.log(block.values - state["offset"] + 1.0),
                columns=block.columns,
                index=block.index,
            )
        else:
            transformed = pd.DataFrame(
                np.log(block.values - state["offsets"][list(block.columns)].values + 1.0),
                columns=block.columns,
                index=block.index,
            )
        return self._finish(table, transformed, self._fit_columns)


# ---------------------------------------------------------------------------
# variance_filter / correlation_filter: column-dropping preprocessors
# ---------------------------------------------------------------------------


def _select_variance_columns(feature_df: pd.DataFrame, threshold: float) -> List[str]:
    """
    Return column names whose variance exceeds ``threshold``.

    Mirrors the v0.1 rule: when no column survives, the highest-variance
    column is kept so the pipeline never produces an empty feature block.
    """
    variances = feature_df.var()
    selected = variances[variances > threshold].index.tolist()
    if not selected:
        selected = [variances.sort_values(ascending=False).index[0]]
    return selected


def _select_correlation_columns(
    feature_df: pd.DataFrame,
    threshold: float,
    corr_method: str,
) -> List[str]:
    """
    Return column names after greedy absolute-correlation pruning.

    Walks columns left-to-right and drops later columns whose absolute
    correlation with a kept column exceeds ``threshold`` (the v0.1 algorithm,
    which favours earlier columns deterministically).
    """
    if feature_df.shape[1] <= 1:
        return list(feature_df.columns)
    corr = feature_df.corr(method=corr_method).abs().fillna(0.0)
    kept_cols = list(feature_df.columns)
    i = 0
    while i < len(kept_cols):
        current = kept_cols[i]
        to_remove = [
            kept_cols[j]
            for j in range(i + 1, len(kept_cols))
            if corr.loc[current, kept_cols[j]] > threshold
        ]
        kept_cols = [col for col in kept_cols if col not in to_remove]
        i += 1
    if not kept_cols:
        kept_cols = [feature_df.columns[0]]
    return kept_cols


class VarianceFilterPreprocessorParams(BaseModel):
    """Constructor parameters for :class:`VarianceFilterPreprocessor`."""

    #: Columns with variance at or below this value are dropped (0 removes
    #: only constant columns).
    variance_threshold: float = 0.0


@TablePreprocessorRegistry.register("variance_filter")
class VarianceFilterPreprocessor(_FittedPreprocessor):
    """
    Drop feature columns whose training variance is at/below a threshold.

    Constant and near-constant features carry no discriminative information
    but still cost model capacity; removing them on TRAINING variances and
    applying the same column subset to prediction data keeps the schema fixed
    between fit and predict.
    """

    _spec_name = "variance_filter"

    def __init__(self, variance_threshold: float = 0.0) -> None:
        super().__init__()
        self._variance_threshold = float(variance_threshold)

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(
            name=self._spec_name,
            params={"variance_threshold": self._variance_threshold},
        )

    def fit(self, table: FeatureTable) -> "VarianceFilterPreprocessor":
        """Learn the surviving column subset from the training table."""
        block = table.frame[list(table.feature_columns)]
        columns = _select_variance_columns(block, self._variance_threshold)
        self._remember_fit(table, {"columns": columns})
        return self

    def transform(self, table: FeatureTable) -> FeatureTable:
        """Restrict the table to the fitted surviving columns."""
        block = self._block_for_transform(table)
        state = self._state
        assert state is not None
        kept: Tuple[str, ...] = tuple(state["columns"])
        return self._finish(table, block[list(kept)], kept)


class CorrelationFilterPreprocessorParams(BaseModel):
    """Constructor parameters for :class:`CorrelationFilterPreprocessor`."""

    #: Absolute-correlation cut-off above which later columns are dropped.
    corr_threshold: float = 0.95
    #: Correlation method for ``DataFrame.corr``.
    corr_method: str = "spearman"


@TablePreprocessorRegistry.register("correlation_filter")
class CorrelationFilterPreprocessor(_FittedPreprocessor):
    """
    Greedily drop redundant, highly correlated feature columns.

    Radiomics feature sets are famously collinear; keeping one representative
    per correlated cluster (computed on TRAINING correlations) reduces
    dimensionality without touching discriminative power. The walk is
    left-to-right, so the surviving subset is deterministic.
    """

    _spec_name = "correlation_filter"

    def __init__(
        self,
        corr_threshold: float = 0.95,
        corr_method: str = "spearman",
    ) -> None:
        super().__init__()
        self._corr_threshold = float(corr_threshold)
        self._corr_method = str(corr_method)

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(
            name=self._spec_name,
            params={
                "corr_threshold": self._corr_threshold,
                "corr_method": self._corr_method,
            },
        )

    def fit(self, table: FeatureTable) -> "CorrelationFilterPreprocessor":
        """Learn the surviving column subset from the training table."""
        block = table.frame[list(table.feature_columns)]
        columns = _select_correlation_columns(
            block, self._corr_threshold, self._corr_method
        )
        self._remember_fit(table, {"columns": columns})
        return self

    def transform(self, table: FeatureTable) -> FeatureTable:
        """Restrict the table to the fitted surviving columns."""
        block = self._block_for_transform(table)
        state = self._state
        assert state is not None
        kept: Tuple[str, ...] = tuple(state["columns"])
        return self._finish(table, block[list(kept)], kept)


# ---------------------------------------------------------------------------
# Parameter schemas (registered after the classes so names resolve)
# ---------------------------------------------------------------------------

TablePreprocessorRegistry.register_params_model("minmax", MinMaxPreprocessorParams)
TablePreprocessorRegistry.register_params_model("zscore", ZScorePreprocessorParams)
TablePreprocessorRegistry.register_params_model("robust", RobustPreprocessorParams)
TablePreprocessorRegistry.register_params_model("binning", BinningPreprocessorParams)
TablePreprocessorRegistry.register_params_model("winsorize", WinsorizePreprocessorParams)
TablePreprocessorRegistry.register_params_model("log", LogPreprocessorParams)
TablePreprocessorRegistry.register_params_model(
    "variance_filter", VarianceFilterPreprocessorParams
)
TablePreprocessorRegistry.register_params_model(
    "correlation_filter", CorrelationFilterPreprocessorParams
)
