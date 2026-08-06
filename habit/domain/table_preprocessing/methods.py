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

These operate on a MODELLING table: one row per subject, with identifier and
outcome columns alongside the features. That is what separates this domain from
``habit.domain.feature_preprocessing``, whose rows are voxels or supervoxels on
the way to a habitat definition.

The arithmetic, however, is identical, so both domains delegate to the same L0
kernel (``habit.kernels.feature_transforms``) rather than each carrying its own
copy. Rescaling is rescaling; a formula written twice is a formula that will
eventually disagree with itself, and a silent disagreement here would mean two
parts of one study normalised their features differently.

What this layer adds on top of the kernel is the table contract: constructor
parameters are explicit typed arguments (validated by a registered Pydantic
schema), ``fit`` learns state from the TRAINING table, and ``transform``
applies that state to any row-aligned table -- which is exactly what a
train/predict split needs to not leak test statistics into features.

Two methods (``variance_filter``, ``correlation_filter``) change the column
set: they learn the surviving columns at fit time and restrict every later
table to that same subset.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, ConfigDict

from habit.exceptions import HABITAPIError
from habit.contracts.table import FeatureTable
from habit.domain.table_preprocessing._base import (
    fit_feature_block,
    replace_feature_values,
)
from habit.domain.table_preprocessing.registry import TablePreprocessorRegistry
from habit.kernels import feature_transforms as _kernel
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


class _ScopedScaler(_FittedPreprocessor):
    """
    Base for the affine scalers, which differ only in which kernel they call.

    Each subclass names its ``fit_*``/``apply_*`` pair; everything else --
    parameter handling, spec, fitted-state bookkeeping -- is identical, and
    writing it once is what keeps the three scalers from drifting apart.
    """

    _fit_kernel: Any = None
    _apply_kernel: Any = None

    def __init__(self, across_features: bool = False) -> None:
        super().__init__()
        self._across_features = bool(across_features)

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(
            name=self._spec_name,
            params={"across_features": self._across_features},
        )

    def fit(self, table: FeatureTable) -> "_ScopedScaler":
        """Learn the scaling statistics from the training table."""
        block = table.frame[list(table.feature_columns)]
        self._remember_fit(
            table, type(self)._fit_kernel(block, self._across_features)
        )
        return self

    def transform(self, table: FeatureTable) -> FeatureTable:
        """Rescale the table's features with the training statistics."""
        block = self._block_for_transform(table)
        state = self._state
        assert state is not None  # guaranteed by _block_for_transform
        return self._finish(
            table, type(self)._apply_kernel(block, state), self._fit_columns
        )


class MinMaxPreprocessorParams(BaseModel):
    """Constructor parameters for :class:`MinMaxPreprocessor`."""

    model_config = ConfigDict(extra="forbid")
    #: When true, learn ONE (min, max) across all feature columns instead of
    #: per-column statistics. Renamed from v0.1's ``global_normalize``, which
    #: read as "use cohort-wide statistics" and never meant that.
    across_features: bool = False


@TablePreprocessorRegistry.register("minmax")
class MinMaxPreprocessor(_ScopedScaler):
    """
    Min-max scaling of every feature to [0, 1].

    Per feature by default (each column scaled by its own training minimum and
    maximum); ``across_features=True`` scales the whole block by one pair,
    matching the v0.1 method of the same name. A constant training column
    divides by 1.0 so it maps to 0 rather than NaN.
    """

    _spec_name = "minmax"
    _fit_kernel = staticmethod(_kernel.fit_minmax)
    _apply_kernel = staticmethod(_kernel.apply_minmax)


class ZScorePreprocessorParams(BaseModel):
    """Constructor parameters for :class:`ZScorePreprocessor`."""

    model_config = ConfigDict(extra="forbid")
    #: When true, learn ONE (mean, std) across all feature columns.
    across_features: bool = False


@TablePreprocessorRegistry.register("zscore")
class ZScorePreprocessor(_ScopedScaler):
    """
    Z-score standardisation of every feature.

    Per feature by default (training mean/std per column);
    ``across_features=True`` standardises by one pair. A zero-variance
    training column divides by 1.0 so it maps to 0 rather than NaN.
    """

    _spec_name = "zscore"
    _fit_kernel = staticmethod(_kernel.fit_zscore)
    _apply_kernel = staticmethod(_kernel.apply_zscore)


class RobustPreprocessorParams(BaseModel):
    """Constructor parameters for :class:`RobustPreprocessor`."""

    model_config = ConfigDict(extra="forbid")
    #: When true, learn ONE (median, IQR) across all feature columns.
    across_features: bool = False


@TablePreprocessorRegistry.register("robust")
class RobustPreprocessor(_ScopedScaler):
    """
    Robust scaling of every feature by training median and IQR.

    The outlier-resistant alternative to z-score: columns are centred on the
    training median and divided by the training interquartile range, so a few
    extreme values do not compress the bulk of the distribution. A zero-IQR
    column divides by 1.0.
    """

    _spec_name = "robust"
    _fit_kernel = staticmethod(_kernel.fit_robust)
    _apply_kernel = staticmethod(_kernel.apply_robust)


# ---------------------------------------------------------------------------
# binning / winsorize / log: value-shape transforms
# ---------------------------------------------------------------------------


class BinningPreprocessorParams(BaseModel):
    """Constructor parameters for :class:`BinningPreprocessor`."""

    model_config = ConfigDict(extra="forbid")
    #: Number of bins per feature (or globally).
    n_bins: int = 10
    #: Binning strategy passed to sklearn's ``KBinsDiscretizer``.
    bin_strategy: str = "uniform"
    #: When true, learn one set of bin edges across all feature columns.
    across_features: bool = False


@TablePreprocessorRegistry.register("binning")
class BinningPreprocessor(_FittedPreprocessor):
    """
    K-bins discretisation of every feature to ordinal bin indices.

    Reproduces sklearn's ``KBinsDiscretizer(encode="ordinal")`` exactly as the
    v0.1 method did: edges are learned on the training table only, and
    prediction tables are binned with those frozen edges. The ``kmeans``
    strategy is stochastic, so this component is
    :class:`~habit.domain.protocols.Seedable`.
    """

    _spec_name = "binning"

    def __init__(
        self,
        n_bins: int = 10,
        bin_strategy: str = "uniform",
        across_features: bool = False,
    ) -> None:
        super().__init__()
        self._n_bins = int(n_bins)
        self._bin_strategy = str(bin_strategy)
        self._across_features = bool(across_features)
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
                "across_features": self._across_features,
            },
        )

    def fit(self, table: FeatureTable) -> "BinningPreprocessor":
        """Learn bin edges from the training table."""
        block = table.frame[list(table.feature_columns)]
        self._remember_fit(
            table,
            _kernel.fit_binning(
                block,
                self._n_bins,
                self._bin_strategy,
                self._across_features,
                random_state=self._seed,
            ),
        )
        return self

    def transform(self, table: FeatureTable) -> FeatureTable:
        """Discretise the table's features with the training bin edges."""
        block = self._block_for_transform(table)
        state = self._state
        assert state is not None
        return self._finish(
            table, _kernel.apply_binning(block, state), self._fit_columns
        )


class WinsorizePreprocessorParams(BaseModel):
    """Constructor parameters for :class:`WinsorizePreprocessor`."""

    model_config = ConfigDict(extra="forbid")
    #: Lower/upper tail fractions clipped at the corresponding quantiles.
    winsor_limits: Tuple[float, float] = (0.05, 0.05)
    #: When true, learn one pair of clip bounds across all feature columns.
    across_features: bool = False


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
        across_features: bool = False,
    ) -> None:
        super().__init__()
        limits = tuple(float(v) for v in winsor_limits)
        if len(limits) != 2 or not all(0.0 <= v < 0.5 for v in limits):
            raise HABITAPIError(
                "winsor_limits must be two fractions in [0, 0.5); got "
                f"{winsor_limits!r}."
            )
        self._winsor_limits = limits
        self._across_features = bool(across_features)

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(
            name=self._spec_name,
            params={
                "winsor_limits": list(self._winsor_limits),
                "across_features": self._across_features,
            },
        )

    def fit(self, table: FeatureTable) -> "WinsorizePreprocessor":
        """Learn clip bounds from the training table."""
        block = table.frame[list(table.feature_columns)]
        self._remember_fit(
            table,
            _kernel.fit_winsorize(
                block, self._winsor_limits, self._across_features
            ),
        )
        return self

    def transform(self, table: FeatureTable) -> FeatureTable:
        """Clip the table's features at the training quantile bounds."""
        block = self._block_for_transform(table)
        state = self._state
        assert state is not None
        return self._finish(
            table, _kernel.apply_winsorize(block, state), self._fit_columns
        )


class LogPreprocessorParams(BaseModel):
    """Constructor parameters for :class:`LogPreprocessor`."""

    model_config = ConfigDict(extra="forbid")
    #: When true, shift by ONE minimum taken across all feature columns.
    across_features: bool = False


@TablePreprocessorRegistry.register("log")
class LogPreprocessor(_ScopedScaler):
    """
    Log transform ``log(x - min_train + 1)`` of every feature.

    The shift by the TRAINING minimum plus one guarantees a positive argument
    for any value seen at fit time, so right-skewed features (voxel counts,
    volumes) become approximately symmetric without hand-tuned offsets.
    """

    _spec_name = "log"
    _fit_kernel = staticmethod(_kernel.fit_log)
    _apply_kernel = staticmethod(_kernel.apply_log)


# ---------------------------------------------------------------------------
# variance_filter / correlation_filter: column-dropping preprocessors
# ---------------------------------------------------------------------------


class VarianceFilterPreprocessorParams(BaseModel):
    """Constructor parameters for :class:`VarianceFilterPreprocessor`."""

    model_config = ConfigDict(extra="forbid")
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
        columns = _kernel.select_variance_columns(block, self._variance_threshold)
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

    model_config = ConfigDict(extra="forbid")
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
        columns = _kernel.select_correlation_columns(
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
