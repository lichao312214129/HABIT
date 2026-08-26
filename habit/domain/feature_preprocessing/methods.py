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
"""The built-in feature-preprocessing methods.

Each method is a ``fit``/``transform`` pair over a unit-by-feature matrix.
Whether the fitted state survives the call is the CHAIN's decision, not the
method's: a per-subject chain throws it away, a cohort chain stores it in the
:class:`~habit.contracts.habitat.HabitatModel`. This is why the same
methods serve voxel features, supervoxel features and both stateless and
stateful use -- v0.1 already worked this way internally
(``apply_stateless_preprocessing`` is literally a fit that discards state),
but its configuration surface split the methods into two named blocks and
hid the fact. The ninth method, ``feature_whitelist``, learns nothing at
all: its column list arrives from outside (e.g. a precision screen).

Registered names match the v0.1 YAML spellings so a legacy configuration
translates without a lookup table.

One rename: v0.1's ``global_normalize`` becomes ``across_features``. The old
name suggested "use global (cohort-wide) statistics", but the flag never had
anything to do with cohorts -- it selects whether statistics are pooled
ACROSS FEATURE COLUMNS or kept per column. With multi-modal features that
distinction is scientific, not cosmetic: pooling preserves the relative
intensity scale between modalities, while per-column scaling erases it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
from pydantic import BaseModel, Field, ConfigDict

from habit.exceptions import HABITAPIError
from habit.domain.feature_preprocessing.registry import (
    FeaturePreprocessingMethodRegistry,
)
from habit.kernels import feature_transforms as _kernel
from habit.spec.specs import Spec

__all__ = [
    "Binning",
    "BinningParams",
    "CorrelationFilter",
    "CorrelationFilterParams",
    "PreciseCorrelationFilter",
    "PreciseCorrelationFilterParams",
    "FeatureWhitelist",
    "FeatureWhitelistParams",
    "Impute",
    "ImputeParams",
    "LogTransform",
    "LogTransformParams",
    "L2Normalizer",
    "L2NormalizerParams",
    "MaxAbsScaling",
    "MaxAbsScalingParams",
    "MinMaxScaling",
    "MinMaxScalingParams",
    "QuantileTransform",
    "QuantileTransformParams",
    "RobustScaling",
    "RobustScalingParams",
    "VarianceFilter",
    "VarianceFilterParams",
    "Winsorizing",
    "WinsorizingParams",
    "ZScoreScaling",
    "ZScoreScalingParams",
]


class ImputeParams(BaseModel):
    """Constructor parameters for :class:`Impute`."""

    model_config = ConfigDict(extra="forbid")
    strategy: str = Field(
        default="mean",
        description="Replacement statistic: mean, median, or zero.",
    )


@FeaturePreprocessingMethodRegistry.register("impute")
class Impute:
    """
    Replace non-finite feature values with a learned per-column statistic.

    Every other method assumes finite input: a quantile computed over a column
    containing infinity is meaningless, and scikit-learn refuses NaN outright.
    So this belongs FIRST in a chain, and both chains insert it automatically
    when a configuration does not name it -- recording it in their spec, so the
    step is never applied invisibly.

    v0.1 ran this logic as a hard-coded prologue rather than a configurable
    step, which left the strategy unreachable from a study's configuration even
    though the underlying helper already supported alternatives.

    Args:
        strategy: ``mean`` or ``median`` of each column's finite values, or
            ``zero``. Columns with no finite value at all impute to 0.0
            regardless, so one unusable modality cannot invalidate a subject.
    """

    _name = "impute"
    changes_columns: bool = False

    def __init__(self, strategy: str = "mean") -> None:
        if strategy not in _kernel.IMPUTE_STRATEGIES:
            raise HABITAPIError(
                f"impute: unknown strategy {strategy!r}; expected one of "
                f"{list(_kernel.IMPUTE_STRATEGIES)}."
            )
        self._strategy = str(strategy)

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(name=self._name, params={"strategy": self._strategy})

    def fit(self, block: pd.DataFrame) -> Dict[str, Any]:
        """
        Learn the per-column replacement values.

        Args:
            block: Unit-by-feature matrix to learn from.

        Returns:
            State to pass to :meth:`transform`.
        """
        return _kernel.fit_impute(block, self._strategy)

    def transform(
        self, block: pd.DataFrame, state: Mapping[str, Any]
    ) -> pd.DataFrame:
        """
        Replace non-finite values with the learned statistics.

        Args:
            block: Matrix to repair.
            state: State from :meth:`fit`.

        Returns:
            A finite matrix.
        """
        return _kernel.apply_impute(block, state)


class _ScopedMethod:
    """
    Shared base for methods parameterised only by feature-column scope.

    Args:
        across_features: Pool statistics across every feature column instead
            of computing them per column.
    """

    #: Registered name; also the spec name and the v0.1 YAML spelling.
    _name: str = ""
    #: Whether ``transform`` may return a different column set.
    changes_columns: bool = False

    def __init__(self, across_features: bool = False) -> None:
        self._across_features = bool(across_features)

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(
            name=self._name, params={"across_features": self._across_features}
        )


class MinMaxScalingParams(BaseModel):
    """Constructor parameters for :class:`MinMaxScaling`."""

    model_config = ConfigDict(extra="forbid")
    across_features: bool = False


@FeaturePreprocessingMethodRegistry.register("minmax")
class MinMaxScaling(_ScopedMethod):
    """
    Scale features to [0, 1].

    The usual final step of a voxel-level chain: distance-based clustering
    treats every feature dimension as commensurable, which is only true once
    the dimensions share a range.
    """

    _name = "minmax"

    def fit(self, block: pd.DataFrame) -> Dict[str, Any]:
        """
        Learn the scaling bounds.

        Args:
            block: Unit-by-feature matrix to learn from.

        Returns:
            State to pass to :meth:`transform`.
        """
        return _kernel.fit_minmax(block, self._across_features)

    def transform(
        self, block: pd.DataFrame, state: Mapping[str, Any]
    ) -> pd.DataFrame:
        """
        Apply the learned scaling.

        Args:
            block: Matrix to transform.
            state: State from :meth:`fit`.

        Returns:
            The scaled matrix.
        """
        return _kernel.apply_minmax(block, state)


class ZScoreScalingParams(BaseModel):
    """Constructor parameters for :class:`ZScoreScaling`."""

    model_config = ConfigDict(extra="forbid")
    across_features: bool = False


@FeaturePreprocessingMethodRegistry.register("zscore")
class ZScoreScaling(_ScopedMethod):
    """
    Standardise features to zero mean and unit variance.

    Preferred over min-max when the downstream algorithm assumes roughly
    Gaussian inputs, since it does not let a single extreme value compress
    everything else into a narrow band.
    """

    _name = "zscore"

    def fit(self, block: pd.DataFrame) -> Dict[str, Any]:
        """
        Learn the mean and standard deviation.

        Args:
            block: Unit-by-feature matrix to learn from.

        Returns:
            State to pass to :meth:`transform`.
        """
        return _kernel.fit_zscore(block, self._across_features)

    def transform(
        self, block: pd.DataFrame, state: Mapping[str, Any]
    ) -> pd.DataFrame:
        """
        Apply the learned standardisation.

        Args:
            block: Matrix to transform.
            state: State from :meth:`fit`.

        Returns:
            The standardised matrix.
        """
        return _kernel.apply_zscore(block, state)


class RobustScalingParams(BaseModel):
    """Constructor parameters for :class:`RobustScaling`."""

    model_config = ConfigDict(extra="forbid")
    across_features: bool = False


@FeaturePreprocessingMethodRegistry.register("robust")
class RobustScaling(_ScopedMethod):
    """
    Centre features on the median and scale by the interquartile range.

    The outlier-resistant standardisation. Useful on radiomics features,
    where a handful of supervoxels can carry values orders of magnitude away
    from the bulk.
    """

    _name = "robust"

    def fit(self, block: pd.DataFrame) -> Dict[str, Any]:
        """
        Learn the median and interquartile range.

        Args:
            block: Unit-by-feature matrix to learn from.

        Returns:
            State to pass to :meth:`transform`.
        """
        return _kernel.fit_robust(block, self._across_features)

    def transform(
        self, block: pd.DataFrame, state: Mapping[str, Any]
    ) -> pd.DataFrame:
        """
        Apply the learned robust scaling.

        Args:
            block: Matrix to transform.
            state: State from :meth:`fit`.

        Returns:
            The robust-scaled matrix.
        """
        return _kernel.apply_robust(block, state)


class MaxAbsScalingParams(BaseModel):
    """Constructor parameters for :class:`MaxAbsScaling`."""

    model_config = ConfigDict(extra="forbid")
    across_features: bool = False


@FeaturePreprocessingMethodRegistry.register("maxabs")
class MaxAbsScaling(_ScopedMethod):
    """
    Scale features by the maximum absolute value.

    Leaves zeros at zero and does not shift the origin, so sparse or
    already-centered radiomics columns keep their sign. A zero-peak column
    divides by 1.0.
    """

    _name = "maxabs"

    def fit(self, block: pd.DataFrame) -> Dict[str, Any]:
        """
        Learn the max-absolute peaks.

        Args:
            block: Unit-by-feature matrix to learn from.

        Returns:
            State to pass to :meth:`transform`.
        """
        return _kernel.fit_maxabs(block, self._across_features)

    def transform(
        self, block: pd.DataFrame, state: Mapping[str, Any]
    ) -> pd.DataFrame:
        """
        Apply the learned max-absolute scaling.

        Args:
            block: Matrix to transform.
            state: State from :meth:`fit`.

        Returns:
            The scaled matrix.
        """
        return _kernel.apply_maxabs(block, state)


class QuantileTransformParams(BaseModel):
    """Constructor parameters for :class:`QuantileTransform`."""

    model_config = ConfigDict(extra="forbid")
    across_features: bool = False
    n_quantiles: int = 1000
    output_distribution: str = "uniform"


@FeaturePreprocessingMethodRegistry.register("quantile")
class QuantileTransform:
    """
    Map each feature onto a uniform or normal distribution by percentile rank.

    Distance-based clustering then sees comparable marginals even when one
    radiomics column is heavy-tailed and another is bounded. Knots are
    learned at fit time; prediction rows are interpolated and clipped to
    the training extrema.
    """

    _name = "quantile"
    changes_columns: bool = False

    def __init__(
        self,
        across_features: bool = False,
        n_quantiles: int = 1000,
        output_distribution: str = "uniform",
    ) -> None:
        dist = str(output_distribution).strip().lower()
        if dist not in _kernel.QUANTILE_DISTRIBUTIONS:
            raise HABITAPIError(
                f"quantile: unknown output_distribution {output_distribution!r}; "
                f"expected one of {list(_kernel.QUANTILE_DISTRIBUTIONS)}."
            )
        if int(n_quantiles) < 2:
            raise HABITAPIError(
                f"quantile: n_quantiles must be >= 2; got {n_quantiles!r}."
            )
        self._across_features = bool(across_features)
        self._n_quantiles = int(n_quantiles)
        self._output_distribution = dist

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(
            name=self._name,
            params={
                "across_features": self._across_features,
                "n_quantiles": self._n_quantiles,
                "output_distribution": self._output_distribution,
            },
        )

    def fit(self, block: pd.DataFrame) -> Dict[str, Any]:
        """
        Learn quantile knots.

        Args:
            block: Unit-by-feature matrix to learn from.

        Returns:
            State to pass to :meth:`transform`.
        """
        return _kernel.fit_quantile(
            block,
            self._across_features,
            n_quantiles=self._n_quantiles,
            output_distribution=self._output_distribution,
        )

    def transform(
        self, block: pd.DataFrame, state: Mapping[str, Any]
    ) -> pd.DataFrame:
        """
        Apply the learned quantile map.

        Args:
            block: Matrix to transform.
            state: State from :meth:`fit`.

        Returns:
            The rank-mapped matrix.
        """
        return _kernel.apply_quantile(block, state)


class L2NormalizerParams(BaseModel):
    """Constructor parameters for :class:`L2Normalizer`."""

    model_config = ConfigDict(extra="forbid")


@FeaturePreprocessingMethodRegistry.register("l2")
class L2Normalizer:
    """
    Scale each row (voxel / supervoxel) to unit Euclidean length.

    After this step clustering compares feature *directions*, not magnitudes.
    A zero-length row stays zero. There are no training statistics: fit only
    records the column count so a later schema change is rejected.
    """

    _name = "l2"
    changes_columns: bool = False

    def __init__(self) -> None:
        return

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(name=self._name, params={})

    def fit(self, block: pd.DataFrame) -> Dict[str, Any]:
        """
        Record the feature width.

        Args:
            block: Unit-by-feature matrix to learn from.

        Returns:
            State to pass to :meth:`transform`.
        """
        return _kernel.fit_l2(block)

    def transform(
        self, block: pd.DataFrame, state: Mapping[str, Any]
    ) -> pd.DataFrame:
        """
        Apply per-row L2 normalisation.

        Args:
            block: Matrix to transform.
            state: State from :meth:`fit`.

        Returns:
            The row-normalised matrix.
        """
        return _kernel.apply_l2(block, state)


class LogTransformParams(BaseModel):
    """Constructor parameters for :class:`LogTransform`."""

    model_config = ConfigDict(extra="forbid")
    across_features: bool = False


@FeaturePreprocessingMethodRegistry.register("log")
class LogTransform(_ScopedMethod):
    """
    Compress right-skewed features with ``log(x - min + 1)``.

    The shift by the learned minimum keeps the argument positive for every
    value seen at fit time, so no hand-tuned offset is needed.
    """

    _name = "log"

    def fit(self, block: pd.DataFrame) -> Dict[str, Any]:
        """
        Learn the shift offsets.

        Args:
            block: Unit-by-feature matrix to learn from.

        Returns:
            State to pass to :meth:`transform`.
        """
        return _kernel.fit_log(block, self._across_features)

    def transform(
        self, block: pd.DataFrame, state: Mapping[str, Any]
    ) -> pd.DataFrame:
        """
        Apply the learned log transform.

        Args:
            block: Matrix to transform.
            state: State from :meth:`fit`.

        Returns:
            The log-transformed matrix.
        """
        return _kernel.apply_log(block, state)


class WinsorizingParams(BaseModel):
    """Constructor parameters for :class:`Winsorizing`."""

    model_config = ConfigDict(extra="forbid")
    winsor_limits: Tuple[float, float] = (0.05, 0.05)
    across_features: bool = False


@FeaturePreprocessingMethodRegistry.register("winsorize")
class Winsorizing:
    """
    Clip extreme values at tail quantiles instead of discarding them.

    Typically the FIRST step of a voxel-level chain: MRI intensity outliers
    (motion, susceptibility artefacts, a few necrotic voxels) would otherwise
    dominate the min-max range that follows and squash the informative middle
    of the distribution.

    Args:
        winsor_limits: Lower and upper tail fractions to clip, each in
            ``[0, 0.5)``.
        across_features: Pool statistics across feature columns.
    """

    _name = "winsorize"
    changes_columns: bool = False

    def __init__(
        self,
        winsor_limits: Tuple[float, float] = (0.05, 0.05),
        across_features: bool = False,
    ) -> None:
        limits = tuple(float(value) for value in winsor_limits)
        if len(limits) != 2 or not all(0.0 <= value < 0.5 for value in limits):
            raise HABITAPIError(
                "winsorize: winsor_limits must be two fractions in [0, 0.5); "
                f"got {winsor_limits!r}."
            )
        self._winsor_limits = (limits[0], limits[1])
        self._across_features = bool(across_features)

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(
            name=self._name,
            params={
                "winsor_limits": list(self._winsor_limits),
                "across_features": self._across_features,
            },
        )

    def fit(self, block: pd.DataFrame) -> Dict[str, Any]:
        """
        Learn the clipping bounds.

        Args:
            block: Unit-by-feature matrix to learn from.

        Returns:
            State to pass to :meth:`transform`.
        """
        return _kernel.fit_winsorize(
            block, self._winsor_limits, self._across_features
        )

    def transform(
        self, block: pd.DataFrame, state: Mapping[str, Any]
    ) -> pd.DataFrame:
        """
        Clip the matrix at the learned bounds.

        Args:
            block: Matrix to transform.
            state: State from :meth:`fit`.

        Returns:
            The clipped matrix.
        """
        return _kernel.apply_winsorize(block, state)


class BinningParams(BaseModel):
    """Constructor parameters for :class:`Binning`."""

    model_config = ConfigDict(extra="forbid")
    n_bins: int = Field(default=10, gt=1, description="Number of bins.")
    bin_strategy: str = Field(
        default="uniform",
        description="Edge rule: uniform, quantile, or kmeans.",
    )
    across_features: bool = Field(
        default=False,
        description="Learn one set of edges from pooled values across feature columns.",
    )


@FeaturePreprocessingMethodRegistry.register("binning")
class Binning:
    """
    Discretise features into ordinal bin indices.

    The characteristic cohort-level step for radiomics-heavy feature sets:
    replacing a continuous value with its bin index discards the fine
    variation that mostly reflects acquisition noise, while keeping the
    ordering that carries biology. Because bin edges come from the pooled
    cohort, the same index means the same thing across subjects.

    Args:
        n_bins: Number of bins.
        bin_strategy: ``uniform``, ``quantile`` or ``kmeans``.
        across_features: Learn one set of edges from the pooled values.
    """

    _name = "binning"
    changes_columns: bool = False

    def __init__(
        self,
        n_bins: int = 10,
        bin_strategy: str = "uniform",
        across_features: bool = False,
    ) -> None:
        if int(n_bins) < 2:
            raise HABITAPIError(f"binning: n_bins must exceed 1; got {n_bins}.")
        self._n_bins = int(n_bins)
        self._bin_strategy = str(bin_strategy)
        self._across_features = bool(across_features)
        self._seed: Optional[int] = None

    def set_random_state(self, seed: int) -> None:
        """Seed the stochastic ``kmeans`` bin strategy."""
        self._seed = int(seed)

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(
            name=self._name,
            params={
                "n_bins": self._n_bins,
                "bin_strategy": self._bin_strategy,
                "across_features": self._across_features,
            },
        )

    def fit(self, block: pd.DataFrame) -> Dict[str, Any]:
        """
        Learn the bin edges.

        Args:
            block: Unit-by-feature matrix to learn from.

        Returns:
            State to pass to :meth:`transform`.
        """
        return _kernel.fit_binning(
            block,
            self._n_bins,
            self._bin_strategy,
            self._across_features,
            random_state=self._seed,
        )

    def transform(
        self, block: pd.DataFrame, state: Mapping[str, Any]
    ) -> pd.DataFrame:
        """
        Map the matrix onto the learned bins.

        Args:
            block: Matrix to transform.
            state: State from :meth:`fit`.

        Returns:
            The binned matrix.
        """
        return _kernel.apply_binning(block, state)


class VarianceFilterParams(BaseModel):
    """Constructor parameters for :class:`VarianceFilter`."""

    model_config = ConfigDict(extra="forbid")
    variance_threshold: float = 0.0


@FeaturePreprocessingMethodRegistry.register("variance_filter")
class VarianceFilter:
    """
    Drop feature columns whose variance is at or below a threshold.

    A column that barely varies cannot separate one region from another, but
    still contributes a dimension to every distance computation downstream.

    Args:
        variance_threshold: Columns with ``var <= threshold`` are dropped;
            ``0.0`` removes only constant columns.
    """

    _name = "variance_filter"
    changes_columns: bool = True

    def __init__(self, variance_threshold: float = 0.0) -> None:
        self._variance_threshold = float(variance_threshold)

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(
            name=self._name,
            params={"variance_threshold": self._variance_threshold},
        )

    def fit(self, block: pd.DataFrame) -> Dict[str, Any]:
        """
        Learn the surviving column subset.

        Args:
            block: Unit-by-feature matrix to learn from.

        Returns:
            State naming the columns to keep.
        """
        return {
            "columns": _kernel.select_variance_columns(
                block, self._variance_threshold
            )
        }

    def transform(
        self, block: pd.DataFrame, state: Mapping[str, Any]
    ) -> pd.DataFrame:
        """
        Restrict the matrix to the learned columns.

        Args:
            block: Matrix to transform.
            state: State from :meth:`fit`.

        Returns:
            The matrix with only the surviving columns. Columns absent from
            ``block`` are skipped rather than raising, matching v0.1.
        """
        kept: List[str] = [
            column for column in state["columns"] if column in block.columns
        ]
        if not kept:
            return block
        return block[kept]


class FeatureWhitelistParams(BaseModel):
    """Constructor parameters for :class:`FeatureWhitelist`."""

    model_config = ConfigDict(extra="forbid")
    features: List[str]


@FeaturePreprocessingMethodRegistry.register("feature_whitelist")
class FeatureWhitelist:
    """
    Restrict the feature matrix to an explicit, externally derived list.

    This is the bridge from a precision screen to habitat computation: the
    :class:`~habit.domain.precision.PreciseFeatureSet` names the features
    that survived, and this method makes a habitat spec cluster exactly
    those -- the workflow of Prior et al. (Radiol Artif Intell
    2024;6(2):e230118), where only precise features may define habitats.

    Unlike the data-driven filters, the column list is a CONSTRUCTOR
    argument: nothing is learned from the matrix, so the method is
    leakage-free by construction and ``fit`` simply echoes the list.

    Args:
        features: Feature names to keep, in output order. At least one is
            required, and every name must be present in the matrix -- a
        missing feature breaks the "same features" contract and raises
        rather than being silently dropped.
    """

    _name = "feature_whitelist"
    changes_columns: bool = True

    def __init__(self, features: Sequence[str]) -> None:
        columns = [str(feature) for feature in features]
        if not columns:
            raise HABITAPIError("feature_whitelist: features must not be empty.")
        self._features = tuple(columns)

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(name=self._name, params={"features": list(self._features)})

    def _restrict(self, block: pd.DataFrame) -> pd.DataFrame:
        """
        Return the matrix restricted to the whitelist, checking presence.

        Args:
            block: Matrix to restrict.

        Returns:
            The whitelist columns, in whitelist order.

        Raises:
            HABITAPIError: If a whitelisted feature is absent.
        """
        missing = [column for column in self._features if column not in block.columns]
        if missing:
            raise HABITAPIError(
                f"feature_whitelist: features absent from the matrix: "
                f"{missing}; available: {list(block.columns)}."
            )
        return block[list(self._features)]

    def fit(self, block: pd.DataFrame) -> Dict[str, Any]:
        """
        Echo the whitelist as the fitted state (nothing is learned).

        Args:
            block: Unit-by-feature matrix, checked against the whitelist.

        Returns:
            State naming the columns to keep.
        """
        self._restrict(block)
        return {"columns": list(self._features)}

    def transform(
        self, block: pd.DataFrame, state: Mapping[str, Any]
    ) -> pd.DataFrame:
        """
        Restrict the matrix to the whitelisted columns.

        Args:
            block: Matrix to transform.
            state: State from :meth:`fit` (the whitelist itself is used).

        Returns:
            The matrix with only the whitelisted columns.
        """
        return self._restrict(block)


class CorrelationFilterParams(BaseModel):
    """Constructor parameters for :class:`CorrelationFilter`."""

    model_config = ConfigDict(extra="forbid")
    corr_threshold: float = 0.95
    corr_method: str = "spearman"


@FeaturePreprocessingMethodRegistry.register("correlation_filter")
class CorrelationFilter:
    """
    Greedily drop redundant, highly correlated feature columns.

    Radiomics feature families are strongly collinear; keeping one
    representative per correlated group cuts dimensionality without losing
    discriminative content. The left-to-right walk makes the surviving subset
    deterministic.

    Args:
        corr_threshold: Absolute-correlation cut-off above which later
            columns are dropped.
        corr_method: ``pearson``, ``spearman`` or ``kendall``.
    """

    _name = "correlation_filter"
    changes_columns: bool = True

    def __init__(
        self,
        corr_threshold: float = 0.95,
        corr_method: str = "spearman",
    ) -> None:
        self._corr_threshold = float(corr_threshold)
        self._corr_method = str(corr_method)

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(
            name=self._name,
            params={
                "corr_threshold": self._corr_threshold,
                "corr_method": self._corr_method,
            },
        )

    def fit(self, block: pd.DataFrame) -> Dict[str, Any]:
        """
        Learn the surviving column subset.

        Args:
            block: Unit-by-feature matrix to learn from.

        Returns:
            State naming the columns to keep.
        """
        return {
            "columns": _kernel.select_correlation_columns(
                block, self._corr_threshold, self._corr_method
            )
        }

    def transform(
        self, block: pd.DataFrame, state: Mapping[str, Any]
    ) -> pd.DataFrame:
        """
        Restrict the matrix to the learned columns.

        Args:
            block: Matrix to transform.
            state: State from :meth:`fit`.

        Returns:
            The matrix with only the surviving columns.
        """
        kept: List[str] = [
            column for column in state["columns"] if column in block.columns
        ]
        if not kept:
            return block
        return block[kept]


class PreciseCorrelationFilterParams(BaseModel):
    """Constructor parameters for :class:`PreciseCorrelationFilter`."""

    model_config = ConfigDict(extra="forbid")
    #: Drop when signed Spearman r is strictly greater than this (Prior code).
    corr_threshold: float = 0.7
    #: Drop only when the Spearman p-value is below this (Prior code: 0.05).
    p_threshold: float = 0.05


@FeaturePreprocessingMethodRegistry.register("precise_correlation_filter")
class PreciseCorrelationFilter:
    """
    Prior 2024 Spearman screen: signed r, p-value, keep the later column.

    ``correlation_filter`` uses |r| and keeps the first column. This method
    copies ``filtering()`` in precise-habitats so a habitat spec can lock
    the same column rule as that paper's published code.

    Args:
        corr_threshold: Signed Spearman cut-off; drop when r is greater.
        p_threshold: Spearman p-value cut-off; drop only when p is smaller.
    """

    _name = "precise_correlation_filter"
    changes_columns: bool = True

    def __init__(
        self,
        corr_threshold: float = 0.7,
        p_threshold: float = 0.05,
    ) -> None:
        self._corr_threshold = float(corr_threshold)
        self._p_threshold = float(p_threshold)

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(
            name=self._name,
            params={
                "corr_threshold": self._corr_threshold,
                "p_threshold": self._p_threshold,
            },
        )

    def fit(self, block: pd.DataFrame) -> Dict[str, Any]:
        """
        Learn the surviving column subset on the training (baseline) matrix.

        Args:
            block: Unit-by-feature matrix to screen.

        Returns:
            State naming the columns to keep.
        """
        return {
            "columns": _kernel.select_precise_correlation_columns(
                block, self._corr_threshold, self._p_threshold
            )
        }

    def transform(
        self, block: pd.DataFrame, state: Mapping[str, Any]
    ) -> pd.DataFrame:
        """
        Restrict the matrix to the learned columns.

        Args:
            block: Matrix to transform.
            state: State from :meth:`fit`.

        Returns:
            The matrix with only the surviving columns.
        """
        kept: List[str] = [
            column for column in state["columns"] if column in block.columns
        ]
        if not kept:
            return block
        return block[kept]


FeaturePreprocessingMethodRegistry.register_params_model("impute", ImputeParams)
FeaturePreprocessingMethodRegistry.register_params_model(
    "minmax", MinMaxScalingParams
)
FeaturePreprocessingMethodRegistry.register_params_model(
    "zscore", ZScoreScalingParams
)
FeaturePreprocessingMethodRegistry.register_params_model(
    "robust", RobustScalingParams
)
FeaturePreprocessingMethodRegistry.register_params_model(
    "maxabs", MaxAbsScalingParams
)
FeaturePreprocessingMethodRegistry.register_params_model(
    "quantile", QuantileTransformParams
)
FeaturePreprocessingMethodRegistry.register_params_model("l2", L2NormalizerParams)
FeaturePreprocessingMethodRegistry.register_params_model("log", LogTransformParams)
FeaturePreprocessingMethodRegistry.register_params_model(
    "winsorize", WinsorizingParams
)
FeaturePreprocessingMethodRegistry.register_params_model("binning", BinningParams)
FeaturePreprocessingMethodRegistry.register_params_model(
    "variance_filter", VarianceFilterParams
)
FeaturePreprocessingMethodRegistry.register_params_model(
    "feature_whitelist", FeatureWhitelistParams
)
FeaturePreprocessingMethodRegistry.register_params_model(
    "correlation_filter", CorrelationFilterParams
)
FeaturePreprocessingMethodRegistry.register_params_model(
    "precise_correlation_filter", PreciseCorrelationFilterParams
)
