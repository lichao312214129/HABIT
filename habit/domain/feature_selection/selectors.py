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
"""Built-in feature selectors (domain ``feature_selector``).

The twelve selectors are numerically equivalent to the v0.1 functions in
``habit.core.machine_learning.feature_selectors``, reshaped around the
:class:`~habit.domain.table_protocols.FeatureSelector` protocol: ``fit``
learns the selected column subset from the TRAINING table (plus, for the ICC
selector, aligned repeat-measurement tables), and ``transform`` restricts any
later table to that same subset -- so prediction data can never be
re-selected with test statistics.

Heavy third-party libraries (sklearn, statsmodels, scipy, mrmr, xgboost) are
imported lazily inside ``fit`` so importing this module stays cheap.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from habit.exceptions import HABITAPIError, OptionalDependencyError
from habit.contracts.table import FeatureTable
from habit.domain.feature_selection._base import (
    FittedSelectorBase,
    outcome_series,
)
from habit.domain.feature_selection.registry import FeatureSelectorRegistry
from habit.domain.evaluation.statistics import repeat_measurement_matrix
from habit.kernels import feature_transforms as _kernel
from habit.kernels.icc import icc2_1, icc3_1
from habit.spec.specs import Spec
from habit.utils.estimator_utils import (
    check_passthrough_accepted,
    validate_estimator_params,
)
from habit.utils.feature_selection_utils import resolve_n_features_to_select
from habit.utils.progress_utils import CustomTqdm

__all__ = [
    "VarianceSelector",
    "VarianceSelectorParams",
    "CorrelationSelector",
    "CorrelationSelectorParams",
    "VifSelector",
    "VifSelectorParams",
    "AnovaSelector",
    "AnovaSelectorParams",
    "Chi2Selector",
    "Chi2SelectorParams",
    "StatisticalTestSelector",
    "StatisticalTestSelectorParams",
    "UnivariateLogisticSelector",
    "UnivariateLogisticSelectorParams",
    "UnivariateCoxSelector",
    "UnivariateCoxSelectorParams",
    "StepwiseSelector",
    "StepwiseSelectorParams",
    "RfecvSelector",
    "RfecvSelectorParams",
    "LassoSelector",
    "LassoSelectorParams",
    "IccSelector",
    "IccSelectorParams",
    "PrecomputedIccSelector",
    "PrecomputedIccSelectorParams",
    "MrmrSelector",
    "MrmrSelectorParams",
]


def _resolve_top_n(
    n_features_to_select: Optional[Union[int, float]],
    n_candidates: int,
    p_threshold: float,
) -> Tuple[Optional[int], float]:
    """
    Resolve the v0.1 dual-notation ``n_features_to_select`` parameter.

    Args:
        n_features_to_select: ``>= 1`` absolute count, ``(0, 1)`` ratio, or
            ``None`` to fall back to the p-value threshold.
        n_candidates: Number of candidate features.
        p_threshold: Fallback p-value threshold.

    Returns:
        Tuple ``(top_n, p_threshold)`` where ``top_n`` is the resolved count
        (or ``None`` when the threshold applies).
    """
    top_n, _ = resolve_n_features_to_select(n_features_to_select, n_candidates)
    return top_n, p_threshold


# ---------------------------------------------------------------------------
# variance: unsupervised variance ranking / threshold
# ---------------------------------------------------------------------------


class VarianceSelectorParams(BaseModel):
    """Constructor parameters for :class:`VarianceSelector`."""

    model_config = ConfigDict(extra="forbid")
    #: Columns with variance at or below this value are dropped.
    threshold: float = 0.0
    #: Keep the ``top_k`` highest-variance columns (overrides ``threshold``).
    top_k: Optional[int] = None
    #: Keep the top ``top_percent`` percent (0-100 scale) highest-variance
    #: columns (overrides ``threshold`` when ``top_k`` is absent).
    top_percent: Optional[float] = None
    #: Keep the single highest-variance column when nothing clears the
    #: threshold. ``False`` here (a selector may legitimately select nothing)
    #: versus ``True`` for the ``variance_filter`` preprocessor.
    keep_at_least_one: bool = False


#: Default of :class:`VarianceSelector`'s fallback parameter. Kept as a module
#: constant because ``spec.params`` records it only when it DEVIATES from this
#: value -- see the note in :meth:`VarianceSelector.spec`.
_VARIANCE_KEEP_AT_LEAST_ONE_DEFAULT = False


@FeatureSelectorRegistry.register("variance")
class VarianceSelector(FittedSelectorBase):
    """
    Unsupervised variance-based selection.

    Three modes, checked in priority order exactly as in v0.1: ``top_k``
    (absolute count of highest-variance features), then ``top_percent``
    (relative count), then the plain variance ``threshold``. Feature
    variances carry no outcome information, so this selector never looks at
    the outcome column.

    The same algorithm is registered a second time, as the
    ``variance_filter`` TABLE PREPROCESSOR. That is an ALIAS, not a second
    implementation: both delegate to
    :func:`habit.kernels.feature_transforms.select_variance_columns`. They
    differ in exactly two documented ways, both preserved deliberately:

    * parameter spelling -- ``threshold`` here, ``variance_threshold`` there;
    * the degenerate case -- ``keep_at_least_one`` defaults to ``False``
      here (selecting nothing is a legitimate finding) and ``True`` there (a
      preprocessing chain must never hand the next step an empty feature
      block). Pass ``keep_at_least_one=True`` to get the preprocessor's
      behaviour from the selector.
    """

    _spec_name = "variance"

    def __init__(
        self,
        threshold: float = 0.0,
        top_k: Optional[int] = None,
        top_percent: Optional[float] = None,
        keep_at_least_one: bool = _VARIANCE_KEEP_AT_LEAST_ONE_DEFAULT,
    ) -> None:
        super().__init__()
        self._threshold = float(threshold)
        self._top_k = None if top_k is None else int(top_k)
        self._top_percent = None if top_percent is None else float(top_percent)
        self._keep_at_least_one = bool(keep_at_least_one)

    @property
    def spec(self) -> Spec:
        """
        Return the algorithm specification.

        ``keep_at_least_one`` appears only when it DEVIATES from the default.
        A ``Spec`` records deviations (see :class:`~habit.spec.specs.Spec`),
        and the asymmetry is load-bearing here: every provenance record and
        golden baseline ever written by HABIT hashes this payload, so
        unconditionally adding a key would move every recorded ``variance``
        fingerprint. Same pattern as the ``estimator_params`` reserved key.
        """
        params: Dict[str, Any] = {
            "threshold": self._threshold,
            "top_k": self._top_k,
            "top_percent": self._top_percent,
        }
        if self._keep_at_least_one != _VARIANCE_KEEP_AT_LEAST_ONE_DEFAULT:
            params["keep_at_least_one"] = self._keep_at_least_one
        return Spec(name=self._spec_name, params=params)

    def fit(
        self,
        table: FeatureTable,
        *,
        repeat_tables: Optional[Sequence[FeatureTable]] = None,
    ) -> "VarianceSelector":
        """Select high-variance columns of the training table."""
        block = table.frame[list(table.feature_columns)]
        selected = _kernel.select_variance_columns(
            block,
            self._threshold,
            top_k=self._top_k,
            top_percent=self._top_percent,
            keep_at_least_one=self._keep_at_least_one,
        )
        self._remember_selection(table, selected)
        return self


# ---------------------------------------------------------------------------
# correlation: unsupervised greedy collinearity pruning
# ---------------------------------------------------------------------------


class CorrelationSelectorParams(BaseModel):
    """Constructor parameters for :class:`CorrelationSelector`."""

    model_config = ConfigDict(extra="forbid")
    #: Absolute-correlation cut-off above which later columns are dropped.
    threshold: float = 0.8
    #: Correlation method for ``DataFrame.corr``.
    method: str = "spearman"


@FeatureSelectorRegistry.register("correlation")
class CorrelationSelector(FittedSelectorBase):
    """
    Unsupervised greedy removal of highly correlated features.

    Walks columns left-to-right and drops later columns whose absolute
    correlation with a kept column exceeds ``threshold`` (the v0.1
    ``correlation`` selector).

    The ``correlation_filter`` TABLE PREPROCESSOR is an ALIAS of this same
    algorithm, not a second implementation: both delegate to
    :func:`habit.kernels.feature_transforms.select_correlation_columns`. They
    differ only in parameter spelling (``threshold`` / ``method`` here,
    ``corr_threshold`` / ``corr_method`` there) and in defaults (0.8 here,
    0.95 there). Unlike variance selection there is no degenerate case to
    disagree about: the greedy walk always keeps the first column.
    """

    _spec_name = "correlation"

    def __init__(self, threshold: float = 0.8, method: str = "spearman") -> None:
        super().__init__()
        self._threshold = float(threshold)
        self._method = str(method)

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(
            name=self._spec_name,
            params={"threshold": self._threshold, "method": self._method},
        )

    def fit(
        self,
        table: FeatureTable,
        *,
        repeat_tables: Optional[Sequence[FeatureTable]] = None,
    ) -> "CorrelationSelector":
        """Select a greedily de-correlated column subset."""
        block = table.frame[list(table.feature_columns)]
        self._remember_selection(
            table,
            _kernel.select_correlation_columns(
                block, self._threshold, self._method
            ),
        )
        return self


# ---------------------------------------------------------------------------
# vif: unsupervised iterative variance-inflation-factor pruning
# ---------------------------------------------------------------------------


class VifSelectorParams(BaseModel):
    """Constructor parameters for :class:`VifSelector`."""

    model_config = ConfigDict(extra="forbid")
    #: Maximum tolerated VIF; the highest-VIF column is removed until every
    #: remaining column is at or below this value.
    max_vif: float = 10.0


@FeatureSelectorRegistry.register("vif")
class VifSelector(FittedSelectorBase):
    """
    Iterative variance-inflation-factor (VIF) pruning of multicollinearity.

    Removes the single highest-VIF feature, recomputes VIFs on the remainder,
    and repeats until every feature is below ``max_vif`` -- the conservative
    one-at-a-time rule from the v0.1 selector, which stops before fewer than
    two features would remain.
    """

    _spec_name = "vif"

    def __init__(self, max_vif: float = 10.0) -> None:
        super().__init__()
        self._max_vif = float(max_vif)

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(name=self._spec_name, params={"max_vif": self._max_vif})

    def fit(
        self,
        table: FeatureTable,
        *,
        repeat_tables: Optional[Sequence[FeatureTable]] = None,
    ) -> "VifSelector":
        """Select columns whose mutual VIFs are all below the threshold."""
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
        except ImportError as exc:
            raise OptionalDependencyError(
                "feature_selector.vif requires the optional statsmodels "
                "dependency; install 'habitat-analysis[ml]' to use it."
            ) from exc

        data = table.frame[list(table.feature_columns)].copy()
        if data.shape[1] > 1:
            vif_values = np.array(
                [
                    variance_inflation_factor(data.values, i)
                    for i in range(data.shape[1])
                ]
            )
            while np.any(vif_values > self._max_vif):
                drop = data.columns[int(np.argmax(vif_values))]
                data = data.drop(columns=[drop])
                if data.shape[1] < 2:
                    # v0.1 stops here: VIF is meaningless on a single column.
                    break
                vif_values = np.array(
                    [
                        variance_inflation_factor(data.values, i)
                        for i in range(data.shape[1])
                    ]
                )
        self._remember_selection(table, list(data.columns))
        return self


# ---------------------------------------------------------------------------
# anova / chi2 / statistical_test: univariate outcome-association tests
# ---------------------------------------------------------------------------


class AnovaSelectorParams(BaseModel):
    """Constructor parameters for :class:`AnovaSelector`."""

    model_config = ConfigDict(extra="forbid")
    #: Keep columns whose ANOVA F-test p-value is below this threshold.
    p_threshold: float = 0.05
    #: ``>= 1`` absolute count or ``(0, 1)`` ratio of top-F columns to keep
    #: (overrides ``p_threshold``).
    n_features_to_select: Optional[float] = None


@FeatureSelectorRegistry.register("anova")
class AnovaSelector(FittedSelectorBase):
    """
    Univariate ANOVA F-value selection against the outcome.

    Ranks features by the sklearn ``f_classif`` F statistic and keeps either
    those with p-value below ``p_threshold`` or the top
    ``n_features_to_select`` (count or ratio), mirroring the v0.1 selector.
    """

    _spec_name = "anova"

    def __init__(
        self,
        p_threshold: float = 0.05,
        n_features_to_select: Optional[Union[int, float]] = None,
    ) -> None:
        super().__init__()
        self._p_threshold = float(p_threshold)
        self._n_features_to_select = n_features_to_select

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(
            name=self._spec_name,
            params={
                "p_threshold": self._p_threshold,
                "n_features_to_select": self._n_features_to_select,
            },
        )

    def fit(
        self,
        table: FeatureTable,
        *,
        repeat_tables: Optional[Sequence[FeatureTable]] = None,
    ) -> "AnovaSelector":
        """Select columns by ANOVA F-test against the training outcome."""
        from sklearn.feature_selection import f_classif

        block = table.frame[list(table.feature_columns)]
        y = outcome_series(table, owner=f"feature_selector.{self._spec_name}")
        f_values, p_values = f_classif(block, y)
        ranking = pd.DataFrame(
            {"feature": list(block.columns), "score": f_values, "pvalue": p_values}
        ).sort_values("score", ascending=False).reset_index(drop=True)
        top_n, _ = _resolve_top_n(
            self._n_features_to_select, len(ranking), self._p_threshold
        )
        if top_n is not None:
            mask = ranking.index < top_n
        else:
            mask = ranking["pvalue"] < self._p_threshold
        self._remember_selection(table, ranking.loc[mask, "feature"].tolist())
        return self


class Chi2SelectorParams(BaseModel):
    """Constructor parameters for :class:`Chi2Selector`."""

    model_config = ConfigDict(extra="forbid")
    #: Keep columns whose chi-square p-value is below this threshold.
    p_threshold: float = 0.05
    #: ``>= 1`` absolute count or ``(0, 1)`` ratio of top-chi2 columns to keep
    #: (overrides ``p_threshold``).
    n_features_to_select: Optional[float] = None


@FeatureSelectorRegistry.register("chi2")
class Chi2Selector(FittedSelectorBase):
    """
    Univariate chi-square selection against a categorical outcome.

    Chi-square statistics require non-negative features; following the v0.1
    selector, negative values are clipped to zero before the test. Keeps
    features below ``p_threshold`` or the top ``n_features_to_select``.
    """

    _spec_name = "chi2"

    def __init__(
        self,
        p_threshold: float = 0.05,
        n_features_to_select: Optional[Union[int, float]] = None,
    ) -> None:
        super().__init__()
        self._p_threshold = float(p_threshold)
        self._n_features_to_select = n_features_to_select

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(
            name=self._spec_name,
            params={
                "p_threshold": self._p_threshold,
                "n_features_to_select": self._n_features_to_select,
            },
        )

    def fit(
        self,
        table: FeatureTable,
        *,
        repeat_tables: Optional[Sequence[FeatureTable]] = None,
    ) -> "Chi2Selector":
        """Select columns by chi-square test against the training outcome."""
        from sklearn.feature_selection import chi2

        block = table.frame[list(table.feature_columns)]
        y = outcome_series(table, owner=f"feature_selector.{self._spec_name}")
        # Chi2 requires non-negative features (v0.1 clipped negatives to 0).
        block = block.clip(lower=0)
        chi2_values, p_values = chi2(block, y)
        ranking = pd.DataFrame(
            {"feature": list(block.columns), "score": chi2_values, "pvalue": p_values}
        ).sort_values("score", ascending=False).reset_index(drop=True)
        top_n, _ = _resolve_top_n(
            self._n_features_to_select, len(ranking), self._p_threshold
        )
        if top_n is not None:
            mask = ranking.index < top_n
        else:
            mask = ranking["pvalue"] < self._p_threshold
        self._remember_selection(table, ranking.loc[mask, "feature"].tolist())
        return self


class StatisticalTestSelectorParams(BaseModel):
    """Constructor parameters for :class:`StatisticalTestSelector`."""

    model_config = ConfigDict(extra="forbid")
    #: Keep columns whose test p-value is below this threshold.
    p_threshold: float = 0.05
    #: ``>= 1`` absolute count or ``(0, 1)`` ratio of top-statistic columns to
    #: keep (overrides ``p_threshold``).
    n_features_to_select: Optional[float] = None
    #: Shapiro-Wilk p-value above which a group counts as normal.
    normality_test_threshold: float = 0.05
    #: Force ``"ttest"`` or ``"mannwhitney"`` instead of the normality-driven
    #: automatic choice.
    force_test: Optional[str] = None


@FeatureSelectorRegistry.register("statistical_test")
class StatisticalTestSelector(FittedSelectorBase):
    """
    Two-group univariate selection with automatic test choice.

    For every feature the outcome's two groups are tested for normality
    (Shapiro-Wilk, skipped for groups with >= 5000 samples as in v0.1); when
    both look normal a Welch t-test is used, otherwise a Mann-Whitney U test.
    Features are ranked by the absolute test statistic and kept by p-value
    threshold or top-``n_features_to_select``.
    """

    _spec_name = "statistical_test"

    def __init__(
        self,
        p_threshold: float = 0.05,
        n_features_to_select: Optional[Union[int, float]] = None,
        normality_test_threshold: float = 0.05,
        force_test: Optional[str] = None,
    ) -> None:
        super().__init__()
        if force_test is not None and force_test not in ("ttest", "mannwhitney"):
            raise HABITAPIError(
                f"force_test must be 'ttest', 'mannwhitney' or None; got "
                f"{force_test!r}."
            )
        self._p_threshold = float(p_threshold)
        self._n_features_to_select = n_features_to_select
        self._normality_test_threshold = float(normality_test_threshold)
        self._force_test = force_test

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(
            name=self._spec_name,
            params={
                "p_threshold": self._p_threshold,
                "n_features_to_select": self._n_features_to_select,
                "normality_test_threshold": self._normality_test_threshold,
                "force_test": self._force_test,
            },
        )

    def fit(
        self,
        table: FeatureTable,
        *,
        repeat_tables: Optional[Sequence[FeatureTable]] = None,
    ) -> "StatisticalTestSelector":
        """Select columns by two-group test against the binary outcome."""
        from scipy import stats

        owner = f"feature_selector.{self._spec_name}"
        block = table.frame[list(table.feature_columns)]
        y = outcome_series(table, owner=owner)
        unique_values = y.unique()
        if len(unique_values) != 2:
            raise HABITAPIError(
                f"{owner} requires a binary outcome; got "
                f"{len(unique_values)} classes."
            )
        class_0 = (y == unique_values[0]).to_numpy()
        class_1 = (y == unique_values[1]).to_numpy()

        test_stats: List[float] = []
        p_values: List[float] = []
        for feature in block.columns:
            values = block[feature].to_numpy()
            group0 = values[class_0]
            group1 = values[class_1]
            if self._force_test is not None:
                test_type = self._force_test
            else:
                _, p0 = stats.shapiro(group0) if len(group0) < 5000 else (0, 0)
                _, p1 = stats.shapiro(group1) if len(group1) < 5000 else (0, 0)
                normal = (
                    p0 > self._normality_test_threshold
                    and p1 > self._normality_test_threshold
                )
                test_type = "ttest" if normal else "mannwhitney"
            if test_type == "ttest":
                stat, p_value = stats.ttest_ind(group0, group1, equal_var=False)
            else:
                stat, p_value = stats.mannwhitneyu(group0, group1)
            test_stats.append(abs(float(stat)))
            p_values.append(float(p_value))

        ranking = pd.DataFrame(
            {
                "feature": list(block.columns),
                "test_statistic": test_stats,
                "pvalue": p_values,
            }
        ).sort_values("test_statistic", ascending=False).reset_index(drop=True)
        top_n, _ = _resolve_top_n(
            self._n_features_to_select, len(ranking), self._p_threshold
        )
        if top_n is not None:
            mask = ranking.index < top_n
        else:
            mask = ranking["pvalue"] < self._p_threshold
        self._remember_selection(table, ranking.loc[mask, "feature"].tolist())
        return self


# ---------------------------------------------------------------------------
# univariate_logistic: per-feature logistic regression p-values
# ---------------------------------------------------------------------------


class UnivariateLogisticSelectorParams(BaseModel):
    """Constructor parameters for :class:`UnivariateLogisticSelector`."""

    model_config = ConfigDict(extra="forbid")
    #: Keep columns whose univariate logistic p-value is below this level.
    alpha: float = 0.05
    #: Show a progress bar over the per-feature model fits.
    #: Default True restores the v0.1 always-on univariate progress bar.
    verbose: bool = True


@FeatureSelectorRegistry.register("univariate_logistic")
class UnivariateLogisticSelector(FittedSelectorBase):
    """
    Per-feature logistic regression against the outcome.

    Fits one statsmodels logit per candidate feature (``outcome ~ feature``)
    and keeps the features whose coefficient p-value is below ``alpha`` --
    the classical radiomics univariate screen, numerically equivalent to the
    v0.1 selector.
    """

    _spec_name = "univariate_logistic"

    def __init__(self, alpha: float = 0.05, verbose: bool = True) -> None:
        super().__init__()
        self._alpha = float(alpha)
        self._verbose = bool(verbose)

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(
            name=self._spec_name,
            params={"alpha": self._alpha, "verbose": self._verbose},
        )

    def fit(
        self,
        table: FeatureTable,
        *,
        repeat_tables: Optional[Sequence[FeatureTable]] = None,
    ) -> "UnivariateLogisticSelector":
        """Select columns by univariate logistic p-value."""
        try:
            import statsmodels.formula.api as smf
        except ImportError as exc:
            raise OptionalDependencyError(
                "feature_selector.univariate_logistic requires the optional "
                "statsmodels dependency; install 'habitat-analysis[ml]' to use it."
            ) from exc

        block = table.frame[list(table.feature_columns)]
        y = outcome_series(table, owner=f"feature_selector.{self._spec_name}")
        progress = CustomTqdm(
            total=block.shape[1], desc="Univariate logistic", disable=not self._verbose
        )
        p_values: Dict[str, float] = {}
        try:
            for feature in block.columns:
                model = smf.logit(
                    formula="event ~ x",
                    data=pd.DataFrame({"event": y, "x": block[feature]}),
                )
                result = model.fit(verbose=0)
                p_values[feature] = float(result.pvalues[1])
                progress.update(1)
        finally:
            progress.close()
        selected = [f for f, p in p_values.items() if p < self._alpha]
        self._remember_selection(table, selected)
        return self


# ---------------------------------------------------------------------------
# stepwise: forward / backward / bidirectional logistic selection
# ---------------------------------------------------------------------------


class StepwiseSelectorParams(BaseModel):
    """Constructor parameters for :class:`StepwiseSelector`."""

    model_config = ConfigDict(extra="forbid")
    #: Selection direction: ``"forward"``, ``"backward"`` or ``"both"``.
    direction: str = "backward"
    #: P-value threshold for inclusion (``criterion="pvalue"`` only).
    threshold_in: float = 0.05
    #: P-value threshold for removal (``criterion="pvalue"`` only).
    threshold_out: float = 0.05
    #: Selection criterion: ``"aic"``, ``"bic"`` or ``"pvalue"``.
    criterion: str = "aic"
    #: Show progress bars over the candidate model fits.
    verbose: bool = False


def _logit_fit(y: pd.Series, X_subset: pd.DataFrame) -> Any:
    """Fit one statsmodels logit with intercept (lazy heavy import)."""
    try:
        import statsmodels.api as sm
        from statsmodels.discrete.discrete_model import Logit
    except ImportError as exc:
        raise OptionalDependencyError(
            "feature_selector.stepwise requires the optional statsmodels "
            "dependency; install 'habitat-analysis[ml]' to use it."
        ) from exc

    return Logit(y, sm.add_constant(X_subset)).fit(disp=0)


def _forward_selection(
    X: pd.DataFrame,
    y: pd.Series,
    threshold_in: float,
    criterion: str,
    verbose: bool,
) -> List[str]:
    """Forward stepwise selection (the v0.1 algorithm, plots removed)."""
    initial_features: List[str] = []
    remaining_features = list(X.columns)
    best_criterion = np.inf if criterion in ("aic", "bic") else 0.0
    while remaining_features:
        best_new_criterion = np.inf if criterion in ("aic", "bic") else 0.0
        best_feature = None
        progress = CustomTqdm(
            total=len(remaining_features),
            desc="Forward selection",
            disable=not verbose,
        )
        try:
            for feature in remaining_features:
                X_subset = X[initial_features + [feature]]
                try:
                    model = _logit_fit(y, X_subset)
                    if criterion in ("aic", "bic"):
                        current = model.aic if criterion == "aic" else model.bic
                        if current < best_new_criterion:
                            best_new_criterion = current
                            best_feature = feature
                    else:
                        pvalue = model.pvalues[feature]
                        if pvalue < threshold_in and pvalue > best_new_criterion:
                            best_new_criterion = pvalue
                            best_feature = feature
                except Exception:
                    # Perfect separation and singular fits simply do not
                    # nominate their feature (v0.1 behaviour).
                    pass
                finally:
                    progress.update(1)
        finally:
            progress.close()
        if criterion in ("aic", "bic"):
            if best_feature is not None and best_new_criterion < best_criterion:
                best_criterion = best_new_criterion
                initial_features.append(best_feature)
                remaining_features.remove(best_feature)
            else:
                break
        else:
            if best_feature is not None:
                initial_features.append(best_feature)
                remaining_features.remove(best_feature)
            else:
                break
    return initial_features


def _backward_elimination(
    X: pd.DataFrame,
    y: pd.Series,
    threshold_out: float,
    criterion: str,
    verbose: bool,
) -> List[str]:
    """Backward stepwise elimination (the v0.1 algorithm, plots removed)."""
    initial_features = list(X.columns)
    try:
        full_model = _logit_fit(y, X[initial_features])
        if criterion == "aic":
            best_criterion = full_model.aic
        elif criterion == "bic":
            best_criterion = full_model.bic
        else:
            best_criterion = 0.0
    except Exception:
        # Singular full model: fall back to forward selection (v0.1 rule).
        return _forward_selection(X, y, threshold_out, criterion, verbose)

    while initial_features:
        if criterion in ("aic", "bic"):
            best_new_criterion = np.inf
            worst_feature = None
            progress = CustomTqdm(
                total=len(initial_features),
                desc=f"Backward elimination ({criterion.upper()})",
                disable=not verbose,
            )
            try:
                for feature in initial_features:
                    model_features = [f for f in initial_features if f != feature]
                    if not model_features:
                        progress.update(1)
                        continue
                    try:
                        model = _logit_fit(y, X[model_features])
                        current = model.aic if criterion == "aic" else model.bic
                        if current < best_new_criterion:
                            best_new_criterion = current
                            worst_feature = feature
                    except Exception:
                        pass
                    finally:
                        progress.update(1)
            finally:
                progress.close()
            if worst_feature is not None and best_new_criterion < best_criterion:
                best_criterion = best_new_criterion
                initial_features.remove(worst_feature)
            else:
                break
        else:
            model = _logit_fit(y, X[initial_features])
            best_pvalue = 0.0
            worst_feature = None
            progress = CustomTqdm(
                total=len(initial_features),
                desc="Backward elimination (p-value)",
                disable=not verbose,
            )
            try:
                for feature in initial_features:
                    pvalue = model.pvalues.get(feature, 0)
                    if pvalue > best_pvalue:
                        best_pvalue = pvalue
                        worst_feature = feature
                    progress.update(1)
            finally:
                progress.close()
            if worst_feature is not None and best_pvalue > threshold_out:
                initial_features.remove(worst_feature)
            else:
                break
    return initial_features


def _stepwise_selection(
    X: pd.DataFrame,
    y: pd.Series,
    threshold_in: float,
    threshold_out: float,
    criterion: str,
    verbose: bool,
) -> List[str]:
    """Bidirectional stepwise selection (the v0.1 algorithm, plots removed)."""
    initial_features: List[str] = []
    remaining_features = list(X.columns)
    best_criterion = np.inf if criterion in ("aic", "bic") else 0.0
    while True:
        changed = False
        # Forward step.
        if criterion in ("aic", "bic"):
            best_new_criterion = np.inf
            best_feature_to_add = None
            for feature in remaining_features:
                try:
                    model = _logit_fit(y, X[initial_features + [feature]])
                    current = model.aic if criterion == "aic" else model.bic
                    if current < best_new_criterion:
                        best_new_criterion = current
                        best_feature_to_add = feature
                except Exception:
                    pass
            if best_feature_to_add is not None and best_new_criterion < best_criterion:
                best_criterion = best_new_criterion
                initial_features.append(best_feature_to_add)
                remaining_features.remove(best_feature_to_add)
                changed = True
        else:
            best_pvalue = threshold_in
            best_feature_to_add = None
            for feature in remaining_features:
                try:
                    model = _logit_fit(y, X[initial_features + [feature]])
                    pvalue = model.pvalues[feature]
                    if pvalue < best_pvalue:
                        best_pvalue = pvalue
                        best_feature_to_add = feature
                except Exception:
                    pass
            if best_feature_to_add is not None and best_pvalue < threshold_in:
                initial_features.append(best_feature_to_add)
                remaining_features.remove(best_feature_to_add)
                changed = True
        # Backward step.
        if initial_features:
            if criterion in ("aic", "bic"):
                worst_criterion = np.inf
                worst_feature = None
                for feature in initial_features:
                    model_features = [f for f in initial_features if f != feature]
                    if not model_features:
                        continue
                    try:
                        model = _logit_fit(y, X[model_features])
                        current = model.aic if criterion == "aic" else model.bic
                        if current < worst_criterion:
                            worst_criterion = current
                            worst_feature = feature
                    except Exception:
                        pass
                if worst_feature is not None and worst_criterion < best_criterion:
                    best_criterion = worst_criterion
                    initial_features.remove(worst_feature)
                    remaining_features.append(worst_feature)
                    changed = True
            else:
                try:
                    model = _logit_fit(y, X[initial_features])
                    worst_pvalue = 0.0
                    worst_feature = None
                    for feature in initial_features:
                        pvalue = model.pvalues.get(feature, 0)
                        if pvalue > worst_pvalue:
                            worst_pvalue = pvalue
                            worst_feature = feature
                    if worst_feature is not None and worst_pvalue > threshold_out:
                        initial_features.remove(worst_feature)
                        remaining_features.append(worst_feature)
                        changed = True
                except Exception:
                    # If the current model fails to fit, keep the set as is.
                    pass
        if not changed:
            break
    return initial_features


class UnivariateCoxSelectorParams(BaseModel):
    """Constructor parameters for :class:`UnivariateCoxSelector`."""

    model_config = ConfigDict(extra="forbid")
    #: Keep columns whose univariate-Cox p-value is below this threshold.
    p_threshold: float = 0.05
    #: ``>= 1`` absolute count or ``(0, 1)`` ratio of top columns to keep
    #: (overrides ``p_threshold``).
    n_features_to_select: Optional[float] = None


@FeatureSelectorRegistry.register("univariate_cox")
class UnivariateCoxSelector(FittedSelectorBase):
    """
    Univariate Cox proportional-hazards selection against a survival outcome.

    Fits one single-covariate Cox model per feature and ranks by the Wald
    p-value of its coefficient -- the survival analogue of the univariate
    logistic selector, and the standard first-pass screen in prognostic
    radiomics. lifelines is imported lazily (optional ``analysis`` extra).

    Requires a :class:`~habit.contracts.outcome.SurvivalOutcome`; any other
    endpoint family is rejected by :func:`survival_target`.
    """

    _spec_name = "univariate_cox"

    def __init__(
        self,
        p_threshold: float = 0.05,
        n_features_to_select: Optional[Union[int, float]] = None,
    ) -> None:
        super().__init__()
        self._p_threshold = float(p_threshold)
        self._n_features_to_select = n_features_to_select

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(
            name=self._spec_name,
            params={
                "p_threshold": self._p_threshold,
                "n_features_to_select": self._n_features_to_select,
            },
        )

    def fit(
        self,
        table: FeatureTable,
        *,
        repeat_tables: Optional[Sequence[FeatureTable]] = None,
    ) -> "UnivariateCoxSelector":
        """Select columns by univariate-Cox p-value against survival."""
        try:
            from lifelines import CoxPHFitter
        except ImportError as exc:
            raise OptionalDependencyError(
                "feature_selector.univariate_cox needs lifelines; install the "
                "'analysis' extra (pip install \"habitat-analysis[analysis]\")."
            ) from exc
        from lifelines.exceptions import ConvergenceError

        from habit.domain.outcome_access import survival_target

        time, event = survival_target(
            table, owner=f"feature_selector.{self._spec_name}"
        )
        rows = []
        for column in table.feature_columns:
            block = pd.DataFrame(
                {
                    "__t__": time.to_numpy(),
                    "__e__": event.astype(int).to_numpy(),
                    column: table.frame[column].to_numpy(),
                }
            )
            p_value = np.nan
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fitter = CoxPHFitter()
                    fitter.fit(block, duration_col="__t__", event_col="__e__")
                p_value = float(fitter.summary.loc[column, "p"])
            except (ConvergenceError, Exception):
                # A feature that cannot be fit (constant, perfect separation)
                # ranks last rather than crashing the whole screen.
                p_value = np.nan
            rows.append({"feature": column, "pvalue": p_value})
        ranking = (
            pd.DataFrame(rows)
            .sort_values("pvalue", ascending=True, na_position="last")
            .reset_index(drop=True)
        )
        top_n, _ = _resolve_top_n(
            self._n_features_to_select, len(ranking), self._p_threshold
        )
        if top_n is not None:
            kept = ranking.loc[ranking.index < top_n, "feature"].tolist()
        else:
            kept = ranking.loc[ranking["pvalue"] < self._p_threshold, "feature"].tolist()
        self._remember_selection(table, kept)
        return self


@FeatureSelectorRegistry.register("stepwise")
class StepwiseSelector(FittedSelectorBase):
    """
    Stepwise logistic-regression selection (forward, backward or both).

    Pure-Python port of the v0.1 selector: candidate models are statsmodels
    logits scored by AIC, BIC or coefficient p-value; features that fail to
    fit (perfect separation, singular matrices) are silently skipped, and a
    backward pass that cannot fit the full model falls back to forward
    selection.
    """

    _spec_name = "stepwise"

    def __init__(
        self,
        direction: str = "backward",
        threshold_in: float = 0.05,
        threshold_out: float = 0.05,
        criterion: str = "aic",
        verbose: bool = False,
    ) -> None:
        super().__init__()
        if direction not in ("forward", "backward", "both"):
            raise HABITAPIError(
                f"direction must be 'forward', 'backward' or 'both'; got "
                f"{direction!r}."
            )
        if criterion not in ("aic", "bic", "pvalue"):
            raise HABITAPIError(
                f"criterion must be 'aic', 'bic' or 'pvalue'; got {criterion!r}."
            )
        self._direction = direction
        self._threshold_in = float(threshold_in)
        self._threshold_out = float(threshold_out)
        self._criterion = criterion
        self._verbose = bool(verbose)

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(
            name=self._spec_name,
            params={
                "direction": self._direction,
                "threshold_in": self._threshold_in,
                "threshold_out": self._threshold_out,
                "criterion": self._criterion,
                "verbose": self._verbose,
            },
        )

    def fit(
        self,
        table: FeatureTable,
        *,
        repeat_tables: Optional[Sequence[FeatureTable]] = None,
    ) -> "StepwiseSelector":
        """Run the configured stepwise search on the training table."""
        block = table.frame[list(table.feature_columns)]
        y = outcome_series(table, owner=f"feature_selector.{self._spec_name}")
        if self._direction == "forward":
            selected = _forward_selection(
                block, y, self._threshold_in, self._criterion, self._verbose
            )
        elif self._direction == "backward":
            selected = _backward_elimination(
                block, y, self._threshold_out, self._criterion, self._verbose
            )
        else:
            selected = _stepwise_selection(
                block,
                y,
                self._threshold_in,
                self._threshold_out,
                self._criterion,
                self._verbose,
            )
        self._remember_selection(table, selected)
        return self


# ---------------------------------------------------------------------------
# rfecv / lasso: model-based selection (Seedable)
# ---------------------------------------------------------------------------


class RfecvSelectorParams(BaseModel):
    """Constructor parameters for :class:`RfecvSelector`."""

    model_config = ConfigDict(extra="forbid")
    #: Estimator name (one of the v0.1 supported set, e.g.
    #: ``"RandomForestClassifier"``, ``"LogisticRegression"``, ``"SVC"``,
    #: ``"GradientBoostingClassifier"``, ``"XGBClassifier"`` and the
    #: regression counterparts).
    estimator: str = "RandomForestClassifier"
    #: Number of features removed per iteration.
    step: int = 1
    #: Cross-validation folds.
    cv: int = 5
    #: Scoring metric optimised across the elimination path.
    scoring: str = "roc_auc"
    #: Minimum number of features the search is allowed to keep.
    min_features_to_select: int = 1
    #: Parallel jobs for the cross-validation.
    n_jobs: int = -1


def _build_rfecv_estimator(name: str, seed: Optional[int]) -> Any:
    """Build the sklearn/xgboost estimator by name (lazy heavy imports)."""
    if name == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(random_state=seed)
    if name == "RandomForestClassifier":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(random_state=seed)
    if name == "SVC":
        from sklearn.svm import SVC

        return SVC(random_state=seed)
    if name == "GradientBoostingClassifier":
        from sklearn.ensemble import GradientBoostingClassifier

        return GradientBoostingClassifier(random_state=seed)
    if name == "XGBClassifier":
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise OptionalDependencyError(
                "rfecv estimator 'XGBClassifier' requires the optional xgboost "
                "dependency; install 'habitat-analysis[ml]' to use it."
            ) from exc

        return xgb.XGBClassifier(random_state=seed)
    if name == "LinearRegression":
        from sklearn.linear_model import LinearRegression

        return LinearRegression()
    if name == "RandomForestRegressor":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(random_state=seed)
    if name == "SVR":
        from sklearn.svm import SVR

        return SVR()
    if name == "GradientBoostingRegressor":
        from sklearn.ensemble import GradientBoostingRegressor

        return GradientBoostingRegressor(random_state=seed)
    if name == "XGBRegressor":
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise OptionalDependencyError(
                "rfecv estimator 'XGBRegressor' requires the optional xgboost "
                "dependency; install 'habitat-analysis[ml]' to use it."
            ) from exc

        return xgb.XGBRegressor(random_state=seed)
    raise HABITAPIError(
        f"Unsupported rfecv estimator {name!r}; supported: LogisticRegression, "
        "RandomForestClassifier, SVC, GradientBoostingClassifier, XGBClassifier, "
        "LinearRegression, RandomForestRegressor, SVR, GradientBoostingRegressor, "
        "XGBRegressor."
    )


@FeatureSelectorRegistry.register("rfecv")
class RfecvSelector(FittedSelectorBase):
    """
    Recursive feature elimination with cross-validation (RFECV).

    Greedily removes the weakest features while cross-validated performance
    of the wrapped estimator keeps improving, exactly as the v0.1 selector.
    The wrapped estimator's stochasticity is controlled through
    :meth:`set_random_state`.
    """

    _spec_name = "rfecv"

    def __init__(
        self,
        estimator: str = "RandomForestClassifier",
        step: int = 1,
        cv: int = 5,
        scoring: str = "roc_auc",
        min_features_to_select: int = 1,
        n_jobs: int = -1,
    ) -> None:
        super().__init__()
        self._estimator = str(estimator)
        self._step = int(step)
        self._cv = int(cv)
        self._scoring = str(scoring)
        self._min_features_to_select = int(min_features_to_select)
        self._n_jobs = int(n_jobs)
        self._seed: Optional[int] = None

    def set_random_state(self, seed: int) -> None:
        """Set the random state of the wrapped estimator."""
        self._seed = int(seed)

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(
            name=self._spec_name,
            params={
                "estimator": self._estimator,
                "step": self._step,
                "cv": self._cv,
                "scoring": self._scoring,
                "min_features_to_select": self._min_features_to_select,
                "n_jobs": self._n_jobs,
            },
        )

    def fit(
        self,
        table: FeatureTable,
        *,
        repeat_tables: Optional[Sequence[FeatureTable]] = None,
    ) -> "RfecvSelector":
        """Run RFECV on the training table."""
        from sklearn.feature_selection import RFECV

        block = table.frame[list(table.feature_columns)]
        y = outcome_series(table, owner=f"feature_selector.{self._spec_name}")
        estimator = _build_rfecv_estimator(self._estimator, self._seed)
        rfecv = RFECV(
            estimator=estimator,
            step=self._step,
            cv=self._cv,
            scoring=self._scoring,
            min_features_to_select=self._min_features_to_select,
            n_jobs=self._n_jobs,
        )
        rfecv.fit(block, y)
        columns = list(block.columns)
        selected = [columns[i] for i in range(len(columns)) if rfecv.support_[i]]
        self._remember_selection(table, selected)
        return self


class LassoSelectorParams(BaseModel):
    """Constructor parameters for :class:`LassoSelector`."""

    model_config = ConfigDict(extra="forbid")
    #: Cross-validation folds for ``LassoCV``.
    cv: int = 10
    #: Number of alpha values along the regularisation path.
    n_alphas: int = 100
    #: Explicit alpha grid (overrides ``n_alphas`` when given).
    alphas: Optional[List[float]] = None
    #: Parallel jobs for the cross-validation.
    n_jobs: int = -1
    #: Vendor kwargs forwarded verbatim to ``LassoCV`` (e.g. ``{"eps": 1e-4}``);
    #: keys colliding with a declared parameter or with the HABIT-injected
    #: ``random_state`` are rejected.
    estimator_params: Dict[str, Any] = Field(default_factory=dict)


@FeatureSelectorRegistry.register("lasso")
class LassoSelector(FittedSelectorBase):
    """
    L1-penalised (Lasso) selection with cross-validated penalty.

    Fits ``LassoCV`` on the training table and keeps the features with a
    non-zero coefficient at the cross-validated optimal alpha -- the v0.1
    rule. Unlike v0.1 the random state is not a constructor parameter; use
    :meth:`set_random_state` (v1.0 naming decisions).

    Args:
        cv: Cross-validation folds for ``LassoCV``.
        n_alphas: Number of alpha values along the regularisation path.
        alphas: Explicit alpha grid (overrides ``n_alphas`` when given).
        n_jobs: Parallel jobs for the cross-validation.
        estimator_params: Extra keyword arguments forwarded verbatim to
            ``LassoCV``, for vendor parameters HABIT does not declare. They
            are validated against the ``LassoCV`` signature at fit time and
            recorded in the spec fingerprint.
    """

    _spec_name = "lasso"

    def __init__(
        self,
        cv: int = 10,
        n_alphas: int = 100,
        alphas: Optional[List[float]] = None,
        n_jobs: int = -1,
        estimator_params: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        self._cv = int(cv)
        self._n_alphas = int(n_alphas)
        self._alphas = None if alphas is None else [float(a) for a in alphas]
        self._n_jobs = int(n_jobs)
        self._estimator_params: Dict[str, Any] = validate_estimator_params(
            estimator_params,
            declared=("cv", "n_alphas", "alphas", "n_jobs"),
            owner=f"feature_selector.{self._spec_name}",
        )
        self._seed: Optional[int] = None

    def set_random_state(self, seed: int) -> None:
        """Set the random state of the underlying ``LassoCV``."""
        self._seed = int(seed)

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        params: Dict[str, Any] = {
            "cv": self._cv,
            "n_alphas": self._n_alphas,
            "alphas": self._alphas,
            "n_jobs": self._n_jobs,
        }
        # Fold the passthrough in only when non-empty so the default
        # configuration keeps its historical fingerprint.
        if self._estimator_params:
            params["estimator_params"] = dict(self._estimator_params)
        return Spec(name=self._spec_name, params=params)

    def fit(
        self,
        table: FeatureTable,
        *,
        repeat_tables: Optional[Sequence[FeatureTable]] = None,
    ) -> "LassoSelector":
        """Fit LassoCV and keep non-zero-coefficient columns."""
        from sklearn.linear_model import LassoCV

        check_passthrough_accepted(
            LassoCV, self._estimator_params, owner=f"feature_selector.{self._spec_name}"
        )
        block = table.frame[list(table.feature_columns)]
        y = outcome_series(table, owner=f"feature_selector.{self._spec_name}")
        lasso_cv = LassoCV(
            cv=self._cv,
            n_alphas=self._n_alphas,
            alphas=self._alphas,
            random_state=self._seed,
            n_jobs=self._n_jobs,
            **self._estimator_params,
        )
        lasso_cv.fit(block, y)
        coefs = np.asarray(lasso_cv.coef_)
        columns = list(block.columns)
        selected = [columns[i] for i in range(len(columns)) if coefs[i] != 0]
        self._remember_selection(table, selected)
        return self


# ---------------------------------------------------------------------------
# icc: test-retest stability selection via the L0 ICC kernels
# ---------------------------------------------------------------------------


class IccSelectorParams(BaseModel):
    """Constructor parameters for :class:`IccSelector`."""

    model_config = ConfigDict(extra="forbid")
    #: Minimum ICC value for a feature to count as stable.
    threshold: float = 0.75
    #: ICC variant: ``"icc3"`` (two-way mixed, consistency; the v0.1 default)
    #: or ``"icc2"`` (two-way random, absolute agreement).
    icc_type: str = "icc3"


@FeatureSelectorRegistry.register("icc")
class IccSelector(FittedSelectorBase):
    """
    Test-retest stability selection by intraclass correlation coefficient.

    Where the v0.1 selector consumed a precomputed JSON of ICC values, the
    v1 selector computes ICCs directly from aligned repeat-measurement
    tables: ``table`` holds the primary measurements and ``repeat_tables``
    one table per repeat session, all sharing the identifier columns. Each
    feature's values across sessions form a (subjects x sessions) matrix
    whose ICC is computed by the L0 kernels; features at or above
    ``threshold`` are kept. Rows with a NaN in any session are omitted per
    feature (pingouin's ``nan_policy="omit"``).
    """

    _spec_name = "icc"

    def __init__(self, threshold: float = 0.75, icc_type: str = "icc3") -> None:
        super().__init__()
        if icc_type not in ("icc3", "icc2"):
            raise HABITAPIError(
                f"icc_type must be 'icc3' or 'icc2'; got {icc_type!r}."
            )
        self._threshold = float(threshold)
        self._icc_type = icc_type

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(
            name=self._spec_name,
            params={"threshold": self._threshold, "icc_type": self._icc_type},
        )

    def fit(
        self,
        table: FeatureTable,
        *,
        repeat_tables: Optional[Sequence[FeatureTable]] = None,
    ) -> "IccSelector":
        """Select features whose across-session ICC clears the threshold."""
        if not repeat_tables:
            raise HABITAPIError(
                "feature_selector.icc requires repeat_tables: one feature table "
                "per repeat measurement session, aligned by identifier columns."
            )
        icc_kernel = icc3_1 if self._icc_type == "icc3" else icc2_1
        selected: List[str] = []
        for feature in table.feature_columns:
            matrix = repeat_measurement_matrix(
                table,
                repeat_tables,
                feature,
                owner=f"feature_selector.{self._spec_name}",
            )
            if matrix.shape[0] < 2:
                continue
            if icc_kernel(matrix) >= self._threshold:
                selected.append(feature)
        self._remember_selection(table, selected)
        return self


# ---------------------------------------------------------------------------
# icc_precomputed: stability selection from a precomputed ICC results JSON
# ---------------------------------------------------------------------------


class PrecomputedIccSelectorParams(BaseModel):
    """Constructor parameters for :class:`PrecomputedIccSelector`."""

    model_config = ConfigDict(extra="forbid")
    #: Path to the ICC results JSON produced by a standalone ICC analysis run.
    icc_results_path: str
    #: Group names in the JSON whose stable-feature sets are intersected.
    groups: List[str]
    #: Minimum ICC value for a feature to count as stable.
    threshold: float = 0.75
    #: ICC metric to read inside nested per-feature maps (e.g. ``"ICC3"``);
    #: ``None`` follows the v0.1 fallback chain.
    metric: Optional[str] = None


@FeatureSelectorRegistry.register("icc_precomputed")
class PrecomputedIccSelector(FittedSelectorBase):
    """
    Stability selection from a PRECOMPUTED ICC results JSON.

    Where :class:`IccSelector` recomputes ICCs from aligned repeat tables,
    this selector trusts the numbers a standalone ICC analysis already
    produced: the JSON maps group names to per-feature ICC values (simple
    ``{feature: value}`` form or nested ``{feature: {metric: {value: x}}}``
    form), and a feature is kept when its ICC clears ``threshold`` in EVERY
    requested group. This is the v0.1 ``icc`` selector's contract, kept for
    legacy configurations whose ICCs come from an external test-retest run:
    the JSON is the measurement record, not a configuration artefact, so
    reading it at fit time is data access, not layer violation.
    """

    _spec_name = "icc_precomputed"

    #: Metric keys tried in order when ``metric`` is not set (v0.1 chain).
    _DEFAULT_METRIC_KEYS = ("ICC3", "ICC2", "icc3", "icc2")

    def __init__(
        self,
        icc_results_path: str,
        groups: Sequence[str],
        threshold: float = 0.75,
        metric: Optional[str] = None,
    ) -> None:
        super().__init__()
        if not icc_results_path:
            raise HABITAPIError(
                "feature_selector.icc_precomputed requires icc_results_path: "
                "the ICC results JSON from a standalone ICC analysis run."
            )
        if not groups:
            raise HABITAPIError(
                "feature_selector.icc_precomputed requires a non-empty "
                "groups list: the JSON groups whose stable-feature sets are "
                "intersected."
            )
        self._icc_results_path = str(icc_results_path)
        self._groups = list(groups)
        self._threshold = float(threshold)
        self._metric = metric

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(
            name=self._spec_name,
            params={
                "icc_results_path": self._icc_results_path,
                "groups": list(self._groups),
                "threshold": self._threshold,
                "metric": self._metric,
            },
        )

    def fit(
        self,
        table: FeatureTable,
        *,
        repeat_tables: Optional[Sequence[FeatureTable]] = None,
    ) -> "PrecomputedIccSelector":
        """Select features stable in every requested group of the JSON."""
        icc_results = self._load_results()
        stable_per_group = [
            self._stable_features(icc_results, group) for group in self._groups
        ]
        selected = set.intersection(*stable_per_group) if stable_per_group else set()
        self._remember_selection(table, sorted(selected))
        return self

    def _load_results(self) -> Dict[str, Any]:
        """Read the ICC results JSON, failing loudly on a bad file."""
        import json

        try:
            with open(self._icc_results_path, "r", encoding="utf-8") as handle:
                results = json.load(handle)
        except FileNotFoundError:
            raise HABITAPIError(
                "feature_selector.icc_precomputed found no ICC results file "
                f"at: {self._icc_results_path}"
            ) from None
        except json.JSONDecodeError as error:
            raise HABITAPIError(
                "feature_selector.icc_precomputed could not decode JSON from "
                f"{self._icc_results_path}: {error}"
            ) from None
        if not isinstance(results, dict):
            raise HABITAPIError(
                "feature_selector.icc_precomputed expects the ICC results "
                f"JSON to map group names to per-feature values; got "
                f"{type(results).__name__} in {self._icc_results_path}."
            )
        return results

    def _resolve_group(self, icc_results: Dict[str, Any], group: str) -> str:
        """Resolve a requested group, falling back to a unique partial match."""
        if group in icc_results:
            return group
        for existing in icc_results:
            if group in existing:
                return existing
        raise HABITAPIError(
            f"feature_selector.icc_precomputed: group {group!r} is not in the "
            f"ICC results file (available: {sorted(icc_results)[:5]})."
        )

    def _stable_features(
        self, icc_results: Dict[str, Any], group: str
    ) -> set:
        """Collect the features clearing the threshold in one group."""
        resolved = self._resolve_group(icc_results, group)
        features_in_group = icc_results[resolved]
        if not isinstance(features_in_group, dict):
            raise HABITAPIError(
                f"feature_selector.icc_precomputed expects group {resolved!r} "
                f"to map feature names to ICC values; got "
                f"{type(features_in_group).__name__}."
            )
        stable = set()
        for feature, metrics in features_in_group.items():
            icc_value = self._feature_icc_value(metrics)
            if icc_value is not None and icc_value >= self._threshold:
                stable.add(feature)
        return stable

    def _feature_icc_value(self, metrics: Any) -> Optional[float]:
        """Extract one ICC value from a per-feature entry (v0.1 chain)."""
        if isinstance(metrics, (int, float)):
            # Simple format: {feature: icc_value}.
            return float(metrics)
        if not isinstance(metrics, dict):
            return None
        if self._metric:
            value = self._metric_entry_value(metrics.get(self._metric))
            if value is not None:
                return value
        for key in self._DEFAULT_METRIC_KEYS:
            value = self._metric_entry_value(metrics.get(key))
            if value is not None:
                return value
        # Last resort: any key containing 'icc' (case-insensitive).
        for key, entry in metrics.items():
            if "icc" in str(key).lower():
                value = self._metric_entry_value(entry)
                if value is not None:
                    return value
        return None

    @staticmethod
    def _metric_entry_value(entry: Any) -> Optional[float]:
        """Read a metric entry, either a bare number or ``{"value": x}``."""
        if isinstance(entry, dict):
            entry = entry.get("value")
        if isinstance(entry, (int, float)):
            return float(entry)
        return None


# ---------------------------------------------------------------------------
# mrmr: minimum redundancy maximum relevance
# ---------------------------------------------------------------------------


class MrmrSelectorParams(BaseModel):
    """Constructor parameters for :class:`MrmrSelector`."""

    model_config = ConfigDict(extra="forbid")
    #: Number of features to select (clipped to the candidate count).
    n_features: int = 10
    #: ``"classification"`` or ``"regression"``.
    task_type: str = "classification"


@FeatureSelectorRegistry.register("mrmr")
class MrmrSelector(FittedSelectorBase):
    """
    Minimum-redundancy-maximum-relevance (MRMR) selection.

    Mutual-information greedy selection from the ``mrmr`` package: every
    chosen feature is maximally relevant to the outcome while minimally
    redundant with the already-chosen set. Where the v0.1 function swallowed
    errors and returned an empty list, the v1 selector lets failures surface
    -- an empty selection should never be mistaken for a successful run.
    """

    _spec_name = "mrmr"

    def __init__(self, n_features: int = 10, task_type: str = "classification") -> None:
        super().__init__()
        if task_type not in ("classification", "regression"):
            raise HABITAPIError(
                f"task_type must be 'classification' or 'regression'; got "
                f"{task_type!r}."
            )
        if int(n_features) <= 0:
            raise HABITAPIError(f"n_features must be positive; got {n_features}.")
        self._n_features = int(n_features)
        self._task_type = task_type

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(
            name=self._spec_name,
            params={"n_features": self._n_features, "task_type": self._task_type},
        )

    def fit(
        self,
        table: FeatureTable,
        *,
        repeat_tables: Optional[Sequence[FeatureTable]] = None,
    ) -> "MrmrSelector":
        """Run MRMR on the training table."""
        try:
            if self._task_type == "classification":
                from mrmr import mrmr_classif as mrmr_fn
            else:
                from mrmr import mrmr_regression as mrmr_fn
        except ImportError as exc:
            raise OptionalDependencyError(
                "feature_selector.mrmr requires the optional mrmr-selection "
                "dependency; install 'habitat-analysis[ml]' to use it."
            ) from exc

        block = table.frame[list(table.feature_columns)]
        y = outcome_series(table, owner=f"feature_selector.{self._spec_name}")
        k = min(self._n_features, block.shape[1])
        selected = list(mrmr_fn(X=block, y=y, K=k))
        self._remember_selection(table, selected)
        return self


# ---------------------------------------------------------------------------
# Parameter schemas (registered after the classes so names resolve)
# ---------------------------------------------------------------------------

FeatureSelectorRegistry.register_params_model("variance", VarianceSelectorParams)
FeatureSelectorRegistry.register_params_model("correlation", CorrelationSelectorParams)
FeatureSelectorRegistry.register_params_model("vif", VifSelectorParams)
FeatureSelectorRegistry.register_params_model("anova", AnovaSelectorParams)
FeatureSelectorRegistry.register_params_model("chi2", Chi2SelectorParams)
FeatureSelectorRegistry.register_params_model(
    "statistical_test", StatisticalTestSelectorParams
)
FeatureSelectorRegistry.register_params_model(
    "univariate_logistic", UnivariateLogisticSelectorParams
)
FeatureSelectorRegistry.register_params_model("stepwise", StepwiseSelectorParams)
FeatureSelectorRegistry.register_params_model("rfecv", RfecvSelectorParams)
FeatureSelectorRegistry.register_params_model("lasso", LassoSelectorParams)
FeatureSelectorRegistry.register_params_model("icc", IccSelectorParams)
FeatureSelectorRegistry.register_params_model(
    "icc_precomputed", PrecomputedIccSelectorParams
)
FeatureSelectorRegistry.register_params_model("mrmr", MrmrSelectorParams)
FeatureSelectorRegistry.register_params_model(
    "univariate_cox", UnivariateCoxSelectorParams
)
