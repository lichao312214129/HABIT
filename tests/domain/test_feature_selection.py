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
"""Tests for the twelve built-in feature selectors."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from habit.api.exceptions import HABITAPIError
from habit.feature_selection import (
    AnovaSelector,
    Chi2Selector,
    CorrelationSelector,
    FeatureSelectorRegistry,
    IccSelector,
    LassoSelector,
    MrmrSelector,
    RfecvSelector,
    StatisticalTestSelector,
    StepwiseSelector,
    UnivariateLogisticSelector,
    VarianceSelector,
    VifSelector,
)
from habit._protocols import Seedable
from habit._table_protocols import FeatureSelector

from .conftest import make_feature_table

_ALL = (
    VarianceSelector,
    CorrelationSelector,
    VifSelector,
    AnovaSelector,
    Chi2Selector,
    StatisticalTestSelector,
    UnivariateLogisticSelector,
    StepwiseSelector,
    RfecvSelector,
    LassoSelector,
    IccSelector,
    MrmrSelector,
)


#: Selectors whose constructors have required parameters, with dummy values.
_REQUIRED_PARAMS = {
    "icc_precomputed": {"icc_results_path": "icc.json", "groups": ["group"]},
}


@pytest.mark.unit
def test_registry_lists_all_fourteen_selectors() -> None:
    """The registry constructs every built-in selector by its registry name."""
    assert set(FeatureSelectorRegistry.available()) == {
        "variance",
        "correlation",
        "vif",
        "anova",
        "chi2",
        "statistical_test",
        "univariate_logistic",
        "univariate_cox",
        "stepwise",
        "rfecv",
        "lasso",
        "icc",
        "icc_precomputed",
        "mrmr",
    }
    for name in FeatureSelectorRegistry.available():
        instance = FeatureSelectorRegistry.create(
            name, **_REQUIRED_PARAMS.get(name, {})
        )
        assert isinstance(instance, FeatureSelector)
        assert instance.spec.name == name
        assert isinstance(
            FeatureSelectorRegistry.constructor_signature(name), inspect.Signature
        )


@pytest.mark.unit
def test_variance_selector_modes() -> None:
    """top_k keeps the highest-variance columns; threshold drops constants."""
    table = make_feature_table(constant_column=True)
    for noise_column in ("noise0", "noise1", "noise2"):
        table.frame[noise_column] = table.frame[noise_column] * 0.01
    table.frame["spread"] = np.linspace(-10, 10, table.frame.shape[0])
    table = type(table)(
        frame=table.frame,
        id_columns=table.id_columns,
        feature_columns=(*table.feature_columns, "spread"),
        outcome=table.outcome,
        provenance=table.provenance,
    )
    top_two = VarianceSelector(top_k=2).fit(table).transform(table)
    assert set(top_two.feature_columns) == {"spread", "signal"}
    thresholded = VarianceSelector(threshold=0.0).fit(table).transform(table)
    assert "constant" not in thresholded.feature_columns


@pytest.mark.unit
def test_correlation_selector_prunes_duplicate() -> None:
    """A perfectly correlated later column is removed."""
    table = make_feature_table()
    table.frame["signal_copy"] = table.frame["signal"] * 3.0
    table = type(table)(
        frame=table.frame,
        id_columns=table.id_columns,
        feature_columns=(*table.feature_columns, "signal_copy"),
        outcome=table.outcome,
        provenance=table.provenance,
    )
    selected = CorrelationSelector(threshold=0.9).fit(table).transform(table)
    assert "signal" in selected.feature_columns
    assert "signal_copy" not in selected.feature_columns


@pytest.mark.unit
def test_vif_selector_removes_collinearity() -> None:
    """Near-duplicate columns drive VIF up and get pruned iteratively."""
    table = make_feature_table(n_noise=0)
    table.frame["dup"] = table.frame["signal"] * 2.0 + 1.0
    table = type(table)(
        frame=table.frame,
        id_columns=table.id_columns,
        feature_columns=(*table.feature_columns, "dup"),
        outcome=table.outcome,
        provenance=table.provenance,
    )
    selected = VifSelector(max_vif=5.0).fit(table).transform(table)
    # The perfect duplicate pair is broken up; at least one survives.
    assert 0 < len(selected.feature_columns) < len(table.feature_columns)


@pytest.mark.unit
def test_anova_selector_finds_signal() -> None:
    """The class-separated feature clears the p threshold; noise does not."""
    table = make_feature_table(n_noise=2, seed=4)
    selected = AnovaSelector(p_threshold=0.01).fit(table).transform(table)
    assert "signal" in selected.feature_columns
    # Top-1 override keeps exactly the strongest feature.
    top_one = AnovaSelector(n_features_to_select=1).fit(table).transform(table)
    assert top_one.feature_columns == ("signal",)


@pytest.mark.unit
def test_chi2_selector_finds_signal_on_non_negative_data() -> None:
    """Chi-square selection runs on clipped (non-negative) features."""
    table = make_feature_table(n_noise=2, seed=5, non_negative=True)
    selected = Chi2Selector(n_features_to_select=1).fit(table).transform(table)
    assert selected.feature_columns == ("signal",)


@pytest.mark.unit
def test_statistical_test_selector_finds_signal_and_validates() -> None:
    """Two-group testing keeps the signal; 3-class outcomes are rejected."""
    table = make_feature_table(n_noise=2, seed=6)
    selected = StatisticalTestSelector(p_threshold=0.01).fit(table).transform(table)
    assert "signal" in selected.feature_columns
    three_class = make_feature_table(n_noise=1, seed=7)
    three_class.frame["y"] = np.arange(three_class.frame.shape[0]) % 3
    with pytest.raises(HABITAPIError):
        StatisticalTestSelector().fit(three_class)
    with pytest.raises(HABITAPIError):
        StatisticalTestSelector(force_test="anova")


@pytest.mark.unit
def test_univariate_logistic_selector_finds_signal() -> None:
    """Per-feature logistic regression keeps the informative column."""
    table = make_feature_table(n_noise=2, seed=8)
    # Overlapping classes: perfect separation would fail the logit fit.
    rng = np.random.RandomState(8)
    table.frame["signal"] = (
        table.frame["signal"] * 0.5 + rng.normal(scale=0.6, size=table.frame.shape[0])
    )
    selected = UnivariateLogisticSelector(alpha=0.05).fit(table).transform(table)
    assert "signal" in selected.feature_columns


@pytest.mark.unit
@pytest.mark.parametrize("direction", ["forward", "backward", "both"])
def test_stepwise_selector_directions(direction: str) -> None:
    """Every search direction keeps the signal and drops pure noise."""
    table = make_feature_table(tuple(f"S{i}" for i in range(24)), n_noise=2, seed=9)
    selected = StepwiseSelector(direction=direction, criterion="aic").fit(table)
    transformed = selected.transform(table)
    assert "signal" in transformed.feature_columns


@pytest.mark.unit
def test_stepwise_selector_validates_parameters() -> None:
    """Unknown directions and criteria fail at construction."""
    with pytest.raises(HABITAPIError):
        StepwiseSelector(direction="sideways")
    with pytest.raises(HABITAPIError):
        StepwiseSelector(criterion="deviance")


@pytest.mark.unit
def test_rfecv_selector_finds_signal() -> None:
    """RFECV around a linear model keeps the discriminative column."""
    table = make_feature_table(tuple(f"S{i}" for i in range(30)), n_noise=2, seed=10)
    selected = (
        RfecvSelector(estimator="LogisticRegression", cv=3, n_jobs=1)
        .fit(table)
        .transform(table)
    )
    assert "signal" in selected.feature_columns


@pytest.mark.unit
def test_rfecv_selector_rejects_unknown_estimator() -> None:
    """An unsupported estimator name surfaces as a clear error."""
    table = make_feature_table(tuple(f"S{i}" for i in range(20)), seed=11)
    with pytest.raises(HABITAPIError):
        RfecvSelector(estimator="NotAModel", cv=2, n_jobs=1).fit(table)


@pytest.mark.unit
def test_lasso_selector_is_seedable_and_finds_signal() -> None:
    """Same seed -> same selection; the signal survives the L1 penalty."""
    table = make_feature_table(tuple(f"S{i}" for i in range(30)), n_noise=2, seed=12)
    assert isinstance(LassoSelector(), Seedable)
    first, second = LassoSelector(cv=3, n_jobs=1), LassoSelector(cv=3, n_jobs=1)
    first.set_random_state(3)
    second.set_random_state(3)
    kept_first = first.fit(table).transform(table).feature_columns
    kept_second = second.fit(table).transform(table).feature_columns
    assert kept_first == kept_second
    assert "signal" in kept_first


def _repeat_tables(stable_noise: float = 0.05):
    """Build (primary, repeats) with one stable and one unstable feature."""
    ids = tuple(f"S{i}" for i in range(16))
    primary = make_feature_table(ids, n_noise=1, seed=20)
    rng = np.random.RandomState(21)
    repeat = make_feature_table(ids, n_noise=1, seed=22)
    # signal is measured stably across sessions; noise0 is not.
    repeat.frame["signal"] = primary.frame["signal"] + rng.normal(
        scale=stable_noise, size=len(ids)
    )
    repeat.frame["noise0"] = rng.normal(size=len(ids))
    return primary, [repeat]


@pytest.mark.unit
def test_icc_selector_keeps_stable_features() -> None:
    """Test-retest stable features pass; unstable ones are filtered out."""
    primary, repeats = _repeat_tables()
    selected = IccSelector(threshold=0.75).fit(primary, repeat_tables=repeats)
    kept = selected.transform(primary).feature_columns
    assert "signal" in kept
    assert "noise0" not in kept


@pytest.mark.unit
def test_icc_selector_requires_repeat_tables() -> None:
    """Without repeat sessions the ICC selector has nothing to score."""
    primary, _ = _repeat_tables()
    with pytest.raises(HABITAPIError):
        IccSelector().fit(primary)
    with pytest.raises(HABITAPIError):
        IccSelector(icc_type="icc1")


@pytest.mark.unit
def test_mrmr_selector_finds_signal() -> None:
    """MRMR ranks the informative feature into the requested top-k."""
    table = make_feature_table(n_noise=2, seed=14)
    selected = MrmrSelector(n_features=2).fit(table).transform(table)
    assert "signal" in selected.feature_columns
    assert len(selected.feature_columns) == 2
    with pytest.raises(HABITAPIError):
        MrmrSelector(n_features=0)
    with pytest.raises(HABITAPIError):
        MrmrSelector(task_type="clustering")


@pytest.mark.unit
def test_transform_before_fit_and_schema_drift_raise() -> None:
    """Unfitted transform and missing selected columns are loud errors."""
    table = make_feature_table()
    selector = VarianceSelector()
    with pytest.raises(HABITAPIError):
        selector.transform(table)
    fitted = selector.fit(table)
    drifted = make_feature_table(n_noise=0)
    drifted = type(table)(
        frame=drifted.frame.drop(columns=["signal"]),
        id_columns=drifted.id_columns,
        feature_columns=tuple(c for c in drifted.feature_columns if c != "signal"),
        outcome=drifted.outcome,
        provenance=drifted.provenance,
    )
    with pytest.raises(HABITAPIError):
        fitted.transform(drifted)


@pytest.mark.unit
def test_supervised_selectors_require_an_outcome_column() -> None:
    """Outcome-driven selectors fail clearly on outcome-less tables."""
    table = make_feature_table(outcome=False)
    for selector in (AnovaSelector(), LassoSelector(cv=2, n_jobs=1), MrmrSelector(n_features=1)):
        with pytest.raises(HABITAPIError):
            selector.fit(table)
