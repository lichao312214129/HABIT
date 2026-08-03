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
"""Tests for the eight built-in table preprocessors."""

from __future__ import annotations

import numpy as np
import pytest

from habit.api.exceptions import HABITAPIError
from habit.domain.table_preprocessing import (
    BinningPreprocessor,
    CorrelationFilterPreprocessor,
    LogPreprocessor,
    MinMaxPreprocessor,
    RobustPreprocessor,
    TablePreprocessorRegistry,
    VarianceFilterPreprocessor,
    WinsorizePreprocessor,
    ZScorePreprocessor,
)
from habit.domain.table_protocols import TablePreprocessor

from .conftest import make_feature_table

_ALL = (
    MinMaxPreprocessor,
    ZScorePreprocessor,
    RobustPreprocessor,
    BinningPreprocessor,
    WinsorizePreprocessor,
    LogPreprocessor,
    VarianceFilterPreprocessor,
    CorrelationFilterPreprocessor,
)


@pytest.mark.unit
def test_registry_lists_all_eight_methods() -> None:
    """The registry constructs every built-in method by its v0.1 name."""
    assert set(TablePreprocessorRegistry.available()) == {
        "minmax",
        "zscore",
        "robust",
        "binning",
        "winsorize",
        "log",
        "variance_filter",
        "correlation_filter",
    }
    for name in TablePreprocessorRegistry.available():
        instance = TablePreprocessorRegistry.create(name)
        assert isinstance(instance, TablePreprocessor)
        assert instance.spec.name == name
        assert TablePreprocessorRegistry.get_params_model(name) is not None


@pytest.mark.unit
def test_minmax_scales_with_training_statistics() -> None:
    """Train maps to [0, 1]; new data uses the TRAINING min/max."""
    table = make_feature_table()
    pre = MinMaxPreprocessor().fit(table)
    transformed = pre.transform(table)
    block = transformed.frame[list(transformed.feature_columns)]
    np.testing.assert_allclose(block.min().to_numpy(), np.zeros(block.shape[1]))
    np.testing.assert_allclose(block.max().to_numpy(), np.ones(block.shape[1]))
    # A new row beyond the training range is scaled with training stats and
    # therefore lands OUTSIDE [0, 1] -- no leakage of test statistics.
    new = make_feature_table(["X1", "X2"], seed=99)
    new.frame.loc[:, "signal"] = [table.frame["signal"].max() * 2, 0.0]
    out = pre.transform(new)
    assert out.frame.loc[0, "signal"] > 1.0


@pytest.mark.unit
def test_minmax_global_mode_uses_one_pair() -> None:
    """Global normalisation scales the whole block by one (min, max)."""
    table = make_feature_table()
    pre = MinMaxPreprocessor(global_normalize=True).fit(table)
    block = pre.transform(table).frame[list(table.feature_columns)]
    assert block.values.min() == pytest.approx(0.0)
    assert block.values.max() == pytest.approx(1.0)


@pytest.mark.unit
def test_zscore_standardises_and_tames_constants() -> None:
    """Per-column mean 0 / std 1; a constant column maps to 0, not NaN."""
    table = make_feature_table(constant_column=True)
    transformed = ZScorePreprocessor().fit(table).transform(table)
    block = transformed.frame[list(transformed.feature_columns)]
    np.testing.assert_allclose(block.mean().to_numpy(), np.zeros(block.shape[1]), atol=1e-12)
    varying = [c for c in block.columns if c != "constant"]
    np.testing.assert_allclose(block[varying].std().to_numpy(), np.ones(len(varying)))
    assert (transformed.frame["constant"] == 0.0).all()


@pytest.mark.unit
def test_robust_scales_by_median_and_iqr() -> None:
    """The transformed training block has median 0 per column."""
    table = make_feature_table()
    transformed = RobustPreprocessor().fit(table).transform(table)
    block = transformed.frame[list(transformed.feature_columns)]
    np.testing.assert_allclose(block.median().to_numpy(), np.zeros(block.shape[1]), atol=1e-12)


@pytest.mark.unit
def test_binning_produces_ordinal_bins() -> None:
    """Binned values are integer ordinals below n_bins, seeded kmeans included."""
    table = make_feature_table()
    transformed = BinningPreprocessor(n_bins=4).fit(table).transform(table)
    block = transformed.frame[list(transformed.feature_columns)]
    assert set(np.unique(block.values)) <= {0.0, 1.0, 2.0, 3.0}
    # The kmeans strategy is stochastic: Seedable makes it deterministic.
    first = BinningPreprocessor(n_bins=4, bin_strategy="kmeans")
    second = BinningPreprocessor(n_bins=4, bin_strategy="kmeans")
    first.set_random_state(7)
    second.set_random_state(7)
    np.testing.assert_allclose(
        first.fit(table).transform(table).frame[list(table.feature_columns)].values,
        second.fit(table).transform(table).frame[list(table.feature_columns)].values,
    )


@pytest.mark.unit
def test_winsorize_clips_at_training_quantiles() -> None:
    """Values beyond the training quantile bounds are clipped to them."""
    table = make_feature_table()
    pre = WinsorizePreprocessor(winsor_limits=(0.1, 0.1)).fit(table)
    block = pre.transform(table).frame[list(table.feature_columns)]
    train = table.frame[list(table.feature_columns)]
    for column in block.columns:
        assert block[column].min() >= train[column].quantile(0.1) - 1e-12
        assert block[column].max() <= train[column].quantile(0.9) + 1e-12
    with pytest.raises(HABITAPIError):
        WinsorizePreprocessor(winsor_limits=(0.6, 0.1))


@pytest.mark.unit
def test_log_transform_shifts_by_training_minimum() -> None:
    """log(x - min_train + 1): the training minimum maps to log(1) == 0."""
    table = make_feature_table(non_negative=True)
    transformed = LogPreprocessor().fit(table).transform(table)
    train = table.frame[list(table.feature_columns)]
    block = transformed.frame[list(table.feature_columns)]
    expected = np.log(train - train.min() + 1.0)
    np.testing.assert_allclose(block.values, expected.values)
    assert block.values.min() == pytest.approx(0.0)


@pytest.mark.unit
def test_variance_filter_drops_constant_column() -> None:
    """Constant columns are dropped; the kept subset is frozen for later tables."""
    table = make_feature_table(constant_column=True)
    pre = VarianceFilterPreprocessor().fit(table)
    transformed = pre.transform(table)
    assert "constant" not in transformed.feature_columns
    assert set(transformed.feature_columns) == {"signal", "noise0", "noise1", "noise2"}
    # The frozen subset applies to new tables too.
    new = make_feature_table(["X1"], seed=3, constant_column=True)
    assert pre.transform(new).feature_columns == transformed.feature_columns


@pytest.mark.unit
def test_variance_filter_keeps_at_least_one_column() -> None:
    """A table of constants still yields its highest-variance column."""
    table = make_feature_table(n_noise=0)
    table.frame.loc[:, "signal"] = 1.0
    transformed = VarianceFilterPreprocessor().fit(table).transform(table)
    assert len(transformed.feature_columns) == 1


@pytest.mark.unit
def test_correlation_filter_drops_redundant_later_column() -> None:
    """A duplicated column is pruned by the greedy left-to-right walk."""
    table = make_feature_table()
    table.frame["signal_copy"] = table.frame["signal"] * 2.0 + 1.0
    table = type(table)(
        frame=table.frame,
        id_columns=table.id_columns,
        feature_columns=(*table.feature_columns, "signal_copy"),
        outcome_column=table.outcome_column,
        provenance=table.provenance,
    )
    transformed = CorrelationFilterPreprocessor(corr_threshold=0.9).fit(table).transform(table)
    assert "signal" in transformed.feature_columns
    assert "signal_copy" not in transformed.feature_columns


@pytest.mark.unit
@pytest.mark.parametrize("cls", _ALL, ids=lambda c: c.__name__)
def test_transform_before_fit_and_schema_drift_raise(cls) -> None:
    """Unfitted transform and missing fit columns are loud errors."""
    table = make_feature_table()
    with pytest.raises(HABITAPIError):
        cls().transform(table)
    fitted = cls().fit(table)
    drifted = make_feature_table(n_noise=0)
    drifted = type(table)(
        frame=drifted.frame.drop(columns=["signal"]),
        id_columns=drifted.id_columns,
        feature_columns=tuple(c for c in drifted.feature_columns if c != "signal"),
        outcome_column=drifted.outcome_column,
        provenance=drifted.provenance,
    )
    with pytest.raises(HABITAPIError):
        fitted.transform(drifted)


@pytest.mark.unit
def test_transform_preserves_identifier_and_outcome_columns() -> None:
    """Only feature values change; ids and outcomes pass through untouched."""
    table = make_feature_table()
    transformed = ZScorePreprocessor().fit(table).transform(table)
    assert transformed.id_columns == table.id_columns
    assert transformed.outcome_column == table.outcome_column
    np.testing.assert_array_equal(
        transformed.frame["subject"].to_numpy(), table.frame["subject"].to_numpy()
    )
    np.testing.assert_array_equal(
        transformed.frame["y"].to_numpy(), table.frame["y"].to_numpy()
    )
