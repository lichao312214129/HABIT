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
"""Tests for the nine built-in metrics and the L3 statistics wrappers."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from habit.api.exceptions import HABITAPIError
from habit.evaluation import (
    AccuracyMetric,
    AucMetric,
    F1ScoreMetric,
    HosmerLemeshowPValueMetric,
    MetricRegistry,
    NpvMetric,
    PpvMetric,
    SensitivityMetric,
    SpecificityMetric,
    SpiegelhalterZPValueMetric,
    auc_confidence_interval,
    calibration_tests,
    delong_test,
    icc_analysis,
    repeat_measurement_matrix,
)
from habit._table_protocols import Metric
from habit.kernels.statistics import hosmer_lemeshow_test, spiegelhalter_z_test

from .conftest import make_feature_table


@pytest.mark.unit
def test_registry_lists_all_nine_metrics() -> None:
    """The registry constructs every built-in metric by its v0.1 name."""
    assert set(MetricRegistry.available()) == {
        "accuracy",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "f1_score",
        "auc",
        "hosmer_lemeshow_p_value",
        "spiegelhalter_z_p_value",
    }
    for name in MetricRegistry.available():
        instance = MetricRegistry.create(name)
        assert isinstance(instance, Metric)
        assert instance.spec.name == name
        signature = inspect.signature(type(instance))
        assert "self" not in signature.parameters


@pytest.mark.unit
def test_needs_proba_flags_split_the_family() -> None:
    """Only AUC and the calibration tests consume probabilities."""
    label_metrics = (
        AccuracyMetric(),
        SensitivityMetric(),
        SpecificityMetric(),
        PpvMetric(),
        NpvMetric(),
        F1ScoreMetric(),
    )
    for metric in label_metrics:
        assert metric.needs_proba is False
        assert metric.greater_is_better is True
    for metric in (
        AucMetric(),
        HosmerLemeshowPValueMetric(),
        SpiegelhalterZPValueMetric(),
    ):
        assert metric.needs_proba is True
        assert metric.greater_is_better is True


@pytest.mark.unit
def test_label_metrics_on_a_known_confusion_matrix() -> None:
    """TN=4, FP=1, FN=2, TP=3 pins every confusion-matrix metric."""
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 1, 0, 0, 1, 1, 1])
    assert AccuracyMetric()(y_true, y_pred) == pytest.approx(0.7)
    assert SensitivityMetric()(y_true, y_pred) == pytest.approx(3 / 5)
    assert SpecificityMetric()(y_true, y_pred) == pytest.approx(4 / 5)
    assert PpvMetric()(y_true, y_pred) == pytest.approx(3 / 4)
    assert NpvMetric()(y_true, y_pred) == pytest.approx(4 / 6)
    expected_f1 = 2 * (3 / 4) * (3 / 5) / ((3 / 4) + (3 / 5))
    assert F1ScoreMetric()(y_true, y_pred) == pytest.approx(expected_f1)


@pytest.mark.unit
def test_sensitivity_macro_averages_multiclass_recalls() -> None:
    """Multi-class sensitivity is the macro mean of per-class recalls."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 0, 2, 0])
    # recalls: class0 = 1.0, class1 = 0.5, class2 = 0.5 -> macro 2/3.
    assert SensitivityMetric()(y_true, y_pred) == pytest.approx(2 / 3)


@pytest.mark.unit
def test_auc_matches_sklearn_binary_and_multiclass() -> None:
    """Binary AUC equals roc_auc_score; a matrix input takes the ovr branch."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.RandomState(0)
    y_true = np.repeat([0, 1], 30)
    scores = rng.uniform(size=60) * 0.4 + y_true * 0.5
    assert AucMetric()(y_true, (scores > 0.5).astype(int), scores) == pytest.approx(
        roc_auc_score(y_true, scores)
    )
    y_multi = np.repeat([0, 1, 2], 20)
    proba = rng.dirichlet(alpha=[2, 2, 2], size=60)
    assert AucMetric()(y_multi, proba.argmax(axis=1), proba) == pytest.approx(
        roc_auc_score(y_multi, proba, multi_class="ovr")
    )


@pytest.mark.unit
def test_calibration_metrics_delegate_to_the_kernels() -> None:
    """Metric p-values equal the kernel values on the same inputs."""
    rng = np.random.RandomState(1)
    probabilities = rng.uniform(0.05, 0.95, size=200)
    y_true = (rng.uniform(size=200) < probabilities).astype(float)
    y_pred = (probabilities > 0.5).astype(float)
    _, kernel_hl = hosmer_lemeshow_test(y_true, probabilities, n_groups=10)
    _, kernel_z = spiegelhalter_z_test(y_true, probabilities)
    assert HosmerLemeshowPValueMetric()(y_true, y_pred, probabilities) == pytest.approx(
        kernel_hl
    )
    assert SpiegelhalterZPValueMetric()(y_true, y_pred, probabilities) == pytest.approx(
        kernel_z
    )
    # n_groups is the one configurable parameter.
    custom = HosmerLemeshowPValueMetric(n_groups=5)
    _, kernel_hl5 = hosmer_lemeshow_test(y_true, probabilities, n_groups=5)
    assert custom(y_true, y_pred, probabilities) == pytest.approx(kernel_hl5)
    assert custom.spec.params["n_groups"] == 5


@pytest.mark.unit
def test_calibration_metrics_fail_soft_to_nan() -> None:
    """Multi-class or degenerate inputs read as NaN, as in v0.1."""
    y_multi = np.repeat([0, 1, 2], 10)
    proba = np.tile([0.2, 0.3, 0.5], (10, 1))
    assert np.isnan(HosmerLemeshowPValueMetric()(y_multi, proba.argmax(axis=1), proba))
    assert np.isnan(SpiegelhalterZPValueMetric()(y_multi, proba.argmax(axis=1), proba))
    # Tied probabilities cannot form ten risk groups -> NaN, not an error.
    y_binary = np.repeat([0, 1], 20)
    assert np.isnan(
        HosmerLemeshowPValueMetric()(y_binary, y_binary, np.full(40, 0.5))
    )


@pytest.mark.unit
def test_delong_wrapper_bundles_aucs_and_pvalue() -> None:
    """The L3 wrapper reports both AUCs and the paired p-value."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.RandomState(2)
    y_true = np.repeat([0, 1], 40)
    scores_a = rng.uniform(size=80) * 0.3 + y_true * 0.6
    scores_b = rng.uniform(size=80)
    result = delong_test(y_true, scores_a, scores_b)
    assert result.auc_a == pytest.approx(roc_auc_score(y_true, scores_a))
    assert result.auc_b == pytest.approx(roc_auc_score(y_true, scores_b))
    assert 0.0 <= result.p_value <= 1.0
    identical = delong_test(y_true, scores_a, scores_a)
    assert identical.p_value == pytest.approx(1.0)


@pytest.mark.unit
def test_auc_confidence_interval_wrapper() -> None:
    """The CI wrapper brackets the AUC and records the level."""
    rng = np.random.RandomState(3)
    y_true = np.repeat([0, 1], 40)
    # Overlapping classes: at AUC == 1 the DeLong variance degenerates.
    scores = rng.uniform(size=80) * 0.7 + y_true * 0.25
    interval = auc_confidence_interval(y_true, scores, alpha=0.9)
    assert interval.alpha == pytest.approx(0.9)
    assert interval.lower <= interval.auc <= interval.upper
    assert interval.upper <= 1.0


@pytest.mark.unit
def test_calibration_tests_wrapper_matches_kernels() -> None:
    """The combined wrapper reports exactly the two kernel outcomes."""
    rng = np.random.RandomState(4)
    probabilities = rng.uniform(0.05, 0.95, size=200)
    y_true = (rng.uniform(size=200) < probabilities).astype(float)
    result = calibration_tests(y_true, probabilities, n_groups=8)
    hl_stat, hl_p = hosmer_lemeshow_test(y_true, probabilities, n_groups=8)
    z, z_p = spiegelhalter_z_test(y_true, probabilities)
    assert result.hl_statistic == pytest.approx(hl_stat)
    assert result.hl_p_value == pytest.approx(hl_p)
    assert result.spiegelhalter_z == pytest.approx(z)
    assert result.spiegelhalter_p_value == pytest.approx(z_p)


def _repeat_tables(nan_row: bool = False):
    """Primary + one repeat session with a stable and an unstable feature."""
    ids = tuple(f"S{i}" for i in range(10))
    primary = make_feature_table(ids, n_noise=1, seed=30)
    rng = np.random.RandomState(31)
    repeat = make_feature_table(ids, n_noise=1, seed=32)
    repeat.frame["signal"] = primary.frame["signal"] + rng.normal(scale=0.01, size=10)
    repeat.frame["noise0"] = rng.normal(size=10)
    if nan_row:
        repeat.frame.loc[3, "signal"] = np.nan
    return primary, [repeat]


@pytest.mark.unit
def test_repeat_measurement_matrix_aligns_and_omits_nan() -> None:
    """Sessions align by id order; NaN-carrying subjects drop out."""
    primary, repeats = _repeat_tables(nan_row=True)
    matrix = repeat_measurement_matrix(primary, repeats, "signal", owner="test")
    assert matrix.shape == (9, 2)  # subject S3 omitted
    np.testing.assert_allclose(matrix[:, 0], primary.frame["signal"].drop(index=3))


@pytest.mark.unit
def test_repeat_measurement_matrix_validation() -> None:
    """Mismatched ids or a missing feature are explicit errors."""
    primary, repeats = _repeat_tables()
    with pytest.raises(HABITAPIError):
        repeat_measurement_matrix(primary, repeats, "missing", owner="test")
    broken = make_feature_table(tuple(f"X{i}" for i in range(10)), n_noise=1, seed=33)
    with pytest.raises(HABITAPIError):
        repeat_measurement_matrix(primary, [broken], "signal", owner="test")


@pytest.mark.unit
def test_icc_analysis_reports_per_feature_iccs() -> None:
    """Stable features score ~1, unstable features low, per variant column."""
    primary, repeats = _repeat_tables()
    result = icc_analysis(primary, repeats)
    assert list(result.columns) == ["feature", "icc2", "icc3"]
    by_feature = result.set_index("feature")
    assert by_feature.loc["signal", "icc3"] > 0.9
    assert by_feature.loc["noise0", "icc3"] < 0.9
    only_icc3 = icc_analysis(primary, repeats, icc_types=("icc3",))
    assert list(only_icc3.columns) == ["feature", "icc3"]


@pytest.mark.unit
def test_icc_analysis_min_subjects_and_validation() -> None:
    """Too-few complete subjects yield NaN; bad requests raise."""
    primary, repeats = _repeat_tables()
    result = icc_analysis(primary, repeats, min_subjects=100)
    assert result[["icc2", "icc3"]].isna().all().all()
    with pytest.raises(HABITAPIError):
        icc_analysis(primary, [])
    with pytest.raises(HABITAPIError):
        icc_analysis(primary, repeats, icc_types=("icc1",))
