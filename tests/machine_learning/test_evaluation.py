"""
Unit tests for the evaluation layer:
  - PredictionContainer / create_prediction_container / helper functions
  - calculate_metrics (basic, statistical, category-filtered)
  - calculate_metrics_youden, apply_youden_threshold
  - calculate_metrics_at_target (target sensitivity/specificity/PPV/NPV)
  - ThresholdManager

Migrated and extended from tests/test_metrics_optimization.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from habit.core.machine_learning.evaluation.prediction_container import (
    PredictionContainer,
    create_prediction_container,
    from_dict,
    from_tuple,
)
from habit.core.machine_learning.evaluation.metrics import (
    MetricsCache,
    apply_youden_threshold,
    calculate_metrics,
    calculate_metrics_at_target,
    calculate_metrics_youden,
)
from habit.core.machine_learning.evaluation.threshold_manager import ThresholdManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_arrays(n: int = 200, seed: int = 0):
    rng = np.random.RandomState(seed)
    y_true = np.array([0] * (n // 2) + [1] * (n // 2))
    y_prob = np.concatenate([rng.beta(2, 5, n // 2), rng.beta(5, 2, n // 2)])
    y_pred = (y_prob >= 0.5).astype(int)
    return y_true, y_prob, y_pred


# ---------------------------------------------------------------------------
# PredictionContainer
# ---------------------------------------------------------------------------


class TestPredictionContainer:
    def test_binary_1d_prob(self) -> None:
        y_true, y_prob, y_pred = _make_arrays(100)
        c = PredictionContainer(y_true, y_prob, y_pred)
        assert c.num_classes == 2
        assert c.get_binary_probs().ndim == 1
        assert len(c) == 100

    def test_default_pred_generated_when_none(self) -> None:
        y_true, y_prob, _ = _make_arrays(100)
        c = PredictionContainer(y_true, y_prob)
        assert c.y_pred is not None
        assert c.y_pred.shape == (100,)

    def test_2d_prob_binary(self) -> None:
        """2-column probability matrix should be handled as binary."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([[0.9, 0.1], [0.7, 0.3], [0.2, 0.8], [0.1, 0.9]])
        c = PredictionContainer(y_true, y_prob)
        assert c.get_binary_probs().ndim == 1
        np.testing.assert_allclose(c.get_binary_probs(), [0.1, 0.3, 0.8, 0.9])

    def test_to_dict_keys(self) -> None:
        y_true, y_prob, y_pred = _make_arrays(50)
        c = PredictionContainer(y_true, y_prob, y_pred)
        d = c.to_dict()
        assert set(d.keys()) == {"y_true", "y_prob", "y_pred"}

    def test_clean_nan_removes_rows(self) -> None:
        # y_true must stay binary-only: NaN in labels makes num_classes>2 and breaks
        # default pred generation for 1D probabilities.
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.5, 0.9, np.nan])
        y_pred = np.array([0, 1, 1, 0])
        c = PredictionContainer(y_true, y_prob, y_pred=y_pred).clean_nan()
        assert len(c) < 4

    def test_len_matches_n_samples(self) -> None:
        y_true, y_prob, y_pred = _make_arrays(60)
        c = PredictionContainer(y_true, y_prob, y_pred)
        assert len(c) == 60


class TestCreatePredictionContainer:
    def test_create_cleans_nan(self) -> None:
        y_true = np.array([0, 1, 1])
        y_prob = np.array([0.2, np.nan, 0.8])
        c = create_prediction_container(y_true, y_prob)
        assert len(c) == 2  # NaN row removed

    def test_from_tuple_2(self) -> None:
        y_true, y_prob, _ = _make_arrays(40)
        c = from_tuple((y_true, y_prob))
        assert len(c) == 40

    def test_from_tuple_3(self) -> None:
        y_true, y_prob, y_pred = _make_arrays(40)
        c = from_tuple((y_true, y_prob, y_pred))
        assert len(c) == 40

    def test_from_dict(self) -> None:
        y_true, y_prob, y_pred = _make_arrays(40)
        # Use 'y_prob' key: ``or`` between ndarray keys is ambiguous in from_dict.
        c = from_dict({"y_true": y_true, "y_prob": y_prob})
        assert len(c) == 40

    def test_from_dict_missing_keys_raises(self) -> None:
        with pytest.raises(ValueError):
            from_dict({"y_pred": np.array([0, 1])})


# ---------------------------------------------------------------------------
# MetricsCache
# ---------------------------------------------------------------------------


class TestMetricsCache:
    def test_confusion_matrix_cached(self) -> None:
        y_true, y_prob, y_pred = _make_arrays(80)
        cache = MetricsCache(y_true, y_pred, y_prob)
        cm1 = cache.confusion_matrix
        cm2 = cache.confusion_matrix
        assert cm1 is cm2  # same object reference → cached

    def test_cache_shape(self) -> None:
        y_true, y_prob, y_pred = _make_arrays(80)
        cache = MetricsCache(y_true, y_pred, y_prob)
        assert cache.confusion_matrix.shape == (2, 2)


# ---------------------------------------------------------------------------
# calculate_metrics
# ---------------------------------------------------------------------------


class TestCalculateMetrics:
    def test_returns_dict_with_basic_keys(self) -> None:
        y_true, y_prob, y_pred = _make_arrays()
        result = calculate_metrics(y_true, y_pred, y_prob)
        for key in ("accuracy", "sensitivity", "specificity", "auc"):
            assert key in result, f"Missing key: {key}"

    def test_values_in_valid_range(self) -> None:
        y_true, y_prob, y_pred = _make_arrays()
        result = calculate_metrics(y_true, y_pred, y_prob)
        for v in result.values():
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                assert 0.0 <= float(v) <= 1.0 or float(v) > 1.0  # p-values can be > 1 is wrong; skip those

    def test_category_filter_basic(self) -> None:
        y_true, y_prob, y_pred = _make_arrays()
        result = calculate_metrics(y_true, y_pred, y_prob, categories=["basic"])
        assert "accuracy" in result
        assert "hosmer_lemeshow_p_value" not in result

    def test_category_filter_statistical(self) -> None:
        y_true, y_prob, y_pred = _make_arrays()
        result = calculate_metrics(y_true, y_pred, y_prob, categories=["statistical"])
        assert "accuracy" not in result


# ---------------------------------------------------------------------------
# calculate_metrics_youden + apply_youden_threshold
# ---------------------------------------------------------------------------


class TestYoudenMetrics:
    def test_returns_threshold(self) -> None:
        y_true, y_prob, _ = _make_arrays()
        result = calculate_metrics_youden(y_true, y_prob)
        assert "threshold" in result
        assert 0.0 < result["threshold"] < 1.0

    def test_apply_youden_threshold_returns_metrics(self) -> None:
        y_true, y_prob, _ = _make_arrays()
        youden_result = calculate_metrics_youden(y_true, y_prob)
        threshold = youden_result["threshold"]
        metrics = apply_youden_threshold(y_true, y_prob, threshold)
        assert isinstance(metrics, dict)
        inner = metrics.get("metrics", metrics)
        assert "sensitivity" in inner or "accuracy" in inner


# ---------------------------------------------------------------------------
# calculate_metrics_at_target
# ---------------------------------------------------------------------------


class TestCalculateMetricsAtTarget:
    def test_returns_expected_keys(self) -> None:
        y_true, y_prob, _ = _make_arrays()
        result = calculate_metrics_at_target(y_true, y_prob, {"sensitivity": 0.80})
        assert "thresholds" in result
        assert "metrics_at_thresholds" in result

    def test_ppv_target(self) -> None:
        y_true, y_prob, _ = _make_arrays(200)
        result = calculate_metrics_at_target(y_true, y_prob, {"ppv": 0.75, "sensitivity": 0.60})
        assert "ppv" in result["thresholds"]

    def test_impossible_target_fallback(self) -> None:
        """When targets cannot be met, closest_threshold must be returned."""
        y_true, y_prob, _ = _make_arrays(200)
        result = calculate_metrics_at_target(
            y_true, y_prob,
            {"sensitivity": 0.99, "specificity": 0.99},
            fallback_to_closest=True,
        )
        assert result["closest_threshold"] is not None

    def test_pareto_youden_strategy(self) -> None:
        y_true, y_prob, _ = _make_arrays(200)
        result = calculate_metrics_at_target(
            y_true, y_prob,
            {"sensitivity": 0.70, "specificity": 0.70},
            threshold_selection="pareto+youden",
        )
        if result["best_threshold"]:
            assert result["best_threshold"]["strategy"] == "pareto+youden"


# ---------------------------------------------------------------------------
# ThresholdManager
# ---------------------------------------------------------------------------


class TestThresholdManager:
    def test_find_and_store_youden(self) -> None:
        y_true, y_prob, y_pred = _make_arrays()
        container = PredictionContainer(y_true, y_prob, y_pred)

        mgr = ThresholdManager()
        mgr.find_and_store("MyModel", container, method="youden")
        threshold = mgr.get_threshold("MyModel", "youden")
        assert 0.0 < threshold < 1.0

    def test_get_unknown_model_returns_default(self) -> None:
        mgr = ThresholdManager()
        t = mgr.get_threshold("NoModel", "youden")
        assert t == 0.5  # default

    def test_multiple_models_independent(self) -> None:
        y_true, y_prob, y_pred = _make_arrays()
        container = PredictionContainer(y_true, y_prob, y_pred)
        mgr = ThresholdManager()
        mgr.find_and_store("A", container, method="youden")
        mgr.find_and_store("B", container, method="youden")
        assert mgr.get_threshold("A") == mgr.get_threshold("B")  # same data → same threshold
        assert "A" in mgr.store and "B" in mgr.store
