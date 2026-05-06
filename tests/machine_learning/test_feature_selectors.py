"""
Integration tests for registered feature-selector callables.

Selectors are exposed via ``register_selector`` + ``run_selector``, not standalone
``*Selector`` sklearn-style classes (legacy tests assumed class APIs).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from habit.core.machine_learning.feature_selectors import (
    get_available_selectors,
    run_selector,
)


def _make_X_y(n: int = 100, n_features: int = 20, seed: int = 0):
    """Build a reproducible binary classification tabular dataset."""
    X_arr, y_arr = make_classification(
        n_samples=n,
        n_features=n_features,
        n_informative=5,
        n_redundant=5,
        random_state=seed,
    )
    X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(n_features)])
    y = pd.Series(y_arr, name="label")
    return X, y


class TestVarianceViaRegistry:
    def test_removes_zero_variance_columns(self) -> None:
        X, y = _make_X_y()
        X["const"] = 1.0
        kept = run_selector(
            "variance",
            X,
            y,
            threshold=0.0,
            plot_variances=False,
        )
        assert "const" not in kept

    def test_keeps_high_variance_columns_when_none_constant(self) -> None:
        X, y = _make_X_y()
        n_cols_before = X.shape[1]
        kept = run_selector(
            "variance",
            X,
            y,
            threshold=0.0,
            plot_variances=False,
        )
        assert len(kept) == n_cols_before

    def test_fit_transform_via_run_selector(self) -> None:
        X, y = _make_X_y()
        X["zero"] = 0.0
        kept = run_selector(
            "variance",
            X,
            y,
            threshold=0.0,
            plot_variances=False,
        )
        X_out = X[kept]
        assert isinstance(X_out, pd.DataFrame)
        assert "zero" not in X_out.columns


class TestCorrelationViaRegistry:
    def test_removes_highly_correlated_columns(self) -> None:
        base = np.linspace(0, 1, 100)
        X = pd.DataFrame({
            "A": base,
            "B": base * 1.001,
            "C": np.random.RandomState(0).randn(100),
        })
        y = pd.Series(np.random.RandomState(0).randint(0, 2, 100))
        kept = run_selector(
            "correlation",
            X,
            y,
            threshold=0.99,
            visualize=False,
        )
        assert len(kept) < 3

    def test_threshold_one_keeps_all(self) -> None:
        X, y = _make_X_y(100, 10)
        kept = run_selector(
            "correlation",
            X,
            y,
            threshold=1.0,
            visualize=False,
        )
        assert len(kept) == X.shape[1]


class TestLassoViaRegistry:
    def test_returns_subset_of_features(self) -> None:
        X, y = _make_X_y(100, 20)
        kept = run_selector(
            "lasso",
            X,
            y,
            cv=3,
            n_alphas=8,
            visualize=False,
            random_state=0,
        )
        assert isinstance(kept, list)
        assert len(kept) <= X.shape[1]

    def test_runs_with_explicit_alpha_grid(self) -> None:
        X, y = _make_X_y(100, 20)
        alphas = np.logspace(-2, 1, 12)
        kept = run_selector(
            "lasso",
            X,
            y,
            cv=3,
            alphas=list(alphas),
            n_alphas=len(alphas),
            visualize=False,
            random_state=0,
        )
        assert len(kept) >= 1


class TestAnovaViaRegistry:
    def test_selects_up_to_k_features(self) -> None:
        X, y = _make_X_y(100, 20)
        k = 5
        kept = run_selector(
            "anova",
            X,
            y,
            n_features_to_select=k,
            plot_importance=False,
        )
        assert len(kept) <= k


class TestVifViaRegistry:
    def test_run_selector_completes(self) -> None:
        X, y = _make_X_y(80, 6)
        kept = run_selector("vif", X, y, max_vif=5.0, visualize=False)
        assert isinstance(kept, list)
        assert 0 < len(kept) <= X.shape[1]

    def test_reduces_near_duplicate_columns(self) -> None:
        rng = np.random.RandomState(0)
        base = rng.randn(80)
        X = pd.DataFrame({
            "A": base,
            "B": base + rng.randn(80) * 0.01,
            "C": rng.randn(80),
        })
        y = pd.Series(rng.randint(0, 2, 80))
        kept = run_selector("vif", X, y, max_vif=5.0, visualize=False)
        assert len(kept) < 3


class TestSelectorRegistryMetadata:
    def test_variance_registered(self) -> None:
        assert "variance" in get_available_selectors()

    def test_registry_run_variance(self) -> None:
        X, y = _make_X_y(30, 8)
        kept = run_selector(
            "variance",
            X,
            y,
            threshold=0.0,
            plot_variances=False,
        )
        assert len(kept) >= 1
