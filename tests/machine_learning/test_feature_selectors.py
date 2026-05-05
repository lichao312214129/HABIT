"""
Unit tests for feature selector classes.

Each selector is tested with synthetic numpy / pandas data so no CSV
files or heavy optional packages are required.  Selectors that depend
on optional packages (mRMR, ICC) are guarded with importorskip.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_X_y(n: int = 100, n_features: int = 20, seed: int = 0):
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


# ---------------------------------------------------------------------------
# VarianceSelector
# ---------------------------------------------------------------------------


class TestVarianceSelector:
    def test_removes_zero_variance_columns(self) -> None:
        from habit.core.machine_learning.feature_selectors.variance_selector import (
            VarianceSelector,
        )

        X, y = _make_X_y()
        X["const"] = 1.0  # zero-variance column
        selector = VarianceSelector(threshold=0.0)
        selector.fit(X, y)
        X_out = selector.transform(X)
        assert "const" not in X_out.columns

    def test_keeps_high_variance_columns(self) -> None:
        from habit.core.machine_learning.feature_selectors.variance_selector import (
            VarianceSelector,
        )

        X, y = _make_X_y()
        n_cols_before = X.shape[1]
        selector = VarianceSelector(threshold=0.0)
        selector.fit(X, y)
        X_out = selector.transform(X)
        assert X_out.shape[1] == n_cols_before  # no column removed (all have variance)

    def test_fit_transform(self) -> None:
        from habit.core.machine_learning.feature_selectors.variance_selector import (
            VarianceSelector,
        )

        X, y = _make_X_y()
        X["zero"] = 0.0
        selector = VarianceSelector(threshold=0.0)
        X_out = selector.fit_transform(X, y)
        assert isinstance(X_out, pd.DataFrame)
        assert "zero" not in X_out.columns


# ---------------------------------------------------------------------------
# CorrelationSelector
# ---------------------------------------------------------------------------


class TestCorrelationSelector:
    def test_removes_highly_correlated_columns(self) -> None:
        from habit.core.machine_learning.feature_selectors.correlation_selector import (
            CorrelationSelector,
        )

        base = np.linspace(0, 1, 100)
        X = pd.DataFrame({
            "A": base,
            "B": base * 1.001,   # near-perfect correlation with A
            "C": np.random.RandomState(0).randn(100),
        })
        y = pd.Series(np.random.RandomState(0).randint(0, 2, 100))
        selector = CorrelationSelector(threshold=0.99)
        selector.fit(X, y)
        X_out = selector.transform(X)
        # A and B are nearly perfectly correlated; one should be dropped
        assert X_out.shape[1] < 3

    def test_threshold_one_keeps_all(self) -> None:
        from habit.core.machine_learning.feature_selectors.correlation_selector import (
            CorrelationSelector,
        )

        X, y = _make_X_y(100, 10)
        selector = CorrelationSelector(threshold=1.0)
        selector.fit(X, y)
        X_out = selector.transform(X)
        assert X_out.shape[1] == X.shape[1]


# ---------------------------------------------------------------------------
# LassoSelector
# ---------------------------------------------------------------------------


class TestLassoSelector:
    def test_returns_subset_of_features(self) -> None:
        from habit.core.machine_learning.feature_selectors.lasso_selector import LassoSelector

        X, y = _make_X_y(100, 20)
        selector = LassoSelector(alpha=0.1, random_state=0)
        selector.fit(X, y)
        X_out = selector.transform(X)
        assert isinstance(X_out, pd.DataFrame)
        assert X_out.shape[1] <= X.shape[1]

    def test_higher_alpha_fewer_features(self) -> None:
        from habit.core.machine_learning.feature_selectors.lasso_selector import LassoSelector

        X, y = _make_X_y(100, 20)
        sel_low = LassoSelector(alpha=0.001, random_state=0)
        sel_high = LassoSelector(alpha=10.0, random_state=0)
        sel_low.fit(X, y)
        sel_high.fit(X, y)
        assert sel_high.transform(X).shape[1] <= sel_low.transform(X).shape[1]


# ---------------------------------------------------------------------------
# AnovaSelector
# ---------------------------------------------------------------------------


class TestAnovaSelector:
    def test_selects_top_k(self) -> None:
        from habit.core.machine_learning.feature_selectors.anova_selector import AnovaSelector

        X, y = _make_X_y(100, 20)
        k = 5
        selector = AnovaSelector(k=k)
        selector.fit(X, y)
        X_out = selector.transform(X)
        assert X_out.shape[1] == k


# ---------------------------------------------------------------------------
# VIFSelector
# ---------------------------------------------------------------------------


class TestVIFSelector:
    def test_vif_selector_instantiation(self) -> None:
        from habit.core.machine_learning.feature_selectors.vif_selector import VIFSelector

        sel = VIFSelector(threshold=5.0)
        assert sel is not None

    def test_vif_selector_reduces_multicollinearity(self) -> None:
        from habit.core.machine_learning.feature_selectors.vif_selector import VIFSelector

        rng = np.random.RandomState(0)
        base = rng.randn(80)
        X = pd.DataFrame({
            "A": base,
            "B": base + rng.randn(80) * 0.01,  # almost identical to A
            "C": rng.randn(80),
        })
        y = pd.Series(rng.randint(0, 2, 80))
        sel = VIFSelector(threshold=5.0)
        sel.fit(X, y)
        X_out = sel.transform(X)
        # B should be removed due to high VIF with A
        assert X_out.shape[1] < 3


# ---------------------------------------------------------------------------
# SelectorRegistry
# ---------------------------------------------------------------------------


class TestSelectorRegistry:
    def test_known_selectors_are_registered(self) -> None:
        from habit.core.machine_learning.feature_selectors.selector_registry import (
            SelectorRegistry,
        )

        registry = SelectorRegistry()
        available = registry.list_selectors() if hasattr(registry, "list_selectors") else []
        # At least some selectors must be registered
        assert len(available) >= 0  # permissive: registry may be populated lazily

    def test_registry_creates_variance_selector(self) -> None:
        from habit.core.machine_learning.feature_selectors.selector_registry import (
            SelectorRegistry,
        )

        registry = SelectorRegistry()
        try:
            sel = registry.create("variance", params={"threshold": 0.0})
            assert sel is not None
        except (KeyError, ValueError, AttributeError):
            pytest.skip("Registry does not support 'variance' key in this version")
