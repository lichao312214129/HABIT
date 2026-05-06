"""
Unit tests for habitat-level feature computation modules:
  - basic_features (volume, proportion, etc.)
  - ITH (intra-tumour heterogeneity) features
  - MSI (metabolic spatial index) features

Uses synthetic habitat maps represented as numpy arrays / pandas DataFrames.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest


def _make_habitat_map(n_voxels: int = 200, n_habitats: int = 3, seed: int = 0) -> np.ndarray:
    """Simulate an integer label array where each value is a habitat ID."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, n_habitats, size=n_voxels)


def _make_habitat_df(n_subjects: int = 10, n_features: int = 6, seed: int = 0) -> pd.DataFrame:
    """Simulate a per-subject feature DataFrame for habitat-level features."""
    rng = np.random.RandomState(seed)
    df = pd.DataFrame(
        rng.randn(n_subjects, n_features),
        columns=[f"hab_{i}_feat_{j}" for i in range(2) for j in range(n_features // 2)],
        index=[f"S{i:03d}" for i in range(n_subjects)],
    )
    return df


# ---------------------------------------------------------------------------
# BasicFeatures
# ---------------------------------------------------------------------------


class TestBasicFeatures:
    def test_import_succeeds(self) -> None:
        from habit.core.habitat_analysis.habitat_features import basic_features  # noqa: F401

    def test_compute_returns_dict_or_series(self) -> None:
        """basic_features computation on a synthetic habitat map should return some mapping."""
        try:
            from habit.core.habitat_analysis.habitat_features.basic_features import (
                compute_basic_features,
            )
        except ImportError:
            pytest.skip("compute_basic_features not importable")

        labels = _make_habitat_map()
        result = compute_basic_features(labels)
        assert isinstance(result, (dict, pd.Series))


# ---------------------------------------------------------------------------
# ITH features
# ---------------------------------------------------------------------------


class TestITHFeatures:
    def test_import_succeeds(self) -> None:
        from habit.core.habitat_analysis.habitat_features import ith_features  # noqa: F401

    def test_ith_score_is_numeric(self) -> None:
        try:
            from habit.core.habitat_analysis.habitat_features.ith_features import (
                compute_ith_score,
            )
        except ImportError:
            pytest.skip("compute_ith_score not importable")

        labels = _make_habitat_map()
        score = compute_ith_score(labels)
        assert isinstance(score, (int, float, np.floating))


# ---------------------------------------------------------------------------
# MSI features
# ---------------------------------------------------------------------------


class TestMSIFeatures:
    def test_import_succeeds(self) -> None:
        from habit.core.habitat_analysis.habitat_features import msi_features  # noqa: F401

    def test_msi_computation_returns_mapping(self) -> None:
        try:
            from habit.core.habitat_analysis.habitat_features.msi_features import (
                compute_msi_features,
            )
        except ImportError:
            pytest.skip("compute_msi_features not importable")

        labels = _make_habitat_map()
        result = compute_msi_features(labels)
        assert isinstance(result, (dict, pd.Series))


# ---------------------------------------------------------------------------
# HabitatAnalyzer (top-level entry point)
# ---------------------------------------------------------------------------


class TestHabitatAnalyzer:
    def test_import_succeeds(self) -> None:
        from habit.core.habitat_analysis.habitat_features.habitat_analyzer import (  # noqa: F401
            HabitatMapAnalyzer,
        )

    def test_instantiation(self, tmp_path: Path) -> None:
        from habit.core.habitat_analysis.habitat_features.habitat_analyzer import HabitatMapAnalyzer

        # out_dir is required: _setup_logging creates the directory and file handlers there.
        analyzer = HabitatMapAnalyzer(out_dir=str(tmp_path))
        assert analyzer is not None


# ---------------------------------------------------------------------------
# feature_preprocessing: VarianceFilter and CorrelationFilter
# ---------------------------------------------------------------------------


class TestFeaturePreprocessing:
    def test_variance_filter_removes_zero_variance(self) -> None:
        from habit.core.habitat_analysis.feature_preprocessing.variance_filter import (
            apply_variance_filter,
        )

        df = pd.DataFrame(
            {"const": [1.0] * 20, "vary": np.linspace(0, 1, 20)}
        )
        filtered = apply_variance_filter(df, threshold=0.0)
        assert "const" not in filtered.columns
        assert "vary" in filtered.columns

    def test_variance_filter_keeps_all_when_threshold_zero_and_no_const(self) -> None:
        from habit.core.habitat_analysis.feature_preprocessing.variance_filter import (
            apply_variance_filter,
        )

        df = pd.DataFrame(np.random.randn(20, 5), columns=list("ABCDE"))
        filtered = apply_variance_filter(df, threshold=0.0)
        assert filtered.shape[1] == 5

    def test_correlation_filter_removes_redundant_features(self) -> None:
        from habit.core.habitat_analysis.feature_preprocessing.correlation_filter import (
            apply_correlation_filter,
        )

        base = np.linspace(0, 1, 30)
        df = pd.DataFrame(
            {"A": base, "B": base * 1.001, "C": np.random.randn(30)}
        )
        filtered = apply_correlation_filter(df, threshold=0.99)
        # A and B are nearly perfectly correlated; one should be dropped
        assert filtered.shape[1] < 3
