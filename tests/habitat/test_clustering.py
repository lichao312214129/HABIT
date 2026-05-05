"""
Unit tests for clustering algorithms in habitat_analysis.

Uses synthetic numpy arrays so no image I/O is required.
sklearn-based algorithms (KMeans, GMM, etc.) are tested with small 2D data.
Algorithms that require optional packages (SLIC) are guarded with importorskip.
"""

from __future__ import annotations

import numpy as np
import pytest

from habit.core.habitat_analysis.clustering.base_clustering import BaseClustering


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_voxel_data(n: int = 100, n_features: int = 3, seed: int = 0) -> np.ndarray:
    """Return a (n, n_features) float array simulating voxel feature vectors."""
    rng = np.random.RandomState(seed)
    return rng.randn(n, n_features).astype(np.float32)


# ---------------------------------------------------------------------------
# BaseClustering interface
# ---------------------------------------------------------------------------


class TestBaseClusteringContract:
    def test_base_clustering_is_abstract(self) -> None:
        """BaseClustering must expose abstract methods to enforce subclass API."""
        abstract = getattr(BaseClustering, "__abstractmethods__", set())
        assert len(abstract) > 0, "BaseClustering should have abstract methods"

    def test_base_clustering_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            BaseClustering()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# KMeans clustering
# ---------------------------------------------------------------------------


class TestKMeansClustering:
    def _make_step(self, n_clusters: int = 3):
        from habit.core.habitat_analysis.clustering.kmeans_clustering import KMeansClustering

        return KMeansClustering(params={"n_clusters": n_clusters, "random_state": 0})

    def test_instantiation(self) -> None:
        step = self._make_step()
        assert step is not None

    def test_fit_returns_self(self) -> None:
        step = self._make_step()
        X = _make_voxel_data(60)
        result = step.fit(X)
        assert result is step

    def test_predict_returns_integer_labels(self) -> None:
        step = self._make_step(n_clusters=3)
        X = _make_voxel_data(60)
        step.fit(X)
        labels = step.predict(X)
        assert labels.shape == (60,)
        assert set(labels).issubset({0, 1, 2})

    def test_fit_predict_consistent(self) -> None:
        step = self._make_step(n_clusters=3)
        X = _make_voxel_data(60)
        labels_fit = step.fit(X).predict(X)
        labels_fp = step.fit_predict(X) if hasattr(step, "fit_predict") else step.fit(X).predict(X)
        np.testing.assert_array_equal(labels_fit, labels_fp)

    def test_n_clusters_respected(self) -> None:
        k = 4
        step = self._make_step(n_clusters=k)
        X = _make_voxel_data(80)
        step.fit(X)
        labels = step.predict(X)
        assert len(set(labels)) <= k


# ---------------------------------------------------------------------------
# GMM clustering
# ---------------------------------------------------------------------------


class TestGMMClustering:
    def _make_step(self, n_components: int = 3):
        from habit.core.habitat_analysis.clustering.gmm_clustering import GMMClustering

        return GMMClustering(params={"n_components": n_components, "random_state": 0})

    def test_instantiation(self) -> None:
        step = self._make_step()
        assert step is not None

    def test_fit_predict_labels(self) -> None:
        step = self._make_step(n_components=3)
        X = _make_voxel_data(90)
        step.fit(X)
        labels = step.predict(X)
        assert labels.shape == (90,)
        assert len(set(labels)) <= 3


# ---------------------------------------------------------------------------
# Hierarchical clustering
# ---------------------------------------------------------------------------


class TestHierarchicalClustering:
    def _make_step(self, n_clusters: int = 3):
        from habit.core.habitat_analysis.clustering.hierarchical_clustering import (
            HierarchicalClustering,
        )

        return HierarchicalClustering(params={"n_clusters": n_clusters})

    def test_instantiation(self) -> None:
        step = self._make_step()
        assert step is not None

    def test_fit_predict(self) -> None:
        step = self._make_step(n_clusters=3)
        X = _make_voxel_data(60)
        step.fit(X)
        labels = step.predict(X)
        assert labels.shape == (60,)


# ---------------------------------------------------------------------------
# Spectral clustering
# ---------------------------------------------------------------------------


class TestSpectralClustering:
    def _make_step(self, n_clusters: int = 3):
        from habit.core.habitat_analysis.clustering.spectral_clustering import (
            SpectralClustering,
        )

        return SpectralClustering(params={"n_clusters": n_clusters, "random_state": 0})

    def test_instantiation(self) -> None:
        step = self._make_step()
        assert step is not None


# ---------------------------------------------------------------------------
# Cluster validation methods
# ---------------------------------------------------------------------------


class TestClusterValidationMethods:
    def test_silhouette_score_available(self) -> None:
        from habit.core.habitat_analysis.clustering.cluster_validation_methods import (
            compute_silhouette_score,
        )

        X = _make_voxel_data(60)
        labels = np.array([0, 1, 2] * 20)
        score = compute_silhouette_score(X, labels)
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0
