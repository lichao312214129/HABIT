"""Tests for automatic cluster-number selection helpers."""

import numpy as np

from habit.core.habitat_analysis.clustering.base_clustering import BaseClustering


class DummyClustering(BaseClustering):
    """Minimal clustering class used to test selection logic only."""

    def fit(self, X: np.ndarray) -> "DummyClustering":
        self.labels_ = np.zeros(X.shape[0], dtype=int)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0], dtype=int)


def test_multi_method_voting_preserves_underscored_method_names() -> None:
    """Method names containing underscores must remain intact during voting."""
    clusterer = DummyClustering()
    clusterer.cluster_range = [2, 3, 4]
    scores = {
        "silhouette": [0.1, 0.9, 0.2],
        "calinski_harabasz": [1.0, 8.0, 2.0],
        "davies_bouldin": [4.0, 2.0, 3.0],
    }

    best_n_clusters = clusterer.auto_select_best_n_clusters(
        scores,
        ["silhouette", "calinski_harabasz", "davies_bouldin"]
    )

    assert best_n_clusters == 3


def test_auto_select_best_index_is_separate_from_cluster_value() -> None:
    """Parameter-search algorithms need the best score index, not a cluster value."""
    clusterer = DummyClustering()
    scores = {
        "silhouette": [0.1, 0.4, 0.2],
        "calinski_harabasz": [1.0, 5.0, 3.0],
    }

    best_idx = clusterer.auto_select_best_index(
        scores,
        ["silhouette", "calinski_harabasz"]
    )

    assert best_idx == 1
