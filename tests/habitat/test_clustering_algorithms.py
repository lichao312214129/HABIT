"""
Unit tests for clustering algorithms, validation-method mapping, and optimal-k selection.

Covers K-Means, GMM, SLIC, and multi-method voting used by habitat analysis.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from habit.core.habitat_analysis.clustering.cluster_validation_methods import (
    get_all_clustering_algorithms,
    get_default_methods,
    get_method_description,
    get_optimization_direction,
    get_validation_methods,
    is_valid_method_for_algorithm,
)
from habit.core.habitat_analysis.clustering.cluster_search_parallel import (
    resolve_cluster_search_workers,
)
from habit.core.habitat_analysis.clustering.gmm_clustering import GMMClustering
from habit.core.habitat_analysis.clustering.kmeans_clustering import KMeansClustering
from habit.core.habitat_analysis.clustering.slic_clustering import SLICClustering
from habit.core.habitat_analysis.config_schemas import HabitatClusteringConfig, OneStepSettings


@pytest.fixture
def blob_data() -> Tuple[np.ndarray, np.ndarray, int]:
    """Synthetic Gaussian blobs with four well-separated clusters."""
    X: np.ndarray
    y: np.ndarray
    X, y = make_blobs(
        n_samples=400,
        centers=4,
        n_features=3,
        cluster_std=0.6,
        random_state=42,
    )
    true_k: int = 4
    return X, y, true_k


@pytest.fixture
def spatial_coords(blob_data: Tuple[np.ndarray, np.ndarray, int]) -> np.ndarray:
    """Fake voxel coordinates aligned with blob samples for SLIC optimal-k tests."""
    X, _, _ = blob_data
    n_samples: int = X.shape[0]
    side: int = int(np.ceil(n_samples ** (1.0 / 3.0)))
    grid_z, grid_y, grid_x = np.mgrid[0:side, 0:side, 0:side]
    coords: np.ndarray = np.column_stack(
        [grid_z.ravel(), grid_y.ravel(), grid_x.ravel()]
    )[:n_samples]
    return coords.astype(np.int32)


@pytest.fixture
def slic_volume_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Small 3D ROI with spatially varying features for SLIC fit/predict tests.

    Returns:
        X: Feature matrix (n_voxels, n_features).
        spatial_coords: Integer voxel coordinates (z, y, x).
        mask_array: Full 3D binary mask.
        n_clusters: Expected minimum meaningful segment count.
    """
    shape: Tuple[int, int, int] = (10, 10, 10)
    mask_array: np.ndarray = np.ones(shape, dtype=np.uint8)
    spatial_coords: np.ndarray = np.column_stack(np.where(mask_array > 0)).astype(np.int32)
    # Normalize coordinates so SLIC can split along spatial axes.
    X: np.ndarray = spatial_coords.astype(np.float32)
    X[:, 0] /= float(shape[0])
    X[:, 1] /= float(shape[1])
    X[:, 2] /= float(shape[2])
    return X, spatial_coords, mask_array, 4


@pytest.mark.unit
@pytest.mark.habitat
class TestValidationMethodsMapping:
    """Tests for algorithm ↔ validation-method registry."""

    def test_known_algorithms_registered(self) -> None:
        algos = get_all_clustering_algorithms()
        assert "kmeans" in algos
        assert "gmm" in algos
        assert "slic" in algos

    def test_kmeans_supports_inertia_not_bic(self) -> None:
        assert is_valid_method_for_algorithm("kmeans", "inertia") is True
        assert is_valid_method_for_algorithm("kmeans", "bic") is False

    def test_gmm_supports_bic_not_inertia(self) -> None:
        assert is_valid_method_for_algorithm("gmm", "bic") is True
        assert is_valid_method_for_algorithm("gmm", "inertia") is False

    def test_slic_supports_kneedle(self) -> None:
        assert is_valid_method_for_algorithm("slic", "kneedle") is True
        assert is_valid_method_for_algorithm("slic", "aic") is False

    def test_unknown_algorithm_falls_back_to_silhouette(self) -> None:
        info = get_validation_methods("unknown_algo")
        assert info["default"] == ["silhouette"]
        assert "silhouette" in info["methods"]

    def test_default_methods_per_algorithm(self) -> None:
        assert "silhouette" in get_default_methods("kmeans")
        assert get_default_methods("gmm") == ["aic"]

    def test_optimization_directions(self) -> None:
        assert get_optimization_direction("kmeans", "silhouette") == "maximize"
        assert get_optimization_direction("gmm", "bic") == "minimize"
        assert get_optimization_direction("kmeans", "elbow") == "elbow"

    def test_method_descriptions_non_empty(self) -> None:
        desc = get_method_description("gmm", "bic")
        assert isinstance(desc, str) and len(desc) > 0


@pytest.mark.unit
@pytest.mark.habitat
class TestOneStepSettingsSchema:
    """OneStep selection_method must accept GMM-specific metrics."""

    @pytest.mark.parametrize("method", ["aic", "bic", "gap"])
    def test_gmm_metrics_allowed(self, method: str) -> None:
        cfg = OneStepSettings(selection_method=method)
        assert cfg.selection_method == method


@pytest.mark.unit
@pytest.mark.habitat
class TestKMeansClustering:
    """Basic K-Means fit/predict and optimal-k search."""

    def test_fit_predict_shape(self, blob_data: Tuple[np.ndarray, np.ndarray, int]) -> None:
        X, _, true_k = blob_data
        model = KMeansClustering(n_clusters=true_k, random_state=42, n_init=10)
        model.fit(X)
        labels: np.ndarray = model.predict(X)
        assert labels.shape == (X.shape[0],)
        assert len(np.unique(labels)) == true_k

    @pytest.mark.parametrize("method", ["silhouette", "elbow", "kneedle"])
    def test_find_optimal_clusters_near_true_k(
        self,
        blob_data: Tuple[np.ndarray, np.ndarray, int],
        method: str,
    ) -> None:
        X, _, true_k = blob_data
        model = KMeansClustering(random_state=42, n_init=10)
        best_k, scores = model.find_optimal_clusters(
            X,
            min_clusters=2,
            max_clusters=8,
            methods=[method],
            show_progress=False,
        )
        assert best_k in model.cluster_range
        assert method in scores
        assert abs(best_k - true_k) <= 2


@pytest.mark.unit
@pytest.mark.habitat
class TestGMMClustering:
    """GMM fit/predict and AIC/BIC-based optimal-k selection."""

    def test_fit_predict_shape(self, blob_data: Tuple[np.ndarray, np.ndarray, int]) -> None:
        X, _, true_k = blob_data
        model = GMMClustering(n_clusters=true_k, random_state=42, n_init=5)
        model.fit(X)
        labels: np.ndarray = model.predict(X)
        assert labels.shape == (X.shape[0],)
        assert len(np.unique(labels)) >= 2

    def test_reproducible_with_random_state(
        self,
        blob_data: Tuple[np.ndarray, np.ndarray, int],
    ) -> None:
        X, _, true_k = blob_data
        model_a = GMMClustering(n_clusters=true_k, random_state=7, n_init=5)
        model_b = GMMClustering(n_clusters=true_k, random_state=7, n_init=5)
        model_a.fit(X)
        model_b.fit(X)
        np.testing.assert_array_equal(model_a.predict(X), model_b.predict(X))

    @pytest.mark.parametrize("method", ["bic", "aic", "silhouette"])
    def test_find_optimal_clusters_near_true_k(
        self,
        blob_data: Tuple[np.ndarray, np.ndarray, int],
        method: str,
    ) -> None:
        X, _, true_k = blob_data
        model = GMMClustering(random_state=42, n_init=5)
        best_k, scores = model.find_optimal_clusters(
            X,
            min_clusters=2,
            max_clusters=8,
            methods=[method],
            show_progress=False,
        )
        assert best_k in model.cluster_range
        assert method in scores
        assert abs(best_k - true_k) <= 2

    def test_bic_skipped_for_kmeans(
        self,
        blob_data: Tuple[np.ndarray, np.ndarray, int],
    ) -> None:
        X, _, _ = blob_data
        model = KMeansClustering(random_state=42, n_init=5)
        scores = model.calculate_bic_scores(X, list(range(2, 6)))
        assert scores is None


@pytest.mark.unit
@pytest.mark.habitat
class TestSLICClustering:
    """SLIC subject-specific clustering and spatial optimal-k search."""

    def test_fit_predict_with_mask_array(
        self,
        slic_volume_data: Tuple[np.ndarray, np.ndarray, np.ndarray, int],
    ) -> None:
        X, spatial_coords, mask_array, n_clusters = slic_volume_data
        model = SLICClustering(n_clusters=n_clusters, random_state=42, compactness=0.01)
        model.fit(X, spatial_coords=spatial_coords, mask_array=mask_array)
        labels: np.ndarray = model.predict(X)
        assert labels.shape == (X.shape[0],)
        assert len(np.unique(labels)) >= 2

    @pytest.mark.parametrize("method", ["silhouette", "kneedle"])
    def test_find_optimal_clusters_with_spatial_coords(
        self,
        blob_data: Tuple[np.ndarray, np.ndarray, int],
        spatial_coords: np.ndarray,
        method: str,
    ) -> None:
        X, _, true_k = blob_data
        model = SLICClustering(random_state=42, compactness=0.1, n_init=5)
        best_k, scores = model.find_optimal_clusters(
            X,
            min_clusters=2,
            max_clusters=8,
            methods=[method],
            show_progress=False,
            spatial_coords=spatial_coords,
        )
        assert best_k in model.cluster_range
        assert method in scores
        assert abs(best_k - true_k) <= 3


@pytest.mark.unit
@pytest.mark.habitat
class TestParallelClusterSearch:
    """Parallel k-search should match serial results for KMeans and GMM."""

    def test_resolve_cluster_search_workers_default(self) -> None:
        assert resolve_cluster_search_workers(None) == 2

    def test_resolve_cluster_search_workers_custom(self) -> None:
        assert resolve_cluster_search_workers(3) == 3

    def test_habitat_config_parallel_defaults(self) -> None:
        cfg = HabitatClusteringConfig()
        assert cfg.parallel_cluster_search is True
        assert cfg.cluster_search_workers is None

    @pytest.mark.parametrize("method", ["silhouette", "elbow"])
    def test_parallel_matches_serial_kmeans(
        self,
        blob_data: Tuple[np.ndarray, np.ndarray, int],
        method: str,
    ) -> None:
        X, _, _ = blob_data
        serial_model = KMeansClustering(random_state=42, n_init=10)
        best_serial, scores_serial = serial_model.find_optimal_clusters(
            X,
            min_clusters=2,
            max_clusters=8,
            methods=[method],
            show_progress=False,
            parallel_cluster_search=False,
        )

        parallel_model = KMeansClustering(random_state=42, n_init=10)
        best_parallel, scores_parallel = parallel_model.find_optimal_clusters(
            X,
            min_clusters=2,
            max_clusters=8,
            methods=[method],
            show_progress=False,
            parallel_cluster_search=True,
            cluster_search_workers=2,
        )

        assert best_serial == best_parallel
        for metric_name, serial_values in scores_serial.items():
            np.testing.assert_allclose(
                serial_values,
                scores_parallel[metric_name],
                rtol=1e-10,
                atol=1e-10,
            )

    @pytest.mark.parametrize("method", ["aic", "silhouette"])
    def test_parallel_matches_serial_gmm(
        self,
        blob_data: Tuple[np.ndarray, np.ndarray, int],
        method: str,
    ) -> None:
        X, _, _ = blob_data
        serial_model = GMMClustering(random_state=42, n_init=5)
        best_serial, scores_serial = serial_model.find_optimal_clusters(
            X,
            min_clusters=2,
            max_clusters=8,
            methods=[method],
            show_progress=False,
            parallel_cluster_search=False,
        )

        parallel_model = GMMClustering(random_state=42, n_init=5)
        best_parallel, scores_parallel = parallel_model.find_optimal_clusters(
            X,
            min_clusters=2,
            max_clusters=8,
            methods=[method],
            show_progress=False,
            parallel_cluster_search=True,
            cluster_search_workers=2,
        )

        assert best_serial == best_parallel
        for metric_name, serial_values in scores_serial.items():
            np.testing.assert_allclose(
                serial_values,
                scores_parallel[metric_name],
                rtol=1e-10,
                atol=1e-10,
            )

    def test_parallel_kmeans_with_max_iter_in_kwargs(
        self,
        blob_data: Tuple[np.ndarray, np.ndarray, int],
    ) -> None:
        """Regression: max_iter passed via **kwargs must not duplicate sklearn args."""
        X, _, _ = blob_data
        model = KMeansClustering(random_state=42, n_init=10, max_iter=250)
        best_k, scores = model.find_optimal_clusters(
            X,
            min_clusters=2,
            max_clusters=6,
            methods=["inertia"],
            show_progress=False,
            parallel_cluster_search=True,
            cluster_search_workers=2,
        )
        assert best_k in model.cluster_range
        assert "inertia" in scores

    def test_parallel_multi_method_voting_matches_serial(
        self,
        blob_data: Tuple[np.ndarray, np.ndarray, int],
    ) -> None:
        X, _, _ = blob_data
        methods = ["silhouette", "calinski_harabasz"]
        serial_model = KMeansClustering(random_state=42, n_init=10)
        best_serial, scores_serial = serial_model.find_optimal_clusters(
            X,
            min_clusters=2,
            max_clusters=8,
            methods=methods,
            show_progress=False,
            parallel_cluster_search=False,
        )

        parallel_model = KMeansClustering(random_state=42, n_init=10)
        best_parallel, scores_parallel = parallel_model.find_optimal_clusters(
            X,
            min_clusters=2,
            max_clusters=8,
            methods=methods,
            show_progress=False,
            parallel_cluster_search=True,
            cluster_search_workers=2,
        )

        assert best_serial == best_parallel
        for metric_name in methods:
            np.testing.assert_allclose(
                scores_serial[metric_name],
                scores_parallel[metric_name],
                rtol=1e-10,
                atol=1e-10,
            )


@pytest.mark.unit
@pytest.mark.habitat
class TestMultiMethodVoting:
    """Multi-metric voting in auto_select_best_n_clusters."""

    def test_voting_picks_consensus(
        self,
        blob_data: Tuple[np.ndarray, np.ndarray, int],
    ) -> None:
        X, _, true_k = blob_data
        model = KMeansClustering(random_state=42, n_init=10)
        best_k, scores = model.find_optimal_clusters(
            X,
            min_clusters=2,
            max_clusters=8,
            methods=["silhouette", "calinski_harabasz"],
            show_progress=False,
        )
        assert best_k in model.cluster_range
        assert "silhouette" in scores
        assert "calinski_harabasz" in scores
        assert abs(best_k - true_k) <= 3

    def test_auto_select_best_index_single_method(
        self,
        blob_data: Tuple[np.ndarray, np.ndarray, int],
    ) -> None:
        X, _, _ = blob_data
        model = KMeansClustering(random_state=42, n_init=10)
        model.find_optimal_clusters(
            X,
            min_clusters=2,
            max_clusters=6,
            methods=["silhouette"],
            show_progress=False,
        )
        best_idx = model.auto_select_best_index(model.scores, "silhouette")
        assert 0 <= best_idx < len(model.cluster_range)
