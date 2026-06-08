# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text. Summary:
#
#   - Non-commercial use (academic, research, education, personal) is permitted
#     provided that copyright notices are retained and HABIT usage is
#     acknowledged in publications, reports, or documentation.
#   - Commercial use requires prior written consent from the copyright holder
#     (lichao19870617@163.com) and public acknowledgment of HABIT usage in
#     product documentation or user-facing materials.
#   - Unauthorized commercial use or removal of attribution is prohibited.
#
"""
Base clustering module for habitat analysis.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional, Union, Callable, Iterator
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import warnings
import os
import importlib
import pkgutil
import inspect
from kneed import KneeLocator

warnings.simplefilter('ignore')

from .cluster_validation_methods import (
    get_default_methods,
    get_method_description,
    get_optimization_direction,
)
from .cluster_search_parallel import (
    parallel_cluster_search_scores,
    resolve_cluster_search_workers,
)

# Registry for clustering algorithms
_CLUSTERING_REGISTRY = {}

def register_clustering(name: str):
    """
    Decorator for registering clustering algorithm classes
    
    Args:
        name (str): Name of the clustering algorithm
    """
    def decorator(cls):
        ClusteringAlgorithmFactory.register(name)(cls)
        return cls
    return decorator


class ClusteringAlgorithmFactory:
    """Factory class for creating clustering algorithm instances."""

    _registry = _CLUSTERING_REGISTRY

    @classmethod
    def register(cls, name: str):
        """
        Register a clustering algorithm class.

        Args:
            name: Name of the clustering algorithm.

        Returns:
            Decorator function.
        """
        def decorator(algorithm_class):
            cls._registry[name.lower()] = algorithm_class
            return algorithm_class
        return decorator

    @classmethod
    def create_algorithm(cls, name: str, **kwargs) -> 'BaseClustering':
        """
        Create a clustering algorithm instance by name.

        Args:
            name: Name of the clustering algorithm.
            **kwargs: Parameters to pass to the clustering algorithm constructor.

        Returns:
            BaseClustering: Instance of the clustering algorithm.
        """
        # Lazy discovery of clustering algorithms to avoid circular imports
        if not cls._registry:
            cls._discover_algorithms()

        # First try to find registered algorithms
        if name.lower() in cls._registry:
            return cls._registry[name.lower()](**kwargs)

        # If not found, try dynamic import
        try:
            # Try to import module with specified name
            module_name = f"{name}_clustering"
            module = importlib.import_module(f".{module_name}", package=__package__)

            # Find clustering algorithm class in the module
            for attr_name, attr_value in inspect.getmembers(module, inspect.isclass):
                if attr_name != 'BaseClustering' and 'BaseClustering' in [base.__name__ for base in attr_value.__mro__ if base.__name__ != 'object']:
                    # Auto-register found class
                    cls._registry[name.lower()] = attr_value
                    return attr_value(**kwargs)
        except (ImportError, ModuleNotFoundError):
            pass

        # If still not found, raise error
        raise ValueError(f"Unknown clustering algorithm: {name}, available algorithms: {list(cls._registry.keys())}")

    @classmethod
    def get_available_algorithms(cls) -> List[str]:
        """
        Get all available clustering algorithm names.

        Returns:
            List[str]: List of clustering algorithm names.
        """
        # Lazy discovery of clustering algorithms to avoid circular imports
        if not cls._registry:
            cls._discover_algorithms()

        return list(cls._registry.keys())

    @classmethod
    def _discover_algorithms(cls) -> None:
        """
        Automatically discover all clustering algorithms defined in the clustering directory.
        """
        # Use current file directory to avoid circular imports
        package_dir = os.path.dirname(os.path.abspath(__file__))

        # Iterate through all modules in the package
        for _, module_name, _ in pkgutil.iter_modules([package_dir]):
            if module_name.endswith('_clustering') and module_name != 'base_clustering':
                try:
                    # Dynamically import module
                    module = importlib.import_module(f".{module_name}", package=__package__)

                    # Find and register clustering algorithms defined in the module
                    for attr_name, attr_value in inspect.getmembers(module, inspect.isclass):
                        if attr_name != 'BaseClustering' and 'BaseClustering' in [base.__name__ for base in attr_value.__mro__ if base.__name__ != 'object']:
                            # Extract algorithm name from module name (e.g., extract kmeans from kmeans_clustering)
                            algo_name = module_name.replace('_clustering', '')
                            cls._registry[algo_name.lower()] = attr_value
                except ImportError:
                    pass


def get_clustering_algorithm(name: str, **kwargs) -> 'BaseClustering':
    """
    Get clustering algorithm class by name.

    Backward-compatible wrapper around ClusteringAlgorithmFactory.create_algorithm.
    """
    return ClusteringAlgorithmFactory.create_algorithm(name, **kwargs)

def get_available_clustering_algorithms() -> List[str]:
    """
    Get all available clustering algorithm names.

    Backward-compatible wrapper around ClusteringAlgorithmFactory.get_available_algorithms.
    """
    return ClusteringAlgorithmFactory.get_available_algorithms()

def discover_clustering_algorithms() -> None:
    """
    Automatically discover all clustering algorithms defined in the clustering directory.

    Backward-compatible wrapper around ClusteringAlgorithmFactory._discover_algorithms.
    """
    ClusteringAlgorithmFactory._discover_algorithms()


class BaseClustering(ABC):
    """
    Base class for clustering algorithms, defining methods that all clustering algorithms must implement
    
    Subclasses must implement the following methods:
    - fit: Train the clustering model based on input data
    - predict: Predict cluster labels for input data
    """
    
    def __init__(self, n_clusters: Optional[int] = None, random_state: int = 0):
        """
        Initialize clustering algorithm
        
        Args:
            n_clusters (Optional[int]): Number of clusters, can be None, indicating it needs to be determined by find_optimal_clusters
            random_state (int): Random seed to ensure reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = None
        self.labels_ = None
        self.cluster_range = None
        self.scores = {}
    
    @abstractmethod
    def fit(self, X: np.ndarray) -> 'BaseClustering':
        """
        Train clustering model
        
        Args:
            X (np.ndarray): Input data with shape (n_samples, n_features)
            
        Returns:
            BaseClustering: Trained model
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for input data
        
        Args:
            X (np.ndarray): Input data with shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Cluster labels with shape (n_samples,)
        """
        pass
    
    def _create_model_for_score(self, n_clusters: int) -> 'BaseClustering':
        """
        Create a temporary model for validation-score calculation.

        Args:
            n_clusters: Number of clusters used by the temporary model.

        Returns:
            BaseClustering: New unfitted model with the same main constructor parameters.
        """
        params: Dict[str, Any] = {
            "n_clusters": n_clusters,
            "random_state": self.random_state,
        }
        for attr_name in ("init", "n_init", "max_iter", "covariance_type"):
            if hasattr(self, attr_name):
                params[attr_name] = getattr(self, attr_name)
        params.update(getattr(self, "kwargs", {}))
        return self.__class__(**params)

    def _algorithm_name(self) -> str:
        """
        Normalize the registered clustering algorithm name for validation lookup.

        Returns:
            str: Algorithm key such as ``kmeans`` or ``gmm``.
        """
        algo_name = self.__class__.__name__.lower()
        if algo_name.endswith("clustering"):
            algo_name = algo_name[:-10]
        return algo_name

    def _cluster_search_model_params(self) -> Dict[str, Any]:
        """
        Serialize constructor kwargs used during cluster-count validation.

        Returns:
            Dict[str, Any]: Parameters shared by every candidate ``k`` fit.
        """
        params: Dict[str, Any] = {
            "random_state": self.random_state,
            "kwargs": dict(getattr(self, "kwargs", {})),
        }
        for attr_name in ("init", "n_init", "max_iter", "covariance_type"):
            if hasattr(self, attr_name):
                params[attr_name] = getattr(self, attr_name)
        return params

    def _supports_parallel_cluster_search(self) -> bool:
        """
        Whether this clusterer can use the parallel k-search backend.

        Returns:
            bool: True for KMeans and GMM group-level search.
        """
        return self._algorithm_name() in {"kmeans", "gmm"}

    def _begin_cluster_search_logging(
        self,
        *,
        min_clusters: int,
        max_clusters: int,
        methods: List[str],
        show_progress: bool,
        logger: Optional[logging.Logger],
    ) -> None:
        """Store logger context for per-k progress messages during model selection."""
        self._cluster_search_logger = logger
        self._cluster_search_show_progress = show_progress
        self._cluster_search_method = None
        if logger is None or not show_progress:
            return
        candidate_count = max_clusters - min_clusters + 1
        logger.info(
            "Searching optimal cluster count: evaluating k=%s..%s "
            "(%s candidate(s), methods=%s)",
            min_clusters,
            max_clusters,
            candidate_count,
            ", ".join(methods),
        )

    def _set_cluster_search_method(self, method_name: str) -> None:
        """Record the active validation method for per-k log lines."""
        self._cluster_search_method = method_name
        logger = getattr(self, "_cluster_search_logger", None)
        show_progress = getattr(self, "_cluster_search_show_progress", False)
        if logger is not None and show_progress:
            logger.info("Validation method: %s", method_name)

    def _iter_cluster_range(self, cluster_range: List[int]) -> Iterator[int]:
        """
        Yield cluster counts in ``cluster_range`` and emit try-k progress logs.

        Args:
            cluster_range: Cluster counts to evaluate.

        Yields:
            int: Next cluster count to fit.
        """
        total_steps = len(cluster_range)
        for step_index, n_clusters in enumerate(cluster_range, start=1):
            logger = getattr(self, "_cluster_search_logger", None)
            show_progress = getattr(self, "_cluster_search_show_progress", False)
            method_name = getattr(self, "_cluster_search_method", None)
            if logger is not None and show_progress:
                if method_name:
                    logger.info(
                        "Trying %s cluster(s) [%s/%s] (%s)",
                        n_clusters,
                        step_index,
                        total_steps,
                        method_name,
                    )
                else:
                    logger.info(
                        "Trying %s cluster(s) [%s/%s]",
                        n_clusters,
                        step_index,
                        total_steps,
                    )
            yield n_clusters

    def _finish_cluster_search_logging(
        self,
        best_n_clusters: int,
        show_progress: bool,
        logger: Optional[logging.Logger],
    ) -> None:
        """Log the end of cluster-count search."""
        if logger is not None and show_progress:
            logger.info(
                "Cluster search finished: selected k=%s",
                best_n_clusters,
            )
        self._cluster_search_logger = None
        self._cluster_search_show_progress = False
        self._cluster_search_method = None

    def calculate_inertia_scores(self, X: np.ndarray, cluster_range: List[int]) -> List[float]:
        """
        Calculate inertia (SSE) for different numbers of clusters (for K-Means)
        
        Args:
            X (np.ndarray): Input data with shape (n_samples, n_features)
            cluster_range (List[int]): Range of cluster numbers to evaluate
            
        Returns:
            List[float]: List of inertia values
        """
        try:
            from sklearn.cluster import KMeans
            if "kmeans" not in self.__class__.__name__.lower():
                warnings.warn(f"calculate_inertia_scores is primarily for KMeans algorithm, but was called on {self.__class__.__name__}")
            
            inertias = []
            for n_clusters in self._iter_cluster_range(cluster_range):
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    random_state=self.random_state,
                    init=getattr(self, 'init', 'k-means++'),
                    n_init=getattr(self, 'n_init', 10),
                    **getattr(self, 'kwargs', {})
                )
                kmeans.fit(X)
                inertias.append(kmeans.inertia_)
            
            return inertias
        except ImportError:
            raise ImportError("sklearn.cluster.KMeans is required for calculate_inertia_scores")
        except Exception as e:
            raise ValueError(f"Error calculating inertia scores: {str(e)}")

    def calculate_kneedle_scores(self, X: np.ndarray, cluster_range: List[int]) -> List[float]:
        """
        Calculate scores for Kneedle-based selection.

        Kneedle needs a monotonically changing curve. For KMeans, the inertia curve
        is the standard choice, so this method reuses inertia scores to build the
        curve that Kneedle will analyze later.

        Args:
            X (np.ndarray): Input data with shape (n_samples, n_features)
            cluster_range (List[int]): Range of cluster numbers to evaluate

        Returns:
            List[float]: List of inertia values used for Kneedle detection
        """
        return self.calculate_inertia_scores(X, cluster_range)

    def calculate_elbow_scores(self, X: np.ndarray, cluster_range: List[int]) -> List[float]:
        """
        Calculate scores for second-derivative elbow selection.

        The elbow method uses the same inertia curve as Kneedle; only the
        post-processing rule for picking the optimal k differs.

        Args:
            X (np.ndarray): Input data with shape (n_samples, n_features)
            cluster_range (List[int]): Range of cluster numbers to evaluate

        Returns:
            List[float]: List of inertia values used for elbow detection
        """
        return self.calculate_inertia_scores(X, cluster_range)

    def calculate_bic_scores(self, X: np.ndarray, cluster_range: List[int]) -> Optional[List[float]]:
        """
        Calculate BIC scores for different numbers of clusters (for GMM only)
        
        Args:
            X (np.ndarray): Input data with shape (n_samples, n_features)
            cluster_range (List[int]): Range of cluster numbers to evaluate
            
        Returns:
            Optional[List[float]]: List of BIC scores, or None if not applicable to this algorithm
        """
        # BIC is only applicable to GMM (probabilistic model with likelihood function)
        if "gmm" not in self.__class__.__name__.lower():
            warnings.warn(
                f"BIC is only applicable to GMM algorithm. "
                f"Skipping BIC calculation for {self.__class__.__name__}. "
                f"Consider using silhouette, calinski_harabasz, or davies_bouldin instead."
            )
            return None
        
        try:
            from sklearn.mixture import GaussianMixture
            
            bic_scores = []
            for n_clusters in self._iter_cluster_range(cluster_range):
                gmm = GaussianMixture(
                    n_components=n_clusters,
                    random_state=self.random_state,
                    covariance_type=getattr(self, 'covariance_type', 'full'),
                    n_init=getattr(self, 'n_init', 1),
                    max_iter=getattr(self, 'max_iter', 100),
                    **getattr(self, 'kwargs', {})
                )
                gmm.fit(X)
                bic_scores.append(gmm.bic(X))
            
            return bic_scores
        except ImportError:
            raise ImportError("sklearn.mixture.GaussianMixture is required for calculate_bic_scores")
        except Exception as e:
            raise ValueError(f"Error calculating BIC scores: {str(e)}")

    def calculate_aic_scores(self, X: np.ndarray, cluster_range: List[int]) -> Optional[List[float]]:
        """
        Calculate AIC scores for different numbers of clusters (for GMM only)
        
        Args:
            X (np.ndarray): Input data with shape (n_samples, n_features)
            cluster_range (List[int]): Range of cluster numbers to evaluate
            
        Returns:
            Optional[List[float]]: List of AIC scores, or None if not applicable to this algorithm
        """
        # AIC is only applicable to GMM (probabilistic model with likelihood function)
        if "gmm" not in self.__class__.__name__.lower():
            warnings.warn(
                f"AIC is only applicable to GMM algorithm. "
                f"Skipping AIC calculation for {self.__class__.__name__}. "
                f"Consider using silhouette, calinski_harabasz, or davies_bouldin instead."
            )
            return None
        
        try:
            from sklearn.mixture import GaussianMixture
            
            aic_scores = []
            for n_clusters in self._iter_cluster_range(cluster_range):
                gmm = GaussianMixture(
                    n_components=n_clusters,
                    random_state=self.random_state,
                    covariance_type=getattr(self, 'covariance_type', 'full'),
                    n_init=getattr(self, 'n_init', 1),
                    max_iter=getattr(self, 'max_iter', 100),
                    **getattr(self, 'kwargs', {})
                )
                gmm.fit(X)
                aic_scores.append(gmm.aic(X))
            
            return aic_scores
        except ImportError:
            raise ImportError("sklearn.mixture.GaussianMixture is required for calculate_aic_scores")
        except Exception as e:
            raise ValueError(f"Error calculating AIC scores: {str(e)}")
    
    def calculate_silhouette_scores(self, X: np.ndarray, cluster_range: List[int]) -> List[float]:
        """
        Calculate silhouette scores for different numbers of clusters
        
        Args:
            X (np.ndarray): Input data
            cluster_range (List[int]): Range of cluster numbers
            
        Returns:
            List[float]: List of silhouette scores
        """
        scores = []
        for n_clusters in self._iter_cluster_range(cluster_range):
            # Reuse algorithm-specific constructor parameters so validation and final fitting match.
            temp_model = self._create_model_for_score(n_clusters)
            temp_model.fit(X)
            labels = temp_model.labels_
            
            # Calculate silhouette score
            if len(np.unique(labels)) > 1:  # Need at least two clusters
                score = silhouette_score(X, labels)
            else:
                score = 0
            scores.append(score)
        
        return scores
    
    def calculate_calinski_harabasz_scores(self, X: np.ndarray, cluster_range: List[int]) -> List[float]:
        """
        Calculate Calinski-Harabasz index for different numbers of clusters
        
        Args:
            X (np.ndarray): Input data
            cluster_range (List[int]): Range of cluster numbers
            
        Returns:
            List[float]: List of Calinski-Harabasz indices
        """
        scores = []
        for n_clusters in self._iter_cluster_range(cluster_range):
            # Reuse algorithm-specific constructor parameters so validation and final fitting match.
            temp_model = self._create_model_for_score(n_clusters)
            temp_model.fit(X)
            labels = temp_model.labels_
            
            # Calculate Calinski-Harabasz index
            if len(np.unique(labels)) > 1:  # Need at least two clusters
                score = calinski_harabasz_score(X, labels)
            else:
                score = 0
            scores.append(score)
        
        return scores
    
    def calculate_davies_bouldin_scores(self, X: np.ndarray, cluster_range: List[int]) -> List[float]:
        """
        Calculate Davies-Bouldin index for different numbers of clusters
        
        Args:
            X (np.ndarray): Input data
            cluster_range (List[int]): Range of cluster numbers
        """
        scores = []
        for n_clusters in self._iter_cluster_range(cluster_range):
            # Reuse algorithm-specific constructor parameters so validation and final fitting match.
            temp_model = self._create_model_for_score(n_clusters)
            temp_model.fit(X)
            labels = temp_model.labels_
            
            # Calculate Davies-Bouldin index
            if len(np.unique(labels)) > 1:  # Need at least two clusters
                score = davies_bouldin_score(X, labels)
            else:
                score = 0
            scores.append(score)
        
        return scores
    
    def calculate_gap_scores(self, X: np.ndarray, cluster_range: List[int]) -> List[float]:
        """
        Calculate Gap statistic for different numbers of clusters
        
        Args:
            X (np.ndarray): Input data
            cluster_range (List[int]): Range of cluster numbers
        """
        scores = []
        for n_clusters in self._iter_cluster_range(cluster_range):
            # Reuse algorithm-specific constructor parameters so validation and final fitting match.
            temp_model = self._create_model_for_score(n_clusters)
            temp_model.fit(X)
            labels = temp_model.labels_
            
            # Calculate Gap statistic
            if len(np.unique(labels)) > 1:  # Need at least two clusters
                score = gap_score(X, labels)
            else:
                score = 0
            scores.append(score)
        
        return scores
    
    def find_optimal_clusters(self, X: np.ndarray, min_clusters: int = 2, 
                              max_clusters: int = 10, 
                              methods: Optional[Union[List[str], str]] = None, 
                              show_progress: bool = True,
                              logger: Optional[logging.Logger] = None,
                              parallel_cluster_search: bool = False,
                              cluster_search_workers: Optional[int] = None) -> Tuple[int, Dict[str, List[float]]]:
        """
        Find the optimal number of clusters
        
        Args:
            X (np.ndarray): Input data with shape (n_samples, n_features)
            min_clusters (int): Minimum number of clusters
            max_clusters (int): Maximum number of clusters
            methods (Optional[Union[List[str], str]]): List of methods to determine the optimal number of clusters. If None, default methods are used
            show_progress (bool): Whether to display progress
            logger (Optional[logging.Logger]): Logger for search progress; falls back to print when None
            parallel_cluster_search (bool): When True and the algorithm supports it,
                evaluate each candidate ``k`` in parallel worker processes.
            cluster_search_workers (Optional[int]): Worker count for parallel search.
                ``None`` uses ``2``.
            
        Returns:
            Tuple[int, Dict[str, List[float]]]: 
                - int: Optimal number of clusters
                - Dict[str, List[float]]: Dictionary of scores for different numbers of clusters
        """
        # Basic validation
        if min_clusters <= 0:
            raise ValueError("min_clusters must be positive")
        if max_clusters <= min_clusters:
            raise ValueError("max_clusters must be greater than min_clusters")
        if X.shape[0] < max_clusters:
            raise ValueError(f"Number of samples ({X.shape[0]}) must be greater than max_clusters ({max_clusters})")
        
        # If methods is None, use default methods from validation module
        if methods is None:
            # Get the clustering algorithm name by checking the class name
            algo_name = self.__class__.__name__.lower()
            # Remove 'clustering' suffix if present
            if algo_name.endswith('clustering'):
                algo_name = algo_name[:-10]
            # Get default methods for this algorithm
            methods = get_default_methods(algo_name)
            if show_progress:
                message = f"Using default validation methods for {algo_name}: {methods}"
                if logger is not None:
                    logger.info(message)
                else:
                    print(message)
        
        # Check and calculate each validation method
        if isinstance(methods, str):
            methods = [methods]

        self.cluster_range = list(range(min_clusters, max_clusters + 1))
        self._begin_cluster_search_logging(
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            methods=list(methods),
            show_progress=show_progress,
            logger=logger,
        )
        
        # Calculate different scores
        self.scores: Dict[str, List[float]] = {}
        valid_methods: List[str] = []

        use_parallel_search = (
            parallel_cluster_search
            and self._supports_parallel_cluster_search()
            and len(self.cluster_range) > 1
        )
        if use_parallel_search:
            n_workers = resolve_cluster_search_workers(cluster_search_workers)
            n_workers = min(n_workers, len(self.cluster_range))
            self.scores, valid_methods = parallel_cluster_search_scores(
                X,
                self.cluster_range,
                self._algorithm_name(),
                self._cluster_search_model_params(),
                methods,
                n_workers=n_workers,
                show_progress=show_progress,
                log=logger,
            )
            for method in valid_methods:
                if show_progress:
                    message = f"{method.capitalize()} calculation completed (parallel)!"
                    if logger is not None:
                        logger.info(message)
                    else:
                        print(message)
            methods = valid_methods
        else:
            for method in methods:
                if hasattr(self, f'calculate_{method}_scores'):
                    self._set_cluster_search_method(method)

                    # Call the specific calculation method
                    # This dynamically gets the method named "calculate_{method}_scores" from the class
                    # For example, if method is "silhouette", it gets "calculate_silhouette_scores" method
                    calculation_method = getattr(self, f'calculate_{method}_scores')
                    scores = calculation_method(X, self.cluster_range)

                    # Skip if method returns None (e.g., AIC/BIC for non-GMM algorithms)
                    if scores is None:
                        if show_progress:
                            message = f"{method.capitalize()} skipped (not applicable to this algorithm)"
                            if logger is not None:
                                logger.info(message)
                            else:
                                print(message)
                        continue

                    self.scores[method] = scores
                    valid_methods.append(method)

                    if show_progress:
                        message = f"{method.capitalize()} calculation completed!"
                        if logger is not None:
                            logger.info(message)
                        else:
                            print(message)

            methods = valid_methods

        # Check if any valid methods remain
        if len(methods) == 0:
            self._finish_cluster_search_logging(min_clusters, show_progress, logger)
            raise ValueError(
                "No valid validation methods available for this clustering algorithm. "
                f"Original methods requested: {methods}. "
                "Please use appropriate methods for your algorithm."
            )
        
        best_n_clusters = self.auto_select_best_n_clusters(self.scores, methods)

        # Sanity-check: auto-select must return a value from the evaluated range.
        # This prevents subtle index-vs-value bugs from silently propagating.
        if best_n_clusters not in self.cluster_range:
            self._finish_cluster_search_logging(best_n_clusters, show_progress, logger)
            raise ValueError(
                "Selected best_n_clusters is not in cluster_range. "
                f"best_n_clusters={best_n_clusters}, cluster_range={self.cluster_range}"
            )
        best_idx = self.cluster_range.index(best_n_clusters)
        
        # Set the best number of clusters
        self.n_clusters = best_n_clusters
        
        if show_progress:
            message = (
                "Automatically selected best number of clusters: "
                f"{best_n_clusters} (index={best_idx})"
            )
            if logger is not None:
                logger.info(message)
            else:
                print(message)

        self._finish_cluster_search_logging(best_n_clusters, show_progress, logger)
        
        return best_n_clusters, self.scores 
    
    @staticmethod
    def _find_best_n_clusters_for_elbow_method(scores: List[float]) -> int:
        """
        Find best cluster number using elbow method.
        
        Args:
            scores: List of scores for different cluster numbers
            
        Returns:
            int: Index of the best cluster number (0-based)
        """
        deltas = np.diff(scores)
        deltas2 = np.diff(deltas)
        best_idx = np.argmax(deltas2) + 1
        if best_idx >= len(scores) - 1:
            best_idx = len(scores) - 2  # Choose the second-to-last point
        return best_idx

    @staticmethod
    def _find_best_n_clusters_for_kneedle_method(scores: List[float]) -> int:
        """
        Find best cluster index using the Kneedle method.

        This implementation assumes a monotonically decreasing curve (e.g., inertia).
        It relies on the kneed package to detect the knee of a convex, decreasing curve.

        Args:
            scores: List of scores for different cluster numbers

        Returns:
            int: Index of the best cluster number (0-based)
        """
        scores_array = np.asarray(scores, dtype=float)
        if scores_array.size == 0:
            return 0
        if scores_array.size < 3:
            # With too few points, fall back to the minimum score index.
            return int(np.argmin(scores_array))

        x_values = np.arange(scores_array.size, dtype=float)

        # For inertia curves in KMeans, the shape is usually convex and decreasing.
        knee_locator = KneeLocator(
            x_values,
            scores_array,
            curve="convex",
            direction="decreasing"
        )
        knee_index = knee_locator.knee

        if knee_index is None:
            # If no knee is detected, return the middle point for stability.
            return int(scores_array.size // 2)

        best_idx = int(knee_index)
        # Avoid selecting endpoints, which are usually not meaningful elbows.
        if best_idx <= 0:
            best_idx = 1
        if best_idx >= scores_array.size - 1:
            best_idx = scores_array.size - 2
        return best_idx

    def _select_best_n_clusters_for_single_method(self, scores: List[float], method: str) -> int:
        """
        Select the best cluster index for a single validation method.
        
        Args:
            scores: List of scores for each cluster number in cluster_range order
            method: Validation method name used to decide optimization direction
            
        Returns:
            int: Index into self.cluster_range corresponding to the best cluster number
        """
        # Get optimization direction for the method
        algo_name = self.__class__.__name__.lower()
        if algo_name.endswith('clustering'):
            algo_name = algo_name[:-10]
        optimization = get_optimization_direction(algo_name, method)

        # Select best cluster index based on optimization direction
        if optimization == 'maximize':
            best_idx = np.argmax(scores)
        elif optimization == 'minimize':
            best_idx = np.argmin(scores)
        elif optimization in ('inertia', 'kneedle'):
            best_idx = self._find_best_n_clusters_for_kneedle_method(scores)
        elif optimization == 'elbow':
            best_idx = self._find_best_n_clusters_for_elbow_method(scores)
        else:
            # Default to maximum value
            best_idx = np.argmax(scores)
        
        return best_idx

    def _select_best_index_by_methods(
        self,
        scores_dict: Dict[str, List[float]],
        methods: Union[List[str], str]
    ) -> int:
        """
        Select the best score index from one or more validation methods.

        Args:
            scores_dict: Mapping from validation method name to score sequence.
            methods: A method name or a list of method names. Lists are treated as
                independent votes and are never split by underscores, because method
                names such as calinski_harabasz contain underscores.

        Returns:
            int: Index into the evaluated parameter range.
        """
        if isinstance(methods, str):
            methods = [methods]
        if not methods:
            raise ValueError("At least one scoring method is required")

        missing_methods = [method for method in methods if method not in scores_dict]
        if missing_methods:
            raise ValueError(
                "Unknown scoring method(s): "
                f"{', '.join(missing_methods)}"
            )

        if len(methods) == 1:
            method = methods[0]
            return self._select_best_n_clusters_for_single_method(
                scores_dict[method],
                method
            )

        votes: Dict[int, int] = {}
        for method in methods:
            best_idx = self._select_best_n_clusters_for_single_method(
                scores_dict[method],
                method
            )
            votes[best_idx] = votes.get(best_idx, 0) + 1

        max_votes = max(votes.values())
        candidates = [idx for idx, count in votes.items() if count == max_votes]
        return min(candidates)

    def auto_select_best_index(
        self,
        scores_dict: Dict[str, List[float]],
        methods: Union[List[str], str] = 'silhouette'
    ) -> int:
        """
        Automatically select the best index in the evaluated range.

        Args:
            scores_dict: Mapping from validation method name to score sequence.
            methods: Validation method name or list of names.

        Returns:
            int: Index into the evaluated range.
        """
        return self._select_best_index_by_methods(scores_dict, methods)

    def auto_select_best_n_clusters(
        self,
        scores_dict: Dict[str, List[float]],
        methods: Union[List[str], str] = 'silhouette'
    ) -> int:
        """
        Automatically select optimal number of clusters based on scores
        
        Args:
            scores_dict (Dict[str, List[float]]): Dictionary of scores, keys are method names, values are score lists
            methods (Union[List[str], str]): Method or methods to use. A list uses voting across methods.
            
        Returns:
            int: Optimal number of clusters
        """
        if self.cluster_range is None:
            raise ValueError("cluster_range must be set before selecting n_clusters")

        best_idx = self._select_best_index_by_methods(scores_dict, methods)
        return self.cluster_range[best_idx]
