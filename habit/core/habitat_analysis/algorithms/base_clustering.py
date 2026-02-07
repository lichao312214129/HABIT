"""
Base clustering module for habitat analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional, Union, Callable
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import warnings
import os
import importlib
import pkgutil
import inspect
from kneed import KneeLocator

warnings.simplefilter('ignore')

from .cluster_validation_methods import get_default_methods, get_method_description, get_optimization_direction

# Registry for clustering algorithms
_CLUSTERING_REGISTRY = {}

def register_clustering(name: str):
    """
    Decorator for registering clustering algorithm classes
    
    Args:
        name (str): Name of the clustering algorithm
    """
    def decorator(cls):
        _CLUSTERING_REGISTRY[name.lower()] = cls
        return cls
    return decorator

def get_clustering_algorithm(name: str, **kwargs) -> 'BaseClustering':
    """
    Get clustering algorithm class by name
    
    Args:
        name (str): Name of the clustering algorithm
        **kwargs: Parameters to pass to the clustering algorithm constructor
    
    Returns:
        BaseClustering: Instance of the clustering algorithm
    """
    # Lazy discovery of clustering algorithms to avoid circular imports
    if not _CLUSTERING_REGISTRY:
        discover_clustering_algorithms()
        
    # First try to find registered algorithms
    if name.lower() in _CLUSTERING_REGISTRY:
        return _CLUSTERING_REGISTRY[name.lower()](**kwargs)
    
    # If not found, try dynamic import
    try:
        # Try to import module with specified name
        module_name = f"{name}_clustering"
        module = importlib.import_module(f".{module_name}", package=__package__)
        
        # Find clustering algorithm class in the module
        for attr_name, attr_value in inspect.getmembers(module, inspect.isclass):
            if attr_name != 'BaseClustering' and 'BaseClustering' in [base.__name__ for base in attr_value.__mro__ if base.__name__ != 'object']:
                # Auto-register found class
                _CLUSTERING_REGISTRY[name.lower()] = attr_value
                return attr_value(**kwargs)
    except (ImportError, ModuleNotFoundError):
        pass
    
    # If still not found, raise error
    raise ValueError(f"Unknown clustering algorithm: {name}, available algorithms: {list(_CLUSTERING_REGISTRY.keys())}")

def get_available_clustering_algorithms() -> List[str]:
    """
    Get all available clustering algorithm names
    
    Returns:
        List[str]: List of clustering algorithm names
    """
    # Lazy discovery of clustering algorithms to avoid circular imports
    if not _CLUSTERING_REGISTRY:
        discover_clustering_algorithms()
        
    return list(_CLUSTERING_REGISTRY.keys())

def discover_clustering_algorithms() -> None:
    """
    Automatically discover all clustering algorithms defined in the clustering directory
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
                        _CLUSTERING_REGISTRY[algo_name.lower()] = attr_value
            except ImportError:
                pass


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
            # 检查是否是KMeans或其子类
            if "kmeans" not in self.__class__.__name__.lower():
                warnings.warn(f"calculate_inertia_scores is primarily for KMeans algorithm, but was called on {self.__class__.__name__}")
            
            inertias = []
            for n_clusters in cluster_range:
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    random_state=self.random_state,
                    # 尝试获取KMeans特有的参数
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
            for n_clusters in cluster_range:
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
            for n_clusters in cluster_range:
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
        for n_clusters in cluster_range:
            # Create temporary model
            temp_model = self.__class__(n_clusters=n_clusters, random_state=self.random_state)
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
        for n_clusters in cluster_range:
            # Create temporary model
            temp_model = self.__class__(n_clusters=n_clusters, random_state=self.random_state)
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
        for n_clusters in cluster_range:
            # Create temporary model
            temp_model = self.__class__(n_clusters=n_clusters, random_state=self.random_state)
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
        for n_clusters in cluster_range:
            # Create temporary model
            temp_model = self.__class__(n_clusters=n_clusters, random_state=self.random_state)
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
                              show_progress: bool = True) -> Tuple[int, Dict[str, List[float]]]:
        """
        Find the optimal number of clusters
        
        Args:
            X (np.ndarray): Input data with shape (n_samples, n_features)
            min_clusters (int): Minimum number of clusters
            max_clusters (int): Maximum number of clusters
            methods (Optional[Union[List[str], str]]): List of methods to determine the optimal number of clusters. If None, default methods are used
            show_progress (bool): Whether to display progress
            
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
                print(f"Using default validation methods for {algo_name}: {methods}")
        
        # Save cluster range
        self.cluster_range = list(range(min_clusters, max_clusters + 1))
        
        # Calculate different scores
        self.scores: Dict[str, List[float]] = {}
        
        if show_progress:
            print("Starting to calculate evaluation metrics for different numbers of clusters...")
            total = max_clusters - min_clusters + 1
        
        # Check and calculate each validation method
        if isinstance(methods, str):
            methods = [methods]
        
        # Track which methods were actually calculated
        valid_methods = []
            
        for method in methods:
            if hasattr(self, f'calculate_{method}_scores'):
                if show_progress:
                    print(f"Calculating {method}...")
                    
                # Call the specific calculation method
                # This dynamically gets the method named "calculate_{method}_scores" from the class
                # For example, if method is "silhouette", it gets "calculate_silhouette_scores" method
                calculation_method = getattr(self, f'calculate_{method}_scores')
                scores = calculation_method(X, self.cluster_range)
                
                # Skip if method returns None (e.g., AIC/BIC for non-GMM algorithms)
                if scores is None:
                    if show_progress:
                        print(f"{method.capitalize()} skipped (not applicable to this algorithm)")
                    continue
                
                self.scores[method] = scores
                valid_methods.append(method)
                
                if show_progress:
                    print(f"{method.capitalize()} calculation completed!")
        
        # Update methods list to only include valid ones
        methods = valid_methods
        
        # Check if any valid methods remain
        if len(methods) == 0:
            raise ValueError(
                "No valid validation methods available for this clustering algorithm. "
                f"Original methods requested: {methods}. "
                "Please use appropriate methods for your algorithm."
            )
        
        # Automatically select the best number of clusters
        if len(methods) == 1:
            best_method = methods[0]
        else:
            # Use combined method
            best_method = '_'.join(methods)
        
        best_n_clusters = self.auto_select_best_n_clusters(self.scores, best_method)

        # Sanity-check: auto-select must return a value from the evaluated range.
        # This prevents subtle index-vs-value bugs from silently propagating.
        if best_n_clusters not in self.cluster_range:
            raise ValueError(
                "Selected best_n_clusters is not in cluster_range. "
                f"best_n_clusters={best_n_clusters}, cluster_range={self.cluster_range}"
            )
        best_idx = self.cluster_range.index(best_n_clusters)
        
        # Set the best number of clusters
        self.n_clusters = best_n_clusters
        
        if show_progress:
            print(
                "Automatically selected best number of clusters: "
                f"{best_n_clusters} (index={best_idx})"
            )
        
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
        elif optimization == 'inertia':
            # Inertia uses Kneedle to detect the elbow on a decreasing curve.
            best_idx = self._find_best_n_clusters_for_kneedle_method(scores)
        elif optimization == 'elbow':
            # Backward compatibility: treat elbow as Kneedle.
            best_idx = self._find_best_n_clusters_for_kneedle_method(scores)
        elif optimization == 'kneedle':
            best_idx = self._find_best_n_clusters_for_kneedle_method(scores)
        else:
            # Default to maximum value
            best_idx = np.argmax(scores)
        
        return best_idx
    
    def auto_select_best_n_clusters(self, scores_dict: Dict[str, List[float]], method: str = 'silhouette') -> int:
        """
        Automatically select optimal number of clusters based on scores
        
        Args:
            scores_dict (Dict[str, List[float]]): Dictionary of scores, keys are method names, values are score lists
            method (str): Method to use, options include 'silhouette', 'calinski_harabasz', 'elbow', etc.
                    If it's a combination of methods, use '_' to connect, e.g., 'silhouette_calinski_harabasz'
            
        Returns:
            int: Optimal number of clusters
        """
        if method not in scores_dict and '_' not in method:
            raise ValueError(f"Unknown scoring method: {method}")
        
        # If it's a single method
        if method in scores_dict:
            scores = scores_dict[method]
            best_idx = self._select_best_n_clusters_for_single_method(scores, method)
            best_n_clusters = self.cluster_range[best_idx]
        # If it's a combination of methods, use voting system
        else:
            methods = method.split('_')
            votes = {}  # Dictionary to store votes: {cluster_index: vote_count}
            
            # Each method votes for its best cluster number
            for m in methods:
                if m not in scores_dict:
                    continue
                
                scores = scores_dict[m]
                # Select best cluster index for this method and map to cluster number
                best_idx = self._select_best_n_clusters_for_single_method(scores, m)
                best_n_clusters = self.cluster_range[best_idx]
                
                # Count the vote
                if best_n_clusters not in votes:
                    votes[best_n_clusters] = 0
                votes[best_n_clusters] += 1
            
            if len(votes) == 0:
                raise ValueError("No valid methods found in the combination")
            
            # Find the cluster number with the most votes
            # If there's a tie, choose the one with the smallest cluster number
            max_votes = max(votes.values())
            candidates = [idx for idx, count in votes.items() if count == max_votes]
            best_n_clusters = min(candidates)  # In case of tie, choose the smallest cluster number
        
        # Add min_clusters offset
        return best_n_clusters 
