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

    def calculate_bic_scores(self, X: np.ndarray, cluster_range: List[int]) -> List[float]:
        """
        Calculate BIC scores for different numbers of clusters (for GMM)
        
        Args:
            X (np.ndarray): Input data with shape (n_samples, n_features)
            cluster_range (List[int]): Range of cluster numbers to evaluate
            
        Returns:
            List[float]: List of BIC scores
        """
        try:
            from sklearn.mixture import GaussianMixture
            # 检查是否是GMM或其子类
            if "gmm" not in self.__class__.__name__.lower():
                warnings.warn(f"calculate_bic_scores is primarily for GMM algorithm, but was called on {self.__class__.__name__}")
            
            bic_scores = []
            for n_clusters in cluster_range:
                gmm = GaussianMixture(
                    n_components=n_clusters,
                    random_state=self.random_state,
                    # 尝试获取GMM特有的参数
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

    def calculate_aic_scores(self, X: np.ndarray, cluster_range: List[int]) -> List[float]:
        """
        Calculate AIC scores for different numbers of clusters (for GMM)
        
        Args:
            X (np.ndarray): Input data with shape (n_samples, n_features)
            cluster_range (List[int]): Range of cluster numbers to evaluate
            
        Returns:
            List[float]: List of AIC scores
        """
        try:
            from sklearn.mixture import GaussianMixture
            # 检查是否是GMM或其子类
            if "gmm" not in self.__class__.__name__.lower():
                warnings.warn(f"calculate_aic_scores is primarily for GMM algorithm, but was called on {self.__class__.__name__}")
            
            aic_scores = []
            for n_clusters in cluster_range:
                gmm = GaussianMixture(
                    n_components=n_clusters,
                    random_state=self.random_state,
                    # 尝试获取GMM特有的参数
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
            
        for method in methods:
            if hasattr(self, f'calculate_{method}_scores'):
                if show_progress:
                    print(f"Calculating {method}...")
                    
                # Call the specific calculation method
                # 这行代码的作用是：通过方法名字符串查找并获取当前类(self)中以"calculate_{method}_scores"命名的方法，
                # 并将其赋值给变量calculation_method，以后可以像函数一样调用它来计算指定指标的聚类分数。
                # 例如，如果method是"silhouette"，则会获取"calculate_silhouette_scores"方法。
                calculation_method = getattr(self, f'calculate_{method}_scores')
                self.scores[method] = calculation_method(X, self.cluster_range)
                
                if show_progress:
                    print(f"{method.capitalize()} calculation completed!")
        
        # Automatically select the best number of clusters
        if len(methods) == 1:
            best_method = methods[0]
        else:
            # Use combined method
            best_method = '_'.join(methods)
        
        best_n_clusters = self.auto_select_best_n_clusters(self.scores, best_method)
        
        # Set the best number of clusters
        self.n_clusters = best_n_clusters
        
        if show_progress:
            print(f"Automatically selected best number of clusters: {best_n_clusters}")
        
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

    def _select_best_n_clusters_for_single_method(self, scores: List[float], method: str) -> int:
        """
        Select optimal number of clusters for a single method.
        
        Args:
            scores: List of scores for different cluster numbers
            method: Method name
            
        Returns:
            int: Best number of clusters
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
        elif optimization == 'elbow':
            best_idx = self._find_best_n_clusters_for_elbow_method(scores)
        else:
            # Default to maximum value
            best_idx = np.argmax(scores)
        
        best_n_clusters = self.cluster_range[best_idx]

        return best_n_clusters
    
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
        
        # If it's a combination of methods, use voting system
        else:
            methods = method.split('_')
            votes = {}  # Dictionary to store votes: {cluster_index: vote_count}
            
            # Each method votes for its best cluster number
            for m in methods:
                if m not in scores_dict:
                    continue
                
                scores = scores_dict[m]
                # Select best cluster index for this method
                best_n_clusters = self._select_best_n_clusters_for_single_method(scores, m)
                
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