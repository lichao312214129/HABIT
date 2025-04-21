"""
Base class for clustering algorithms
"""

from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import importlib
import inspect
import os
import pkgutil
from typing import Dict, List, Any, Type, Optional, Tuple, Union

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
        module_name = f"habitat_clustering.clustering.{name}_clustering"
        module = importlib.import_module(module_name)
        
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
                module = importlib.import_module(f"habitat_clustering.clustering.{module_name}")
                
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
    - find_optimal_clusters: Find the optimal number of clusters
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
    
    @abstractmethod
    def find_optimal_clusters(self, X: np.ndarray, min_clusters: int, max_clusters: int, methods: Optional[List[str]] = None) -> Tuple[int, Dict[str, List[float]]]:
        """
        Find optimal number of clusters
        
        Args:
            X (np.ndarray): Input data with shape (n_samples, n_features)
            min_clusters (int): Minimum number of clusters
            max_clusters (int): Maximum number of clusters
            methods (Optional[List[str]]): List of methods to determine optimal number of clusters, if None, use default methods
            
        Returns:
            Tuple[int, Dict[str, List[float]]]: Optimal number of clusters and scores for different numbers of clusters
        """
        pass
    
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
            
            if method in ['silhouette', 'calinski_harabasz']:
                # These methods are better with higher values
                best_idx = np.argmax(scores)
            elif method in ['inertia', 'bic', 'aic']:
                # These methods are better with lower values, but need to use elbow method
                # Calculate first-order differences
                deltas = np.diff(scores)
                # Calculate second-order differences (elbow is usually the point with maximum second-order difference)
                deltas2 = np.diff(deltas)
                # Elbow is the point after the maximum second-order difference
                best_idx = np.argmax(deltas2) + 1
                if best_idx >= len(scores) - 1:
                    best_idx = len(scores) - 2  # Choose the second-to-last point
            else:
                # Default to using maximum value
                best_idx = np.argmax(scores)
        
        # If it's a combination of methods
        else:
            methods = method.split('_')
            rankings = np.zeros(len(list(scores_dict.values())[0]))
            
            for m in methods:
                if m not in scores_dict:
                    continue
                
                scores = scores_dict[m]
                
                if m in ['silhouette', 'calinski_harabasz']:
                    # Calculate normalized rankings (higher is better)
                    norm_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-10)
                    rankings += norm_scores
                elif m in ['inertia', 'bic', 'aic']:
                    # Calculate normalized rankings (lower is better)
                    norm_scores = 1 - (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-10)
                    rankings += norm_scores
            
            # Choose the one with highest combined ranking
            best_idx = np.argmax(rankings)
        
        # Add min_clusters offset
        min_clusters = min(self.cluster_range)
        best_n_clusters = best_idx + min_clusters
        
        return best_n_clusters 
    
    def plot_scores(self, scores_dict: Optional[Dict[str, List[float]]] = None, methods: Optional[Union[List[str], str]] = None, 
                   min_clusters: int = 2, max_clusters: int = 10, figsize: Tuple[int, int] = (12, 8), 
                   save_path: Optional[str] = None, show: bool = True):
        """
        Plot scores for different numbers of clusters
        
        Args:
            scores_dict (Optional[Dict[str, List[float]]]): Dictionary of scores, keys are method names, values are score lists, if None, use self.scores
            methods (Optional[Union[List[str], str]]): List of scoring methods to plot, if None, plot all methods in scores_dict
            min_clusters (int): Minimum number of clusters
            max_clusters (int): Maximum number of clusters
            figsize (Tuple[int, int]): Figure size
            save_path (Optional[str]): Path to save the figure, if None, don't save
            show (bool): Whether to display the figure
            
        Returns:
            matplotlib.figure.Figure: Matplotlib figure object
        """
        import matplotlib.pyplot as plt

        # Support for Chinese characters
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # Set cluster range
        self.cluster_range = list(range(min_clusters, max_clusters + 1))
        
        # If scores_dict is None, use self.scores
        if scores_dict is None:
            scores_dict = self.scores
        
        # If methods is None, use all methods in scores_dict
        if methods is None:
            methods = list(scores_dict.keys())
        else:
            # Filter out methods not in scores_dict
            if isinstance(methods, str):  # If methods is a string, convert to list
                methods = [methods]
            methods = [m for m in methods if m in scores_dict]
        
        if not methods:
            raise ValueError("No scoring methods to plot")
        
        # Create figure
        fig, axes = plt.subplots(len(methods), 1, figsize=figsize, constrained_layout=True)
        
        # If there's only one method, convert axes to list for uniform handling
        if len(methods) == 1:
            axes = [axes]
        
        # Iterate through each scoring method
        for i, method in enumerate(methods):
            ax = axes[i]
            scores = scores_dict[method]
                        
            # Plot score curve
            ax.plot(self.cluster_range, scores, 'o-', linewidth=2, markersize=8)

            
            # Mark optimal number of clusters
            if method in ['silhouette', 'calinski_harabasz']:
                # These methods are better with higher values
                best_idx = np.argmax(scores)
                best_score = scores[best_idx]
                criterion = "Maximum Value"
            elif method in ['inertia', 'bic', 'aic']:
                # These methods are better with lower values, but need to use elbow method
                # Calculate first-order differences
                deltas = np.diff(scores)
                # Calculate second-order differences (elbow is usually the point with maximum second-order difference)
                deltas2 = np.diff(deltas)
                # Elbow is the point after the maximum second-order difference
                best_idx = np.argmax(deltas2) + 1
                if best_idx >= len(scores) - 1:
                    best_idx = len(scores) - 2
                best_score = scores[best_idx]
                criterion = "Elbow Method"
            else:
                # Default to using maximum value
                best_idx = np.argmax(scores)
                best_score = scores[best_idx]
                criterion = "Maximum Value"
            
            best_n_clusters = self.cluster_range[best_idx]
            ax.plot(best_n_clusters, best_score, 'rx', markersize=12, markeredgewidth=3)
            
            # Set title and labels
            method_names = {
                'silhouette': 'Silhouette Score',
                'calinski_harabasz': 'Calinski-Harabasz Index',
                'inertia': 'Inertia',
                'bic': 'BIC Index',
                'aic': 'AIC Index'
            }
            
            method_name = method_names.get(method, method)
            ax.set_title(f'{method_name} ({criterion} selects clusters: {best_n_clusters})', fontsize=14)
            ax.set_xlabel('Number of Clusters', fontsize=12)
            ax.set_ylabel('Score', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add annotations
            for j, (x, y) in enumerate(zip(self.cluster_range, scores)):
                ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", 
                            xytext=(0, 10), ha='center', fontsize=9)
            
            # Set x-axis ticks to integers
            ax.set_xticks(self.cluster_range)
        
        plt.suptitle('Comparison of Evaluation Metrics for Different Numbers of Clusters', fontsize=16)
        
        # Save figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Display figure
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig 