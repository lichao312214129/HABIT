"""
KMeans clustering algorithm implementation
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
from habitat_clustering.clustering.base_clustering import BaseClustering, register_clustering


@register_clustering('kmeans')
class KMeansClustering(BaseClustering):
    """
    Implementation of KMeans clustering algorithm
    """
    
    def __init__(self, n_clusters: Optional[int] = None, random_state: int = 0, 
                 init: str = 'k-means++', n_init: int = 50, **kwargs: Any):
        """
        Initialize KMeans clustering algorithm
        
        Args:
            n_clusters (Optional[int]): Number of clusters. Can be None, indicating it needs to be determined by find_optimal_clusters
            random_state (int): Random seed to ensure reproducibility
            init (str): Initialization method, default is 'k-means++'
            n_init (int): Number of times to run the algorithm, selecting the best result
            **kwargs (Any): Additional parameters to pass to KMeans
        """
        super().__init__(n_clusters, random_state)
        self.init = init
        self.n_init = n_init
        self.kwargs = kwargs
    
    def fit(self, X: np.ndarray) -> 'KMeansClustering':
        """
        Train the KMeans clustering model
        
        Args:
            X (np.ndarray): Input data with shape (n_samples, n_features)
            
        Returns:
            KMeansClustering: Trained model instance
        """
        if self.n_clusters is None:
            raise ValueError("n_clusters must be specified or determined using find_optimal_clusters method")
        
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            init=self.init,
            n_init=self.n_init,
            **self.kwargs
        )
        self.model.fit(X)
        # 移除X中nan inf的行
        X1 = X[~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1)]
        # 找到X中nan inf的行
        nan_inf_rows = np.where(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1))[0]
        X[nan_inf_rows]

        self.labels_ = self.model.labels_
        self.cluster_centers_ = self.model.cluster_centers_
        self.inertia_ = self.model.inertia_
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for input data
        
        Args:
            X (np.ndarray): Input data with shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Cluster labels with shape (n_samples,)
        """
        if self.model is None:
            raise ValueError("fit method must be called first")
        
        return self.model.predict(X)
    
    def calculate_inertia(self, X: np.ndarray, cluster_range: List[int]) -> List[float]:
        """
        Calculate inertia (SSE) for different numbers of clusters
        
        Args:
            X (np.ndarray): Input data
            cluster_range (List[int]): Range of cluster numbers to evaluate
            
        Returns:
            List[float]: List of inertia values
        """
        inertias = []
        for n_clusters in cluster_range:
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                init=self.init,
                n_init=self.n_init,
                **self.kwargs
            )
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        
        return inertias
    
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
        # If methods is None, use default methods
        if methods is None:
            methods = ['silhouette', 'calinski_harabasz', 'inertia']
        
        # Save cluster range
        self.cluster_range = list(range(min_clusters, max_clusters + 1))
        
        # Calculate different scores
        self.scores: Dict[str, List[float]] = {}
        
        if show_progress:
            print("Starting to calculate evaluation metrics for different numbers of clusters...")
            total = max_clusters - min_clusters + 1
        
        # Calculate inertia (if needed)
        if 'inertia' in methods:
            if show_progress:
                print("Calculating inertia...")
            inertias = []
            for i, n_clusters in enumerate(self.cluster_range):
                if show_progress:
                    progress = int((i + 1) / total * 50)
                    bar = "█" * progress + "-" * (50 - progress)
                    percent = (i + 1) / total * 100
                    print(f"\r[{bar}] {percent:.2f}% ({i+1}/{total})", end="")
                
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    random_state=self.random_state,
                    init=self.init,
                    n_init=self.n_init,
                    **self.kwargs
                )
                kmeans.fit(X)
                inertias.append(kmeans.inertia_)
            
            self.scores['inertia'] = inertias
            
            if show_progress:
                print("\nInertia calculation completed!")
        
        # Calculate silhouette coefficient (if needed)
        if 'silhouette' in methods:
            if show_progress:
                print("Calculating silhouette coefficient...")
            self.scores['silhouette'] = self.calculate_silhouette_scores(X, self.cluster_range)
            if show_progress:
                print("Silhouette coefficient calculation completed!")
        
        # Calculate Calinski-Harabasz index (if needed)
        if 'calinski_harabasz' in methods:
            if show_progress:
                print("Calculating Calinski-Harabasz index...")
            self.scores['calinski_harabasz'] = self.calculate_calinski_harabasz_scores(X, self.cluster_range)
            if show_progress:
                print("Calinski-Harabasz index calculation completed!")
        
        # Automatically select the best number of clusters
        if isinstance(methods, str):
            methods = [methods]
            
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