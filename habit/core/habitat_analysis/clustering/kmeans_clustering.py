"""
K-Means Clustering Implementation for Habitat Analysis.
"""

import numpy as np
from sklearn.cluster import KMeans
from typing import Optional, Any
from .base_clustering import BaseClustering, register_clustering


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