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