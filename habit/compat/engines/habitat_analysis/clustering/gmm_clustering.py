# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Gaussian Mixture Model Clustering for Habitat Analysis.
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from typing import Optional, Any, Tuple
from .base_clustering import BaseClustering, ClusteringAlgorithmFactory


@ClusteringAlgorithmFactory.register('gmm')
class GMMClustering(BaseClustering):
    """
    Gaussian Mixture Model (GMM) clustering algorithm implementation
    """
    
    def __init__(self, n_clusters: Optional[int] = None, random_state: int = 0, 
                 covariance_type: str = 'full', n_init: int = 50, max_iter: int = 100, **kwargs: Any):
        """
        Initialize GMM clustering algorithm
        
        Args:
            n_clusters (Optional[int]): Number of clusters, can be None if determined by find_optimal_clusters
            random_state (int): Random seed to ensure reproducibility
            covariance_type (str): Covariance type, options: 'full', 'tied', 'diag', 'spherical'
            n_init (int): Number of times the algorithm will run, selecting the best result
            max_iter (int): Maximum number of iterations
            **kwargs (Any): Additional parameters for GaussianMixture
        """
        super().__init__(n_clusters, random_state)
        self.covariance_type = covariance_type
        self.n_init = n_init
        self.max_iter = max_iter
        self.kwargs = kwargs
    
    def fit(self, X: np.ndarray) -> 'GMMClustering':
        """
        Train the GMM clustering model
        
        Args:
            X (np.ndarray): Input data with shape (n_samples, n_features)
            
        Returns:
            GMMClustering: Trained model
        """
        if self.n_clusters is None:
            raise ValueError("n_clusters must be specified or determined by find_optimal_clusters method")
        
        self.model = GaussianMixture(
            n_components=self.n_clusters,
            random_state=self.random_state,
            covariance_type=self.covariance_type,
            n_init=self.n_init,
            max_iter=self.max_iter,
            **self.kwargs
        )
        self.model.fit(X)
        self.labels_ = self.model.predict(X)
        self.means_ = self.model.means_
        self.covariances_ = self.model.covariances_
        self.weights_ = self.model.weights_
        self.bic_ = self.model.bic(X)
        self.aic_ = self.model.aic(X)
        
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