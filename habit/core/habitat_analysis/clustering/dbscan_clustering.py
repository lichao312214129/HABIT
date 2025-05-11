"""
Implementation of DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm
"""

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from typing import Tuple, Dict, Any, Optional, List
from habitat_clustering.clustering.base_clustering import BaseClustering, register_clustering

@register_clustering('dbscan')
class DBSCANClustering(BaseClustering):
    """
    DBSCAN Clustering implementation
    
    Parameters:
    -----------
    eps : float, optional (default=0.5)
        The maximum distance between two samples for one to be considered
        as in the neighborhood of the other
        
    min_samples : int, optional (default=5)
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point
        
    metric : str, optional (default='euclidean')
        The metric to use when calculating distance between instances in a feature array
        - 'euclidean'
        - 'manhattan'
        - 'cosine'
        - etc.
        
    algorithm : str, optional (default='auto')
        The algorithm to be used by the NearestNeighbors module to compute pointwise distances
        and find nearest neighbors
        - 'auto'
        - 'ball_tree'
        - 'kd_tree'
        - 'brute'
    """
    
    def __init__(self, eps: float = 0.5, min_samples: int = 5,
                 metric: str = 'euclidean', algorithm: str = 'auto',
                 random_state: int = 0, **kwargs):
        super().__init__(n_clusters=None, random_state=random_state)  # DBSCAN doesn't require n_clusters
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.algorithm = algorithm
        self.kwargs = kwargs
        self.model = None
        self.labels_ = None
    
    def fit(self, X: np.ndarray) -> 'DBSCANClustering':
        """
        Fit the DBSCAN clustering model
        
        Args:
            X : np.ndarray
                Training data of shape (n_samples, n_features)
                
        Returns:
            self : DBSCANClustering
                Returns the instance itself
        """
        self.model = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric,
            algorithm=self.algorithm
        )
        self.labels_ = self.model.fit_predict(X)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the closest cluster each sample in X belongs to
        
        Args:
            X : np.ndarray
                New data to predict of shape (n_samples, n_features)
                
        Returns:
            labels : np.ndarray
                Cluster labels for each sample. Noise points are labeled as -1
        """
        if self.model is None:
            raise ValueError("Must call fit method first")
        return self.model.fit_predict(X)
    
    def calculate_silhouette_scores(self, X: np.ndarray, eps_range: List[float]) -> np.ndarray:
        """
        Calculate silhouette scores for different eps values
        
        Args:
            X : np.ndarray
                Input data of shape (n_samples, n_features)
            eps_range : List[float]
                List of eps values to evaluate
                
        Returns:
            scores : np.ndarray
                Silhouette scores for each eps value
        """
        scores = []
        for eps in eps_range:
            model = DBSCAN(
                eps=eps,
                min_samples=self.min_samples,
                metric=self.metric,
                algorithm=self.algorithm
            )
            labels = model.fit_predict(X)
            # Only calculate score if there are at least 2 clusters and not all points are noise
            if len(np.unique(labels)) > 1 and not np.all(labels == -1):
                score = silhouette_score(X, labels)
                scores.append(score)
            else:
                scores.append(-1)  # Invalid clustering
        return np.array(scores)
    
    def calculate_calinski_harabasz_scores(self, X: np.ndarray, eps_range: List[float]) -> np.ndarray:
        """
        Calculate Calinski-Harabasz scores for different eps values
        
        Args:
            X : np.ndarray
                Input data of shape (n_samples, n_features)
            eps_range : List[float]
                List of eps values to evaluate
                
        Returns:
            scores : np.ndarray
                Calinski-Harabasz scores for each eps value
        """
        scores = []
        for eps in eps_range:
            model = DBSCAN(
                eps=eps,
                min_samples=self.min_samples,
                metric=self.metric,
                algorithm=self.algorithm
            )
            labels = model.fit_predict(X)
            # Only calculate score if there are at least 2 clusters and not all points are noise
            if len(np.unique(labels)) > 1 and not np.all(labels == -1):
                score = calinski_harabasz_score(X, labels)
                scores.append(score)
            else:
                scores.append(-1)  # Invalid clustering
        return np.array(scores)
    
    def find_optimal_clusters(self, X: np.ndarray, min_eps: float = 0.1, max_eps: float = 1.0,
                             n_eps: int = 10, methods: List[str] = None,
                             show_progress: bool = True) -> Tuple[float, Dict[str, np.ndarray]]:
        """
        Find the optimal eps value for DBSCAN
        
        Args:
            X : np.ndarray
                Input data of shape (n_samples, n_features)
            min_eps : float, optional (default=0.1)
                Minimum eps value to consider
            max_eps : float, optional (default=1.0)
                Maximum eps value to consider
            n_eps : int, optional (default=10)
                Number of eps values to evaluate
            methods : List[str], optional (default=None)
                List of methods to use for determining optimal eps
            show_progress : bool, optional (default=True)
                Whether to show progress during calculation
                
        Returns:
            best_eps : float
                Optimal eps value
            scores : Dict[str, np.ndarray]
                Dictionary of scores for each method
        """
        if methods is None:
            methods = ['silhouette', 'calinski_harabasz']
            
        eps_range = np.linspace(min_eps, max_eps, n_eps)
        self.eps_range = eps_range
        self.scores = {}
        
        if 'silhouette' in methods:
            if show_progress:
                print("Calculating silhouette scores...")
            self.scores['silhouette'] = self.calculate_silhouette_scores(X, eps_range)
            if show_progress:
                print("Silhouette score calculation completed!")
                
        if 'calinski_harabasz' in methods:
            if show_progress:
                print("Calculating Calinski-Harabasz scores...")
            self.scores['calinski_harabasz'] = self.calculate_calinski_harabasz_scores(X, eps_range)
            if show_progress:
                print("Calinski-Harabasz score calculation completed!")
                
        if len(methods) == 1:
            best_method = methods[0]
        else:
            best_method = '_'.join(methods)
            
        best_eps_idx = self.auto_select_best_n_clusters(self.scores, best_method)
        best_eps = eps_range[best_eps_idx]
        self.eps = best_eps
        
        if show_progress:
            print(f"Automatically selected best eps value: {best_eps}")
            
        return best_eps, self.scores
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get parameters for this estimator
        
        Returns:
            params : dict
                Parameter names mapped to their values
        """
        params = super().get_params()
        params.update({
            'eps': self.eps,
            'min_samples': self.min_samples,
            'metric': self.metric,
            'algorithm': self.algorithm
        })
        return params
    
    def set_params(self, **params) -> 'DBSCANClustering':
        """
        Set the parameters of this estimator
        
        Args:
            **params : dict
                Estimator parameters
            
        Returns:
            self : DBSCANClustering
                Returns the instance itself
        """
        super().set_params(**params)
        if 'eps' in params:
            self.eps = params['eps']
        if 'min_samples' in params:
            self.min_samples = params['min_samples']
        if 'metric' in params:
            self.metric = params['metric']
        if 'algorithm' in params:
            self.algorithm = params['algorithm']
        return self 