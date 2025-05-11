"""
Implementation of Mean Shift Clustering algorithm
"""

import numpy as np
from sklearn.cluster import MeanShift
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from typing import Tuple, Dict, Any, Optional, List
from habitat_clustering.clustering.base_clustering import BaseClustering, register_clustering

@register_clustering('mean_shift')
class MeanShiftClustering(BaseClustering):
    """
    Mean Shift Clustering implementation
    
    Parameters:
    -----------
    bandwidth : float, optional (default=None)
        Bandwidth used in the RBF kernel. If not given, the bandwidth is estimated
        using sklearn.cluster.estimate_bandwidth
        
    seeds : array, shape=[n_samples, n_features], optional (default=None)
        Seeds used to initialize kernels. If not set, the seeds are calculated by
        clustering.get_bin_seeds with bandwidth as the grid size and default values
        for other parameters
        
    bin_seeding : bool, optional (default=False)
        If true, initial kernel locations are not locations of all points, but
        rather the location of the discretized version of points, where points are
        binned onto a grid whose coarseness corresponds to the bandwidth
        
    min_bin_freq : int, optional (default=1)
        To speed up the algorithm, accept only those bins with at least min_bin_freq
        points as seeds
        
    cluster_all : bool, optional (default=True)
        If true, then all points are clustered, even those orphans that are not within
        any kernel. Orphans are assigned to the nearest kernel. If false, then orphans
        are given cluster label -1
    """
    
    def __init__(self, bandwidth: Optional[float] = None, seeds: Optional[np.ndarray] = None,
                 bin_seeding: bool = False, min_bin_freq: int = 1,
                 cluster_all: bool = True, random_state: int = 0, **kwargs):
        super().__init__(n_clusters=None, random_state=random_state)  # MeanShift doesn't require n_clusters
        self.bandwidth = bandwidth
        self.seeds = seeds
        self.bin_seeding = bin_seeding
        self.min_bin_freq = min_bin_freq
        self.cluster_all = cluster_all
        self.kwargs = kwargs
        self.model = None
        self.labels_ = None
    
    def fit(self, X: np.ndarray) -> 'MeanShiftClustering':
        """
        Fit the mean shift clustering model
        
        Args:
            X : np.ndarray
                Training data of shape (n_samples, n_features)
                
        Returns:
            self : MeanShiftClustering
                Returns the instance itself
        """
        self.model = MeanShift(
            bandwidth=self.bandwidth,
            seeds=self.seeds,
            bin_seeding=self.bin_seeding,
            min_bin_freq=self.min_bin_freq,
            cluster_all=self.cluster_all
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
                Cluster labels for each sample
        """
        if self.model is None:
            raise ValueError("Must call fit method first")
        return self.model.fit_predict(X)
    
    def calculate_silhouette_scores(self, X: np.ndarray, bandwidth_range: List[float]) -> np.ndarray:
        """
        Calculate silhouette scores for different bandwidth values
        
        Args:
            X : np.ndarray
                Input data of shape (n_samples, n_features)
            bandwidth_range : List[float]
                List of bandwidth values to evaluate
                
        Returns:
            scores : np.ndarray
                Silhouette scores for each bandwidth value
        """
        scores = []
        for bandwidth in bandwidth_range:
            model = MeanShift(
                bandwidth=bandwidth,
                seeds=self.seeds,
                bin_seeding=self.bin_seeding,
                min_bin_freq=self.min_bin_freq,
                cluster_all=self.cluster_all
            )
            labels = model.fit_predict(X)
            # Only calculate score if there are at least 2 clusters
            if len(np.unique(labels)) > 1:
                score = silhouette_score(X, labels)
                scores.append(score)
            else:
                scores.append(-1)  # Invalid clustering
        return np.array(scores)
    
    def calculate_calinski_harabasz_scores(self, X: np.ndarray, bandwidth_range: List[float]) -> np.ndarray:
        """
        Calculate Calinski-Harabasz scores for different bandwidth values
        
        Args:
            X : np.ndarray
                Input data of shape (n_samples, n_features)
            bandwidth_range : List[float]
                List of bandwidth values to evaluate
                
        Returns:
            scores : np.ndarray
                Calinski-Harabasz scores for each bandwidth value
        """
        scores = []
        for bandwidth in bandwidth_range:
            model = MeanShift(
                bandwidth=bandwidth,
                seeds=self.seeds,
                bin_seeding=self.bin_seeding,
                min_bin_freq=self.min_bin_freq,
                cluster_all=self.cluster_all
            )
            labels = model.fit_predict(X)
            # Only calculate score if there are at least 2 clusters
            if len(np.unique(labels)) > 1:
                score = calinski_harabasz_score(X, labels)
                scores.append(score)
            else:
                scores.append(-1)  # Invalid clustering
        return np.array(scores)
    
    def calculate_davies_bouldin_scores(self, X: np.ndarray, bandwidth_range: List[float]) -> np.ndarray:
        """
        Calculate Davies-Bouldin scores for different bandwidth values
        
        Args:
            X : np.ndarray
                Input data of shape (n_samples, n_features)
            bandwidth_range : List[float]
                List of bandwidth values to evaluate
                
        Returns:
            scores : np.ndarray
                Davies-Bouldin scores for each bandwidth value
        """
        scores = []
        for bandwidth in bandwidth_range:
            model = MeanShift(
                bandwidth=bandwidth,
                seeds=self.seeds,
                bin_seeding=self.bin_seeding,
                min_bin_freq=self.min_bin_freq,
                cluster_all=self.cluster_all
            )
            labels = model.fit_predict(X)
            # Only calculate score if there are at least 2 clusters
            if len(np.unique(labels)) > 1:
                score = davies_bouldin_score(X, labels)
                scores.append(score)
            else:
                scores.append(float('inf'))  # Invalid clustering
        return np.array(scores)
    
    def find_optimal_clusters(self, X: np.ndarray, min_bandwidth: float = 0.1, max_bandwidth: float = 1.0,
                             n_bandwidth: int = 10, methods: List[str] = None,
                             show_progress: bool = True) -> Tuple[float, Dict[str, np.ndarray]]:
        """
        Find the optimal bandwidth value for Mean Shift
        
        Args:
            X : np.ndarray
                Input data of shape (n_samples, n_features)
            min_bandwidth : float, optional (default=0.1)
                Minimum bandwidth value to consider
            max_bandwidth : float, optional (default=1.0)
                Maximum bandwidth value to consider
            n_bandwidth : int, optional (default=10)
                Number of bandwidth values to evaluate
            methods : List[str], optional (default=None)
                List of methods to use for determining optimal bandwidth
            show_progress : bool, optional (default=True)
                Whether to show progress during calculation
                
        Returns:
            best_bandwidth : float
                Optimal bandwidth value
            scores : Dict[str, np.ndarray]
                Dictionary of scores for each method
        """
        if methods is None:
            methods = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
            
        bandwidth_range = np.linspace(min_bandwidth, max_bandwidth, n_bandwidth)
        self.bandwidth_range = bandwidth_range
        self.scores = {}
        
        if 'silhouette' in methods:
            if show_progress:
                print("Calculating silhouette scores...")
            self.scores['silhouette'] = self.calculate_silhouette_scores(X, bandwidth_range)
            if show_progress:
                print("Silhouette score calculation completed!")
                
        if 'calinski_harabasz' in methods:
            if show_progress:
                print("Calculating Calinski-Harabasz scores...")
            self.scores['calinski_harabasz'] = self.calculate_calinski_harabasz_scores(X, bandwidth_range)
            if show_progress:
                print("Calinski-Harabasz score calculation completed!")
                
        if 'davies_bouldin' in methods:
            if show_progress:
                print("Calculating Davies-Bouldin scores...")
            self.scores['davies_bouldin'] = self.calculate_davies_bouldin_scores(X, bandwidth_range)
            if show_progress:
                print("Davies-Bouldin score calculation completed!")
                
        if len(methods) == 1:
            best_method = methods[0]
        else:
            best_method = '_'.join(methods)
            
        best_bandwidth_idx = self.auto_select_best_n_clusters(self.scores, best_method)
        best_bandwidth = bandwidth_range[best_bandwidth_idx]
        self.bandwidth = best_bandwidth
        
        if show_progress:
            print(f"Automatically selected best bandwidth value: {best_bandwidth}")
            
        return best_bandwidth, self.scores
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get parameters for this estimator
        
        Returns:
            params : dict
                Parameter names mapped to their values
        """
        params = super().get_params()
        params.update({
            'bandwidth': self.bandwidth,
            'seeds': self.seeds,
            'bin_seeding': self.bin_seeding,
            'min_bin_freq': self.min_bin_freq,
            'cluster_all': self.cluster_all
        })
        return params
    
    def set_params(self, **params) -> 'MeanShiftClustering':
        """
        Set the parameters of this estimator
        
        Args:
            **params : dict
                Estimator parameters
            
        Returns:
            self : MeanShiftClustering
                Returns the instance itself
        """
        super().set_params(**params)
        if 'bandwidth' in params:
            self.bandwidth = params['bandwidth']
        if 'seeds' in params:
            self.seeds = params['seeds']
        if 'bin_seeding' in params:
            self.bin_seeding = params['bin_seeding']
        if 'min_bin_freq' in params:
            self.min_bin_freq = params['min_bin_freq']
        if 'cluster_all' in params:
            self.cluster_all = params['cluster_all']
        return self 