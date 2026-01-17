"""
Implementation of Spectral Clustering algorithm
"""

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from typing import Tuple, Dict, Any, Optional, List
from ..algorithms.base_clustering import BaseClustering, register_clustering

@register_clustering('spectral')
class SpectralClustering(BaseClustering):
    """
    Spectral Clustering implementation
    
    Parameters:
    -----------
    n_clusters : int, optional (default=None)
        The number of clusters to find. If None, it will be determined by find_optimal_clusters method
        
    affinity : str, optional (default='rbf')
        How to construct the affinity matrix
        - 'nearest_neighbors': construct the affinity matrix by computing a
          graph of nearest neighbors
        - 'rbf': construct the affinity matrix using a radial basis function (RBF) kernel
        - 'precomputed': interpret X as a precomputed affinity matrix
        - 'precomputed_nearest_neighbors': interpret X as a sparse graph of precomputed distances
        
    gamma : float, optional (default=1.0)
        Kernel coefficient for rbf, poly, sigmoid, laplacian and chi2 kernels
        
    n_neighbors : int, optional (default=10)
        Number of neighbors to use when constructing the affinity matrix using
        the nearest neighbors method
        
    eigen_solver : str, optional (default='arpack')
        The eigenvalue decomposition strategy to use. ARPACK can handle both
        sparse and dense problems, while 'lobpcg' is recommended for very large
        and sparse problems
    """
    
    def __init__(self, n_clusters: int = None, affinity: str = 'rbf',
                 gamma: float = 1.0, n_neighbors: int = 10,
                 eigen_solver: str = 'arpack', random_state: int = 0, **kwargs):
        super().__init__(n_clusters=n_clusters, random_state=random_state)
        self.affinity = affinity
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.eigen_solver = eigen_solver
        self.kwargs = kwargs
        self.model = None
        self.labels_ = None
    
    def fit(self, X: np.ndarray) -> 'SpectralClustering':
        """
        Fit the spectral clustering model
        
        Args:
            X : np.ndarray
                Training data of shape (n_samples, n_features)
                
        Returns:
            self : MySpectralClustering
                Returns the instance itself
        """
        if self.n_clusters is None:
            raise ValueError("n_clusters must be specified or determined by find_optimal_clusters method")
            
        self.model = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity=self.affinity,
            gamma=self.gamma,
            n_neighbors=self.n_neighbors,
            eigen_solver=self.eigen_solver,
            random_state=self.random_state
        )
        return self.model.fit(X)
    
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
    
    def calculate_silhouette_scores(self, X: np.ndarray, cluster_range: List[int]) -> np.ndarray:
        """
        Calculate silhouette scores for different numbers of clusters
        
        Args:
            X : np.ndarray
                Input data of shape (n_samples, n_features)
            cluster_range : List[int]
                List of cluster numbers to evaluate
                
        Returns:
            scores : np.ndarray
                Silhouette scores for each number of clusters
        """
        scores = []
        for n in cluster_range:
            model = SpectralClustering(
                n_clusters=n,
                affinity=self.affinity,
                gamma=self.gamma,
                n_neighbors=self.n_neighbors,
                eigen_solver=self.eigen_solver,
                random_state=self.random_state
            )
            labels = model.fit_predict(X)
            score = silhouette_score(X, labels)
            scores.append(score)
        return np.array(scores)
    
    def calculate_calinski_harabasz_scores(self, X: np.ndarray, cluster_range: List[int]) -> np.ndarray:
        """
        Calculate Calinski-Harabasz scores for different numbers of clusters
        
        Args:
            X : np.ndarray
                Input data of shape (n_samples, n_features)
            cluster_range : List[int]
                List of cluster numbers to evaluate
                
        Returns:
            scores : np.ndarray
                Calinski-Harabasz scores for each number of clusters
        """
        scores = []
        for n in cluster_range:
            model = SpectralClustering(
                n_clusters=n,
                affinity=self.affinity,
                gamma=self.gamma,
                n_neighbors=self.n_neighbors,
                eigen_solver=self.eigen_solver,
                random_state=self.random_state
            )
            labels = model.fit_predict(X)
            score = calinski_harabasz_score(X, labels)
            scores.append(score)
        return np.array(scores)
    
    def calculate_davies_bouldin_scores(self, X: np.ndarray, cluster_range: List[int]) -> np.ndarray:
        """
        Calculate Davies-Bouldin scores for different numbers of clusters
        
        Args:
            X : np.ndarray
                Input data of shape (n_samples, n_features)
            cluster_range : List[int]
                List of cluster numbers to evaluate
                
        Returns:
            scores : np.ndarray
                Davies-Bouldin scores for each number of clusters
        """
        scores = []
        for n in cluster_range:
            model = SpectralClustering(
                n_clusters=n,
                affinity=self.affinity,
                gamma=self.gamma,
                n_neighbors=self.n_neighbors,
                eigen_solver=self.eigen_solver,
                random_state=self.random_state
            )
            labels = model.fit_predict(X)
            score = davies_bouldin_score(X, labels)
            scores.append(score)
        return np.array(scores)
    
    def find_optimal_clusters(self, X: np.ndarray, min_clusters: int = 2, max_clusters: int = 10, 
                             methods: List[str] = None, show_progress: bool = True) -> Tuple[int, Dict[str, np.ndarray]]:
        """
        Find the optimal number of clusters
        
        Args:
            X : np.ndarray
                Input data of shape (n_samples, n_features)
            min_clusters : int, optional (default=2)
                Minimum number of clusters to consider
            max_clusters : int, optional (default=10)
                Maximum number of clusters to consider
            methods : List[str], optional (default=None)
                List of methods to use for determining optimal clusters
            show_progress : bool, optional (default=True)
                Whether to show progress during calculation
                
        Returns:
            best_n_clusters : int
                Optimal number of clusters
            scores : Dict[str, np.ndarray]
                Dictionary of scores for each method
        """
        if methods is None:
            methods = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
            
        self.cluster_range = list(range(min_clusters, max_clusters + 1))
        self.scores = {}
        
        if 'silhouette' in methods:
            if show_progress:
                print("Calculating silhouette scores...")
            self.scores['silhouette'] = self.calculate_silhouette_scores(X, self.cluster_range)
            if show_progress:
                print("Silhouette score calculation completed!")
                
        if 'calinski_harabasz' in methods:
            if show_progress:
                print("Calculating Calinski-Harabasz scores...")
            self.scores['calinski_harabasz'] = self.calculate_calinski_harabasz_scores(X, self.cluster_range)
            if show_progress:
                print("Calinski-Harabasz score calculation completed!")
                
        if 'davies_bouldin' in methods:
            if show_progress:
                print("Calculating Davies-Bouldin scores...")
            self.scores['davies_bouldin'] = self.calculate_davies_bouldin_scores(X, self.cluster_range)
            if show_progress:
                print("Davies-Bouldin score calculation completed!")
                
        if len(methods) == 1:
            best_method = methods[0]
        else:
            best_method = '_'.join(methods)
            
        best_n_clusters = self.auto_select_best_n_clusters(self.scores, best_method)
        self.n_clusters = best_n_clusters
        
        if show_progress:
            print(f"Automatically selected best number of clusters: {best_n_clusters}")
            
        return best_n_clusters, self.scores
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get parameters for this estimator
        
        Returns:
            params : dict
                Parameter names mapped to their values
        """
        params = super().get_params()
        params.update({
            'affinity': self.affinity,
            'gamma': self.gamma,
            'n_neighbors': self.n_neighbors,
            'eigen_solver': self.eigen_solver
        })
        return params
    
    def set_params(self, **params) -> 'SpectralClustering':
        """
        Set the parameters of this estimator
        
        Args:
            **params : dict
                Estimator parameters
            
        Returns:
            self : MySpectralClustering
                Returns the instance itself
        """
        super().set_params(**params)
        if 'affinity' in params:
            self.affinity = params['affinity']
        if 'gamma' in params:
            self.gamma = params['gamma']
        if 'n_neighbors' in params:
            self.n_neighbors = params['n_neighbors']
        if 'eigen_solver' in params:
            self.eigen_solver = params['eigen_solver']
        return self 