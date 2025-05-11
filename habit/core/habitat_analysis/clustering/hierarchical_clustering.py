"""
Implementation of Hierarchical Clustering algorithm
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from typing import Tuple, Dict, Any, Optional, List
from habitat_clustering.clustering.base_clustering import BaseClustering, register_clustering

@register_clustering('hierarchical')
class HierarchicalClustering(BaseClustering):
    """
    Hierarchical Clustering implementation
    
    Parameters:
    -----------
    n_clusters : int, optional (default=None)
        The number of clusters to find. If None, it will be determined by find_optimal_clusters method
        
    linkage : str, optional (default='ward')
        The linkage criterion to use. The linkage criterion determines which
        distance to use between sets of observation. The algorithm will merge
        the pairs of cluster that minimize this criterion.
        - 'ward' minimizes the variance of the clusters being merged
        - 'complete' or 'maximum' linkage uses the maximum distances between all observations of the two sets
        - 'average' uses the average of the distances of each observation of the two sets
        - 'single' uses the minimum of the distances between all observations of the two sets
        
    affinity : str, optional (default='euclidean')
        Metric used to compute the linkage. Can be 'euclidean', 'l1', 'l2', 'manhattan',
        'cosine', or 'precomputed'.
    """
    
    def __init__(self, n_clusters: int = None, linkage: str = 'ward', 
                 affinity: str = 'euclidean', random_state: int = 0, **kwargs):
        super().__init__(n_clusters=n_clusters, random_state=random_state)
        self.linkage = linkage
        self.affinity = affinity
        self.kwargs = kwargs
        self.model = None
        self.labels_ = None
    
    def fit(self, X: np.ndarray) -> 'HierarchicalClustering':
        """
        Fit the hierarchical clustering model
        
        Args:
            X : np.ndarray
                Training data of shape (n_samples, n_features)
                
        Returns:
            self : HierarchicalClustering
                Returns the instance itself
        """
        if self.n_clusters is None:
            raise ValueError("n_clusters must be specified or determined by find_optimal_clusters method")
            
        self.model = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage=self.linkage,
            affinity=self.affinity
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
            model = AgglomerativeClustering(
                n_clusters=n,
                linkage=self.linkage,
                affinity=self.affinity
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
            model = AgglomerativeClustering(
                n_clusters=n,
                linkage=self.linkage,
                affinity=self.affinity
            )
            labels = model.fit_predict(X)
            score = calinski_harabasz_score(X, labels)
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
            methods = ['silhouette', 'calinski_harabasz']
            
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
            'linkage': self.linkage,
            'affinity': self.affinity
        })
        return params
    
    def set_params(self, **params) -> 'HierarchicalClustering':
        """
        Set the parameters of this estimator
        
        Args:
            **params : dict
                Estimator parameters
            
        Returns:
            self : HierarchicalClustering
                Returns the instance itself
        """
        super().set_params(**params)
        if 'linkage' in params:
            self.linkage = params['linkage']
        if 'affinity' in params:
            self.affinity = params['affinity']
        return self 