"""
Gaussian Mixture Model (GMM) clustering algorithm implementation
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
from habitat_clustering.clustering.base_clustering import BaseClustering, register_clustering


@register_clustering('gmm')
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
    
    def calculate_bic_aic(self, X: np.ndarray, cluster_range: List[int]) -> Tuple[List[float], List[float]]:
        """
        Calculate BIC and AIC values for different numbers of clusters
        
        Args:
            X (np.ndarray): Input data
            cluster_range (List[int]): Range of cluster numbers to evaluate
            
        Returns:
            Tuple[List[float], List[float]]: BIC scores list and AIC scores list
        """
        bic_scores = []
        aic_scores = []
        
        for n_clusters in cluster_range:
            gmm = GaussianMixture(
                n_components=n_clusters,
                random_state=self.random_state,
                covariance_type=self.covariance_type,
                n_init=self.n_init,
                max_iter=self.max_iter,
                **self.kwargs
            )
            gmm.fit(X)
            bic_scores.append(gmm.bic(X))
            aic_scores.append(gmm.aic(X))
        
        return bic_scores, aic_scores
    
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
            methods (Optional[Union[List[str], str]]): List of methods to determine the optimal number of clusters, 
                                     if None, default methods will be used
            show_progress (bool): Whether to display progress
            
        Returns:
            Tuple[int, Dict[str, List[float]]]: Best number of clusters and scores for different cluster numbers
        """
        # Use default methods if methods is None
        if methods is None:
            methods = ['bic', 'aic', 'silhouette']
        
        # Save cluster range
        self.cluster_range = list(range(min_clusters, max_clusters + 1))
        
        # Calculate different scores
        self.scores: Dict[str, List[float]] = {}
        
        if show_progress:
            print("Starting to calculate evaluation metrics for different cluster numbers...")
            total = max_clusters - min_clusters + 1
        
        # Calculate BIC and AIC (if needed)
        if 'bic' in methods or 'aic' in methods:
            bic_scores = []
            aic_scores = []
            for i, n_clusters in enumerate(self.cluster_range):
                if show_progress:
                    progress = int((i + 1) / total * 50)
                    bar = "█" * progress + "-" * (50 - progress)
                    percent = (i + 1) / total * 100
                    print(f"\r[{bar}] {percent:.2f}% ({i+1}/{total})", end="")
                
                gmm = GaussianMixture(
                    n_components=n_clusters,
                    random_state=self.random_state,
                    covariance_type=self.covariance_type,
                    n_init=self.n_init,
                    max_iter=self.max_iter,
                    **self.kwargs
                )
                gmm.fit(X)
                bic_scores.append(gmm.bic(X))
                aic_scores.append(gmm.aic(X))
            
            if 'bic' in methods:
                self.scores['bic'] = bic_scores
            if 'aic' in methods:
                self.scores['aic'] = aic_scores
            
            if show_progress:
                print("\nBIC/AIC calculation completed!")
        
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