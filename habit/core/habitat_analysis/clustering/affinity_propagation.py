"""
Implementation of Affinity Propagation Clustering algorithm
"""

import numpy as np
from sklearn.cluster import AffinityPropagation
from typing import Tuple, Dict, Any, Optional

from .base_clustering import BaseClustering

class AffinityPropagationClustering(BaseClustering):
    """
    Affinity Propagation Clustering implementation
    
    Parameters:
    -----------
    damping : float, optional (default=0.5)
        Damping factor between 0.5 and 1
        
    max_iter : int, optional (default=200)
        Maximum number of iterations
        
    convergence_iter : int, optional (default=15)
        Number of iterations with no change in the number of estimated clusters
        that stops the convergence
        
    preference : array-like, shape (n_samples,) or float, optional (default=None)
        Preferences for each point - points with larger values of preferences are
        more likely to be chosen as exemplars. The number of exemplars, i.e. of
        clusters, is influenced by the input preferences value. If the preferences
        are not passed as arguments, they will be set to the median of the input
        similarities
        
    affinity : str, optional (default='euclidean')
        Which affinity to use. At the moment 'precomputed' and 'euclidean' are
        supported. 'euclidean' uses the negative squared euclidean distance between
        points
    """
    
    def __init__(self, damping: float = 0.5, max_iter: int = 200,
                 convergence_iter: int = 15, preference: Optional[float] = None,
                 affinity: str = 'euclidean', **kwargs):
        super().__init__(n_clusters=None, **kwargs)  # AffinityPropagation doesn't require n_clusters
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.preference = preference
        self.affinity = affinity
        self.model = AffinityPropagation(
            damping=damping,
            max_iter=max_iter,
            convergence_iter=convergence_iter,
            preference=preference,
            affinity=affinity
        )
    
    def fit(self, X: np.ndarray) -> 'AffinityPropagationClustering':
        """
        Fit the affinity propagation clustering model
        
        Args:
            X : np.ndarray
                Training data of shape (n_samples, n_features)
                
        Returns:
            self : AffinityPropagationClustering
                Returns the instance itself
        """
        self.model.fit(X)
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
        return self.model.fit_predict(X)
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get parameters for this estimator
        
        Returns:
            params : dict
                Parameter names mapped to their values
        """
        params = super().get_params()
        params.update({
            'damping': self.damping,
            'max_iter': self.max_iter,
            'convergence_iter': self.convergence_iter,
            'preference': self.preference,
            'affinity': self.affinity
        })
        return params
    
    def set_params(self, **params) -> 'AffinityPropagationClustering':
        """
        Set the parameters of this estimator
        
        Args:
            **params : dict
                Estimator parameters
            
        Returns:
            self : AffinityPropagationClustering
                Returns the instance itself
        """
        super().set_params(**params)
        if 'damping' in params:
            self.damping = params['damping']
        if 'max_iter' in params:
            self.max_iter = params['max_iter']
        if 'convergence_iter' in params:
            self.convergence_iter = params['convergence_iter']
        if 'preference' in params:
            self.preference = params['preference']
        if 'affinity' in params:
            self.affinity = params['affinity']
            
        self.model = AffinityPropagation(
            damping=self.damping,
            max_iter=self.max_iter,
            convergence_iter=self.convergence_iter,
            preference=self.preference,
            affinity=self.affinity
        )
        return self 