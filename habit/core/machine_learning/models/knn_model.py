"""
K-Nearest Neighbors Model

Wrapper for sklearn's KNeighborsClassifier model
"""
from sklearn.neighbors import KNeighborsClassifier
from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from .base import BaseModel
from .factory import ModelFactory

@ModelFactory.register('KNN')
class KNNModel(BaseModel):
    """Wrapper for sklearn's KNeighborsClassifier model"""
    
    @property
    def model_type(self) -> str:
        """
        Get the type of the model
        
        Returns:
            str: Model type ('distance-based' for KNN)
        """
        return 'distance-based'
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model
        
        Args:
            config: Configuration dictionary with model parameters
        """
        super().__init__(config)
        
        # Extract parameters from config
        params = config.get('params', {})
        
        # Create model with parameters
        self.model = KNeighborsClassifier(
            n_neighbors=params.get('n_neighbors', 5),
            weights=params.get('weights', 'uniform'),
            algorithm=params.get('algorithm', 'auto'),
            leaf_size=params.get('leaf_size', 30),
            p=params.get('p', 2),
            metric=params.get('metric', 'minkowski'),
            n_jobs=params.get('n_jobs', -1),
            **{k: v for k, v in params.items() if k not in ['n_neighbors', 'weights', 'algorithm', 'leaf_size', 'p', 'metric', 'n_jobs']}
        )
        
    def fit(self, X: Union[pd.DataFrame, np.ndarray], 
             y: Union[pd.Series, np.ndarray]) -> None:
        """
        Train the model
        
        Args:
            X: Training features
            y: Training labels
        """
        # Save feature names if X is a DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
        
        # Train the model
        self.model.fit(X, y)
        
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features
            
        Returns:
            np.ndarray: Predicted class labels
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict(X)
        
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Get prediction probabilities
        
        Args:
            X: Features
            
        Returns:
            np.ndarray: Predicted probabilities for positive class
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict_proba(X)
        
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores
        
        Note: KNN does not have inherent feature importance.
        This method returns empty dict or can use permutation importance.
        
        Returns:
            Dict[str, float]: Empty dict (KNN doesn't provide feature importance)
        """
        # KNN doesn't have feature importance
        # Could implement permutation importance here if needed
        return {}

