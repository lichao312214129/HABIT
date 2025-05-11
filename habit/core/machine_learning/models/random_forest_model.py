"""
Random Forest Model

Implementation of Random Forest classifier
"""
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any, Optional, Union, List
import numpy as np
import pandas as pd
from .base import BaseModel
from .factory import ModelFactory

@ModelFactory.register('RandomForest')
class RandomForestModel(BaseModel):
    """
    Random Forest Model implementation
    
    This class implements a Random Forest classifier with configurable parameters
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Random Forest model
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        super().__init__(config)
        self.model = self._create_model()
    
    def _create_model(self) -> RandomForestClassifier:
        """
        Create and configure the Random Forest model
        
        Returns:
            RandomForestClassifier: Configured model instance
        """
        # Get parameters from config or use defaults
        params = {
            'n_estimators': self.config.get('n_estimators', 100),
            'max_depth': self.config.get('max_depth', None),
            'min_samples_split': self.config.get('min_samples_split', 2),
            'min_samples_leaf': self.config.get('min_samples_leaf', 1),
            'max_features': self.config.get('max_features', 'sqrt'),
            'bootstrap': self.config.get('bootstrap', True),
            'class_weight': self.config.get('class_weight', None),
            'random_state': self.config.get('random_state', 42)
        }
        
        # Add any additional parameters from config
        params.update({k: v for k, v in self.config.items() 
                      if k not in params and not k.startswith('_')})
        
        return RandomForestClassifier(**params)
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], 
            y: Union[pd.Series, np.ndarray]) -> None:
        """
        Train the model
        
        Args:
            X: Training features
            y: Training labels
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        self.model.fit(X, y)
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features
            
        Returns:
            np.ndarray: Predicted class labels
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Get prediction probabilities
        
        Args:
            X: Features
            
        Returns:
            np.ndarray: Predicted probabilities for positive class
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict_proba(X)[:, 1]
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores
        
        Returns:
            Dict[str, float]: Feature importance scores
        """
        if self.feature_names is None:
            return {}
        return dict(zip(self.feature_names, self.model.feature_importances_)) 