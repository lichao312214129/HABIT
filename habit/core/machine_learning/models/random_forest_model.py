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
    
    @property
    def model_type(self) -> str:
        """
        Get the type of the model
        
        Returns:
            str: Model type ('tree' for Random Forest)
        """
        return 'tree'
    
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
        # Get parameters from config['params'] if it exists, otherwise from config directly
        config_params = self.config.get('params', self.config)
        
        # Get parameters from config or use defaults
        params = {
            'n_estimators': config_params.get('n_estimators', 100),
            'max_depth': config_params.get('max_depth', None),
            'min_samples_split': config_params.get('min_samples_split', 2),
            'min_samples_leaf': config_params.get('min_samples_leaf', 1),
            'max_features': config_params.get('max_features', 'sqrt'),
            'bootstrap': config_params.get('bootstrap', True),
            'class_weight': config_params.get('class_weight', None),
            'random_state': config_params.get('random_state', 42)
        }
        
        # Add any additional parameters from config_params (excluding 'params' key itself)
        params.update({k: v for k, v in config_params.items() 
                      if k not in params and k != 'params' and not k.startswith('_')})
        
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