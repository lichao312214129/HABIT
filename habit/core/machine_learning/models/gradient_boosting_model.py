"""
Gradient Boosting Model

Wrapper for sklearn's GradientBoostingClassifier model
"""
from sklearn.ensemble import GradientBoostingClassifier
from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from .base import BaseModel
from .factory import ModelFactory

@ModelFactory.register('GradientBoosting')
class GradientBoostingModel(BaseModel):
    """Wrapper for sklearn's GradientBoostingClassifier model"""
    
    @property
    def model_type(self) -> str:
        """
        Get the type of the model
        
        Returns:
            str: Model type ('tree' for Gradient Boosting)
        """
        return 'tree'
    
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
        self.model = GradientBoostingClassifier(
            loss=params.get('loss', 'log_loss'),
            learning_rate=params.get('learning_rate', 0.1),
            n_estimators=params.get('n_estimators', 100),
            subsample=params.get('subsample', 1.0),
            criterion=params.get('criterion', 'friedman_mse'),
            min_samples_split=params.get('min_samples_split', 2),
            min_samples_leaf=params.get('min_samples_leaf', 1),
            max_depth=params.get('max_depth', 3),
            max_features=params.get('max_features', None),
            random_state=params.get('random_state', 42),
            **{k: v for k, v in params.items() if k not in [
                'loss', 'learning_rate', 'n_estimators', 'subsample', 'criterion',
                'min_samples_split', 'min_samples_leaf', 'max_depth', 'max_features', 'random_state'
            ]}
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
        
        Returns:
            Dict[str, float]: Feature importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
            
        if not hasattr(self.model, 'feature_importances_'):
            return {}
            
        # Get feature names
        feature_names = self.feature_names or [f"feature_{i}" for i in range(len(self.model.feature_importances_))]
        
        # Return as dictionary
        return dict(zip(feature_names, self.model.feature_importances_))

