"""
AdaBoost Model

Wrapper for sklearn's AdaBoostClassifier model
"""
from sklearn.ensemble import AdaBoostClassifier
from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from .base import BaseModel
from .factory import ModelFactory

@ModelFactory.register('AdaBoost')
class AdaBoostModel(BaseModel):
    """Wrapper for sklearn's AdaBoostClassifier model"""
    
    @property
    def model_type(self) -> str:
        """
        Get the type of the model
        
        Returns:
            str: Model type ('ensemble' for AdaBoost)
        """
        return 'ensemble'
    
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
        self.model = AdaBoostClassifier(
            n_estimators=params.get('n_estimators', 50),
            learning_rate=params.get('learning_rate', 1.0),
            algorithm=params.get('algorithm', 'SAMME.R'),
            random_state=params.get('random_state', 42),
            **{k: v for k, v in params.items() if k not in [
                'n_estimators', 'learning_rate', 'algorithm', 'random_state'
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

