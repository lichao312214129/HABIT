"""
Logistic Regression Model

Wrapper for sklearn's LogisticRegression model
"""
from sklearn.linear_model import LogisticRegression
from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from .base import BaseModel
from .factory import ModelFactory

@ModelFactory.register('LogisticRegression')
class LogisticRegressionModel(BaseModel):
    """Wrapper for sklearn's LogisticRegression model"""
    
    @property
    def model_type(self) -> str:
        """
        Get the type of the model
        
        Returns:
            str: Model type ('linear' for Logistic Regression)
        """
        return 'linear'
    
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
        self.model = LogisticRegression(
            C=params.get('C', 1.0),
            penalty=params.get('penalty', 'l2'),
            solver=params.get('solver', 'liblinear'),
            max_iter=params.get('max_iter', 1000),
            random_state=params.get('random_state', 42),
            class_weight=params.get('class_weight', None),
            **{k: v for k, v in params.items() if k not in ['C', 'penalty', 'solver', 'max_iter', 'random_state', 'class_weight']}
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
            raise ValueError("Model not trained. Call train() first.")
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
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)
        
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores
        
        Returns:
            Dict[str, float]: Feature importance scores (coefficient values)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        if not hasattr(self.model, 'coef_'):
            return {}
            
        # Get feature names
        feature_names = self.feature_names or [f"feature_{i}" for i in range(self.model.coef_.shape[1])]
        
        # Get coefficients
        coef = self.model.coef_[0]
        
        # Return as dictionary
        return dict(zip(feature_names, coef)) 