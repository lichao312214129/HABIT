"""
SVM Model

Wrapper for sklearn's SVC model
"""
from sklearn.svm import SVC
from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from .base import BaseModel
from .factory import ModelFactory

@ModelFactory.register('SVM')
class SVMModel(BaseModel):
    """Wrapper for sklearn's SVC model"""
    
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
        self.model = SVC(
            C=params.get('C', 1.0),
            kernel=params.get('kernel', 'rbf'),
            degree=params.get('degree', 3),
            gamma=params.get('gamma', 'scale'),
            probability=params.get('probability', True),
            random_state=params.get('random_state', 42),
            class_weight=params.get('class_weight', None),
            **{k: v for k, v in params.items() if k not in ['C', 'kernel', 'degree', 'gamma', 'probability', 'random_state', 'class_weight']}
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
            
        # Check if probability=True was set during initialization
        if not hasattr(self.model, 'predict_proba'):
            # Use decision function and convert to probability
            decision_values = self.model.decision_function(X)
            return 1 / (1 + np.exp(-decision_values))
            
        return self.model.predict_proba(X)
        
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores
        
        Returns:
            Dict[str, float]: Feature importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        # For linear SVM, coefficients can be used as feature importance
        if hasattr(self.model, 'coef_') and self.model.kernel == 'linear':
            # Get feature names
            feature_names = self.feature_names or [f"feature_{i}" for i in range(self.model.coef_.shape[1])]
            
            # Get coefficients
            coef = self.model.coef_[0]
            
            # Return as dictionary
            return dict(zip(feature_names, coef))
        
        # For non-linear SVM, there's no direct feature importance
        # We could implement permutation importance or other methods here
        return {} 