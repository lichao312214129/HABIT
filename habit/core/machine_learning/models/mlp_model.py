"""
Multi-layer Perceptron Model

Wrapper for sklearn's MLPClassifier model
"""
from sklearn.neural_network import MLPClassifier
from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from .base import BaseModel
from .factory import ModelFactory

@ModelFactory.register('MLP')
class MLPModel(BaseModel):
    """Wrapper for sklearn's MLPClassifier model"""
    
    @property
    def model_type(self) -> str:
        """
        Get the type of the model
        
        Returns:
            str: Model type ('neural-network' for MLP)
        """
        return 'neural-network'
    
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
        self.model = MLPClassifier(
            hidden_layer_sizes=params.get('hidden_layer_sizes', (100,)),
            activation=params.get('activation', 'relu'),
            solver=params.get('solver', 'adam'),
            alpha=params.get('alpha', 0.0001),
            batch_size=params.get('batch_size', 'auto'),
            learning_rate=params.get('learning_rate', 'constant'),
            learning_rate_init=params.get('learning_rate_init', 0.001),
            max_iter=params.get('max_iter', 200),
            shuffle=params.get('shuffle', True),
            random_state=params.get('random_state', 42),
            early_stopping=params.get('early_stopping', False),
            validation_fraction=params.get('validation_fraction', 0.1),
            **{k: v for k, v in params.items() if k not in [
                'hidden_layer_sizes', 'activation', 'solver', 'alpha', 'batch_size',
                'learning_rate', 'learning_rate_init', 'max_iter', 'shuffle', 
                'random_state', 'early_stopping', 'validation_fraction'
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
        
        Note: MLP does not have inherent feature importance like tree-based models.
        This method returns empty dict or can use weight-based importance.
        
        Returns:
            Dict[str, float]: Empty dict (MLP doesn't provide direct feature importance)
        """
        # MLP doesn't have direct feature importance
        # Could implement connection weight analysis here if needed
        return {}

