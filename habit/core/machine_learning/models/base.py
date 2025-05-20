"""
Base Model

Abstract base class for all models, defining common interface
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd

class BaseModel(ABC):
    """Abstract base class for all models"""
    
    @property
    @abstractmethod
    def model_type(self) -> str:
        """
        Get the type of the model
        
        Returns:
            str: Model type ('tree', 'linear', or other)
        """
        pass
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model with configuration
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config
        self.model = None
        self.feature_names = None
        
    @abstractmethod
    def fit(self, X: Union[pd.DataFrame, np.ndarray], 
             y: Union[pd.Series, np.ndarray]) -> None:
        """
        Train the model
        
        Args:
            X: Training features
            y: Training labels
        """
        pass
        
    @abstractmethod
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features
            
        Returns:
            np.ndarray: Predicted class labels
        """
        pass
        
    @abstractmethod
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Get prediction probabilities
        
        Args:
            X: Features
            
        Returns:
            np.ndarray: Predicted probabilities for positive class
        """
        pass
        
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores
        
        Returns:
            Dict[str, float]: Feature importance scores
        """
        pass
        
    def get_model(self):
        """
        Get the underlying model instance
        
        Returns:
            The underlying model object (e.g., sklearn estimator)
        """
        return self.model
        
    def save(self, filepath: str) -> None:
        """
        Save model to file
        
        Args:
            filepath: Path to save model
        """
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
            
    @classmethod
    def load(cls, filepath: str) -> 'BaseModel':
        """
        Load model from file
        
        Args:
            filepath: Path to load model from
            
        Returns:
            BaseModel: Loaded model
        """
        import pickle
        with open(filepath, 'rb') as f:
            return pickle.load(f) 