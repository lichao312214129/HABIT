"""
Naive Bayes Models

Wrapper for sklearn's Naive Bayes models (GaussianNB, MultinomialNB, BernoulliNB)
"""
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from .base import BaseModel
from .factory import ModelFactory

@ModelFactory.register('GaussianNB')
class GaussianNBModel(BaseModel):
    """Wrapper for sklearn's GaussianNB model"""
    
    @property
    def model_type(self) -> str:
        """
        Get the type of the model
        
        Returns:
            str: Model type ('probabilistic' for Naive Bayes)
        """
        return 'probabilistic'
    
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
        self.model = GaussianNB(
            priors=params.get('priors', None),
            var_smoothing=params.get('var_smoothing', 1e-9),
            **{k: v for k, v in params.items() if k not in ['priors', 'var_smoothing']}
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
        
        Note: Naive Bayes does not have inherent feature importance.
        This method returns empty dict.
        
        Returns:
            Dict[str, float]: Empty dict (Naive Bayes doesn't provide feature importance)
        """
        # Naive Bayes doesn't have feature importance
        return {}


@ModelFactory.register('MultinomialNB')
class MultinomialNBModel(BaseModel):
    """Wrapper for sklearn's MultinomialNB model"""
    
    @property
    def model_type(self) -> str:
        """
        Get the type of the model
        
        Returns:
            str: Model type ('probabilistic' for Naive Bayes)
        """
        return 'probabilistic'
    
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
        self.model = MultinomialNB(
            alpha=params.get('alpha', 1.0),
            fit_prior=params.get('fit_prior', True),
            class_prior=params.get('class_prior', None),
            **{k: v for k, v in params.items() if k not in ['alpha', 'fit_prior', 'class_prior']}
        )
        
    def fit(self, X: Union[pd.DataFrame, np.ndarray], 
             y: Union[pd.Series, np.ndarray]) -> None:
        """
        Train the model
        
        Args:
            X: Training features (must be non-negative for MultinomialNB)
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
        
        Note: Naive Bayes does not have inherent feature importance.
        This method returns empty dict.
        
        Returns:
            Dict[str, float]: Empty dict (Naive Bayes doesn't provide feature importance)
        """
        # Naive Bayes doesn't have feature importance
        return {}


@ModelFactory.register('BernoulliNB')
class BernoulliNBModel(BaseModel):
    """Wrapper for sklearn's BernoulliNB model"""
    
    @property
    def model_type(self) -> str:
        """
        Get the type of the model
        
        Returns:
            str: Model type ('probabilistic' for Naive Bayes)
        """
        return 'probabilistic'
    
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
        self.model = BernoulliNB(
            alpha=params.get('alpha', 1.0),
            binarize=params.get('binarize', 0.0),
            fit_prior=params.get('fit_prior', True),
            class_prior=params.get('class_prior', None),
            **{k: v for k, v in params.items() if k not in ['alpha', 'binarize', 'fit_prior', 'class_prior']}
        )
        
    def fit(self, X: Union[pd.DataFrame, np.ndarray], 
             y: Union[pd.Series, np.ndarray]) -> None:
        """
        Train the model
        
        Args:
            X: Training features (will be binarized if binarize parameter is set)
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
        
        Note: Naive Bayes does not have inherent feature importance.
        This method returns empty dict.
        
        Returns:
            Dict[str, float]: Empty dict (Naive Bayes doesn't provide feature importance)
        """
        # Naive Bayes doesn't have feature importance
        return {}

