"""
SVM Model

Wrapper for sklearn's LinearSVC model for faster training
"""
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from .base import BaseModel
from .factory import ModelFactory
from scipy.special import expit  # sigmoid function

@ModelFactory.register('SVM')
class SVMModel(BaseModel):
    """Wrapper for sklearn's LinearSVC model with probability calibration"""
    
    @property
    def model_type(self) -> str:
        """
        Get the type of the model
        
        Returns:
            str: Model type ('linear' for Linear SVM)
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
        
        # Create base model with parameters
        self.model = LinearSVC(
            C=params.get('C', 1.0),
            class_weight=params.get('class_weight', None),
            random_state=params.get('random_state', 42),
            max_iter=params.get('max_iter', 1000),
            **{k: v for k, v in params.items() if k not in ['C', 'class_weight', 'random_state', 'max_iter']}
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
        
        # Store classes
        self.classes_ = np.unique(y)
        
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
        Get prediction probabilities using the decision function values
        
        Args:
            X: Features
            
        Returns:
            np.ndarray: Predicted probabilities for each class
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get decision function values
        decision_values = self.model.decision_function(X)
        
        # For binary classification
        if len(self.classes_) == 2:
            # Convert to probabilities using sigmoid function
            proba = expit(decision_values)
            # Return probabilities for both classes
            return np.vstack([1 - proba, proba]).T
        else:
            # For multi-class, use softmax on decision values
            # Subtract max for numerical stability
            exp_decision = np.exp(decision_values - np.max(decision_values, axis=1, keepdims=True))
            return exp_decision / np.sum(exp_decision, axis=1, keepdims=True)
        
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores
        
        Returns:
            Dict[str, float]: Feature importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        try:
            # Get coefficients from the model
            if len(self.classes_) == 2:
                coef = self.model.coef_[0]
            else:
                # For multiclass, average the coefficients across classes
                coef = np.mean(self.model.coef_, axis=0)
            
            # Get feature names
            feature_names = self.feature_names or [f"feature_{i}" for i in range(len(coef))]
            
            # Return as dictionary
            return dict(zip(feature_names, coef))
        except AttributeError:
            # If we can't get coefficients, return empty dict
            return {} 