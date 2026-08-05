# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Naive Bayes Models

Wrapper for sklearn's Naive Bayes models (GaussianNB, MultinomialNB, BernoulliNB)
"""
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from habit.utils.estimator_utils import build_estimator_params
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
        
        # Create model with parameters. Naive Bayes has no random_state, so the
        # seed injected globally by the pipeline builder is filtered out here.
        self.model = GaussianNB(
            **build_estimator_params(
                GaussianNB,
                defaults={'priors': None, 'var_smoothing': 1e-9},
                user_params=params,
                model_name='GaussianNB',
            )
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
        
        # Create model with parameters. Naive Bayes has no random_state, so the
        # seed injected globally by the pipeline builder is filtered out here.
        self.model = MultinomialNB(
            **build_estimator_params(
                MultinomialNB,
                defaults={'alpha': 1.0, 'fit_prior': True, 'class_prior': None},
                user_params=params,
                model_name='MultinomialNB',
            )
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
        
        # Create model with parameters. Naive Bayes has no random_state, so the
        # seed injected globally by the pipeline builder is filtered out here.
        self.model = BernoulliNB(
            **build_estimator_params(
                BernoulliNB,
                defaults={
                    'alpha': 1.0,
                    'binarize': 0.0,
                    'fit_prior': True,
                    'class_prior': None,
                },
                user_params=params,
                model_name='BernoulliNB',
            )
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

