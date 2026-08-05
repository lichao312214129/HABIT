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
Random Forest Model

Implementation of Random Forest classifier
"""
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any, Optional, Union, List
import numpy as np
import pandas as pd
from habit.utils.estimator_utils import build_estimator_params
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

        # Internal bookkeeping keys must never reach the estimator.
        user_params = {
            key: value
            for key, value in config_params.items()
            if key != 'params' and not key.startswith('_')
        }

        return RandomForestClassifier(
            **build_estimator_params(
                RandomForestClassifier,
                defaults={
                    'n_estimators': 100,
                    'max_depth': None,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'max_features': 'sqrt',
                    'bootstrap': True,
                    'class_weight': None,
                    'random_state': 42,
                },
                user_params=user_params,
                model_name='RandomForest',
            )
        )
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], 
            y: Union[pd.Series, np.ndarray]) -> 'RandomForestModel':
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
        return self
    
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
            np.ndarray: Class-probability matrix with shape
            ``(n_samples, n_classes)``.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores
        
        Returns:
            Dict[str, float]: Feature importance scores
        """
        if self.feature_names is None:
            return {}
        return dict(zip(self.feature_names, self.model.feature_importances_)) 