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
K-Nearest Neighbors Model

Wrapper for sklearn's KNeighborsClassifier model
"""
from sklearn.neighbors import KNeighborsClassifier
from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from habit.utils.estimator_utils import build_estimator_params
from .base import BaseModel
from .factory import ModelFactory

@ModelFactory.register('KNN')
class KNNModel(BaseModel):
    """Wrapper for sklearn's KNeighborsClassifier model"""
    
    @property
    def model_type(self) -> str:
        """
        Get the type of the model
        
        Returns:
            str: Model type ('distance-based' for KNN)
        """
        return 'distance-based'
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model
        
        Args:
            config: Configuration dictionary with model parameters
        """
        super().__init__(config)
        
        # Extract parameters from config
        params = config.get('params', {})

        # Create model with parameters. KNN has no random_state, so the seed
        # injected globally by the pipeline builder is filtered out here.
        self.model = KNeighborsClassifier(
            **build_estimator_params(
                KNeighborsClassifier,
                defaults={
                    'n_neighbors': 5,
                    'weights': 'uniform',
                    'algorithm': 'auto',
                    'leaf_size': 30,
                    'p': 2,
                    'metric': 'minkowski',
                    'n_jobs': -1,
                },
                user_params=params,
                model_name='KNN',
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
        
        Note: KNN does not have inherent feature importance.
        This method returns empty dict or can use permutation importance.
        
        Returns:
            Dict[str, float]: Empty dict (KNN doesn't provide feature importance)
        """
        # KNN doesn't have feature importance
        # Could implement permutation importance here if needed
        return {}

