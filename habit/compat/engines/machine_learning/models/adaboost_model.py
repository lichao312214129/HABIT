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
AdaBoost Model

Wrapper for sklearn's AdaBoostClassifier model
"""
from sklearn.ensemble import AdaBoostClassifier
from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from habit.utils.estimator_utils import build_estimator_params
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

        # ``algorithm`` is intentionally not defaulted here. sklearn removed the
        # long-standing 'SAMME.R' value and is phasing the parameter out
        # entirely, so pinning any value in HABIT would break on upgrade. When
        # the user does not set it, sklearn's own default applies.
        self.model = AdaBoostClassifier(
            **build_estimator_params(
                AdaBoostClassifier,
                defaults={
                    'n_estimators': 50,
                    'learning_rate': 1.0,
                    'random_state': 42,
                },
                user_params=params,
                model_name='AdaBoost',
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

