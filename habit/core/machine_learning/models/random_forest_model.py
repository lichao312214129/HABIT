# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text. Summary:
#
#   - Non-commercial use (academic, research, education, personal) is permitted
#     provided that copyright notices are retained and HABIT usage is
#     acknowledged in publications, reports, or documentation.
#   - Commercial use requires prior written consent from the copyright holder
#     (lichao19870617@163.com) and public acknowledgment of HABIT usage in
#     product documentation or user-facing materials.
#   - Unauthorized commercial use or removal of attribution is prohibited.
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