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
Custom Model Example

This file demonstrates how to create and register a custom model
"""
from sklearn.linear_model import LogisticRegression
from typing import Dict, Any, Optional, Union, List
import numpy as np
import pandas as pd
from .base import BaseModel
from .factory import ModelFactory

@ModelFactory.register('CustomEnsemble')
class CustomEnsembleClassifier(BaseModel):
    """
    A custom ensemble classifier that combines multiple base models
    """
    def __init__(self, 
                 base_models: List,
                 weights: Optional[List[float]] = None,
                 voting: str = 'soft'):
        """
        Initialize custom ensemble
        
        Args:
            base_models: List of base estimator objects
            weights: Model weights for voting
            voting: 'hard' or 'soft' voting
        """
        self.base_models = base_models
        self.weights = weights
        self.voting = voting
        self._fitted = False
        
    def fit(self, X, y):
        """
        Fit all base models
        
        Args:
            X: Training data
            y: Target values
        """
        # Fit each base model
        for model in self.base_models:
            model.fit(X, y)
        
        self._fitted = True
        return self
        
    def predict(self, X):
        """
        Predict using voting
        
        Args:
            X: Test data
        """
        if not self._fitted:
            raise ValueError("Model not fitted yet")
            
        if self.voting == 'hard':
            # Hard voting: majority rule
            predictions = np.asarray([model.predict(X) for model in self.base_models])
            maj = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, 
                                       weights=self.weights)), 
                                       axis=0, 
                                       arr=predictions.astype('int'))
            return maj
        else:
            # Soft voting: weighted probabilities
            return self.predict_proba(X)[:, 1] >= 0.5
            
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Args:
            X: Test data
        """
        if not self._fitted:
            raise ValueError("Model not fitted yet")
            
        # Get predicted probabilities from each model
        probas = np.asarray([model.predict_proba(X) for model in self.base_models])
        
        # Apply weights if specified
        if self.weights is not None:
            probas = np.average(probas, axis=0, weights=self.weights)
        else:
            probas = np.average(probas, axis=0)
            
        return probas

