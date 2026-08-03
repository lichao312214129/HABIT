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
SVM Models

Two registry entries with different trade-offs:

* ``SVM`` — ``LinearSVC``, fast, linear decision boundary only. Probabilities
  are approximated from the decision function.
* ``SVC`` — kernel SVC (rbf/poly/sigmoid/linear) with native probability
  estimates, slower but able to model non-linear boundaries.
"""
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from habit.utils.estimator_utils import build_estimator_params
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
            **build_estimator_params(
                LinearSVC,
                defaults={
                    'C': 1.0,
                    'class_weight': None,
                    'random_state': 42,
                    'max_iter': 1000,
                },
                user_params=params,
                model_name='SVM',
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
        # ``classes_`` comes from ``BaseModel`` forwarding to ``self.model`` — do not assign here.
        
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


@ModelFactory.register('SVC')
class SVCModel(BaseModel):
    """
    Wrapper for sklearn's kernel SVC.

    Use this instead of ``SVM`` (which is a ``LinearSVC``) when a non-linear
    kernel is needed. ``probability=True`` is the default because HABIT relies
    on ``predict_proba`` for ROC/AUC reporting; note that it makes training
    noticeably slower, since sklearn fits an internal calibration model.
    """

    @property
    def model_type(self) -> str:
        """
        Get the type of the model

        Returns:
            str: Model type ('kernel' for kernel SVC)
        """
        return 'kernel'

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model

        Args:
            config: Configuration dictionary with model parameters
        """
        super().__init__(config)

        params = config.get('params', {})

        self.model = SVC(
            **build_estimator_params(
                SVC,
                defaults={
                    'C': 1.0,
                    'kernel': 'rbf',
                    'gamma': 'scale',
                    'class_weight': None,
                    'probability': True,
                    'random_state': 42,
                },
                user_params=params,
                model_name='SVC',
            )
        )

    def fit(self, X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, np.ndarray]) -> 'SVCModel':
        """
        Train the model

        Args:
            X: Training features
            y: Training labels

        Returns:
            SVCModel: The fitted model instance.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
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
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            X: Features

        Returns:
            np.ndarray: Class-probability matrix with shape
            ``(n_samples, n_classes)``.

        Raises:
            ValueError: If the model was configured with ``probability=False``,
                in which case sklearn cannot produce probabilities.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        if not getattr(self.model, 'probability', False):
            raise ValueError(
                "SVC was configured with probability=False, so predict_proba is "
                "unavailable. Set probability: true in the model params."
            )
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.

        Only a linear kernel exposes coefficients; other kernels have no direct
        per-feature importance.

        Returns:
            Dict[str, float]: Feature importance scores, empty for non-linear
            kernels.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        try:
            coef = self.model.coef_
        except AttributeError:
            # Non-linear kernels do not expose coef_.
            return {}

        weights = coef[0] if len(self.classes_) == 2 else np.mean(coef, axis=0)
        feature_names = self.feature_names or [
            f"feature_{i}" for i in range(len(weights))
        ]
        return dict(zip(feature_names, weights))