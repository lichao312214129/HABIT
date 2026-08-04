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
Base Model

Abstract base class for all models, defining common interface
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

class BaseModel(ClassifierMixin, BaseEstimator, ABC):
    """
    Abstract base class for all models.

    Base order follows the sklearn convention of listing mixins before
    ``BaseEstimator``. Since sklearn 1.6 the estimator type is reported by
    ``__sklearn_tags__``, which ``ClassifierMixin`` overrides; if
    ``BaseEstimator`` came first, its default implementation would win the MRO
    and leave ``estimator_type`` unset, so ``is_classifier()`` would return
    False and sklearn utilities would refuse probability-based scoring
    (``permutation_importance``/``cross_val_score`` with ``roc_auc``,
    ``GridSearchCV``, ``CalibratedClassifierCV``). ``ABC`` stays last; the
    metaclass is still ``ABCMeta``, so ``@abstractmethod`` remains enforced.
    """
    
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
             y: Union[pd.Series, np.ndarray]) -> 'BaseModel':
        """
        Train the model
        
        Args:
            X: Training features
            y: Training labels

        Returns:
            The fitted model instance, following the scikit-learn estimator
            contract required by pipeline composition and cloning.
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
            np.ndarray: Class-probability matrix with shape
            ``(n_samples, n_classes)``.
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

    @property
    def classes_(self) -> np.ndarray:
        """
        Expose classifier classes for sklearn compatibility.

        Why this is needed:
        - sklearn-compatible wrappers may need `estimator.classes_` after fitting.
        - HABIT wraps sklearn estimators with custom model classes, so we forward
          this attribute to the underlying sklearn estimator.

        Returns:
            np.ndarray: Class labels from the fitted underlying estimator.

        Raises:
            AttributeError: If the underlying estimator is not fitted yet.
        """
        if self.model is None or not hasattr(self.model, "classes_"):
            raise AttributeError(
                f"{self.__class__.__name__} does not expose classes_ before fitting."
            )
        return self.model.classes_
        
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