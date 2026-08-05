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
Multi-layer Perceptron Model

Wrapper for sklearn's MLPClassifier model
"""
from sklearn.neural_network import MLPClassifier
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from habit.utils.estimator_utils import build_estimator_params
from .base import BaseModel
from .factory import ModelFactory


def _parse_hidden_layer_sizes(
    value: Union[int, str, Sequence[int], None],
) -> Tuple[int, ...]:
    """
    Normalize ``hidden_layer_sizes`` into a form ``MLPClassifier`` accepts.

    YAML users spell this parameter in several natural ways, while sklearn only
    accepts an int or an array-like of ints. All accepted spellings are mapped
    to a tuple of layer widths:

    * ``None`` -> ``(100,)`` (single hidden layer, the sklearn default)
    * ``100`` -> ``(100,)``
    * ``[100, 50]`` -> ``(100, 50)``
    * ``"100,50"`` or ``"(100, 50)"`` -> ``(100, 50)``

    Args:
        value: Raw value taken from the YAML ``params`` block.

    Returns:
        Tuple[int, ...]: Width of each hidden layer.

    Raises:
        ValueError: When the value cannot be read as one or more layer widths.
    """
    if value is None:
        return (100,)
    if isinstance(value, bool):
        raise ValueError(f"hidden_layer_sizes must be int or list of int, got {value!r}")
    if isinstance(value, int):
        return (value,)
    if isinstance(value, str):
        tokens = [token for token in value.strip(" ()[]").split(',') if token.strip()]
        if not tokens:
            raise ValueError("hidden_layer_sizes is empty")
        try:
            return tuple(int(token) for token in tokens)
        except ValueError as exc:
            raise ValueError(
                f"hidden_layer_sizes must contain integers, got {value!r}"
            ) from exc
    if isinstance(value, Sequence):
        if not value:
            raise ValueError("hidden_layer_sizes is empty")
        return tuple(int(item) for item in value)
    raise ValueError(f"Unsupported hidden_layer_sizes value: {value!r}")

@ModelFactory.register('MLP')
class MLPModel(BaseModel):
    """Wrapper for sklearn's MLPClassifier model"""
    
    @property
    def model_type(self) -> str:
        """
        Get the type of the model
        
        Returns:
            str: Model type ('neural-network' for MLP)
        """
        return 'neural-network'
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model
        
        Args:
            config: Configuration dictionary with model parameters
        """
        super().__init__(config)
        
        # Extract parameters from config
        params = dict(config.get('params', {}))

        # YAML allows several spellings for the layer widths; sklearn accepts
        # only an int or an array-like of ints.
        params['hidden_layer_sizes'] = _parse_hidden_layer_sizes(
            params.get('hidden_layer_sizes')
        )

        # Create model with parameters
        self.model = MLPClassifier(
            **build_estimator_params(
                MLPClassifier,
                defaults={
                    'activation': 'relu',
                    'solver': 'adam',
                    'alpha': 0.0001,
                    'batch_size': 'auto',
                    'learning_rate': 'constant',
                    'learning_rate_init': 0.001,
                    'max_iter': 200,
                    'shuffle': True,
                    'random_state': 42,
                    'early_stopping': False,
                    'validation_fraction': 0.1,
                },
                user_params=params,
                model_name='MLP',
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
        
        Note: MLP does not have inherent feature importance like tree-based models.
        This method returns empty dict or can use weight-based importance.
        
        Returns:
            Dict[str, float]: Empty dict (MLP doesn't provide direct feature importance)
        """
        # MLP doesn't have direct feature importance
        # Could implement connection weight analysis here if needed
        return {}

