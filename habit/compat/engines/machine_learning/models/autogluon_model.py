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
AutoGluon TabularPredictor Model

Wrapper for AutoGluon's TabularPredictor model
"""
from typing import Any, Dict, FrozenSet, List, Optional, Tuple, Union
import inspect
import os
import random

import numpy as np
import pandas as pd

from habit.utils.estimator_utils import get_accepted_params
from habit.utils.log_utils import get_module_logger
from .base import BaseModel
from .factory import ModelFactory

logger = get_module_logger(__name__)

try:
    from autogluon.tabular import TabularPredictor
except ImportError:
    raise ImportError(
        "AutoML support is not installed. Windows lightweight-release users "
        "can run 'launchers/一键启用HABIT-AutoML.bat'; package users can install "
        "'HABIT[automl]'."
    )

# HABIT-level keys that configure the wrapper rather than AutoGluon itself.
_HABIT_ONLY_PARAMS = frozenset({'feature_importance', 'random_state', 'seed'})


def _split_autogluon_params(
    params: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Split configured parameters across AutoGluon's two-part API.

    ``TabularPredictor(...)`` defines the task while ``.fit(...)`` controls
    training, and both accept ``**kwargs``. Explicit ``predictor:`` / ``fit:``
    blocks are used as-is. Any remaining flat key is routed by looking at which
    side names it explicitly; keys named by neither go to ``fit``, which is
    where AutoGluon's advanced options (``num_bag_folds``,
    ``excluded_model_types``, ...) live.

    Args:
        params: Validated ``params`` block for the model.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: Keyword arguments for the
        constructor and for ``fit`` respectively.
    """
    predictor_params: Dict[str, Any] = dict(params.get('predictor') or {})
    fit_params: Dict[str, Any] = dict(params.get('fit') or {})

    predictor_names, _ = get_accepted_params(TabularPredictor)
    fit_names = _fit_param_names()

    for key, value in params.items():
        if key in ('predictor', 'fit') or key in _HABIT_ONLY_PARAMS:
            continue
        # An explicit block always wins over the flat spelling of the same key.
        if key in predictor_params or key in fit_params:
            continue
        if key in predictor_names:
            predictor_params[key] = value
        elif key in fit_names:
            fit_params[key] = value
        else:
            fit_params[key] = value

    return predictor_params, fit_params


def _fit_param_names() -> FrozenSet[str]:
    """
    Return the parameter names ``TabularPredictor.fit`` names explicitly.

    Returns:
        FrozenSet[str]: Accepted names excluding the training data argument.
    """
    try:
        signature = inspect.signature(TabularPredictor.fit)
    except (TypeError, ValueError):
        return frozenset()
    return frozenset(
        name
        for name, parameter in signature.parameters.items()
        if name not in ('self', 'train_data')
        and parameter.kind
        not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    )


@ModelFactory.register('AutoGluonTabular')
class AutoGluonTabularModel(BaseModel):
    """Wrapper for AutoGluon's TabularPredictor model"""
    
    @property
    def model_type(self) -> str:
        """
        Get the type of the model
        
        Returns:
            str: Model type ('ensemble' for AutoGluon)
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
        params = config.get('params', config)

        # HABIT-level parameters, never forwarded to AutoGluon.
        self.feature_importance = params.get('feature_importance', 'auto')
        self.random_state = int(
            params.get(
                'random_state',
                params.get('seed', config.get('random_state', 42)),
            )
        )

        # Split the remaining parameters across AutoGluon's two-part API.
        self.predictor_params, self.fit_params = _split_autogluon_params(params)

        # ``label`` is resolved from the training target when not configured.
        self.label = self.predictor_params.get('label')

        # Create TabularPredictor instance
        self.model = None  # Will be initialized during fit
        
    def fit(self, X: Union[pd.DataFrame, np.ndarray], 
            y: Union[pd.Series, np.ndarray]) -> None:
        """
        Train the model
        
        Args:
            X: Training features
            y: Training labels
        """
        # Convert numpy arrays to pandas if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        
        if isinstance(y, np.ndarray):
            y = pd.Series(y, name=self.label or 'target')
        
        # Save feature names
        self.feature_names = list(X.columns)
        
        # Combine features and target into a single dataframe
        train_data = X.copy()
        if self.label is None:
            self.label = y.name if hasattr(y, 'name') and y.name else 'target'
        train_data[self.label] = y

        predictor_params = dict(self.predictor_params)
        predictor_params['label'] = self.label

        # Initialize and train TabularPredictor
        self.model = TabularPredictor(**predictor_params)

        # AutoGluon TabularPredictor.fit() does not accept random_state (v1.3+).
        # Seed Python/NumPy before fit so config random_state still improves
        # reproducibility for libraries that honor global RNG state.
        random.seed(self.random_state)
        np.random.seed(self.random_state)

        # Train the model
        self.model.fit(train_data=train_data, **self.fit_params)

        # Save the leaderboard after model training for later analysis. The
        # leaderboard summarizes every model AutoGluon trained, with metrics.
        # When no path was configured, AutoGluon generates one, so read it back
        # off the fitted predictor.
        output_dir = (
            getattr(self.model, 'path', None)
            or self.predictor_params.get('path')
            or "./"
        )
        leaderboard_path = os.path.join(output_dir, "leaderboard.csv")
        leaderboard_df = self.model.leaderboard(silent=True)
        print(leaderboard_df)
        leaderboard_df.to_csv(leaderboard_path, index=False)
        
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
            
        # Convert numpy array to pandas if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names or [f"feature_{i}" for i in range(X.shape[1])])
            
        return self.model.predict(X).values
        
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Get prediction probabilities
        
        Args:
            X: Features
            
        Returns:
            np.ndarray: Predicted probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
            
        # Convert numpy array to pandas if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names or [f"feature_{i}" for i in range(X.shape[1])])
            
        try:
            # For classification problems
            return self.model.predict_proba(X).values
        except:
            # For regression problems where predict_proba isn't available
            return self.model.predict(X).values.reshape(-1, 1)
        
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores
        
        Returns:
            Dict[str, float]: Feature importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
            
        try:
            # Try to get feature importance
            importance_df = self.model.feature_importance(self.feature_importance)
            
            # Return as dictionary
            return dict(zip(importance_df.index, importance_df['importance'].values))
        except Exception as e:
            logger.error(f"Failed to get feature importance: {e}")
            return {}
    
    def save(self, filepath: str) -> None:
        """
        Save model to file (override base save method)
        
        Args:
            filepath: Directory path to save model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
            
        # AutoGluon has its own saving mechanism
        os.makedirs(filepath, exist_ok=True)
        self.model.save(filepath)
        
    @classmethod
    def load(cls, filepath: str, config: Dict[str, Any] = None) -> 'AutoGluonTabularModel':
        """
        Load model from file (override base load method)
        
        Args:
            filepath: Directory path to load model from
            config: Configuration dictionary
            
        Returns:
            AutoGluonTabularModel: Loaded model
        """
        # Create a new instance
        config = config or {}
        instance = cls(config)
        
        # Load the AutoGluon model
        instance.model = TabularPredictor.load(filepath)
        return instance 