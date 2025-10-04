"""
AutoGluon TabularPredictor Model

Wrapper for AutoGluon's TabularPredictor model
"""
from typing import Dict, Any, Optional, Union, List
import numpy as np
import pandas as pd
import os
from .base import BaseModel
from .factory import ModelFactory

try:
    from autogluon.tabular import TabularPredictor
except ImportError:
    raise ImportError("AutoGluon is not installed. Install it using: pip install autogluon")

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
        params = config.get('params', {})
        
        # Extract predictor specific parameters
        self.label = params.get('label', None)
        self.problem_type = params.get('problem_type', None)  # 'binary', 'multiclass', 'regression'
        self.eval_metric = params.get('eval_metric', None)
        self.path = params.get('path', './AutogluonModels')
        
        # Model training parameters
        self.time_limit = params.get('time_limit', 60)  # Time limit in seconds
        self.presets = params.get('presets', 'medium_quality')  # quality/performance tradeoff
        self.hyperparameters = params.get('hyperparameters', None)
        self.feature_importance = params.get('feature_importance', 'auto')
        
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
        
        # Initialize and train TabularPredictor
        self.model = TabularPredictor(
            label=self.label,
            problem_type=self.problem_type,
            eval_metric=self.eval_metric,
            path=self.path
        )
        
        # Train the model
        self.model.fit(
            train_data=train_data,
            time_limit=self.time_limit,
            presets=self.presets,
            hyperparameters=self.hyperparameters
        )

        # Save the leaderboard after model training for later analysis
        # The leaderboard provides a summary of all models trained by AutoGluon, including their performance metrics
        leaderboard_path = os.path.join(self.path if self.path else "./", "leaderboard.csv")
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
            print(f"Failed to get feature importance: {e}")
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