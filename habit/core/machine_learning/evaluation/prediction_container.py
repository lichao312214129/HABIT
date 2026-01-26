"""
PredictionContainer Module
Encapsulates prediction results and handles binary/multiclass abstraction.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any, Tuple, List

class PredictionContainer:
    """
    A unified container for model predictions.
    Ensures consistency between probabilities and predicted labels.
    """
    def __init__(self, y_true: np.ndarray, y_prob: np.ndarray, y_pred: Optional[np.ndarray] = None):
        self.y_true = np.array(y_true)
        self.y_prob = np.array(y_prob)
        
        # Detect classes
        self.classes = np.unique(self.y_true)
        self.num_classes = len(self.classes)
        
        # Priority: 
        # 1. Provided y_pred (from model.predict())
        # 2. Argmax/Threshold fallback
        if y_pred is not None:
            self.y_pred = np.array(y_pred)
        else:
            self.y_pred = self._generate_default_preds()

    def _generate_default_preds(self) -> np.ndarray:
        """Fallback prediction logic if y_pred is not provided."""
        if self.num_classes == 2:
            return (self.get_binary_probs() >= 0.5).astype(int)
        else:
            return np.argmax(self.y_prob, axis=1)

    def get_binary_probs(self) -> np.ndarray:
        """
        Returns a 1D array of probabilities for positive class.
        """
        if self.y_prob.ndim == 1:
            return self.y_prob
        if self.y_prob.ndim == 2:
            # For multiclass, index 1 is only meaningful if it's actually binary data
            return self.y_prob[:, 1]
        return self.y_prob

    def get_eval_probs(self) -> np.ndarray:
        """
        Returns probabilities optimized for evaluation metrics.
        1D for binary, 2D for multiclass.
        """
        if self.num_classes == 2:
            return self.get_binary_probs()
        return self.y_prob

    def to_dict(self):
        """Converts results to a serializable dictionary."""
        return {
            'y_true': self.y_true.tolist(),
            'y_prob': self.get_eval_probs().tolist(),
            'y_pred': self.y_pred.tolist()
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert prediction data to a DataFrame.
        
        Returns:
            DataFrame with columns: y_true, y_pred_proba, y_pred
        """
        return pd.DataFrame({
            'y_true': self.y_true,
            'y_pred_proba': self.get_eval_probs(),
            'y_pred': self.y_pred
        })
    
    def clean_nan(self) -> 'PredictionContainer':
        """
        Remove rows with NaN values.
        
        Returns:
            New PredictionContainer with cleaned data
        """
        df = self.to_dataframe()
        df = df.dropna()
        
        return PredictionContainer(
            y_true=df['y_true'].values,
            y_prob=df['y_pred_proba'].values,
            y_pred=df['y_pred'].values
        )
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.y_true)


def create_prediction_container(
    y_true: Union[np.ndarray, List],
    y_pred_proba: Union[np.ndarray, List],
    y_pred: Optional[Union[np.ndarray, List]] = None
) -> PredictionContainer:
    """
    Create a PredictionContainer from raw arrays with automatic cleaning.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        y_pred: Optional predicted labels
        
    Returns:
        PredictionContainer instance
    """
    container = PredictionContainer(
        y_true=np.array(y_true),
        y_prob=np.array(y_pred_proba),
        y_pred=np.array(y_pred) if y_pred is not None else None
    )
    
    return container.clean_nan()


def from_tuple(data: Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]) -> PredictionContainer:
    """
    Create a PredictionContainer from a tuple (y_true, y_pred_proba, y_pred).
    
    Args:
        data: Tuple containing (y_true, y_pred_proba, y_pred)
        
    Returns:
        PredictionContainer instance
    """
    if len(data) == 3:
        y_true, y_pred_proba, y_pred = data
    elif len(data) == 2:
        y_true, y_pred_proba = data
        y_pred = None
    else:
        raise ValueError(f"Expected tuple of length 2 or 3, got {len(data)}")
    
    return create_prediction_container(y_true, y_pred_proba, y_pred)


def from_dict(data: Dict[str, Any]) -> PredictionContainer:
    """
    Create a PredictionContainer from a dictionary.
    
    Args:
        data: Dictionary containing prediction data
        
    Returns:
        PredictionContainer instance
    """
    y_true = data.get('y_true')
    y_pred_proba = data.get('y_pred_proba') or data.get('y_prob')
    y_pred = data.get('y_pred')
    
    if y_true is None or y_pred_proba is None:
        raise ValueError("Dictionary must contain 'y_true' and 'y_pred_proba' (or 'y_prob') keys")
    
    return create_prediction_container(y_true, y_pred_proba, y_pred)


def convert_models_data_to_containers(
    models_data: Dict[str, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]
) -> Dict[str, PredictionContainer]:
    """
    Convert a dictionary of model data tuples to PredictionContainers.
    
    Args:
        models_data: Dictionary mapping model names to (y_true, y_pred_proba, y_pred) tuples
        
    Returns:
        Dictionary mapping model names to PredictionContainer instances
    """
    containers = {}
    for model_name, data_tuple in models_data.items():
        try:
            containers[model_name] = from_tuple(data_tuple)
        except Exception as e:
            raise ValueError(f"Failed to create container for model '{model_name}': {str(e)}")
    
    return containers


def convert_containers_to_models_data(
    containers: Dict[str, PredictionContainer]
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Convert a dictionary of PredictionContainers back to model data tuples.
    
    Args:
        containers: Dictionary mapping model names to PredictionContainer instances
        
    Returns:
        Dictionary mapping model names to (y_true, y_pred_proba, y_pred) tuples
    """
    models_data = {}
    for model_name, container in containers.items():
        models_data[model_name] = (
            container.y_true,
            container.get_eval_probs(),
            container.y_pred
        )
    
    return models_data