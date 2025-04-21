"""
Random Forest Model

Provides a factory function for creating Random Forest models
"""
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any, Optional, Union
from .factory import ModelFactory

@ModelFactory.register('RandomForest')
def random_forest_model(
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        max_features: Optional[Union[str, int, float]] = 'sqrt',
        bootstrap: bool = True,
        class_weight: Optional[Union[Dict, str]] = None,
        random_state: int = 42,
        **kwargs
    ) -> RandomForestClassifier:
    """
    Create a Random Forest model
    
    Args:
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of trees
        min_samples_split: Minimum samples required to split a node
        min_samples_leaf: Minimum samples required at a leaf node
        max_features: Number of features to consider for best split
        bootstrap: Use bootstrap samples
        class_weight: Weights for classes (None, 'balanced', 'balanced_subsample', or dict)
        random_state: Random seed for reproducibility
        **kwargs: Additional parameters passed to RandomForestClassifier
        
    Returns:
        RandomForestClassifier: Configured model instance
    """
    # Create parameter dictionary with default values
    params = {
        'n_estimators': n_estimators,
        'random_state': random_state,
        'bootstrap': bootstrap
    }
    
    # Add optional parameters if specified
    if max_depth is not None:
        params['max_depth'] = max_depth
    
    if min_samples_split != 2:
        params['min_samples_split'] = min_samples_split
        
    if min_samples_leaf != 1:
        params['min_samples_leaf'] = min_samples_leaf
    
    if max_features is not None:
        params['max_features'] = max_features
    
    # Add class_weight if specified
    if class_weight is not None:
        params['class_weight'] = class_weight
        
    # Add any additional parameters
    params.update(kwargs)
    
    # Create and return model
    return RandomForestClassifier(**params) 