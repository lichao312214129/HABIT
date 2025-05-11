"""
Model Utilities

Provides utility functions for working with machine learning models
"""
from .base import BaseModel
from .factory import ModelFactory

# Export functions for model management
def create_model(name, **kwargs):
    """
    Create a model instance
    
    Args:
        name (str): Model name
        **kwargs: Additional arguments passed to the model constructor
        
    Returns:
        BaseModel: Model instance
    """
    config = {'params': kwargs} if kwargs else {}
    return ModelFactory.create_model(name, config)

def get_available_models():
    """
    Get list of all available models
    
    Returns:
        List[str]: List of model names
    """
    return ModelFactory.get_available_models()

# Import all model classes to ensure they're registered
def register_all_models():
    """
    Register all available model classes
    
    This function attempts to import all known model classes
    to ensure they are registered with the factory
    """
    models_loaded = 0
    
    try:
        from .logistic_regression_model import LogisticRegressionModel
        models_loaded += 1
    except ImportError:
        print("Warning: Failed to import LogisticRegressionModel")

    try:
        from .svm_model import SVMModel
        models_loaded += 1
    except ImportError:
        print("Warning: Failed to import SVMModel")

    try:
        from .random_forest_model import RandomForestModel
        models_loaded += 1
    except ImportError:
        print("Warning: Failed to import RandomForestModel")

    try:
        from .xgboost_model import XGBoostModel
        models_loaded += 1
    except ImportError:
        print("Warning: Failed to import XGBoostModel")
        
    return models_loaded 