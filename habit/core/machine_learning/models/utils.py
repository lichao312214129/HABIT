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