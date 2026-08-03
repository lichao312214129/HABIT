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
Model Utilities

Provides utility functions for working with machine learning models
"""
from .base import BaseModel
from .factory import ModelFactory

from habit.utils.log_utils import get_module_logger
logger = get_module_logger(__name__)

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
    return ModelFactory.create(name, config)

def get_available_models():
    """
    Get list of all available models
    
    Returns:
        List[str]: List of model names
    """
    return ModelFactory.available()

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
        logger.warning("Failed to import LogisticRegressionModel")

    try:
        from .svm_model import SVMModel
        models_loaded += 1
    except ImportError:
        logger.warning("Failed to import SVMModel")

    try:
        from .random_forest_model import RandomForestModel
        models_loaded += 1
    except ImportError:
        logger.warning("Failed to import RandomForestModel")

    try:
        from .xgboost_model import XGBoostModel
        models_loaded += 1
    except ImportError:
        logger.warning("Failed to import XGBoostModel")
        
    return models_loaded 