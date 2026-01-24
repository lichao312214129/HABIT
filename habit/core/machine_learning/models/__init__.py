"""
Machine Learning Models Package
This package contains various machine learning model implementations
"""

from .base import BaseModel
from .factory import ModelFactory
from .logistic_regression_model import LogisticRegressionModel
from .random_forest_model import RandomForestModel
from .svm_model import SVMModel
from .xgboost_model import XGBoostModel

_all_models = [
    'BaseModel',
    'ModelFactory',
    'LogisticRegressionModel',
    'RandomForestModel',
    'SVMModel',
    'XGBoostModel',
]

# Handle optional AutoGluon dependency
try:
    from .autogluon_model import AutoGluonTabularModel
    _all_models.append('AutoGluonTabularModel')
except ImportError:
    # autogluon is not installed, which is acceptable
    AutoGluonTabularModel = None

__all__ = _all_models
