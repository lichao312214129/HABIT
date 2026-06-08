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
