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
Machine Learning Models Package
This package contains various machine learning model implementations
"""

from .base import BaseModel
from .factory import ModelFactory
from .logistic_regression_model import LogisticRegressionModel
from .random_forest_model import RandomForestModel
from .svm_model import SVMModel

_all_models = [
    'BaseModel',
    'ModelFactory',
    'LogisticRegressionModel',
    'RandomForestModel',
    'SVMModel',
]

# Handle optional XGBoost dependency (extra 'ml'); the model module itself
# also guards its xgboost import, so this never raises at package import.
try:
    from .xgboost_model import XGBoostModel
    _all_models.append('XGBoostModel')
except ImportError:
    # xgboost is not installed, which is acceptable
    XGBoostModel = None

# Handle optional AutoGluon dependency
try:
    from .autogluon_model import AutoGluonTabularModel
    _all_models.append('AutoGluonTabularModel')
except ImportError:
    # autogluon is not installed, which is acceptable
    AutoGluonTabularModel = None

__all__ = _all_models
