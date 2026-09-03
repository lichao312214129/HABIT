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
"""Built-in regressors (domain ``regressor``)."""

from __future__ import annotations

from habit.regression.models import (
    ElasticNetRegressor,
    GradientBoostingRegressor,
    LassoRegressor,
    RandomForestRegressor,
    RidgeRegressor,
    SvrRegressor,
)
from habit.regression.registry import RegressorRegistry

from habit._table_protocols import Regressor

__all__ = [
    "Regressor",
    "RegressorRegistry",
    "RidgeRegressor",
    "LassoRegressor",
    "ElasticNetRegressor",
    "SvrRegressor",
    "RandomForestRegressor",
    "GradientBoostingRegressor",
]
