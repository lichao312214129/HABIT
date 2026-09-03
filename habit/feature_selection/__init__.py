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
"""Built-in feature selectors and the ``feature_selector`` registry."""

from __future__ import annotations

from habit.feature_selection.registry import FeatureSelectorRegistry
from habit.feature_selection.selectors import (
    AnovaSelector,
    Chi2Selector,
    CorrelationSelector,
    IccSelector,
    LassoSelector,
    MrmrSelector,
    RfecvSelector,
    StatisticalTestSelector,
    StepwiseSelector,
    UnivariateCoxSelector,
    UnivariateLogisticSelector,
    VarianceSelector,
    VifSelector,
)

from habit._table_protocols import FeatureSelector

__all__ = [
    "FeatureSelector",
    "FeatureSelectorRegistry",
    "VarianceSelector",
    "CorrelationSelector",
    "VifSelector",
    "AnovaSelector",
    "Chi2Selector",
    "StatisticalTestSelector",
    "UnivariateLogisticSelector",
    "UnivariateCoxSelector",
    "StepwiseSelector",
    "RfecvSelector",
    "LassoSelector",
    "IccSelector",
    "MrmrSelector",
]
