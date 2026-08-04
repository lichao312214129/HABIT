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

from habit.domain.feature_selection.registry import FeatureSelectorRegistry
from habit.domain.feature_selection.selectors import (
    AnovaSelector,
    AnovaSelectorParams,
    Chi2Selector,
    Chi2SelectorParams,
    CorrelationSelector,
    CorrelationSelectorParams,
    IccSelector,
    IccSelectorParams,
    LassoSelector,
    LassoSelectorParams,
    MrmrSelector,
    MrmrSelectorParams,
    RfecvSelector,
    RfecvSelectorParams,
    StatisticalTestSelector,
    StatisticalTestSelectorParams,
    StepwiseSelector,
    StepwiseSelectorParams,
    UnivariateCoxSelector,
    UnivariateCoxSelectorParams,
    UnivariateLogisticSelector,
    UnivariateLogisticSelectorParams,
    VarianceSelector,
    VarianceSelectorParams,
    VifSelector,
    VifSelectorParams,
)

__all__ = [
    "FeatureSelectorRegistry",
    "VarianceSelector",
    "VarianceSelectorParams",
    "CorrelationSelector",
    "CorrelationSelectorParams",
    "VifSelector",
    "VifSelectorParams",
    "AnovaSelector",
    "AnovaSelectorParams",
    "Chi2Selector",
    "Chi2SelectorParams",
    "StatisticalTestSelector",
    "StatisticalTestSelectorParams",
    "UnivariateLogisticSelector",
    "UnivariateLogisticSelectorParams",
    "UnivariateCoxSelector",
    "UnivariateCoxSelectorParams",
    "StepwiseSelector",
    "StepwiseSelectorParams",
    "RfecvSelector",
    "RfecvSelectorParams",
    "LassoSelector",
    "LassoSelectorParams",
    "IccSelector",
    "IccSelectorParams",
    "MrmrSelector",
    "MrmrSelectorParams",
]
