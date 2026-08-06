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
"""Feature preprocessing for clustering inputs (voxel and supervoxel).

Two chains over one shared set of methods. The chains differ by whether their
state crosses subject boundaries, not by what they process:

* :class:`SubjectPreprocessingChain` -- stateless, per subject, usable at
  either granularity, removes between-subject variation.
* :class:`CohortPreprocessingChain` -- stateful, fitted on the training
  cohort, makes subjects comparable, state travels in the habitat model.

Distinguish this domain from ``table_preprocessor``: that one preprocesses
the MODELLING table (one row per subject, with identifier and outcome
columns) on the way to an outcome model. This one preprocesses the CLUSTERING
input (one row per voxel or supervoxel) on the way to a habitat definition.

Chains speak plain ``DataFrame``. The bridge to the typed contracts is the
symmetric pair ``feature_frame()`` / ``with_feature_frame()`` carried by both
:class:`~habit.contracts.habitat.VoxelFeatureField` and
:class:`~habit.contracts.habitat.Supervoxelization`, so a chain never needs to
know which one it is processing.
"""

from __future__ import annotations

from habit.domain.feature_preprocessing.chains import (
    CohortPreprocessingChain,
    SubjectPreprocessingChain,
    build_methods,
)
from habit.domain.feature_preprocessing.methods import (
    Binning,
    BinningParams,
    CorrelationFilter,
    CorrelationFilterParams,
    FeatureWhitelist,
    FeatureWhitelistParams,
    Impute,
    ImputeParams,
    LogTransform,
    LogTransformParams,
    MinMaxScaling,
    MinMaxScalingParams,
    RobustScaling,
    RobustScalingParams,
    VarianceFilter,
    VarianceFilterParams,
    Winsorizing,
    WinsorizingParams,
    ZScoreScaling,
    ZScoreScalingParams,
)
from habit.domain.feature_preprocessing.registry import (
    FeaturePreprocessingMethodRegistry,
)

__all__ = [
    "Binning",
    "BinningParams",
    "CohortPreprocessingChain",
    "CorrelationFilter",
    "CorrelationFilterParams",
    "FeaturePreprocessingMethodRegistry",
    "FeatureWhitelist",
    "FeatureWhitelistParams",
    "Impute",
    "ImputeParams",
    "LogTransform",
    "LogTransformParams",
    "MinMaxScaling",
    "MinMaxScalingParams",
    "RobustScaling",
    "RobustScalingParams",
    "SubjectPreprocessingChain",
    "VarianceFilter",
    "VarianceFilterParams",
    "Winsorizing",
    "WinsorizingParams",
    "ZScoreScaling",
    "ZScoreScalingParams",
    "build_methods",
]
