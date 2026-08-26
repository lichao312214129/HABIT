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
"""Built-in table preprocessors and the ``table_preprocessor`` registry."""

from __future__ import annotations

from habit.domain.table_preprocessing.methods import (
    BinningPreprocessor,
    BinningPreprocessorParams,
    CorrelationFilterPreprocessor,
    CorrelationFilterPreprocessorParams,
    PreciseCorrelationFilterPreprocessor,
    PreciseCorrelationFilterPreprocessorParams,
    L2Preprocessor,
    L2PreprocessorParams,
    LogPreprocessor,
    LogPreprocessorParams,
    MaxAbsPreprocessor,
    MaxAbsPreprocessorParams,
    MinMaxPreprocessor,
    MinMaxPreprocessorParams,
    QuantilePreprocessor,
    QuantilePreprocessorParams,
    RobustPreprocessor,
    RobustPreprocessorParams,
    VarianceFilterPreprocessor,
    VarianceFilterPreprocessorParams,
    WinsorizePreprocessor,
    WinsorizePreprocessorParams,
    ZScorePreprocessor,
    ZScorePreprocessorParams,
)
from habit.domain.table_preprocessing.registry import TablePreprocessorRegistry

__all__ = [
    "TablePreprocessorRegistry",
    "MinMaxPreprocessor",
    "MinMaxPreprocessorParams",
    "ZScorePreprocessor",
    "ZScorePreprocessorParams",
    "RobustPreprocessor",
    "RobustPreprocessorParams",
    "MaxAbsPreprocessor",
    "MaxAbsPreprocessorParams",
    "QuantilePreprocessor",
    "QuantilePreprocessorParams",
    "L2Preprocessor",
    "L2PreprocessorParams",
    "BinningPreprocessor",
    "BinningPreprocessorParams",
    "WinsorizePreprocessor",
    "WinsorizePreprocessorParams",
    "LogPreprocessor",
    "LogPreprocessorParams",
    "VarianceFilterPreprocessor",
    "VarianceFilterPreprocessorParams",
    "CorrelationFilterPreprocessor",
    "CorrelationFilterPreprocessorParams",
    "PreciseCorrelationFilterPreprocessor",
    "PreciseCorrelationFilterPreprocessorParams",
]
