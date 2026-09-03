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

from habit.table_preprocessing.methods import (
    BinningPreprocessor,
    CorrelationFilterPreprocessor,
    PreciseCorrelationFilterPreprocessor,
    L2Preprocessor,
    LogPreprocessor,
    MaxAbsPreprocessor,
    MinMaxPreprocessor,
    QuantilePreprocessor,
    RobustPreprocessor,
    VarianceFilterPreprocessor,
    WinsorizePreprocessor,
    ZScorePreprocessor,
)
from habit.table_preprocessing.registry import TablePreprocessorRegistry

from habit._table_protocols import TablePreprocessor

__all__ = [
    "TablePreprocessor",
    "TablePreprocessorRegistry",
    "MinMaxPreprocessor",
    "ZScorePreprocessor",
    "RobustPreprocessor",
    "MaxAbsPreprocessor",
    "QuantilePreprocessor",
    "L2Preprocessor",
    "BinningPreprocessor",
    "WinsorizePreprocessor",
    "LogPreprocessor",
    "VarianceFilterPreprocessor",
    "CorrelationFilterPreprocessor",
    "PreciseCorrelationFilterPreprocessor",
]
