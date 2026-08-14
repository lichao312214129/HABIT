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
"""Image-volume preprocessors and the ``preprocessor`` registry."""

from __future__ import annotations

from habit.domain.image_preprocessing.methods import (
    AdaptiveHistogramEqualization,
    HistogramStandardization,
    N4Correction,
    Registration,
    Reorientation,
    Resample,
    ZScoreNormalization,
)
from habit.domain.image_preprocessing.registry import PreprocessorRegistry

__all__ = [
    "PreprocessorRegistry",
    "ZScoreNormalization",
    "Resample",
    "Reorientation",
    "N4Correction",
    "HistogramStandardization",
    "AdaptiveHistogramEqualization",
    "Registration",
]
