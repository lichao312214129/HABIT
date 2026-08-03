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
"""Built-in habitat model fitters and their registry."""

from __future__ import annotations

from habit.domain.habitat_model.gmm import GmmHabitatModelFitter, GmmHabitatModelFitterParams
from habit.domain.habitat_model.kmeans import (
    KMeansHabitatModelFitter,
    KMeansHabitatModelFitterParams,
)
from habit.domain.habitat_model.registry import HabitatModelFitterRegistry

__all__ = [
    "GmmHabitatModelFitter",
    "GmmHabitatModelFitterParams",
    "KMeansHabitatModelFitter",
    "KMeansHabitatModelFitterParams",
    "HabitatModelFitterRegistry",
]
