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
"""Registry for the ``feature_preprocessing_method`` plugin domain."""

from __future__ import annotations

from typing import Type

from habit.registry.core import ComponentRegistry

__all__ = ["FeaturePreprocessingMethodRegistry"]


class FeaturePreprocessingMethodRegistry(ComponentRegistry[Type[object]]):
    """
    Name-to-implementation registry for feature-matrix preprocessing methods.

    One registry serves both chains. A method describes a column-wise
    computation over a unit-by-feature matrix and is deliberately ignorant of
    whether those units are voxels or supervoxels, and of whether the chain
    holding it discards its state (per subject) or keeps it (per cohort).
    That ignorance is what lets a study name ``winsorize`` once and use it at
    either granularity.
    """

    domain = "feature_preprocessing_method"
    kind = "feature preprocessing method"
