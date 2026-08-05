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
"""sklearn estimator bridge to compat engine configurators (L1 compat)."""

from __future__ import annotations

from typing import Any, Type

__all__ = ["get_habitat_configurator_class", "get_ml_pipeline_builder_class"]


def get_habitat_configurator_class() -> Type[Any]:
    """Return the habitat configurator used by sklearn-style estimators."""
    from habit.compat.engines.habitat_analysis.configurator import HabitatConfigurator

    return HabitatConfigurator


def get_ml_pipeline_builder_class() -> Type[Any]:
    """Return the ML ``PipelineBuilder`` class."""
    from habit.compat.engines.machine_learning.pipeline_builder import PipelineBuilder

    return PipelineBuilder
