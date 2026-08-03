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
"""
Shared configurator infrastructure.

``BaseConfigurator`` lives in ``habit.core.common.configurators.base`` as shared assembly
infrastructure. Domain-specific configurators live with their domains:

* ``habit.core.habitat_analysis.configurator.HabitatConfigurator``
* ``habit.core.machine_learning.configurator.MLConfigurator``
* ``habit.core.preprocessing.configurator.PreprocessingConfigurator``
"""

from .base import BaseConfigurator

__all__ = [
    'BaseConfigurator',
]
