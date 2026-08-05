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
from habit.registry.base import ClassRegistry
from .base_preprocessor import BasePreprocessor


class PreprocessorFactory(ClassRegistry[BasePreprocessor]):
    """
    Factory for creating preprocessors.

    Uses the shared :class:`~habit.core.common.registry.ClassRegistry` contract:
    ``register`` / ``create`` / ``get`` / ``available`` /
    ``register_params_model`` / ``get_params_model``. Register with::

        @PreprocessorFactory.register("resample")
        class ResamplePreprocessor(BasePreprocessor):
            ...
    """

    kind = "preprocessor"
