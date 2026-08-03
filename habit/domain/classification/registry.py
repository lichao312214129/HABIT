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
"""Registry for the ``classifier`` plugin domain."""

from __future__ import annotations

from typing import Type

from habit.domain.table_protocols import Classifier
from habit.registry.core import ComponentRegistry

__all__ = ["ClassifierRegistry"]


class ClassifierRegistry(ComponentRegistry[Type[Classifier]]):
    """
    Name-to-implementation registry for outcome classifiers.

    The domain is ``classifier`` (NOT ``model``) so it can never be confused
    with :class:`~habit.contracts.habitat.HabitatModel`, the
    habitat-definition artefact (v1.0 naming decisions).
    """

    domain = "classifier"
    kind = "classifier"
