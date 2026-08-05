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
Runner context shared between workflow and execution layers.

A :class:`RunnerContext` bundles the *minimum* set of collaborators a runner
needs in order to execute a training/evaluation loop.  Bundling them here
removes the previous reverse dependency where ``BaseRunner`` reached back
into ``self.workflow`` for ``data_manager``, ``pipeline_builder``, the
logger and workflow-owned helper methods.

With an explicit context object:

* the runner depends on a small, named contract (``RunnerContext``) instead
  of ``Any``;
* runners are independently constructible/testable: tests can build a
  context with stubs without instantiating a full workflow;
* dependency direction stays one-way: ``workflow -> context -> runner``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ..data_manager import DataManager
from ..pipeline_builder import PipelineBuilder


@dataclass
class RunnerContext:
    """
    Bundle of collaborators injected into a runner.

    Attributes
    ----------
    data_manager:
        Source of features/labels and split logic.
    pipeline_builder:
        Builder that produces a fresh sklearn pipeline per model.
    logger:
        Logger used for status messages.
    config:
        Validated ML configuration object (``MLConfig``).  Stored as ``Any``
        to avoid a circular import; runners only read from it.
    """

    data_manager: DataManager
    pipeline_builder: PipelineBuilder
    logger: logging.Logger
    config: Any
