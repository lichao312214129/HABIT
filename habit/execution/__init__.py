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
"""Execution backends and checkpoints: optional accelerators, never required.

An entire study can run without ever constructing anything from this
package: ``op(subject)`` handles one subject and ``Cohort.map(op)`` handles
the cohort with an implicit serial backend. Explicit backends are
constructed only when parallelism, per-subject timeouts, or resume are
actually wanted.
"""

from __future__ import annotations

from habit.execution.backends import SerialBackend
from habit.execution.checkpoint import CheckpointStore
from habit.execution.checkpoint_migrate import (
    LegacyCheckpointMigrationReport,
    is_v01_checkpoint_layout,
    migrate_v01_checkpoint_if_needed,
)
from habit.execution.process_pool import ProcessPoolBackend, SubjectTimeoutError
from habit.execution.selection import backend_from_policy, should_use_process_pool

__all__ = [
    "SerialBackend",
    "ProcessPoolBackend",
    "SubjectTimeoutError",
    "CheckpointStore",
    "LegacyCheckpointMigrationReport",
    "is_v01_checkpoint_layout",
    "migrate_v01_checkpoint_if_needed",
    "backend_from_policy",
    "should_use_process_pool",
]
