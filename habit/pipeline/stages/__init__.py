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
"""Stage role resolution and the shared habitat dataflow executor (L3)."""

from __future__ import annotations

from habit.pipeline.stages.executor import (
    ensure_habitat_spec_resolved,
    execute_habitat_dataflow,
    normalize_spec_for_execution,
    run_subject_stage_prefix,
)
from habit.pipeline.stages.resolve import (
    ResolvedStage,
    design_from_stages,
    resolve_habitat_stages,
)

__all__ = [
    "ResolvedStage",
    "design_from_stages",
    "resolve_habitat_stages",
    "ensure_habitat_spec_resolved",
    "execute_habitat_dataflow",
    "normalize_spec_for_execution",
    "run_subject_stage_prefix",
]
