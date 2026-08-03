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
Structural contracts shared across the machine-learning subpackage.

The :class:`WorkflowResult` protocol defines the *minimum* shape that any
runner output must satisfy in order to be consumed by reporting components
(``ModelStore``, ``ReportWriter``, ``PlotComposer``).  Encoding the shape as
a :class:`typing.Protocol` (rather than a base dataclass) lets concrete
result objects stay frozen dataclasses with their own fields while still
plugging into the same reporting seam.
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable

from .plan import WorkflowPlan


@runtime_checkable
class WorkflowResult(Protocol):
    """
    Minimum surface every workflow result must expose.

    Reporting components only depend on this protocol, not on a particular
    dataclass.  This keeps holdout / k-fold / inference results
    interchangeable from the writers' point of view.
    """

    plan: WorkflowPlan
    summary_rows: List[Dict[str, Any]]
    created_at: str

    def to_legacy_results(self) -> Dict[str, Any]:
        """Expose a dict-style payload compatible with legacy reporting code."""
        ...
