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
"""L4 ICC reliability-analysis recipe (thin assembly).

Stage-5 scope: wire the ``icc`` CLI through a recipe instead of importing
``habit.core`` directly. The recipe delegates to the public
:func:`habit.api.analysis.run_icc_analysis` workflow helper (which still
executes the v0.1 engine internally), keeping ``habit.recipes`` free of
direct ``habit.core`` imports per the architecture gate.

Callers who already hold aligned measurement sessions as
:class:`~habit.contracts.table.FeatureTable` objects do not need this
config-driven path: :func:`habit.domain.evaluation.statistics.icc_analysis`
computes the same per-feature ICC(2,1)/ICC(3,1) panel directly on tables via
the L0 kernels, with no files involved.
"""

from __future__ import annotations

from typing import Any

__all__ = ["icc_analysis"]


def icc_analysis(config: Any) -> Any:
    """
    Run ICC reliability analysis from a validated config (``habit icc``).

    Args:
        config: Validated ICC configuration (v0.1 schema object or mapping
            accepted by :class:`~habit.api.analysis.ICCConfig`).

    Returns:
        :class:`~habit.api.contracts.WorkflowResult` with the ICC JSON path
        in ``artifacts["icc_result"]`` and a run manifest path.
    """
    from habit.api.analysis import run_icc_analysis

    return run_icc_analysis(config)
