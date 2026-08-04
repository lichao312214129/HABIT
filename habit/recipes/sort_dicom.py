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
"""L4 DICOM sort recipe (thin assembly).

Stage-4 scope: wire the ``sort-dicom`` CLI command through a recipe instead
of importing ``habit.core.dicom_sort.run`` directly. The recipe delegates
to the public :func:`habit.api.dicom_sort.run_dicom_sort` workflow helper
(which still executes the v0.1 engine internally), keeping ``habit.recipes``
free of direct ``habit.core`` imports per the architecture gate.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

__all__ = ["sort_dicom"]


def sort_dicom(
    config: Any,
    *,
    logger: Optional[logging.Logger] = None,
) -> Any:
    """
    Run the standalone DICOM sort pipeline (``habit sort-dicom`` recipe).

    Args:
        config: Validated DICOM sort configuration (v0.1 schema object or
            mapping accepted by
            :class:`~habit.api.dicom_sort.DicomSortConfig`).
        logger: Optional run logger forwarded to the workflow helper.

    Returns:
        :class:`~habit.api.contracts.WorkflowResult` with output directory
        metadata and a run manifest path.
    """
    from habit.api.dicom_sort import run_dicom_sort

    return run_dicom_sort(config, logger=logger)
