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
"""L4 image-preprocessing recipe (thin assembly).

Stage-4 scope: wire the ``preprocess`` CLI command through a recipe instead
of importing ``habit.core.preprocessing.run`` directly. The recipe delegates
to the public :func:`habit.api.preprocessing.run_preprocess` workflow helper
(which still executes the v0.1 engine internally), keeping ``habit.recipes``
free of direct ``habit.core`` imports per the architecture gate.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

__all__ = ["preprocess_images"]


def preprocess_images(
    config: Any,
    *,
    logger: Optional[logging.Logger] = None,
) -> Any:
    """
    Run the batch image-preprocessing pipeline (``habit preprocess`` recipe).

    Args:
        config: Validated preprocessing configuration (v0.1 schema object or
            mapping accepted by
            :class:`~habit.api.preprocessing.PreprocessingConfig`).
        logger: Optional run logger forwarded to the workflow helper.

    Returns:
        :class:`~habit.api.contracts.WorkflowResult` with output directory
        metadata and a run manifest path.
    """
    from habit.api.preprocessing import run_preprocess

    return run_preprocess(config, logger=logger)
