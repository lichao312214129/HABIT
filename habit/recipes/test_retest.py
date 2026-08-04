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
"""L4 test-retest reproducibility recipe (thin assembly).

Stage-5 scope: wire the ``retest`` CLI through a recipe instead of importing
``habit.core`` directly. The recipe delegates to the public
:func:`habit.api.analysis.run_test_retest_analysis` workflow helper (which
still executes the v0.1 habitat-label mapper internally), keeping
``habit.recipes`` free of direct ``habit.core`` imports per the architecture
gate. A domain-native remapping operator -- habitat maps in, remapped maps
out, no NRRD globbing -- is deferred until the v0.1 mapper is replaced.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

__all__ = ["test_retest_analysis"]


def test_retest_analysis(
    config: Any,
    *,
    logger: Optional[logging.Logger] = None,
) -> Any:
    """
    Map retest habitat labels onto test labels (``habit retest`` recipe).

    Args:
        config: Validated test-retest configuration (v0.1 schema object or
            mapping accepted by
            :class:`~habit.api.analysis.TestRetestConfig`).
        logger: Optional run logger forwarded to the workflow helper.

    Returns:
        :class:`~habit.api.contracts.WorkflowResult` with the
        ``{retest_label: test_label}`` mapping in ``data`` and the
        remapped-image directory in ``output_dir``.
    """
    from habit.api.analysis import run_test_retest_analysis

    return run_test_retest_analysis(config, logger=logger)
