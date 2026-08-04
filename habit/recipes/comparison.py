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
"""L4 model-comparison recipes (thin assembly).

Stage-4 scope: wire the ``compare`` CLI through recipes instead of importing
``habit.core`` directly. A full domain-native orchestrator -- cohort merge,
split-aware evaluation, plotting, and report export assembled from L3
components -- is deferred until the v0.1 ``ModelComparison`` batch loop is
replaced. Until then :func:`compare_models` delegates to the public
``habit.api.machine_learning`` workflow helper (which still executes the v0.1
engine internally), keeping ``habit.recipes`` free of direct ``habit.core``
imports per the architecture gate.

:func:`pairwise_delong_test` exposes the v1 DeLong kernel path for callers who
only need paired ROC-AUC comparison on aligned score vectors.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from habit.domain.evaluation.statistics import DelongResult, delong_test

__all__ = ["compare_models", "pairwise_delong_test"]


def compare_models(
    config: Any,
    *,
    logger: Optional[logging.Logger] = None,
    output_dir: Optional[str] = None,
) -> Any:
    """
    Compare multiple trained models from a validated config (``habit compare``).

    Args:
        config: Validated model-comparison configuration (v0.1 schema object
            or mapping accepted by
            :class:`~habit.api.machine_learning.ModelComparisonConfig`).
        logger: Optional run logger forwarded to the workflow helper.
        output_dir: Optional output directory override for plots and reports.

    Returns:
        :class:`~habit.api.contracts.WorkflowResult` with comparison metrics
        in ``data`` and the output directory in ``artifacts``.
    """
    from habit.api.machine_learning import run_model_comparison

    return run_model_comparison(config, logger=logger, output_dir=output_dir)


def pairwise_delong_test(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> DelongResult:
    """
    Compare two models' ROC AUCs on the same subjects (paired DeLong test).

    This is the notebook-friendly surface over
    :func:`habit.kernels.statistics.delong_roc_test`, bundled with the two
    point AUC estimates a report needs.

    Args:
        y_true: Binary ground-truth labels (0/1), both classes present.
        scores_a: Probability-of-class-1 scores of the first model.
        scores_b: Probability-of-class-1 scores of the second model, aligned
            to ``scores_a``.

    Returns:
        Frozen :class:`~habit.domain.evaluation.statistics.DelongResult`.
    """
    return delong_test(y_true, scores_a, scores_b)
