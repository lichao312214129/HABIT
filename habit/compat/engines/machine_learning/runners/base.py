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
Base runner definitions for machine-learning workflows.

The base class only owns the ``context`` and ``plan`` references.  Concrete
runners (:class:`HoldoutRunner`, :class:`KFoldRunner`, :class:`InferenceRunner`)
implement their own ``run`` signature because their inputs/outputs differ.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import pandas as pd

from habit.utils.random_utils import resolve_random_state

from ..contracts.plan import WorkflowPlan
from .context import RunnerContext


class BaseRunner:
    """
    Shared runner helper that depends on a :class:`RunnerContext`.

    Concrete runners extend this class and add their own ``run`` method with
    the signature appropriate to their workflow mode (no shared abstract
    ``run`` is enforced because the input shapes differ).
    """

    def __init__(self, context: RunnerContext, plan: WorkflowPlan) -> None:
        """
        Args:
        context:
            Bundle of collaborators (data manager, pipeline builder,
            resampler, logger, config).
        plan:
            Immutable execution-plan snapshot.
        """
        self.context = context
        self.plan = plan

    def load_dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load features/labels via the context's data manager.

        Returns:
        Tuple[pd.DataFrame, pd.Series]
            Feature matrix (without label column) and target vector.
        """
        data_manager = self.context.data_manager
        if data_manager.data is None:
            data_manager.load_data()
        if data_manager.data is None or data_manager.label_col is None:
            raise ValueError(
                "DataManager returned empty dataset or missing label column."
            )
        X: pd.DataFrame = data_manager.data.drop(columns=[data_manager.label_col])
        y: pd.Series = data_manager.data[data_manager.label_col]
        return X, y

    def bootstrap_options(self) -> Optional[Dict[str, Any]]:
        """
        Resolve bootstrap keyword arguments from the run configuration.

        Returns:
            Optional[Dict[str, Any]]: Keyword arguments for
            :func:`~habit.compat.engines.machine_learning.evaluation.metrics.bootstrap_metrics`,
            or ``None`` when confidence intervals are disabled.
        """
        bootstrap_config = getattr(self.context.config, "bootstrap", None)
        if bootstrap_config is None or not getattr(bootstrap_config, "enabled", False):
            return None
        return {
            "n_iterations": int(getattr(bootstrap_config, "n_iterations", 1000)),
            "ci_level": float(getattr(bootstrap_config, "ci_level", 0.95)),
            "stratified": bool(getattr(bootstrap_config, "stratified", True)),
            "random_state": resolve_random_state(
                getattr(bootstrap_config, "random_state", None),
                self.plan.random_state,
            ),
        }
