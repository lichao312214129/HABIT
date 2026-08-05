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
Holdout machine-learning workflow (train / test split).

The class is a thin orchestration shell: it builds an execution plan, hands
control to a runner (:class:`HoldoutRunner` for ``run_mode='train'``,
:class:`InferenceRunner` for ``run_mode='predict'``) and pipes the
resulting :class:`RunResult` / :class:`InferenceResult` through the
reporting layer.

The class was previously named ``MachineLearningWorkflow`` - that name
suggested it was the *only* ML workflow when in fact it implements the
holdout strategy specifically.  The class is now called
:class:`HoldoutWorkflow`; ``MachineLearningWorkflow`` remains as a
deprecated alias so external scripts and the static API tests keep working.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from ..config_schemas import MLConfig
from ..contracts.plan import WorkflowPlan
from ..contracts.results import InferenceResult, RunResult
from ..reporting.model_store import ModelStore
from ..reporting.plot_composer import PlotComposer
from ..reporting.report_writer import ReportWriter
from ..runners.holdout import HoldoutRunner
from ..runners.inference import InferenceRunner
from .base import BaseWorkflow


class HoldoutWorkflow(BaseWorkflow):
    """
    Hold-out train/test workflow.

    Public entry points
    -------------------
    fit:
        Train + persist pipelines, write reports, render plots.
    predict:
        Load a saved ``*_final_pipeline.pkl`` and predict on
        ``config.input[0].path``.
    run:
        Dispatcher; routes to :meth:`fit` / :meth:`predict` based on
        ``config.run_mode``.  This is the entry used by the CLI
        (``habit model``).
    """

    def __init__(self, config: MLConfig) -> None:
        super().__init__(config, module_name="ml_standard")
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self._plan = WorkflowPlan(
            config=self.config_obj,
            output_dir=self.output_dir,
            random_state=self.random_state,
        )
        self._train_runner = HoldoutRunner(
            context=self.runner_context, plan=self._plan
        )
        self._inference_runner = InferenceRunner(
            context=self.runner_context, plan=self._plan
        )
        self._run_result: Optional[RunResult] = None
        self._inference_result: Optional[InferenceResult] = None

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Route to :meth:`fit` or :meth:`predict` according to ``run_mode``."""
        run_mode = getattr(self.config_obj, "run_mode", "train")
        if run_mode == "train":
            self.fit()
        elif run_mode == "predict":
            self.predict()
        else:
            raise ValueError(
                f"HoldoutWorkflow: unsupported run_mode={run_mode!r}; "
                "expected 'train' or 'predict'."
            )

    # ------------------------------------------------------------------
    # Train path
    # ------------------------------------------------------------------

    def fit(self) -> None:
        """Train every configured model and persist outputs in explicit order."""
        self.logger.info("Starting Holdout ML pipeline (mode=train)...")
        self.data_manager.load_data()
        self._run_result = self._train_runner.run()

        # Backward-compatible attributes consumed by some external scripts.
        self.X_train = self._run_result.dataset.x_train
        self.X_test = self._run_result.dataset.x_test
        self.results = self._run_result.to_legacy_results()

        # Persist outputs: models -> reports -> plots.
        model_store = ModelStore(
            output_dir=self.output_dir,
            is_save_model=bool(getattr(self.config_obj, "is_save_model", True)),
        )
        report_writer = ReportWriter(
            output_dir=self.output_dir,
            module_name=self.module_name,
        )
        plot_composer = PlotComposer(
            plot_manager=self.plot_manager,
            is_visualize=bool(getattr(self.config_obj, "is_visualize", True)),
        )

        saved_paths = model_store.save(self._run_result)
        for model_name, path in saved_paths.items():
            self.logger.info("Model saved [%s]: %s", model_name, path)

        report_writer.write(self._run_result)
        plot_composer.render(self._run_result)

        self.logger.info(
            "Holdout ML workflow completed (train). Results saved to %s",
            self.output_dir,
        )

    # ------------------------------------------------------------------
    # Predict path
    # ------------------------------------------------------------------

    def predict(self) -> None:
        """
        Run :class:`InferenceRunner`, then persist reports and optional plots.

        Evaluation figures are generated only when ``is_visualize=true`` and
        the inference result contains ground-truth labels
        (``evaluate=true`` with a resolvable ``label_col``).
        """
        self.logger.info("Starting Holdout ML pipeline (mode=predict)...")
        self._inference_result = self._inference_runner.run()

        report_writer = ReportWriter(
            output_dir=self.output_dir, module_name=self.module_name
        )
        report_writer.write(self._inference_result)

        plot_composer = PlotComposer(
            plot_manager=self.plot_manager,
            is_visualize=bool(getattr(self.config_obj, "is_visualize", True)),
        )
        plot_composer.render(self._inference_result)

        self.logger.info(
            "Holdout ML workflow completed (predict). Output dir: %s",
            self.output_dir,
        )


# ---------------------------------------------------------------------------
# Backward-compatible deprecation shim
# ---------------------------------------------------------------------------


class MachineLearningWorkflow(HoldoutWorkflow):
    """
    Deprecated alias for :class:`HoldoutWorkflow`.

    This class is kept as a thin subclass so external scripts that import
    ``MachineLearningWorkflow`` keep working.  All behaviour lives on
    :class:`HoldoutWorkflow`; new code should import that name directly.
    """

    def run(self) -> None:
        """
        Run the workflow.  Identical to :meth:`HoldoutWorkflow.run`.

        Emits a :class:`DeprecationWarning` so callers know to migrate to
        :class:`HoldoutWorkflow`.
        """
        import warnings

        warnings.warn(
            "MachineLearningWorkflow is deprecated; use HoldoutWorkflow instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().run()
