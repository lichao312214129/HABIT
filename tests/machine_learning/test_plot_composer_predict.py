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
"""Unit tests for prediction-mode plotting in PlotComposer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from habit.core.machine_learning.contracts.plan import WorkflowPlan
from habit.core.machine_learning.contracts.results import InferenceResult
from habit.core.machine_learning.reporting.plot_composer import PlotComposer


def _make_inference_result(
    *,
    with_labels: bool,
    pipeline_path: str = "/tmp/LogisticRegression_final_pipeline.pkl",
) -> InferenceResult:
    """Build a minimal InferenceResult for PlotComposer tests."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0])
    y_prob = np.array([0.1, 0.9, 0.2, 0.4])
    predictions = pd.DataFrame(
        {
            "subject_id": ["s1", "s2", "s3", "s4"],
            "label": y_true,
            "predicted_label": y_pred,
            "predicted_probability": y_prob,
        }
    )
    plan = MagicMock(spec=WorkflowPlan)
    return InferenceResult.create(
        plan=plan,
        pipeline_path=pipeline_path,
        predictions=predictions,
        metrics={"auc": 0.75} if with_labels else {},
        label_col="label" if with_labels else None,
        y_true=y_true if with_labels else None,
        y_pred=y_pred if with_labels else None,
        y_prob=y_prob if with_labels else None,
        fitted_estimator=object(),
        feature_frame=pd.DataFrame({"f1": [1, 2, 3, 4], "f2": [0, 1, 0, 1]}),
    )


class TestPlotComposerPredict:
    def test_render_inference_calls_plot_manager_when_labels_exist(self) -> None:
        """Labeled predict results should trigger PlotManager once."""
        plot_manager = MagicMock()
        composer = PlotComposer(plot_manager=plot_manager, is_visualize=True)
        result = _make_inference_result(with_labels=True)

        composer.render(result)

        plot_manager.run_workflow_plots.assert_called_once()
        args: tuple = plot_manager.run_workflow_plots.call_args.args
        kwargs: Dict[str, Any] = plot_manager.run_workflow_plots.call_args.kwargs
        payload = args[0]
        assert "LogisticRegression" in payload
        assert "y_true" in payload["LogisticRegression"]["raw"]
        assert kwargs["prefix"] == "predict_"
        assert kwargs["dataset_type"] == "raw"

    def test_render_inference_skips_when_no_labels(self) -> None:
        """Without evaluation arrays, predict mode must not call PlotManager."""
        plot_manager = MagicMock()
        composer = PlotComposer(plot_manager=plot_manager, is_visualize=True)
        result = _make_inference_result(with_labels=False)

        composer.render(result)

        plot_manager.run_workflow_plots.assert_not_called()

    def test_render_respects_is_visualize_false(self) -> None:
        """Global visualization off must suppress inference plots."""
        plot_manager = MagicMock()
        composer = PlotComposer(plot_manager=plot_manager, is_visualize=False)
        result = _make_inference_result(with_labels=True)

        composer.render(result)

        plot_manager.run_workflow_plots.assert_not_called()

    def test_infer_model_name_from_pipeline_path(self) -> None:
        """Pipeline filename suffixes should be stripped for plot titles."""
        name = PlotComposer._infer_model_name(
            str(Path("models") / "RandomForest_final_pipeline.pkl")
        )
        assert name == "RandomForest"
