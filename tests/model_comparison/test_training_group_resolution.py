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
"""Unit tests for training-split label resolution in model comparison."""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from habit.compat.engines.machine_learning.workflows.comparison_workflow import ModelComparison


def _build_comparison(split_groups: Dict[str, Any]) -> ModelComparison:
    """Create a lightweight ModelComparison instance for label-resolution tests."""
    comparison = ModelComparison(
        config={
            "output_dir": "tmp_compare",
            "files_config": [
                {
                    "path": "dummy.csv",
                    "name": "model_a",
                    "subject_id_col": "subjID",
                    "label_col": "true_label",
                    "prob_col": "prob",
                }
            ],
        },
        evaluator=MagicMock(),
        reporter=MagicMock(),
        threshold_manager=MagicMock(),
        plot_manager=MagicMock(),
        metrics_store=MagicMock(),
        logger=MagicMock(),
    )
    comparison.split_groups = split_groups
    return comparison


@pytest.mark.unit
@pytest.mark.parametrize(
    "train_label",
    [
        "train",
        "Train",
        "training",
        "Training set",
        "training_set",
        "train-set",
    ],
)
def test_get_training_group_name_accepts_common_aliases(train_label: str) -> None:
    """Production CSVs use several equivalent training-split labels."""
    comparison = _build_comparison(
        {
            train_label: {"model_a": MagicMock()},
            "Test set": {"model_a": MagicMock()},
        }
    )

    assert comparison._get_training_group_name() == train_label


@pytest.mark.unit
def test_get_training_group_name_rejects_ambiguous_train_vs_test() -> None:
    """Labels that mix train and test must not be treated as the training set."""
    comparison = _build_comparison(
        {
            "train_vs_test": {"model_a": MagicMock()},
            "Test set": {"model_a": MagicMock()},
        }
    )

    assert comparison._get_training_group_name() is None
    comparison.logger.warning.assert_called()
