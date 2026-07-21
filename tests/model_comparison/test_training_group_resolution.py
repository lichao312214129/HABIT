"""Unit tests for training-split label resolution in model comparison."""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from habit.core.machine_learning.workflows.comparison_workflow import ModelComparison


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
