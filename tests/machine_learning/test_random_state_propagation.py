"""Tests for random_state propagation in ML pipeline builder."""

from __future__ import annotations

from habit.core.machine_learning.config_schemas import MLConfig, ModelConfig
from habit.core.machine_learning.pipeline_builder import PipelineBuilder
from habit.utils.random_utils import merge_random_state_into_params


def test_merge_random_state_into_model_params() -> None:
    merged = merge_random_state_into_params({"C": 1.0}, global_seed=55)
    assert merged["random_state"] == 55
    assert merged["C"] == 1.0


def test_pipeline_builder_injects_model_random_state() -> None:
    config = MLConfig(
        input=[{"path": "dummy.csv", "subject_id_col": "id", "label_col": "y"}],
        output="./out",
        random_state=77,
        models={"LogisticRegression": ModelConfig(params={"C": 1.0})},
    )
    builder = PipelineBuilder(config, output_dir="./out")
    pipeline = builder.build(
        "LogisticRegression",
        dict(config.models["LogisticRegression"].params),
        feature_names=["f1"],
    )
    model_step = pipeline.named_steps["model"]
    assert model_step.model.random_state == 77
