"""
Contract tests for IndividualLevelStep subclasses and HabitatPipeline.

Migrated and extended from tests/test_individual_step_contract.py.
Tests pin the per-subject step API introduced in the habitat_analysis rework:
  - Every concrete step must implement transform_one(subject_id, subject_data).
  - The inherited transform() must iterate and delegate to transform_one().
  - HabitatPipeline._process_single_subject must drive steps via transform_one.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from habit.core.habitat_analysis.pipelines.base_pipeline import (
    HabitatPipeline,
    IndividualLevelStep,
)
from habit.core.habitat_analysis.pipelines.subject_state import SubjectHabitatState
from habit.core.habitat_analysis.pipelines.steps import (
    CalculateMeanVoxelFeaturesStep,
    IndividualClusteringStep,
    MergeSupervoxelFeaturesStep,
    SubjectPreprocessingStep,
    SupervoxelFeatureExtractionStep,
    VoxelFeatureExtractor,
)

INDIVIDUAL_STEP_TYPES = [
    VoxelFeatureExtractor,
    SubjectPreprocessingStep,
    IndividualClusteringStep,
    SupervoxelFeatureExtractionStep,
    CalculateMeanVoxelFeaturesStep,
    MergeSupervoxelFeaturesStep,
]


# ---------------------------------------------------------------------------
# Inheritance contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("step_cls", INDIVIDUAL_STEP_TYPES)
def test_step_subclasses_individual_level_step(step_cls: type) -> None:
    """Every concrete step must inherit from IndividualLevelStep."""
    assert issubclass(step_cls, IndividualLevelStep), (
        f"{step_cls.__name__} should inherit from IndividualLevelStep"
    )


# ---------------------------------------------------------------------------
# transform_one signature contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("step_cls", INDIVIDUAL_STEP_TYPES)
def test_step_implements_transform_one(step_cls: type) -> None:
    """transform_one must be defined on the subclass (not only on the base)."""
    assert "transform_one" in step_cls.__dict__, (
        f"{step_cls.__name__} must override transform_one(subject_id, subject_data)"
    )
    sig = inspect.signature(step_cls.transform_one)
    params = list(sig.parameters)
    assert params[:3] == ["self", "subject_id", "subject_data"], (
        f"{step_cls.__name__}.transform_one must take (self, subject_id, subject_data); got {params}"
    )


@pytest.mark.parametrize("step_cls", INDIVIDUAL_STEP_TYPES)
def test_step_does_not_override_transform(step_cls: type) -> None:
    """
    Subclasses must NOT override transform(); that is the base's responsibility.
    Per-subject logic belongs in transform_one().
    """
    assert "transform" not in step_cls.__dict__, (
        f"{step_cls.__name__} should NOT override transform(); "
        "use transform_one() for per-subject logic."
    )


# ---------------------------------------------------------------------------
# Base-class behaviour
# ---------------------------------------------------------------------------


def test_default_fit_is_stateless() -> None:
    """Base-class fit must mark the step as fitted without side effects."""

    class _Dummy(IndividualLevelStep):
        def transform_one(self, subject_id: str, subject_data: Any) -> Any:
            return subject_data

    step = _Dummy()
    assert step.fitted_ is False
    out = step.fit({"S1": object()})
    assert out is step
    assert step.fitted_ is True


def test_default_transform_iterates_via_transform_one() -> None:
    """Base-class transform must call transform_one for every subject in order."""
    calls: list = []

    class _Recorder(IndividualLevelStep):
        def transform_one(self, subject_id: str, subject_data: Any) -> Any:
            calls.append((subject_id, subject_data))
            return f"out:{subject_id}"

    step = _Recorder()
    step.fitted_ = True
    inputs: Dict[str, Any] = {"S1": "a", "S2": "b", "S3": "c"}
    out = step.transform(inputs)

    assert calls == [("S1", "a"), ("S2", "b"), ("S3", "c")]
    assert out == {"S1": "out:S1", "S2": "out:S2", "S3": "out:S3"}


def test_transform_preserves_subject_order() -> None:
    """Output dict keys must follow insertion order of the input."""
    class _Identity(IndividualLevelStep):
        def transform_one(self, subject_id: str, subject_data: Any) -> Any:
            return subject_data

    step = _Identity()
    step.fitted_ = True
    keys = [f"P{i:03d}" for i in range(20)]
    inputs = {k: i for i, k in enumerate(keys)}
    out = step.transform(inputs)
    assert list(out.keys()) == keys


def test_default_transform_propagates_per_subject_errors() -> None:
    """transform must surface the original exception from transform_one."""

    class _Failing(IndividualLevelStep):
        def transform_one(self, subject_id: str, subject_data: Any) -> Any:
            if subject_id == "bad":
                raise RuntimeError("boom")
            return subject_data

    step = _Failing()
    step.fitted_ = True
    with pytest.raises(RuntimeError, match="boom"):
        step.transform({"good": 1, "bad": 2})


# ---------------------------------------------------------------------------
# HabitatPipeline._process_single_subject
# ---------------------------------------------------------------------------


def test_pipeline_drives_steps_via_transform_one() -> None:
    """HabitatPipeline._process_single_subject must call transform_one directly."""
    calls: list = []

    class _Recorder(IndividualLevelStep):
        def __init__(self, tag: str) -> None:
            super().__init__()
            self.tag = tag

        def transform_one(self, subject_id: str, subject_data: Any) -> Any:
            calls.append((self.tag, subject_id, subject_data))
            return f"{self.tag}->{subject_data}"

    step1 = _Recorder("a")
    step2 = _Recorder("b")
    pipeline = HabitatPipeline(steps=[("a", step1), ("b", step2)], config=None)

    out_id, out_value = pipeline._process_single_subject(("S1", "INPUT"))
    assert out_id == "S1"
    assert out_value == "b->a->INPUT"
    assert calls == [("a", "S1", "INPUT"), ("b", "S1", "a->INPUT")]


# ---------------------------------------------------------------------------
# MergeSupervoxelFeaturesStep custom fit validation
# ---------------------------------------------------------------------------


def test_merge_step_keeps_explicit_fit_validation() -> None:
    """MergeSupervoxelFeaturesStep.fit must fail-fast on missing advanced features."""
    cfg = MagicMock()
    cfg.FeatureConstruction.supervoxel_level.method = "supervoxel_radiomics()"
    cfg.HabitatSegmentation.clustering_mode = "two_step"
    cfg.verbose = False

    step = MergeSupervoxelFeaturesStep(cfg)
    assert step.use_advanced_features is True

    with pytest.raises(ValueError, match="advanced supervoxel features"):
        step.fit({"S1": SubjectHabitatState(mean_voxel_features=object())})

    # With advanced features present, fit must succeed.
    step.fit({"S1": SubjectHabitatState(supervoxel_features=object())})
    assert step.fitted_ is True
