"""
Contract tests for `IndividualLevelStep` subclasses.

These tests pin down the per-subject contract introduced in PR-3 of the
``habitat_analysis`` rework:

* Every concrete individual-level step MUST implement
  :meth:`IndividualLevelStep.transform_one(subject_id, subject_data)`.
* The default :meth:`IndividualLevelStep.transform` MUST iterate over the
  input dict and delegate to ``transform_one`` (i.e. it must NOT be
  overridden by subclasses, except when the step needs an explicit batch
  fit hook such as :class:`MergeSupervoxelFeaturesStep`).
* :class:`HabitatPipeline._process_single_subject` MUST drive each step via
  ``transform_one`` directly (no more single-subject dict wrap/unwrap).

The tests use minimal stubs and do not depend on heavy optional packages
(``SimpleITK``, ``radiomics``, ``sklearn``); they only import the pipeline
infrastructure and the step types.
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


@pytest.mark.parametrize("step_cls", INDIVIDUAL_STEP_TYPES)
def test_step_subclasses_individual_level_step(step_cls: type) -> None:
    """Every concrete step listed above must inherit from IndividualLevelStep."""
    assert issubclass(step_cls, IndividualLevelStep), (
        f"{step_cls.__name__} should inherit from IndividualLevelStep"
    )


@pytest.mark.parametrize("step_cls", INDIVIDUAL_STEP_TYPES)
def test_step_implements_transform_one(step_cls: type) -> None:
    """transform_one must be defined ON the subclass, not inherited as abstract."""
    assert "transform_one" in step_cls.__dict__, (
        f"{step_cls.__name__} must override transform_one(subject_id, subject_data)"
    )

    sig = inspect.signature(step_cls.transform_one)
    params = list(sig.parameters)
    # Expect (self, subject_id, subject_data).
    assert params[:3] == ["self", "subject_id", "subject_data"], (
        f"{step_cls.__name__}.transform_one should take (self, subject_id, subject_data); "
        f"got {params}"
    )


@pytest.mark.parametrize("step_cls", INDIVIDUAL_STEP_TYPES)
def test_step_does_not_override_transform_with_dict_loop(step_cls: type) -> None:
    """
    Subclasses must NOT override the dict-iterating ``transform`` — that is
    now the base-class responsibility. The single legitimate exception
    would be a subclass that needs cross-subject behaviour during transform,
    which would belong in GroupLevelStep instead.
    """
    assert "transform" not in step_cls.__dict__, (
        f"{step_cls.__name__} should NOT override transform(); "
        "iterate via the inherited base-class implementation and put per-subject "
        "logic in transform_one() instead."
    )


def test_default_fit_is_stateless() -> None:
    """
    Base-class fit must be a no-op that just marks the step as fitted, so
    subclasses that don't need batch fit hooks can omit the override.
    """

    class _Dummy(IndividualLevelStep):
        def transform_one(self, subject_id: str, subject_data: Any) -> Any:
            return subject_data

    step = _Dummy()
    assert step.fitted_ is False
    out = step.fit({"S1": object()})
    assert out is step
    assert step.fitted_ is True


def test_default_transform_iterates_via_transform_one() -> None:
    """
    Base-class transform must call transform_one for every subject and
    preserve subject order in the returned dict.
    """

    calls: list[tuple[str, Any]] = []

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


def test_pipeline_drives_steps_via_transform_one() -> None:
    """
    HabitatPipeline._process_single_subject MUST call transform_one directly
    on each individual-level step (no more "wrap-as-dict, then unwrap").
    """

    calls: list[tuple[str, str, Any]] = []

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
    assert out_value == "a->b->a->INPUT"
    assert calls == [("a", "S1", "INPUT"), ("b", "S1", "a->INPUT")]


def test_merge_step_keeps_explicit_fit_validation() -> None:
    """
    MergeSupervoxelFeaturesStep is the one step that overrides fit() to
    fail-fast when advanced features are requested but not produced.
    Verify that contract still holds after the refactor.
    """
    cfg = MagicMock()
    cfg.FeatureConstruction.supervoxel_level.method = "supervoxel_radiomics()"
    cfg.HabitatsSegmention.clustering_mode = "two_step"
    cfg.verbose = False

    step = MergeSupervoxelFeaturesStep(cfg)
    assert step.use_advanced_features is True

    with pytest.raises(ValueError, match="advanced supervoxel features"):
        step.fit({"S1": {"mean_voxel_features": object()}})

    # Same step, but a subject DOES carry advanced features -> fit succeeds.
    step.fit({"S1": {"supervoxel_features": object()}})
    assert step.fitted_ is True
