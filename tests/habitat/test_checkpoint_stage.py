"""Tests for IndividualCheckpointStage auto-retry behavior."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

import pytest

from habit.core.habitat_analysis.checkpoint.stage import IndividualCheckpointStage
from habit.core.habitat_analysis.config_schemas import (
    FeatureConstructionConfig,
    HabitatAnalysisConfig,
    HabitatSegmentationConfig,
    VoxelLevelConfig,
)
from habit.core.habitat_analysis.pipelines.habitat_subject_data import HabitatSubjectData
from habit.utils.parallel_utils import ProcessingResult


def _train_config(
    tmp_path: Path,
    *,
    auto_retry_rounds: int = 2,
    on_subject_failure: str = "continue",
    run_mode: str = "train",
    pipeline_path: Path | None = None,
) -> HabitatAnalysisConfig:
    if run_mode == "predict":
        assert pipeline_path is not None
        return HabitatAnalysisConfig(
            data_dir=str(tmp_path / "data"),
            out_dir=str(tmp_path / "out"),
            run_mode="predict",
            pipeline_path=str(pipeline_path),
            resume=False,
            individual_subject_auto_retry_rounds=auto_retry_rounds,
            individual_subject_parallel_mode="isolated",
            on_subject_failure=on_subject_failure,
            HabitatSegmentation=HabitatSegmentationConfig(clustering_mode="two_step"),
        )

    return HabitatAnalysisConfig(
        data_dir=str(tmp_path / "data"),
        out_dir=str(tmp_path / "out"),
        run_mode="train",
        resume=False,
        individual_subject_auto_retry_rounds=auto_retry_rounds,
        individual_subject_parallel_mode="isolated",
        on_subject_failure=on_subject_failure,
        FeatureConstruction=FeatureConstructionConfig(
            voxel_level=VoxelLevelConfig(method="mean_voxel_features()", params={}),
        ),
        HabitatSegmentation=HabitatSegmentationConfig(clustering_mode="two_step"),
    )


class _StubPipeline:
    """Minimal pipeline stub for stage orchestration tests."""

    individual_steps: List[Any] = []
    mask_info_cache: Dict[str, Any] = {}

    def __init__(self, config: HabitatAnalysisConfig) -> None:
        self.config = config
        self._train_checkpoint = None

    def _process_single_subject(
        self,
        item: Tuple[str, Any],
    ) -> Tuple[str, HabitatSubjectData]:
        subject_id, _payload = item
        return subject_id, HabitatSubjectData(supervoxel_df=None)


@pytest.fixture
def stub_pipeline_factory(tmp_path: Path):
    pipeline_path = tmp_path / "habitat_pipeline.pkl"
    pipeline_path.write_bytes(b"pipeline")

    def _factory(**kwargs: Any) -> _StubPipeline:
        run_mode = kwargs.pop("run_mode", "train")
        if run_mode == "predict":
            config = _train_config(
                tmp_path,
                run_mode="predict",
                pipeline_path=pipeline_path,
                **kwargs,
            )
        else:
            config = _train_config(tmp_path, run_mode="train", **kwargs)
        return _StubPipeline(config)

    return _factory


def _build_parallel_map_side_effect(
    failing_subjects_by_pass: Dict[int, List[str]],
) -> Any:
    pass_index = {"value": 0}

    def _parallel_map(
        func,
        items,
        *,
        n_processes,
        desc,
        logger,
        show_progress,
        on_item_done=None,
        **kwargs,
    ):
        current_pass = pass_index["value"]
        pass_index["value"] += 1
        failing = set(failing_subjects_by_pass.get(current_pass, []))

        successful_results: List[ProcessingResult] = []
        failed_subjects: List[str] = []

        for subject_id, payload in items:
            if subject_id in failing:
                failed_subjects.append(subject_id)
                if on_item_done is not None:
                    on_item_done(
                        ProcessingResult(
                            item_id=subject_id,
                            error=RuntimeError("simulated failure"),
                        )
                    )
                continue

            result = func((subject_id, payload))
            successful_results.append(
                ProcessingResult(item_id=subject_id, result=result[1])
            )
            if on_item_done is not None:
                on_item_done(
                    ProcessingResult(item_id=subject_id, result=result[1])
                )

        return successful_results, failed_subjects

    return _parallel_map


def test_auto_retry_succeeds_on_second_pass(
    monkeypatch: pytest.MonkeyPatch,
    stub_pipeline_factory,
) -> None:
    pipeline = stub_pipeline_factory(auto_retry_rounds=2)
    stage = IndividualCheckpointStage(pipeline, logger=MagicMock())

    monkeypatch.setattr(
        "habit.core.habitat_analysis.checkpoint.stage.parallel_map",
        _build_parallel_map_side_effect({0: ["sub002"], 1: []}),
    )

    results = stage.run(
        {
            "sub001": HabitatSubjectData.empty(),
            "sub002": HabitatSubjectData.empty(),
        }
    )

    assert set(results.keys()) == {"sub001", "sub002"}
    assert stage.checkpoint is not None
    assert stage.checkpoint.manifest.failed_subjects == []
    assert set(stage.checkpoint.manifest.completed_subjects) == {"sub001", "sub002"}


def test_auto_retry_disabled_when_zero(
    monkeypatch: pytest.MonkeyPatch,
    stub_pipeline_factory,
) -> None:
    pipeline = stub_pipeline_factory(auto_retry_rounds=0)
    stage = IndividualCheckpointStage(pipeline, logger=MagicMock())
    call_count = {"value": 0}

    def _parallel_map(*args, **kwargs):
        call_count["value"] += 1
        return _build_parallel_map_side_effect({0: ["sub002"]})(*args, **kwargs)

    monkeypatch.setattr(
        "habit.core.habitat_analysis.checkpoint.stage.parallel_map",
        _parallel_map,
    )

    results = stage.run(
        {
            "sub001": HabitatSubjectData.empty(),
            "sub002": HabitatSubjectData.empty(),
        }
    )

    assert call_count["value"] == 1
    assert set(results.keys()) == {"sub001"}
    assert stage.checkpoint is not None
    assert stage.checkpoint.manifest.failed_subjects == ["sub002"]


def test_auto_retry_respects_max_rounds(
    monkeypatch: pytest.MonkeyPatch,
    stub_pipeline_factory,
) -> None:
    pipeline = stub_pipeline_factory(auto_retry_rounds=2)
    stage = IndividualCheckpointStage(pipeline, logger=MagicMock())

    monkeypatch.setattr(
        "habit.core.habitat_analysis.checkpoint.stage.parallel_map",
        _build_parallel_map_side_effect({0: ["sub002"], 1: ["sub002"], 2: ["sub002"]}),
    )

    results = stage.run({"sub001": HabitatSubjectData.empty(), "sub002": HabitatSubjectData.empty()})

    assert set(results.keys()) == {"sub001"}
    assert stage.checkpoint is not None
    assert stage.checkpoint.manifest.failed_subjects == ["sub002"]


def test_fail_fast_after_auto_retry_exhausted(
    monkeypatch: pytest.MonkeyPatch,
    stub_pipeline_factory,
) -> None:
    pipeline = stub_pipeline_factory(auto_retry_rounds=1, on_subject_failure="fail_fast")
    stage = IndividualCheckpointStage(pipeline, logger=MagicMock())

    monkeypatch.setattr(
        "habit.core.habitat_analysis.checkpoint.stage.parallel_map",
        _build_parallel_map_side_effect({0: ["sub002"], 1: ["sub002"]}),
    )

    with pytest.raises(RuntimeError, match="fail_fast"):
        stage.run({"sub001": HabitatSubjectData.empty(), "sub002": HabitatSubjectData.empty()})


def test_parallel_processes_capped_to_torch_gpu_pool(
    monkeypatch: pytest.MonkeyPatch,
    stub_pipeline_factory,
    tmp_path: Path,
) -> None:
    pipeline = stub_pipeline_factory(auto_retry_rounds=0)
    pipeline.config.processes = 4
    pipeline.config.FeatureConstruction.voxel_level.params = {
        "useTorchRadiomics": "true",
        "torchGpus": [0, 1],
    }
    stage = IndividualCheckpointStage(pipeline, logger=MagicMock())

    observed_processes: list[int] = []

    def _parallel_map(*args, **kwargs):
        observed_processes.append(kwargs["n_processes"])
        return _build_parallel_map_side_effect({0: []})(*args, **kwargs)

    monkeypatch.setattr(
        "habit.core.habitat_analysis.checkpoint.stage.parallel_map",
        _parallel_map,
    )

    stage.run(
        {
            "sub001": HabitatSubjectData.empty(),
            "sub002": HabitatSubjectData.empty(),
            "sub003": HabitatSubjectData.empty(),
        }
    )

    assert observed_processes == [2]


def test_predict_auto_retry_succeeds_on_second_pass(
    monkeypatch: pytest.MonkeyPatch,
    stub_pipeline_factory,
) -> None:
    pipeline = stub_pipeline_factory(auto_retry_rounds=2, run_mode="predict")
    stage = IndividualCheckpointStage(pipeline, logger=MagicMock())

    monkeypatch.setattr(
        "habit.core.habitat_analysis.checkpoint.stage.parallel_map",
        _build_parallel_map_side_effect({0: ["sub002"], 1: []}),
    )

    results = stage.run(
        {
            "sub001": HabitatSubjectData.empty(),
            "sub002": HabitatSubjectData.empty(),
        }
    )

    assert set(results.keys()) == {"sub001", "sub002"}
    assert stage.checkpoint is not None
    assert stage.checkpoint.run_mode == "predict"
    assert stage.checkpoint.manifest.failed_subjects == []

