"""Tests for parallel GPU slot utilities."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from habit.core.habitat_analysis.config_schemas import (
    FeatureConstructionConfig,
    HabitatAnalysisConfig,
    HabitatSegmentationConfig,
    VoxelLevelConfig,
)
from habit.utils.parallel_gpu_utils import (
    HABIT_GPU_SLOT_INDEX_ENV,
    apply_gpu_pool_process_cap,
    cap_processes_to_gpu_pool,
    inject_worker_gpu_slot_index,
    read_worker_gpu_slot_index,
    resolve_habitat_torch_gpu_pool,
)


def _minimal_config(tmp_path: Path, **voxel_params: object) -> HabitatAnalysisConfig:
    return HabitatAnalysisConfig(
        data_dir=str(tmp_path / "data"),
        out_dir=str(tmp_path / "out"),
        FeatureConstruction=FeatureConstructionConfig(
            voxel_level=VoxelLevelConfig(
                method="mean_voxel_features()",
                params=dict(voxel_params),
            ),
        ),
        HabitatSegmentation=HabitatSegmentationConfig(clustering_mode="two_step"),
    )


def test_read_worker_gpu_slot_index_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(HABIT_GPU_SLOT_INDEX_ENV, "2")
    assert read_worker_gpu_slot_index() == 2


def test_inject_worker_gpu_slot_index_respects_existing_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(HABIT_GPU_SLOT_INDEX_ENV, "1")
    params = inject_worker_gpu_slot_index({"gpuSlotIndex": 0, "subject": "sub001"})
    assert params["gpuSlotIndex"] == 0


def test_inject_worker_gpu_slot_index_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(HABIT_GPU_SLOT_INDEX_ENV, "1")
    params = inject_worker_gpu_slot_index({"subject": "sub001"})
    assert params["gpuSlotIndex"] == 1


def test_resolve_habitat_torch_gpu_pool_from_torch_gpus(
    tmp_path: Path,
) -> None:
    config = _minimal_config(
        tmp_path,
        useTorchRadiomics="true",
        torchGpus=[0, 1, 2],
        torchGpuCount=2,
    )
    assert resolve_habitat_torch_gpu_pool(config) == [0, 1]


def test_resolve_habitat_torch_gpu_pool_cpu_only_returns_empty(
    tmp_path: Path,
) -> None:
    config = _minimal_config(tmp_path, useTorchRadiomics="false")
    assert resolve_habitat_torch_gpu_pool(config) == []


def test_resolve_habitat_torch_gpu_pool_auto_cuda_defaults_to_device_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _minimal_config(tmp_path, useTorchRadiomics="auto")
    monkeypatch.setattr(
        "habit.utils.parallel_gpu_utils.is_cuda_available",
        lambda: True,
    )
    assert resolve_habitat_torch_gpu_pool(config) == [0]


def test_cap_processes_to_gpu_pool() -> None:
    logger = MagicMock()
    capped = cap_processes_to_gpu_pool(4, 2, log=logger, gpu_pool=[0, 1])
    assert capped == 2
    logger.warning.assert_called_once()


def test_apply_gpu_pool_process_cap_respects_flag(tmp_path: Path) -> None:
    config = _minimal_config(
        tmp_path,
        useTorchRadiomics="true",
        torchGpus=[0],
    )
    config.processes = 8
    config.cap_processes_to_gpu_pool = True
    assert apply_gpu_pool_process_cap(8, config) == 1

    config.cap_processes_to_gpu_pool = False
    assert apply_gpu_pool_process_cap(8, config) == 8


def test_apply_gpu_pool_process_cap_skips_cpu_only_backend(tmp_path: Path) -> None:
    config = _minimal_config(tmp_path, useTorchRadiomics="false")
    config.processes = 8
    config.cap_processes_to_gpu_pool = True
    assert apply_gpu_pool_process_cap(8, config) == 8


def _capture_gpu_slot_task(item: tuple[str, int]) -> tuple[str, int]:
    """Module-level task for spawn pickling in GPU slot integration tests."""
    subject_id, value = item
    slot = read_worker_gpu_slot_index()
    return subject_id, (slot, value)


def test_worker_wrapper_sets_gpu_slot_env() -> None:
    from habit.utils.isolated_runner import _worker_wrapper

    def _task(item: tuple[str, int]) -> tuple[str, tuple[int | None, int]]:
        subject_id, value = item
        return subject_id, (read_worker_gpu_slot_index(), value)

    result = _worker_wrapper(
        (_task, ("sub001", 7), None, logging.INFO, 1)
    )
    assert result.success
    assert result.item_id == "sub001"
    assert result.result == (1, 7)


def test_parallel_map_sets_worker_gpu_slot_env() -> None:
    from habit.utils.parallel_utils import parallel_map

    observed_slots: list[int | None] = []

    def _collect_result(proc_result) -> None:
        if proc_result.success:
            slot, _value = proc_result.result
            observed_slots.append(slot)

    items = [(f"s{i}", i) for i in range(4)]
    successful, failed = parallel_map(
        _capture_gpu_slot_task,
        items,
        n_processes=2,
        desc="gpu-slot-test",
        show_progress=False,
        per_item_timeout_sec=None,
        on_item_done=_collect_result,
    )
    assert failed == []
    assert len(successful) == 4
    assert set(observed_slots) <= {0, 1}
    assert len(observed_slots) == 4
