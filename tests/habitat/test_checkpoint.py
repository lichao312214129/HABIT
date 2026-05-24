"""Tests for habitat training checkpoint/resume."""

from __future__ import annotations

from pathlib import Path

import pytest

from habit.core.habitat_analysis.checkpoint.manager import (
    HabitatTrainCheckpoint,
    build_individual_stage_config_payload,
    compute_config_hash,
    compute_legacy_config_hash,
    is_checkpoint_config_compatible,
)
from habit.core.habitat_analysis.config_schemas import (
    FeatureConstructionConfig,
    HabitatAnalysisConfig,
    HabitatSegmentationConfig,
    PreprocessingConfig,
    PreprocessingMethod,
    VoxelLevelConfig,
)
from habit.core.habitat_analysis.pipelines.habitat_subject_data import HabitatSubjectData


def _minimal_train_config(tmp_path: Path, *, data_dir: str | None = None) -> HabitatAnalysisConfig:
    return HabitatAnalysisConfig(
        data_dir=data_dir or str(tmp_path / "data"),
        out_dir=str(tmp_path / "out"),
        FeatureConstruction=FeatureConstructionConfig(
            voxel_level=VoxelLevelConfig(method="mean_voxel_features()", params={}),
        ),
        HabitatSegmentation=HabitatSegmentationConfig(clustering_mode="two_step"),
    )


def test_compute_config_hash_changes_when_feature_config_changes(
    tmp_path: Path,
) -> None:
    config_a = _minimal_train_config(tmp_path)
    config_b = _minimal_train_config(tmp_path)
    config_b.FeatureConstruction.voxel_level.method = "other_method()"
    assert compute_config_hash(config_a) != compute_config_hash(config_b)


def test_compute_config_hash_ignores_group_preprocessing_change(
    tmp_path: Path,
) -> None:
    config_a = _minimal_train_config(tmp_path)
    config_b = _minimal_train_config(tmp_path)
    config_b.FeatureConstruction.preprocessing_for_group_level = PreprocessingConfig(
        methods=[
            PreprocessingMethod(
                method="variance_filter",
                variance_threshold=0.01,
                global_normalize=False,
            )
        ]
    )
    assert compute_config_hash(config_a) == compute_config_hash(config_b)


def test_compute_config_hash_ignores_group_habitat_change_for_two_step(
    tmp_path: Path,
) -> None:
    config_a = _minimal_train_config(tmp_path)
    config_b = _minimal_train_config(tmp_path)
    config_b.HabitatSegmentation.habitat.fixed_n_clusters = 6
    config_b.HabitatSegmentation.habitat.algorithm = "gmm"
    assert compute_config_hash(config_a) == compute_config_hash(config_b)


def test_legacy_hash_changes_when_group_preprocessing_changes(
    tmp_path: Path,
) -> None:
    config_a = _minimal_train_config(tmp_path)
    config_b = _minimal_train_config(tmp_path)
    config_b.FeatureConstruction.preprocessing_for_group_level = PreprocessingConfig(
        methods=[PreprocessingMethod(method="binning", n_bins=5, global_normalize=False)]
    )
    assert compute_legacy_config_hash(config_a) != compute_legacy_config_hash(config_b)


def test_resume_keeps_checkpoint_when_only_group_preprocessing_changes(
    tmp_path: Path,
) -> None:
    config_old = _minimal_train_config(tmp_path)
    config_old.FeatureConstruction.preprocessing_for_group_level = PreprocessingConfig(
        methods=[PreprocessingMethod(method="binning", n_bins=10, global_normalize=False)]
    )
    checkpoint_dir = tmp_path / "ckpt"
    manager = HabitatTrainCheckpoint(checkpoint_dir, config_old)
    manager.initialize_for_run(resume=False)
    manager.record_success("sub001", HabitatSubjectData())

    # Simulate legacy v1 manifest that stored full-config hash.
    legacy_hash = compute_legacy_config_hash(config_old)
    manager.manifest.config_hash = legacy_hash
    manager.manifest.individual_config_hash = ""
    manager._write_manifest()

    config_new = _minimal_train_config(tmp_path)
    config_new.FeatureConstruction.preprocessing_for_group_level = PreprocessingConfig(
        methods=[
            PreprocessingMethod(
                method="correlation_filter",
                corr_threshold=0.9,
                corr_method="spearman",
                global_normalize=False,
            )
        ]
    )
    assert compute_config_hash(config_new) == compute_config_hash(config_old)
    assert compute_legacy_config_hash(config_new) != legacy_hash

    manager_new = HabitatTrainCheckpoint(checkpoint_dir, config_new)
    manager_new.initialize_for_run(resume=True)
    assert manager_new.manifest.completed_subjects == ["sub001"]
    assert (checkpoint_dir / "subjects" / "sub001.pkl").exists()
    assert manager_new.manifest.config_hash == compute_config_hash(config_new)
    assert manager_new.manifest.individual_config_hash == compute_config_hash(config_new)


def test_is_checkpoint_config_compatible_with_legacy_manifest(
    tmp_path: Path,
) -> None:
    config = _minimal_train_config(tmp_path)
    legacy_hash = compute_legacy_config_hash(config)
    assert is_checkpoint_config_compatible(legacy_hash, config)


def test_individual_payload_excludes_group_preprocessing(
    tmp_path: Path,
) -> None:
    config = _minimal_train_config(tmp_path)
    config.FeatureConstruction.preprocessing_for_group_level = PreprocessingConfig(
        methods=[PreprocessingMethod(method="binning", n_bins=10, global_normalize=False)]
    )
    payload = build_individual_stage_config_payload(config)
    assert "preprocessing_for_group_level" not in payload["FeatureConstruction"]


def test_resume_skips_completed_and_failed_subjects(tmp_path: Path) -> None:
    config = _minimal_train_config(tmp_path)
    checkpoint_dir = tmp_path / "ckpt"
    manager = HabitatTrainCheckpoint(checkpoint_dir, config)
    manager.initialize_for_run(resume=False)

    subject_data = HabitatSubjectData(supervoxel_df=None)
    manager.record_success("sub001", subject_data)
    manager.record_failure("sub002")

    manager.initialize_for_run(resume=True)
    pending = manager.pending_subjects(
        ["sub001", "sub002", "sub003"],
        resume=True,
    )
    assert pending == ["sub003"]


def test_config_change_clears_checkpoint_and_restarts(tmp_path: Path) -> None:
    config = _minimal_train_config(tmp_path)
    checkpoint_dir = tmp_path / "ckpt"
    manager = HabitatTrainCheckpoint(checkpoint_dir, config)
    manager.initialize_for_run(resume=False)
    manager.record_success("sub001", HabitatSubjectData())

    changed = _minimal_train_config(tmp_path)
    changed.FeatureConstruction.voxel_level.method = "changed_method()"
    manager_changed = HabitatTrainCheckpoint(checkpoint_dir, changed)
    manager_changed.initialize_for_run(resume=True)

    assert manager_changed.manifest.completed_subjects == []
    assert not (checkpoint_dir / "subjects" / "sub001.pkl").exists()


def test_force_rerun_subjects_requeues_completed_subject(tmp_path: Path) -> None:
    config = _minimal_train_config(tmp_path)
    checkpoint_dir = tmp_path / "ckpt"
    manager = HabitatTrainCheckpoint(checkpoint_dir, config)
    manager.initialize_for_run(resume=False)
    manager.record_success("sub001", HabitatSubjectData())

    manager.initialize_for_run(resume=True)
    pending = manager.pending_subjects(
        ["sub001", "sub002"],
        resume=True,
        force_rerun_subjects=["sub001"],
    )
    assert pending == ["sub001", "sub002"]
    assert "sub001" not in manager.manifest.completed_subjects


def test_retry_failed_subjects_requeues_all_failed(tmp_path: Path) -> None:
    config = _minimal_train_config(tmp_path)
    checkpoint_dir = tmp_path / "ckpt"
    manager = HabitatTrainCheckpoint(checkpoint_dir, config)
    manager.initialize_for_run(resume=False)
    manager.record_success("sub001", HabitatSubjectData())
    manager.record_failure("sub002")
    manager.record_failure("sub003")

    manager.initialize_for_run(resume=True)
    pending = manager.pending_subjects(
        ["sub001", "sub002", "sub003", "sub004"],
        resume=True,
        retry_failed_subjects=True,
    )
    assert pending == ["sub002", "sub003", "sub004"]
    assert manager.manifest.failed_subjects == []
    assert "sub001" in manager.manifest.completed_subjects


def test_retry_failed_subjects_false_skips_failed(tmp_path: Path) -> None:
    config = _minimal_train_config(tmp_path)
    checkpoint_dir = tmp_path / "ckpt"
    manager = HabitatTrainCheckpoint(checkpoint_dir, config)
    manager.initialize_for_run(resume=False)
    manager.record_failure("sub002")

    manager.initialize_for_run(resume=True)
    pending = manager.pending_subjects(
        ["sub001", "sub002"],
        resume=True,
        retry_failed_subjects=False,
    )
    assert pending == ["sub001"]
    assert manager.manifest.failed_subjects == ["sub002"]


def test_load_subject_pkl_reads_single_subject(tmp_path: Path) -> None:
    config = _minimal_train_config(tmp_path)
    checkpoint_dir = tmp_path / "ckpt"
    manager = HabitatTrainCheckpoint(checkpoint_dir, config)
    manager.initialize_for_run(resume=False)

    cached = HabitatSubjectData(supervoxel_df=None)
    manager.save_subject_pkl("sub010", cached)

    loaded = manager.load_subject_pkl("sub010")
    assert isinstance(loaded, HabitatSubjectData)


def test_load_completed_results_reads_cached_subject_data(tmp_path: Path) -> None:
    config = _minimal_train_config(tmp_path)
    checkpoint_dir = tmp_path / "ckpt"
    manager = HabitatTrainCheckpoint(checkpoint_dir, config)
    manager.initialize_for_run(resume=False)

    cached = HabitatSubjectData(supervoxel_df=None)
    manager.record_success("sub010", cached)

    loaded = manager.load_completed_results()
    assert "sub010" in loaded
    assert isinstance(loaded["sub010"], HabitatSubjectData)


def test_record_success_manifest_without_rewriting_pkl(tmp_path: Path) -> None:
    config = _minimal_train_config(tmp_path)
    checkpoint_dir = tmp_path / "ckpt"
    manager = HabitatTrainCheckpoint(checkpoint_dir, config)
    manager.initialize_for_run(resume=False)

    data = HabitatSubjectData(supervoxel_df=None)
    manager.save_subject_pkl("sub001", data)
    subject_path = checkpoint_dir / "subjects" / "sub001.pkl"
    assert subject_path.exists()
    mtime_before = subject_path.stat().st_mtime

    manager.record_success_manifest("sub001")
    assert "sub001" in manager.manifest.completed_subjects
    assert subject_path.stat().st_mtime == mtime_before


def test_clear_removes_checkpoint_directory(tmp_path: Path) -> None:
    config = _minimal_train_config(tmp_path)
    checkpoint_dir = tmp_path / "ckpt"
    manager = HabitatTrainCheckpoint(checkpoint_dir, config)
    manager.initialize_for_run(resume=False)
    manager.record_success("sub001", HabitatSubjectData())
    assert checkpoint_dir.exists()

    manager.clear()
    assert not checkpoint_dir.exists()


def test_checkpoint_save_step_writes_pkl_and_returns_payload(tmp_path: Path) -> None:
    from habit.core.habitat_analysis.checkpoint.step import CheckpointSaveStep

    config = _minimal_train_config(tmp_path)
    step = CheckpointSaveStep(config)
    checkpoint_dir = tmp_path / "ckpt"
    manager = HabitatTrainCheckpoint(checkpoint_dir, config)
    manager.initialize_for_run(resume=False)
    step.set_checkpoint(manager)

    subject_data = HabitatSubjectData(supervoxel_df=None)
    payload = step.transform_one("sub001", subject_data)

    assert payload is subject_data
    loaded = manager.load_subject_pkl("sub001")
    assert isinstance(loaded, HabitatSubjectData)
    assert loaded.supervoxel_df is None
