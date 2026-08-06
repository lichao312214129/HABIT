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
"""Tests for v0.1 → v1 CheckpointStore migration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import pytest

from habit.contracts.habitat import Supervoxelization
from habit.exceptions import CompatibilityError
from habit.execution import (
    CheckpointStore,
    SerialBackend,
    is_v01_checkpoint_layout,
    migrate_v01_checkpoint_if_needed,
)
from habit.contracts import Subject


@dataclass
class _FakeHabitatSubjectData:
    """Minimal duck-typed stand-in for v0.1 HabitatSubjectData pickles."""

    features: Optional[pd.DataFrame] = None
    raw: Optional[pd.DataFrame] = None
    mask_info: Optional[Dict[str, Any]] = None
    supervoxel_labels: Optional[np.ndarray] = None
    mean_voxel_features: Optional[pd.DataFrame] = None
    supervoxel_features: Optional[pd.DataFrame] = None
    supervoxel_df: Optional[pd.DataFrame] = None


def _write_v01_manifest(
    root: Path,
    *,
    completed: list[str],
    failed: list[str],
    clustering_mode: str = "two_step",
) -> None:
    """Write a synthetic v0.1 manifest.json."""
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "config_hash": "deadbeef",
        "individual_config_hash": "deadbeef",
        "clustering_mode": clustering_mode,
        "run_mode": "train",
        "completed_subjects": completed,
        "failed_subjects": failed,
        "stage": "individual",
    }
    (root / "manifest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def _convertible_payload(subject_id: str) -> _FakeHabitatSubjectData:
    """Build a payload that can become a Supervoxelization."""
    labels = np.zeros((4, 4, 4), dtype=np.int32)
    labels[1, 1, 1] = 1
    labels[1, 1, 2] = 2
    frame = pd.DataFrame(
        {
            "subject": [subject_id, subject_id],
            "supervoxel": [1, 2],
            "feat_a": [0.1, 0.2],
            "feat_b": [1.0, 2.0],
        }
    )
    mask_info = {
        "mask_array": (labels > 0).astype(np.uint8),
        "spacing": (1.0, 1.0, 1.0),
        "origin": (0.0, 0.0, 0.0),
        "direction": tuple(float(v) for v in np.eye(3).ravel()),
    }
    return _FakeHabitatSubjectData(
        supervoxel_labels=labels,
        supervoxel_df=frame,
        mask_info=mask_info,
    )


def _slim_payload(subject_id: str) -> _FakeHabitatSubjectData:
    """Slim Stage-1 pickle: supervoxel_df only (not scientifically reusable)."""
    return _FakeHabitatSubjectData(
        supervoxel_df=pd.DataFrame(
            {
                "subject": [subject_id],
                "supervoxel": [1],
                "feat_a": [0.5],
            }
        )
    )


@pytest.mark.unit
def test_is_v01_checkpoint_layout_detects_markers(tmp_path: Path) -> None:
    """manifest.json or subjects/ marks a v0.1 tree."""
    root = tmp_path / "ckpt"
    root.mkdir()
    assert is_v01_checkpoint_layout(root) is False
    (root / "manifest.json").write_text("{}", encoding="utf-8")
    assert is_v01_checkpoint_layout(root) is True


@pytest.mark.unit
def test_migrate_failures_and_convertible_success(tmp_path: Path) -> None:
    """Failed IDs become .failed; convertible completed become Supervoxelization."""
    root = tmp_path / "ckpt"
    _write_v01_manifest(
        root, completed=["s_ok"], failed=["s_bad"], clustering_mode="two_step"
    )
    subjects = root / "subjects"
    subjects.mkdir()
    joblib.dump(_convertible_payload("s_ok"), subjects / "s_ok.pkl")

    report = migrate_v01_checkpoint_if_needed(
        root, run_fingerprint="fp123", clustering_mode="two_step"
    )

    assert report.migrated is True
    assert report.completed_migrated == ["s_ok"]
    assert report.failed_migrated == ["s_bad"]
    assert report.completed_payload_skipped == []
    assert not (root / "manifest.json").exists()
    assert not (root / "subjects").exists()
    assert (root / ".v01_legacy_archive").is_dir()
    assert (root / "v01_migration_report.json").is_file()
    assert (root / "run_fingerprint.json").is_file()

    store = CheckpointStore(root, run_fingerprint="fp123", strict=True)
    cached = store.get("habitat.units:fp123:s_ok")
    assert isinstance(cached, Supervoxelization)
    assert cached.subject_id == "s_ok"
    assert store.get_failure("habitat.units:fp123:s_bad") is not None


@pytest.mark.unit
def test_migrate_slim_payload_skips_scientific_reuse(tmp_path: Path) -> None:
    """Slim supervoxel_df-only pickles are logged and left to recompute."""
    root = tmp_path / "ckpt"
    _write_v01_manifest(
        root, completed=["s_slim"], failed=[], clustering_mode="two_step"
    )
    subjects = root / "subjects"
    subjects.mkdir()
    joblib.dump(_slim_payload("s_slim"), subjects / "s_slim.pkl")

    report = migrate_v01_checkpoint_if_needed(root, run_fingerprint="fp")
    assert report.completed_migrated == []
    assert report.completed_payload_skipped == ["s_slim"]
    assert any("cannot be scientifically reused" in n for n in report.notes)

    store = CheckpointStore(root, run_fingerprint="fp")
    assert store.get("habitat.units:fp:s_slim") is None


@pytest.mark.unit
def test_store_open_auto_migrates_then_resume_skips_failures(
    tmp_path: Path,
) -> None:
    """Opening CheckpointStore migrates v0.1 layout; resume skips failures."""
    root = tmp_path / "ckpt"
    _write_v01_manifest(
        root, completed=[], failed=["s1"], clustering_mode="two_step"
    )

    store = CheckpointStore(
        root,
        run_fingerprint="fp",
        strict=False,
        clustering_mode="two_step",
    )
    assert not is_v01_checkpoint_layout(root)

    calls: list[str] = []

    def op(subject: Subject) -> str:
        calls.append(subject.subject_id)
        return subject.subject_id.upper()

    items = [
        Subject(subject_id="s0", images={}, masks={}),
        Subject(subject_id="s1", images={}, masks={}),
    ]
    # Operators without cache_key fall back to type:subject_id; wrap with
    # an explicit key matching the migrated units prefix.
    class _UnitsOp:
        def __call__(self, subject: Subject) -> str:
            return op(subject)

        def cache_key(self, subject: Subject) -> str:
            return f"habitat.units:fp:{subject.subject_id}"

    results = list(SerialBackend().map(_UnitsOp(), items, checkpoint=store))
    assert calls == ["s0"]  # s1 skipped via migrated failure
    assert results[0].result() == "S0"
    assert results[1].from_cache is True
    assert "recorded checkpoint failure" in str(results[1].error)


@pytest.mark.unit
def test_one_step_mode_uses_one_step_key_prefix(tmp_path: Path) -> None:
    """one_step clustering_mode writes habitat.one_step keys."""
    root = tmp_path / "ckpt"
    _write_v01_manifest(
        root, completed=[], failed=["x"], clustering_mode="one_step"
    )
    migrate_v01_checkpoint_if_needed(
        root, run_fingerprint="abc", clustering_mode="one_step"
    )
    store = CheckpointStore(root, run_fingerprint="abc")
    assert store.get_failure("habitat.one_step:abc:x") is not None
    assert store.get_failure("habitat.units:abc:x") is None


@pytest.mark.unit
def test_corrupt_manifest_raises_compatibility_error(tmp_path: Path) -> None:
    """Unreadable manifest.json is a hard CompatibilityError."""
    root = tmp_path / "ckpt"
    root.mkdir()
    (root / "manifest.json").write_text("not-json{", encoding="utf-8")
    with pytest.raises(CompatibilityError, match="corrupt/unreadable"):
        migrate_v01_checkpoint_if_needed(root, run_fingerprint="fp")


@pytest.mark.unit
def test_migrate_is_idempotent_after_archive(tmp_path: Path) -> None:
    """Second open finds no v0.1 markers and does not re-archive."""
    root = tmp_path / "ckpt"
    _write_v01_manifest(root, completed=[], failed=["a"])
    first = migrate_v01_checkpoint_if_needed(root, run_fingerprint="fp")
    assert first.migrated is True
    second = migrate_v01_checkpoint_if_needed(root, run_fingerprint="fp")
    assert second.migrated is False
    # Only one archive directory from the first migration.
    archives = list(root.glob(".v01_legacy_archive*"))
    assert len(archives) == 1
