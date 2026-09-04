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
"""Backend selection from RunPolicy (timeout uncoupled from spawn)."""

from __future__ import annotations

from typing import Any

import pytest

from habit.execution import (
    ProcessPoolBackend,
    SerialBackend,
    backend_from_policy,
    should_use_process_pool,
)
from habit.exceptions import CompatibilityError
from habit.execution.checkpoint import CheckpointStore
from habit.spec import RunPolicy


@pytest.mark.unit
def test_workers_one_with_default_timeout_stays_serial() -> None:
    """Default timeout must not force spawn when workers==1 and serial."""
    policy = RunPolicy(
        workers=1,
        backend="serial",
        subject_timeout_sec=900.0,
    )
    assert should_use_process_pool(policy) is False
    assert isinstance(backend_from_policy(policy), SerialBackend)


@pytest.mark.unit
def test_workers_one_without_timeout_stays_serial() -> None:
    """Null timeout + serial backend stays in-process."""
    policy = RunPolicy(
        workers=1,
        backend="serial",
        subject_timeout_sec=None,
    )
    assert should_use_process_pool(policy) is False
    assert isinstance(backend_from_policy(policy), SerialBackend)


@pytest.mark.unit
def test_explicit_process_backend_always_process_pool() -> None:
    """backend=process selects ProcessPool regardless of timeout."""
    policy = RunPolicy(
        workers=1,
        backend="process",
        subject_timeout_sec=None,
    )
    assert should_use_process_pool(policy) is True
    assert isinstance(backend_from_policy(policy), ProcessPoolBackend)


@pytest.mark.unit
def test_isolated_mode_selects_process_pool() -> None:
    """parallel_mode=isolated needs a child even when workers==1."""
    policy = RunPolicy(
        workers=1,
        backend="serial",
        parallel_mode="isolated",
        subject_timeout_sec=900.0,
    )
    assert should_use_process_pool(policy) is True
    backend = backend_from_policy(policy)
    assert isinstance(backend, ProcessPoolBackend)
    assert backend.policy.backend == "process"


@pytest.mark.unit
def test_workers_gt_one_selects_process_pool() -> None:
    """workers>1 forces process even if YAML left backend=serial."""
    policy = RunPolicy(
        workers=2,
        backend="serial",
        subject_timeout_sec=None,
    )
    assert should_use_process_pool(policy) is True
    backend = backend_from_policy(policy)
    assert isinstance(backend, ProcessPoolBackend)
    assert backend.workers == 2
    assert backend.policy.backend == "process"


@pytest.mark.unit
def test_from_policy_copies_persistent_worker_knobs() -> None:
    """persistent_worker_* fields survive from_policy transcription."""
    policy = RunPolicy(
        workers=2,
        backend="process",
        persistent_worker_max_consecutive_failures=3,
        persistent_worker_recycle_after_tasks=5,
    )
    backend = ProcessPoolBackend.from_policy(policy)
    assert backend.policy.persistent_worker_max_consecutive_failures == 3
    assert backend.policy.persistent_worker_recycle_after_tasks == 5


@pytest.mark.unit
def test_strict_checkpoint_hash_raises_on_fingerprint_mismatch(tmp_path: Any) -> None:
    """strict=True raises CompatibilityError when fingerprints diverge."""
    from pathlib import Path

    root = Path(tmp_path) / "ckpt"
    CheckpointStore(root, run_fingerprint="fp-a", strict=False)
    with pytest.raises(CompatibilityError, match="fingerprint mismatch"):
        CheckpointStore(root, run_fingerprint="fp-b", strict=True)


@pytest.mark.unit
def test_strict_checkpoint_hash_migrates_legacy_layout(tmp_path: Any) -> None:
    """strict=True auto-migrates a readable v0.1 layout instead of refusing it."""
    from pathlib import Path

    root = Path(tmp_path) / "legacy"
    root.mkdir()
    (root / "manifest.json").write_text(
        '{"completed_subjects": [], "failed_subjects": ["s_bad"], '
        '"clustering_mode": "two_step"}',
        encoding="utf-8",
    )
    store = CheckpointStore(root, run_fingerprint="fp", strict=True)
    assert not (root / "manifest.json").exists()
    assert store.get_failure("habitat.units:fp:s_bad") is not None


@pytest.mark.unit
def test_strict_checkpoint_hash_raises_on_corrupt_legacy_manifest(
    tmp_path: Any,
) -> None:
    """Corrupt v0.1 manifest.json still raises CompatibilityError."""
    from pathlib import Path

    root = Path(tmp_path) / "legacy_bad"
    root.mkdir()
    (root / "manifest.json").write_text("{not-json", encoding="utf-8")
    with pytest.raises(CompatibilityError, match="corrupt/unreadable"):
        CheckpointStore(root, run_fingerprint="fp", strict=True)
