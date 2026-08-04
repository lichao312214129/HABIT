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
"""Contract tests for SerialBackend, SubjectResult and CheckpointStore."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from habit.contracts import Subject, SubjectResult
from habit.execution import CheckpointStore, SerialBackend


def _items() -> List[Subject]:
    """Three identity-only subjects for backend contract tests."""
    return [Subject(subject_id=f"s{i}", images={}, masks={}) for i in range(3)]


@pytest.mark.unit
def test_subject_result_returns_value_or_reraises() -> None:
    """result() mirrors concurrent.futures.Future.result semantics."""
    ok = SubjectResult(subject_id="a", value=10, error=None)
    assert ok.result() == 10

    failure = SubjectResult(subject_id="b", value=None, error=ValueError("nope"))
    with pytest.raises(ValueError, match="nope"):
        failure.result()


@pytest.mark.unit
def test_serial_backend_yields_in_input_order_with_progress() -> None:
    """Serial execution preserves order and reports (completed, total)."""
    backend = SerialBackend()
    progress: List[tuple[int, int]] = []

    results = list(
        backend.map(lambda s: s.subject_id.upper(), _items(), progress=lambda c, t: progress.append((c, t)))
    )

    assert [r.result() for r in results] == ["S0", "S1", "S2"]
    assert [r.subject_id for r in results] == ["s0", "s1", "s2"]
    assert progress[-1] == (3, 3)
    assert all(not r.from_cache for r in results)


@pytest.mark.unit
def test_serial_backend_continue_captures_subject_failure() -> None:
    """The continue policy isolates a failure in its SubjectResult slot."""

    def op(subject: Subject) -> str:
        if subject.subject_id == "s1":
            raise RuntimeError("boom")
        return subject.subject_id

    results = list(SerialBackend(on_subject_failure="continue").map(op, _items()))

    assert results[0].result() == "s0"
    assert isinstance(results[1].error, RuntimeError)
    assert results[2].result() == "s2"


@pytest.mark.unit
def test_serial_backend_fail_fast_reraises() -> None:
    """The fail_fast policy re-raises the original exception immediately."""

    def op(subject: Subject) -> str:
        raise KeyError("immediate")

    with pytest.raises(KeyError):
        list(SerialBackend(on_subject_failure="fail_fast").map(op, _items()[:1]))


@pytest.mark.unit
def test_serial_backend_resume_skips_completed_subjects(tmp_path: Path) -> None:
    """A checkpointed second run restores values without recomputation."""
    store = CheckpointStore(tmp_path / "ckpt")
    calls: List[str] = []

    def op(subject: Subject) -> str:
        calls.append(subject.subject_id)
        return subject.subject_id.upper()

    first = list(SerialBackend().map(op, _items(), checkpoint=store))
    second = list(SerialBackend().map(op, _items(), checkpoint=store))

    assert [r.result() for r in first] == ["S0", "S1", "S2"]
    assert [r.result() for r in second] == ["S0", "S1", "S2"]
    assert all(r.from_cache for r in second)
    assert calls == ["s0", "s1", "s2"]  # computed exactly once per subject


@pytest.mark.unit
def test_serial_backend_records_failure_and_skips_it_on_resume(tmp_path: Path) -> None:
    """A terminal failure is recorded, then skipped like v0.1 on resume."""
    store = CheckpointStore(tmp_path / "ckpt")
    calls: List[str] = []

    def op(subject: Subject) -> str:
        calls.append(subject.subject_id)
        if subject.subject_id == "s1":
            raise RuntimeError("boom")
        return subject.subject_id

    first = list(SerialBackend().map(op, _items(), checkpoint=store))
    assert isinstance(first[1].error, RuntimeError)
    assert store.failed_keys() != ()

    calls.clear()
    second = list(SerialBackend().map(op, _items(), checkpoint=store))
    # Nothing is recomputed: successes come from the store, the recorded
    # failure is surfaced as a cached failure result.
    assert calls == []
    assert all(r.from_cache for r in second)
    assert second[0].result() == "s0"
    assert "recorded checkpoint failure" in str(second[1].error)


@pytest.mark.unit
def test_serial_backend_retry_failed_subjects_recomputes(tmp_path: Path) -> None:
    """retry_failed_subjects re-runs recorded failures; success clears them."""
    store = CheckpointStore(tmp_path / "ckpt")
    calls: List[str] = []
    fail = {"s1": True}

    def op(subject: Subject) -> str:
        calls.append(subject.subject_id)
        if fail.get(subject.subject_id):
            raise RuntimeError("boom")
        return subject.subject_id

    list(SerialBackend().map(op, _items(), checkpoint=store))
    calls.clear()

    third = list(
        SerialBackend(retry_failed_subjects=True).map(op, _items(), checkpoint=store)
    )
    assert calls == ["s1"]
    assert isinstance(third[1].error, RuntimeError)

    # When the retry finally succeeds, the failure record disappears.
    fail["s1"] = False
    calls.clear()
    fourth = list(
        SerialBackend(retry_failed_subjects=True).map(op, _items(), checkpoint=store)
    )
    assert calls == ["s1"]
    assert fourth[1].result() == "s1"
    assert store.failed_keys() == ()


@pytest.mark.unit
def test_serial_backend_force_rerun_recomputes_cached_success(tmp_path: Path) -> None:
    """Forced subjects recompute even when a checkpoint success exists."""
    store = CheckpointStore(tmp_path / "ckpt")
    calls: List[str] = []

    def op(subject: Subject) -> str:
        calls.append(subject.subject_id)
        return subject.subject_id

    list(SerialBackend().map(op, _items(), checkpoint=store))
    calls.clear()
    second = list(
        SerialBackend(force_rerun_subjects=("s1",)).map(op, _items(), checkpoint=store)
    )
    assert calls == ["s1"]
    assert [r.from_cache for r in second] == [True, False, True]


@pytest.mark.unit
def test_serial_backend_resume_false_reads_nothing_but_still_writes(tmp_path: Path) -> None:
    """resume=False disables skipping, not recording (v0.1 semantics)."""
    store = CheckpointStore(tmp_path / "ckpt")
    calls: List[str] = []

    def op(subject: Subject) -> str:
        calls.append(subject.subject_id)
        return subject.subject_id

    list(SerialBackend().map(op, _items(), checkpoint=store))
    assert len(store) == 3

    second = list(SerialBackend(resume=False).map(op, _items(), checkpoint=store))
    assert calls == ["s0", "s1", "s2"] * 2
    assert all(not r.from_cache for r in second)
    assert len(store) == 3


@pytest.mark.unit
def test_serial_backend_clear_checkpoint_on_success(tmp_path: Path) -> None:
    """The store is cleared only after a run with zero failures."""
    store = CheckpointStore(tmp_path / "ckpt")

    def op(subject: Subject) -> str:
        return subject.subject_id

    list(SerialBackend(clear_checkpoint_on_success=True).map(op, _items(), checkpoint=store))
    assert len(store) == 0

    def flaky(subject: Subject) -> str:
        if subject.subject_id == "s1":
            raise RuntimeError("boom")
        return subject.subject_id

    list(SerialBackend(clear_checkpoint_on_success=True).map(flaky, _items(), checkpoint=store))
    assert len(store) == 2
    assert store.failed_keys() != ()


@pytest.mark.unit
def test_checkpoint_store_treats_corrupt_entry_as_miss(tmp_path: Path) -> None:
    """A corrupt checkpoint is a cache miss, never a crash."""
    store = CheckpointStore(tmp_path)
    store.put("key", {"value": 1})
    path = store._path_for("key")  # noqa: SLF001 - deliberate white-box check
    path.write_bytes(b"garbage")

    assert store.get("key") is None
    assert not path.exists()
    store.put("key", 2)
    assert store.get("key") == 2


@pytest.mark.unit
def test_backend_rejects_invalid_failure_policy() -> None:
    """Only 'continue' / 'fail_fast' are accepted policies."""
    with pytest.raises(ValueError):
        SerialBackend(on_subject_failure="ignore")


@pytest.mark.unit
def test_checkpoint_store_tracks_failures_separately(tmp_path: Path) -> None:
    """Failure records list original keys, retrievable and discardable."""
    store = CheckpointStore(tmp_path / "ckpt")
    assert store.failed_keys() == ()
    assert store.get_failure("k1") is None

    store.put_failure("k2", "MemoryError: simulated OOM")
    store.put_failure("k1", "RuntimeError: boom")

    assert store.get_failure("k1") == "RuntimeError: boom"
    assert store.failed_keys() == ("k1", "k2")  # sorted, original keys

    store.discard_failure("k1")
    assert store.get_failure("k1") is None
    assert store.failed_keys() == ("k2",)


@pytest.mark.unit
def test_checkpoint_store_success_clears_failure_record(tmp_path: Path) -> None:
    """A retried subject that finally succeeds resumes cleanly."""
    store = CheckpointStore(tmp_path / "ckpt")
    store.put_failure("k1", "RuntimeError: boom")
    store.put("k1", 42)

    assert store.get("k1") == 42
    assert store.get_failure("k1") is None
    assert store.failed_keys() == ()


@pytest.mark.unit
def test_checkpoint_store_treats_corrupt_failure_as_absent(tmp_path: Path) -> None:
    """A corrupt failure record is a cache miss, never a crash."""
    store = CheckpointStore(tmp_path / "ckpt")
    store.put_failure("k1", "RuntimeError: boom")
    path = store._failure_path_for("k1")  # noqa: SLF001 - deliberate white-box check
    path.write_bytes(b"garbage")

    assert store.get_failure("k1") is None
    assert not path.exists()
    assert store.failed_keys() == ()


@pytest.mark.unit
def test_checkpoint_store_contains_len_and_clear(tmp_path: Path) -> None:
    """Whole-store operations: contains, __len__, clear."""
    store = CheckpointStore(tmp_path / "ckpt")
    assert len(store) == 0
    assert not store.contains("k1")

    store.put("k1", 1)
    store.put("k2", 2)
    store.put_failure("k3", "RuntimeError: boom")
    assert store.contains("k1")
    assert len(store) == 2

    store.clear()
    assert len(store) == 0
    assert store.failed_keys() == ()
    assert not store.contains("k1")
