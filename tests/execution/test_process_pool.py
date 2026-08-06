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
"""Contract tests for ProcessPoolBackend with synthetic operators.

All operators are defined at module level so they survive pickling into
spawned child processes; no imaging data is involved anywhere. Operators
that need observable side effects (attempt counting) persist them in small
counter files, which is the only state that can cross the process boundary
reliably in a test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import FrozenSet, List, Tuple

import pytest

from habit.api.exceptions import HABITAPIError
from habit.contracts import Subject
from habit.execution import (
    CheckpointStore,
    ProcessPoolBackend,
    SubjectTimeoutError,
)
from habit.spec.policy import RunPolicy


@dataclass(frozen=True)
class _WorkItem:
    """Minimal subject-scoped payload: light, immutable, picklable."""

    subject_id: str
    value: int


def _items(count: int = 3) -> List[_WorkItem]:
    """Build ``count`` synthetic subjects with distinct values."""
    return [_WorkItem(subject_id=f"s{i}", value=i + 1) for i in range(count)]


class SquareOp:
    """Synthetic operator returning the square of the item value."""

    def __call__(self, item: _WorkItem) -> int:
        return item.value * item.value

    def cache_key(self, item: _WorkItem) -> str:
        return f"square:{item.subject_id}:{item.value}"


class FlakyOp:
    """Synthetic operator that raises for a fixed set of subject ids."""

    def __init__(self, failing: FrozenSet[str]) -> None:
        self.failing = frozenset(failing)

    def __call__(self, item: _WorkItem) -> int:
        if item.subject_id in self.failing:
            raise RuntimeError(f"boom:{item.subject_id}")
        return item.value

    def cache_key(self, item: _WorkItem) -> str:
        return f"flaky:{item.subject_id}"


class SleeperOp:
    """Synthetic operator that sleeps before answering (timeout tests)."""

    def __init__(self, seconds: float) -> None:
        self.seconds = float(seconds)

    def __call__(self, item: _WorkItem) -> int:
        import time

        time.sleep(self.seconds)
        return item.value


class OomOp:
    """Synthetic operator raising MemoryError for a fixed set of subjects."""

    def __init__(self, ooming: FrozenSet[str]) -> None:
        self.ooming = frozenset(ooming)

    def __call__(self, item: _WorkItem) -> int:
        if item.subject_id in self.ooming:
            raise MemoryError("simulated OOM")
        return item.value


class SuicideOp:
    """Synthetic operator whose worker dies without reporting an outcome."""

    def __call__(self, item: _WorkItem) -> int:
        import os

        os._exit(1)
        return item.value  # pragma: no cover - unreachable


class LambdaResultOp:
    """Synthetic operator whose result cannot cross the process boundary."""

    def __call__(self, item: _WorkItem) -> object:
        return lambda: item.value  # lambdas are not picklable


class AttemptCountingOp:
    """
    Synthetic operator persisting per-subject attempt counts in files.

    File-backed counters are the only state a spawned child can mutate
    observably, which makes them the backbone of the resume/retry tests:
    the parent asserts afterwards how often each subject was attempted.

    Args:
        counter_dir: Directory holding one ``<subject_id>.count`` file per
            attempted subject.
        fail_attempts: Number of leading attempts that raise before the
            operator starts succeeding (0 = never fails).
    """

    def __init__(self, counter_dir: object, fail_attempts: int = 0) -> None:
        self.counter_dir = str(counter_dir)
        self.fail_attempts = int(fail_attempts)

    def _record_attempt(self, item: _WorkItem) -> int:
        """Increment and return the attempt counter of one subject."""
        path = Path(self.counter_dir) / f"{item.subject_id}.count"
        try:
            count = int(path.read_text(encoding="utf-8").strip())
        except FileNotFoundError:
            count = 0
        count += 1
        path.write_text(str(count), encoding="utf-8")
        return count

    def __call__(self, item: _WorkItem) -> int:
        if self._record_attempt(item) <= self.fail_attempts:
            raise RuntimeError(f"attempt failed for {item.subject_id}")
        return item.value * 10

    def cache_key(self, item: _WorkItem) -> str:
        return f"count:{item.subject_id}"


def _attempts_of(counter_dir: Path, subject_id: str) -> int:
    """Read the recorded attempt count of one subject (0 = never ran)."""
    path = counter_dir / f"{subject_id}.count"
    if not path.is_file():
        return 0
    return int(path.read_text(encoding="utf-8").strip())


# ---------------------------------------------------------------------------
# Policy surface (no child processes spawned)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_from_policy_transcribes_every_field() -> None:
    """from_policy is a pure field-by-field transcription of RunPolicy."""
    policy = RunPolicy(
        workers=3,
        backend="process",
        subject_timeout_sec=42.0,
        subject_spawn_timeout_sec=7.0,
        graceful_shutdown_sec=3.0,
        on_subject_failure="fail_fast",
        oom_backoff=False,
        oom_reduce_workers_by=2,
        cap_workers_to_gpu_pool=False,
        resume=False,
        parallel_mode="isolated",
        auto_retry_rounds=0,
        retry_failed_subjects=True,
        force_rerun_subjects=("s9",),
        clear_checkpoint_on_success=True,
        persistent_worker_max_consecutive_failures=4,
        persistent_worker_recycle_after_tasks=9,
    )
    backend = ProcessPoolBackend.from_policy(policy)
    snapshot = backend.policy
    assert snapshot.workers == 3
    assert snapshot.backend == "process"
    assert snapshot.subject_timeout_sec == 42.0
    assert snapshot.subject_spawn_timeout_sec == 7.0
    assert snapshot.graceful_shutdown_sec == 3.0
    assert snapshot.on_subject_failure == "fail_fast"
    assert snapshot.oom_backoff is False
    assert snapshot.oom_reduce_workers_by == 2
    assert snapshot.resume is False
    assert snapshot.parallel_mode == "isolated"
    assert snapshot.auto_retry_rounds == 0
    assert snapshot.retry_failed_subjects is True
    assert snapshot.force_rerun_subjects == ("s9",)
    assert snapshot.clear_checkpoint_on_success is True
    assert snapshot.persistent_worker_max_consecutive_failures == 4
    assert snapshot.persistent_worker_recycle_after_tasks == 9
    assert backend.workers == 3


@pytest.mark.unit
def test_constructor_validates_like_run_policy() -> None:
    """Invalid policy values are rejected at the backend boundary."""
    with pytest.raises(HABITAPIError):
        ProcessPoolBackend(workers=0)
    with pytest.raises(HABITAPIError):
        ProcessPoolBackend(on_subject_failure="ignore")
    with pytest.raises(HABITAPIError):
        ProcessPoolBackend(parallel_mode="bogus")
    with pytest.raises(HABITAPIError):
        ProcessPoolBackend(subject_timeout_sec=-1.0)


@pytest.mark.unit
def test_gpu_pool_capping(monkeypatch: pytest.MonkeyPatch) -> None:
    """Workers clamp to the detected GPU pool; an undetectable pool is a no-op."""
    from habit.execution import process_pool

    monkeypatch.setattr(process_pool, "_detect_gpu_pool_size", lambda: 2)
    capped = ProcessPoolBackend(workers=8, cap_workers_to_gpu_pool=True)
    assert capped.workers == 2
    assert capped.gpu_pool_size == 2

    monkeypatch.setattr(process_pool, "_detect_gpu_pool_size", lambda: 0)
    uncapped = ProcessPoolBackend(workers=8, cap_workers_to_gpu_pool=True)
    assert uncapped.workers == 8

    disabled = ProcessPoolBackend(workers=8, cap_workers_to_gpu_pool=False)
    assert disabled.gpu_pool_size == 0
    assert disabled.workers == 8


# ---------------------------------------------------------------------------
# Execution across child processes (synthetic operators only)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("mode", ("persistent", "isolated"))
def test_both_modes_compute_every_subject(mode: str) -> None:
    """Both worker lifecycles return one result per subject, in any order."""
    backend = ProcessPoolBackend(
        workers=2, parallel_mode=mode, auto_retry_rounds=0
    )
    progress: List[Tuple[int, int]] = []
    results = list(
        backend.map(SquareOp(), _items(3), progress=lambda c, t: progress.append((c, t)))
    )

    by_subject = {r.subject_id: r for r in results}
    assert set(by_subject) == {"s0", "s1", "s2"}
    assert by_subject["s0"].result() == 1
    assert by_subject["s1"].result() == 4
    assert by_subject["s2"].result() == 9
    assert all(r.error is None for r in results)
    assert progress[-1] == (3, 3)


@pytest.mark.integration
def test_continue_isolates_subject_failure() -> None:
    """A failing subject lands in its result slot; the rest still succeed."""
    backend = ProcessPoolBackend(
        workers=2, parallel_mode="persistent", auto_retry_rounds=0
    )
    results = list(backend.map(FlakyOp(failing={"s1"}), _items(3)))

    by_subject = {r.subject_id: r for r in results}
    assert isinstance(by_subject["s1"].error, RuntimeError)
    assert by_subject["s0"].result() == 1
    assert by_subject["s2"].result() == 3


@pytest.mark.integration
def test_fail_fast_reraises_first_failure() -> None:
    """fail_fast aborts the run with the original subject exception."""
    backend = ProcessPoolBackend(
        workers=1,
        parallel_mode="persistent",
        on_subject_failure="fail_fast",
        auto_retry_rounds=0,
    )
    with pytest.raises(RuntimeError, match="boom"):
        list(backend.map(FlakyOp(failing={"s0"}), _items(3)))


@pytest.mark.integration
def test_subject_timeout_marks_failure_without_hanging() -> None:
    """A subject exceeding its wall-clock budget is terminated and reported."""
    backend = ProcessPoolBackend(
        workers=1,
        parallel_mode="persistent",
        subject_timeout_sec=2.0,
        subject_spawn_timeout_sec=None,
        graceful_shutdown_sec=2.0,
        auto_retry_rounds=0,
    )
    results = list(backend.map(SleeperOp(seconds=30.0), _items(1)))

    assert len(results) == 1
    assert isinstance(results[0].error, SubjectTimeoutError)


@pytest.mark.integration
def test_worker_death_is_reported_per_subject() -> None:
    """A worker that dies mid-subject yields an error, never a hang."""
    backend = ProcessPoolBackend(
        workers=1, parallel_mode="persistent", auto_retry_rounds=0
    )
    results = list(backend.map(SuicideOp(), _items(1)))

    assert len(results) == 1
    assert results[0].error is not None
    assert "exited" in str(results[0].error)


@pytest.mark.integration
def test_oom_backoff_still_completes_the_run() -> None:
    """A fatal memory error is isolated; remaining subjects still complete."""
    backend = ProcessPoolBackend(
        workers=2, parallel_mode="persistent", auto_retry_rounds=0
    )
    results = list(backend.map(OomOp(ooming={"s1"}), _items(3)))

    by_subject = {r.subject_id: r for r in results}
    assert isinstance(by_subject["s1"].error, MemoryError)
    assert by_subject["s0"].result() == 1
    assert by_subject["s2"].result() == 3


@pytest.mark.integration
def test_unpicklable_result_surfaces_as_subject_error() -> None:
    """A result that cannot cross the boundary is an isolated failure."""
    backend = ProcessPoolBackend(
        workers=1, parallel_mode="persistent", auto_retry_rounds=0
    )
    results = list(backend.map(LambdaResultOp(), _items(1)))

    assert len(results) == 1
    assert isinstance(results[0].error, HABITAPIError)
    assert "not picklable" in str(results[0].error)


@pytest.mark.integration
def test_unpicklable_payload_fails_fast_in_parent() -> None:
    """An unpicklable item is rejected before any child is spawned."""
    backend = ProcessPoolBackend(
        workers=1, parallel_mode="persistent", auto_retry_rounds=0
    )
    bad_item = _WorkItem(subject_id="bad", value=1)
    object.__setattr__(bad_item, "closure", lambda: 1)  # not picklable
    results = list(backend.map(SquareOp(), [bad_item, *_items(1)]))

    by_subject = {r.subject_id: r for r in results}
    assert isinstance(by_subject["bad"].error, HABITAPIError)
    assert "not picklable" in str(by_subject["bad"].error)
    assert by_subject["s0"].result() == 1


@pytest.mark.integration
def test_auto_retry_rounds_recovers_flaky_subject(tmp_path: Path) -> None:
    """A subject failing once succeeds in the same run's retry round."""
    backend = ProcessPoolBackend(
        workers=1, parallel_mode="persistent", auto_retry_rounds=2
    )
    op = AttemptCountingOp(tmp_path, fail_attempts=1)
    results = list(backend.map(op, _items(2)))

    by_subject = {r.subject_id: r for r in results}
    assert by_subject["s0"].result() == 10
    assert by_subject["s1"].result() == 20
    assert _attempts_of(tmp_path, "s0") == 2  # initial attempt + one retry


@pytest.mark.integration
def test_resume_skips_checkpointed_subjects(tmp_path: Path) -> None:
    """A second run restores checkpointed values without recomputation."""
    store = CheckpointStore(tmp_path / "ckpt")
    counters = tmp_path / "counters"
    counters.mkdir()
    backend = ProcessPoolBackend(
        workers=2, parallel_mode="persistent", auto_retry_rounds=0
    )
    op = AttemptCountingOp(counters)

    first = list(backend.map(op, _items(3), checkpoint=store))
    second = list(backend.map(op, _items(3), checkpoint=store))

    assert [r.result() for r in sorted(first, key=lambda r: r.subject_id)] == [10, 20, 30]
    assert all(r.from_cache for r in second)
    assert all(
        _attempts_of(counters, f"s{i}") == 1 for i in range(3)
    )  # computed exactly once per subject


@pytest.mark.integration
def test_recorded_failure_is_skipped_on_resume(tmp_path: Path) -> None:
    """The v0.1 rule: checkpoint-failed subjects skip unless asked to retry."""
    store = CheckpointStore(tmp_path / "ckpt")
    counters = tmp_path / "counters"
    counters.mkdir()
    store.put_failure("count:s1", "RuntimeError: earlier crash")

    backend = ProcessPoolBackend(
        workers=1, parallel_mode="persistent", auto_retry_rounds=0
    )
    results = list(backend.map(AttemptCountingOp(counters), _items(2), checkpoint=store))

    by_subject = {r.subject_id: r for r in results}
    assert by_subject["s0"].result() == 10
    assert by_subject["s1"].error is not None
    assert by_subject["s1"].from_cache
    assert "recorded checkpoint failure" in str(by_subject["s1"].error)
    assert _attempts_of(counters, "s1") == 0  # never recomputed


@pytest.mark.integration
def test_retry_failed_subjects_recomputes_and_clears(tmp_path: Path) -> None:
    """retry_failed_subjects re-runs recorded failures and clears their record."""
    store = CheckpointStore(tmp_path / "ckpt")
    counters = tmp_path / "counters"
    counters.mkdir()
    store.put_failure("count:s1", "RuntimeError: earlier crash")

    backend = ProcessPoolBackend(
        workers=1,
        parallel_mode="persistent",
        auto_retry_rounds=0,
        retry_failed_subjects=True,
    )
    results = list(backend.map(AttemptCountingOp(counters), _items(2), checkpoint=store))

    by_subject = {r.subject_id: r for r in results}
    assert by_subject["s1"].result() == 20
    assert _attempts_of(counters, "s1") == 1
    assert store.get_failure("count:s1") is None


@pytest.mark.integration
def test_force_rerun_subjects_recomputes_checkpoint_success(tmp_path: Path) -> None:
    """Forced subjects recompute even with a checkpoint success present."""
    store = CheckpointStore(tmp_path / "ckpt")
    counters = tmp_path / "counters"
    counters.mkdir()
    backend = ProcessPoolBackend(
        workers=1, parallel_mode="persistent", auto_retry_rounds=0
    )
    op = AttemptCountingOp(counters)
    list(backend.map(op, _items(2), checkpoint=store))

    forced = ProcessPoolBackend(
        workers=1,
        parallel_mode="persistent",
        auto_retry_rounds=0,
        force_rerun_subjects=("s1",),
    )
    results = list(forced.map(op, _items(2), checkpoint=store))

    by_subject = {r.subject_id: r for r in results}
    assert by_subject["s0"].from_cache
    assert not by_subject["s1"].from_cache
    assert _attempts_of(counters, "s0") == 1
    assert _attempts_of(counters, "s1") == 2  # recomputed once more


@pytest.mark.integration
def test_clear_checkpoint_on_success_empties_store(tmp_path: Path) -> None:
    """A fully successful run clears the checkpoint store when asked to."""
    store = CheckpointStore(tmp_path / "ckpt")
    backend = ProcessPoolBackend(
        workers=1,
        parallel_mode="persistent",
        auto_retry_rounds=0,
        clear_checkpoint_on_success=True,
    )
    results = list(backend.map(SquareOp(), _items(2), checkpoint=store))

    assert all(r.error is None for r in results)
    assert len(store) == 0
    assert store.failed_keys() == ()


class _SubjectUpperOp:
    """Module-level operator for the Subject pickling-boundary check."""

    def __call__(self, subject: Subject) -> str:
        return subject.subject_id.upper()


@pytest.mark.integration
def test_subject_contract_payloads_cross_the_boundary() -> None:
    """Real Subject payloads pickle into workers (the Phase 1 boundary claim)."""
    subjects = [
        Subject(subject_id=f"sub{i}", images={}, masks={}) for i in range(2)
    ]
    backend = ProcessPoolBackend(
        workers=2, parallel_mode="persistent", auto_retry_rounds=0
    )
    results = list(backend.map(_SubjectUpperOp(), subjects))

    assert sorted(r.result() for r in results) == ["SUB0", "SUB1"]
