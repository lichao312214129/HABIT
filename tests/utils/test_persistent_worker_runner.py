"""Tests for persistent long-lived spawn worker pools."""

from __future__ import annotations

import time
from typing import Any, Tuple

import pytest

from habit.utils.parallel_utils import parallel_map
from habit.utils.persistent_worker_runner import PersistentWorkerPoolSession


def _task_ok(item: Tuple[str, int]) -> Tuple[str, int]:
    subject_id, value = item
    return subject_id, value * 2


def _task_slow(item: Tuple[str, float]) -> Tuple[str, str]:
    subject_id, delay_sec = item
    time.sleep(delay_sec)
    return subject_id, "done"


def _task_fail(item: Tuple[str, int]) -> Tuple[str, int]:
    subject_id, _ = item
    raise ValueError(f"boom-{subject_id}")


def _task_memory_error(item: Tuple[str, int]) -> Tuple[str, int]:
    subject_id, _ = item
    raise MemoryError(f"simulated OOM-{subject_id}")


def _mixed_ok_fail_task(item: Tuple[str, int]) -> Tuple[str, int]:
    subject_id, should_fail = item
    if should_fail:
        raise ValueError(f"NaN-like failure for {subject_id}")
    return subject_id, should_fail


@pytest.mark.parametrize("n_processes", [1, 2])
def test_persistent_parallel_map_success(n_processes: int) -> None:
    items = [(f"s{i}", i) for i in range(6)]
    successful, failed = parallel_map(
        _task_ok,
        items,
        n_processes=n_processes,
        desc="test",
        show_progress=False,
        parallel_mode="persistent",
    )
    assert failed == []
    assert len(successful) == 6
    by_id = {r.item_id: r.result for r in successful}
    assert by_id["s2"] == 4


def test_persistent_pool_session_reused_across_map_calls() -> None:
    pool = PersistentWorkerPoolSession(
        max_workers=2,
        func=_task_ok,
        per_item_timeout_sec=None,
    )
    pool.start(logger=None)
    try:
        first_ok, first_failed = pool.map_items(
            items_list=[("a", 1), ("b", 2)],
            desc="batch-1",
            logger=None,
            show_progress=False,
        )
        second_ok, second_failed = pool.map_items(
            items_list=[("c", 3)],
            desc="batch-2",
            logger=None,
            show_progress=False,
        )
    finally:
        pool.shutdown(logger=None)

    assert first_failed == []
    assert second_failed == []
    assert len(first_ok) == 2
    assert len(second_ok) == 1
    assert second_ok[0].result == 6


def test_persistent_parallel_map_records_exception() -> None:
    items = [("good", 1), ("bad", 2)]
    successful, failed = parallel_map(
        _task_fail,
        items,
        n_processes=2,
        show_progress=False,
        parallel_mode="persistent",
    )
    assert successful == []
    assert set(failed) == {"good", "bad"}


def test_persistent_recovers_without_restart_on_value_error() -> None:
    """Recoverable failures must not block subsequent subjects on other slots."""
    items = [("ok_a", 0), ("bad_nan", 1), ("ok_b", 0)]
    successful, failed = parallel_map(
        _mixed_ok_fail_task,
        items,
        n_processes=2,
        show_progress=False,
        parallel_mode="persistent",
        per_item_timeout_sec=30.0,
        max_consecutive_failures=1,
    )
    assert failed == ["bad_nan"]
    assert len(successful) == 2
    assert {result.item_id for result in successful} == {"ok_a", "ok_b"}


def test_requires_worker_restart_distinguishes_recoverable_errors() -> None:
    from habit.utils.isolated_runner import ProcessingResult, requires_worker_restart

    recoverable = ProcessingResult(item_id="sub1", error=ValueError("NaN"))
    assert not requires_worker_restart(recoverable, worker_alive=True)
    assert requires_worker_restart(recoverable, worker_alive=False)

    fatal = ProcessingResult(item_id="sub2", error=MemoryError("OOM"))
    assert requires_worker_restart(fatal, worker_alive=True)


def test_persistent_timeout_restarts_worker_slot() -> None:
    items = [("fast", 0.1), ("slow", 5.0)]
    successful, failed = parallel_map(
        _task_slow,
        items,
        n_processes=2,
        per_item_timeout_sec=1.0,
        graceful_shutdown_sec=5.0,
        show_progress=False,
        parallel_mode="persistent",
    )
    assert "slow" in failed
    assert len(successful) == 1
    assert successful[0].item_id == "fast"


def test_persistent_memory_error_marks_item_failed() -> None:
    items = [("oom-subject", 1)]
    successful, failed = parallel_map(
        _task_memory_error,
        items,
        n_processes=1,
        per_item_timeout_sec=30.0,
        show_progress=False,
        parallel_mode="persistent",
    )
    assert successful == []
    assert failed == ["oom-subject"]


def test_persistent_falls_back_to_inprocess_when_single_worker_no_timeout() -> None:
    items = [("a", 3)]
    successful, failed = parallel_map(
        _task_ok,
        items,
        n_processes=1,
        per_item_timeout_sec=None,
        show_progress=False,
        parallel_mode="persistent",
    )
    assert failed == []
    assert successful[0].result == 6
