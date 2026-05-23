"""Tests for isolated per-item spawn execution."""

from __future__ import annotations

import time
from typing import Any, Tuple

import pytest

from habit.utils.isolated_runner import (
    IsolatedTaskRunner,
    ProcessingResult,
    is_fatal_memory_error,
)
from habit.utils.parallel_utils import parallel_map


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


def _task_ok_or_oom(item: Tuple[str, str]) -> Tuple[str, Any]:
    subject_id, mode = item
    if mode == "oom":
        raise MemoryError(f"simulated OOM-{subject_id}")
    return _task_ok((subject_id, 1))


@pytest.mark.parametrize("n_processes", [1, 2])
def test_parallel_map_success(n_processes: int) -> None:
    items = [(f"s{i}", i) for i in range(6)]
    successful, failed = parallel_map(
        _task_ok,
        items,
        n_processes=n_processes,
        desc="test",
        show_progress=False,
    )
    assert failed == []
    assert len(successful) == 6
    by_id = {r.item_id: r.result for r in successful}
    assert by_id["s2"] == 4


def test_parallel_map_records_exception() -> None:
    items = [("good", 1), ("bad", 2)]
    successful, failed = parallel_map(
        _task_fail,
        items,
        n_processes=2,
        show_progress=False,
    )
    assert successful == []
    assert len(failed) == 2


def test_timeout_terminates_slow_item() -> None:
    items = [("fast", 0.1), ("slow", 5.0)]
    successful, failed = parallel_map(
        _task_slow,
        items,
        n_processes=2,
        per_item_timeout_sec=1.0,
        graceful_shutdown_sec=5.0,
        show_progress=False,
    )
    assert "slow" in failed
    assert len(successful) == 1
    assert successful[0].item_id == "fast"


def test_isolated_runner_bounded_workers() -> None:
    runner = IsolatedTaskRunner(max_workers=2, per_item_timeout_sec=None)
    items = [(f"s{i}", 0.2) for i in range(4)]
    successful, failed = runner.map_items(
        func=_task_slow,
        items_list=items,
        desc="test",
        logger=None,
        show_progress=False,
        log_file_path=None,
        log_level=20,
    )
    assert failed == []
    assert len(successful) == 4


def test_inprocess_when_single_worker_no_timeout() -> None:
    items = [("a", 3)]
    successful, failed = parallel_map(
        _task_ok,
        items,
        n_processes=1,
        per_item_timeout_sec=None,
        show_progress=False,
    )
    assert failed == []
    assert successful[0].result == 6


def _task_large_payload(item: Tuple[str, int]) -> Tuple[str, bytes]:
    """Return a multi-megabyte payload to stress queue IPC on Windows spawn."""
    subject_id, nbytes = item
    return subject_id, b"x" * nbytes


def test_large_result_survives_queue_ipc() -> None:
    """Large worker payloads must reach the parent (Windows queue flush)."""
    items = [(f"large-{i}", 4 * 1024 * 1024) for i in range(3)]
    successful, failed = parallel_map(
        _task_large_payload,
        items,
        n_processes=2,
        show_progress=False,
    )
    assert failed == []
    assert len(successful) == 3
    assert len(successful[0].result) == 4 * 1024 * 1024


def test_memory_error_releases_slot_without_waiting_for_timeout() -> None:
    """OOM subjects must fail fast so pending items can start immediately."""
    items = [("oom-sub", "oom"), ("ok-sub", "ok")]
    started_at = time.monotonic()
    successful, failed = parallel_map(
        _task_ok_or_oom,
        items,
        n_processes=2,
        per_item_timeout_sec=600.0,
        show_progress=False,
    )
    elapsed_sec = time.monotonic() - started_at

    assert elapsed_sec < 10.0
    assert "oom-sub" in failed
    assert len(successful) == 1
    assert successful[0].item_id == "ok-sub"
    assert successful[0].result == 2


def test_is_fatal_memory_error_recognizes_array_memory_error_by_name() -> None:
    class ArrayMemoryError(Exception):
        """Minimal stand-in for numpy.core.exceptions.ArrayMemoryError."""

    exc = ArrayMemoryError("Unable to allocate 15.7 GiB")
    assert is_fatal_memory_error(exc)


def test_release_child_skips_terminate_on_fatal_memory_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminate_calls: list[Any] = []

    def _track_terminate(
        self: IsolatedTaskRunner,
        proc: Any,
        logger: Any = None,
    ) -> None:
        terminate_calls.append(proc)

    monkeypatch.setattr(
        IsolatedTaskRunner,
        "_terminate_process",
        _track_terminate,
    )

    class _FakeProc:
        def is_alive(self) -> bool:
            return True

        def join(self, timeout: float | None = None) -> None:
            return None

    runner = IsolatedTaskRunner(max_workers=1)
    oom_result = ProcessingResult(item_id="oom-sub", error=MemoryError("simulated OOM"))
    runner._release_child_after_queue_result(_FakeProc(), oom_result)

    assert terminate_calls == []


def test_terminate_process_swallows_oserror() -> None:
    class _FakeProc:
        pid = 4242

        def is_alive(self) -> bool:
            return True

        def terminate(self) -> None:
            raise PermissionError(5, "Access is denied")

        def join(self, timeout: float | None = None) -> None:
            return None

        def kill(self) -> None:
            raise PermissionError(5, "Access is denied")

    runner = IsolatedTaskRunner(max_workers=1)
    runner._terminate_process(_FakeProc())


def test_work_timeout_fires_while_other_slot_still_running() -> None:
    """Per-item timeout must be polled even when another worker is active."""
    items = [("slow", 8.0), ("fast", 0.05)]
    successful, failed = parallel_map(
        _task_slow,
        items,
        n_processes=2,
        per_item_timeout_sec=1.0,
        graceful_shutdown_sec=5.0,
        show_progress=False,
    )
    assert "slow" in failed
    assert len(successful) == 1
    assert successful[0].item_id == "fast"
