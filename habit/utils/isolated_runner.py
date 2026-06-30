# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text. Summary:
#
#   - Non-commercial use (academic, research, education, personal) is permitted
#     provided that copyright notices are retained and HABIT usage is
#     acknowledged in publications, reports, or documentation.
#   - Commercial use requires prior written consent from the copyright holder
#     (lichao19870617@163.com) and public acknowledgment of HABIT usage in
#     product documentation or user-facing materials.
#   - Unauthorized commercial use or removal of attribution is prohibited.
#
"""
Bounded parallel execution with one OS process per work item (spawn).

Each item runs in a dedicated child process so timeouts terminate only that
process and failures do not break a shared ProcessPoolExecutor.

Child startup (``proc.start()``) runs in a background thread so the poll loop
can still enforce per-item and spawn-startup timeouts while other workers run.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty
from typing import Any, Callable, List, Optional, Tuple, TypeVar

from habit.utils.job_cancel import is_job_cancelled, JobCancelledError
from habit.utils.log_utils import restore_logging_in_subprocess, shutdown_subprocess_logging
from habit.utils.parallel_gpu_utils import HABIT_GPU_SLOT_INDEX_ENV
from habit.utils.progress_utils import CustomTqdm

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class ProcessingResult:
    """
    Container for processing result with error handling.

    Attributes:
        item_id: Identifier for the processed item
        result: The processing result (None if failed)
        error: Exception if processing failed (None if successful)
    """

    item_id: Any
    result: Optional[Any] = None
    error: Optional[Exception] = None

    @property
    def success(self) -> bool:
        """Check if processing was successful."""
        return self.error is None

    def unwrap(self) -> Any:
        """Return result or raise the stored error."""
        if self.error is not None:
            raise self.error
        return self.result


def is_fatal_memory_error(exc: Optional[BaseException]) -> bool:
    """
    Return True when a worker failure should trigger fast slot release.

    Args:
        exc: Exception captured in the child process, if any.

    Returns:
        True for MemoryError and similar unrecoverable allocation failures.
    """
    if exc is None:
        return False
    if isinstance(exc, MemoryError):
        return True
    # NumPy may expose ArrayMemoryError under different module paths across versions.
    return type(exc).__name__ == "ArrayMemoryError"


def requires_worker_restart(
    proc_result: Optional[ProcessingResult],
    *,
    worker_alive: bool,
) -> bool:
    """
    Decide whether a persistent worker slot must be replaced after a task.

    Recoverable Python exceptions (for example NaN validation errors) keep the
    long-lived worker running. Restarts are reserved for missing results, fatal
    memory failures, and worker process death.

    Args:
        proc_result: Result received from the worker, if any.
        worker_alive: Whether the worker process is still alive.

    Returns:
        True when the parent should terminate and respawn the slot worker.
    """
    if not worker_alive:
        return True
    if proc_result is None:
        return True
    if not proc_result.success and is_fatal_memory_error(proc_result.error):
        return True
    return False


def _worker_wrapper(args: Tuple[Any, ...]) -> ProcessingResult:
    """
    Run user ``func`` in a child process with logging restored.

    Args:
        args: ``(func, item, log_queue, log_level, gpu_slot_index)`` or
        ``(func, item, log_queue, log_file_path, log_level, gpu_slot_index)``.

    Returns:
        ``ProcessingResult`` for the item (success or caught exception).
    """
    func, item, log_queue, log_file_path, log_level, gpu_slot_index = _unpack_worker_args(args)

    if log_queue is not None or log_file_path is not None:
        restore_logging_in_subprocess(
            log_file_path=log_file_path,
            log_level=log_level,
            log_queue=log_queue,
        )

    previous_slot = os.environ.get(HABIT_GPU_SLOT_INDEX_ENV)
    os.environ[HABIT_GPU_SLOT_INDEX_ENV] = str(gpu_slot_index)
    try:
        if hasattr(item, "__getitem__") and len(item) > 0:
            item_id = item[0] if isinstance(item, (list, tuple)) else item
        else:
            item_id = item

        result = func(item)

        if isinstance(result, tuple) and len(result) == 2:
            item_id, actual_result = result
            if isinstance(actual_result, Exception):
                return ProcessingResult(item_id=item_id, error=actual_result)
            return ProcessingResult(item_id=item_id, result=actual_result)

        return ProcessingResult(item_id=item_id, result=result)

    except Exception as exc:
        if hasattr(item, "__getitem__") and len(item) > 0:
            item_id = item[0] if isinstance(item, (list, tuple)) else item
        else:
            item_id = item
        return ProcessingResult(item_id=item_id, error=exc)
    finally:
        if previous_slot is None:
            os.environ.pop(HABIT_GPU_SLOT_INDEX_ENV, None)
        else:
            os.environ[HABIT_GPU_SLOT_INDEX_ENV] = previous_slot
        shutdown_subprocess_logging()


def _unpack_worker_args(
    args: Tuple[Any, ...],
) -> Tuple[Callable[..., Any], Any, Any, Optional[Path], int, int]:
    """
    Normalize worker argument tuples for isolated child processes.

    Args:
        args: ``(func, item, log_queue, log_level[, gpu_slot_index])`` or
        ``(func, item, log_queue, log_file_path, log_level[, gpu_slot_index])``.

    Returns:
        Tuple of func, item, log queue, optional log path, log level, GPU slot index.
    """
    if len(args) == 5:
        func, item, log_queue, log_level, gpu_slot_index = args
        return func, item, log_queue, None, log_level, int(gpu_slot_index)
    if len(args) == 6:
        func, item, log_queue, log_file_path, log_level, gpu_slot_index = args
        return func, item, log_queue, log_file_path, log_level, int(gpu_slot_index)
    raise ValueError(
        f"worker args must be length 5 or 6; got {len(args)} element(s)"
    )


def _item_id_from_worker_args(worker_args: Tuple[Any, ...]) -> Any:
    """Extract item id from ``(func, item, log_file_path, log_level)``."""
    item = worker_args[1]
    if isinstance(item, (list, tuple)) and len(item) > 0:
        return item[0]
    return item


def _flush_result_queue(result_queue: Any) -> None:
    """
    Ensure the queue payload is fully sent before the child process exits.

    On Windows ``spawn``, the child can exit while the queue feeder thread is
    still pickling/sending, which makes the parent see exit code 0 with no result.
    """
    try:
        result_queue.close()
        result_queue.join_thread()
    except (OSError, ValueError, AttributeError):
        pass


def _put_worker_wrapper_result(
    worker_args: Tuple[Any, ...],
    result_queue: Any,
) -> None:
    """
    Top-level child entry: run wrapper, publish result, then hard-exit on OOM.

    After a fatal memory error the child skips Python teardown of large arrays
    via ``os._exit`` so the parent can release the worker slot immediately.
    """
    result = _worker_wrapper(worker_args)
    result_queue.put(result)
    _flush_result_queue(result_queue)
    if result.error is not None and is_fatal_memory_error(result.error):
        os._exit(2)


DEFAULT_GRACEFUL_SHUTDOWN_SEC: float = 15.0
DEFAULT_POLL_INTERVAL_SEC: float = 0.25
DEFAULT_SPAWN_STARTUP_TIMEOUT_SEC: float = 120.0
QUEUE_RESULT_GRACE_SEC: float = 15.0


@dataclass
class _ActiveSlot:
    """One in-flight isolated child process (spawn may still be starting)."""

    item_id: Any
    worker_args: Tuple[Any, ...]
    spawn_started_at: float
    result_ready_event: threading.Event = field(default_factory=threading.Event, repr=False)
    proc: Optional[multiprocessing.Process] = None
    reader: Optional[threading.Thread] = None
    recv_bucket: Optional[List[Optional[ProcessingResult]]] = None
    started_at: Optional[float] = None
    proc_exited_at: Optional[float] = None
    spawn_complete: bool = False
    spawn_error: Optional[BaseException] = None
    _spawn_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


def _wait_for_any_result(
    active_slots: List[_ActiveSlot],
    max_wait_sec: float,
) -> None:
    """
    Block until at least one active slot receives a result or *max_wait_sec* elapses.

    Uses per-slot :class:`threading.Event` set by reader threads so the main poll
    loop reacts to newly-available results within milliseconds instead of waiting
    for the full poll interval.
    """
    deadline = time.monotonic() + max_wait_sec
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        for slot in active_slots:
            if slot.result_ready_event.is_set():
                slot.result_ready_event.clear()
                return
        wait_sec = min(remaining, 0.05)
        time.sleep(wait_sec)


class IsolatedTaskRunner:
    """
    Run items with at most ``max_workers`` concurrent spawn children.

    Each item uses a fresh ``spawn`` process. Optional wall-clock timeout
    terminates (then kills) only that child.
    """

    def __init__(
        self,
        max_workers: int,
        per_item_timeout_sec: Optional[float] = None,
        graceful_shutdown_sec: float = DEFAULT_GRACEFUL_SHUTDOWN_SEC,
        poll_interval_sec: float = DEFAULT_POLL_INTERVAL_SEC,
        mp_start_method: str = "spawn",
        oom_backoff: bool = True,
        oom_reduce_workers_by: int = 1,
        spawn_startup_timeout_sec: Optional[float] = DEFAULT_SPAWN_STARTUP_TIMEOUT_SEC,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if oom_reduce_workers_by < 1:
            raise ValueError("oom_reduce_workers_by must be >= 1")
        if spawn_startup_timeout_sec is not None and spawn_startup_timeout_sec <= 0:
            raise ValueError("spawn_startup_timeout_sec must be positive when set")
        self.max_workers = max_workers
        self.per_item_timeout_sec = per_item_timeout_sec
        self.graceful_shutdown_sec = graceful_shutdown_sec
        self.poll_interval_sec = poll_interval_sec
        self.oom_backoff = oom_backoff
        self.oom_reduce_workers_by = oom_reduce_workers_by
        self.spawn_startup_timeout_sec = spawn_startup_timeout_sec
        self._ctx = multiprocessing.get_context(mp_start_method)

    def map_items(
        self,
        func: Callable[[T], R],
        items_list: List[T],
        desc: str,
        logger: Optional[logging.Logger],
        show_progress: bool,
        log_file_path: Optional[Path] = None,
        log_queue: Any = None,
        log_level: int = logging.INFO,
        on_item_done: Optional[Callable[[ProcessingResult], None]] = None,
    ) -> Tuple[List[ProcessingResult], List[Any]]:
        """
        Map ``func`` over ``items_list`` with bounded isolated processes.

        Args:
            func: Callable applied to each item in the child process.
            items_list: Work items (already materialised).
            desc: Progress bar description.
            logger: Optional logger for failures and timeouts.
            show_progress: Whether to show ``CustomTqdm``.
            log_file_path: Legacy log file restored in each child when queue is unavailable.
            log_queue: Multiprocessing queue for ordered logging in the parent.
            log_level: Logging level for children.

        Returns:
            ``(successful_results, failed_item_ids)`` — same contract as ``parallel_map``.
        """
        total = len(items_list)
        if total == 0:
            return [], []

        successful_results: List[ProcessingResult] = []
        failed_items: List[Any] = []
        pending_items = list(items_list)
        active: List[_ActiveSlot] = []
        max_slots = min(self.max_workers, total)

        if logger:
            timeout_msg = (
                f", per-item timeout={self.per_item_timeout_sec}s"
                if self.per_item_timeout_sec is not None
                else ", no per-item timeout"
            )
            spawn_msg = (
                f", spawn startup timeout={self.spawn_startup_timeout_sec}s"
                if self.spawn_startup_timeout_sec is not None
                else ", no spawn startup timeout"
            )
            oom_msg = (
                f", oom_backoff=on (reduce by {self.oom_reduce_workers_by})"
                if self.oom_backoff
                else ", oom_backoff=off"
            )
            logger.info(
                "Using isolated spawn workers: max_workers=%s%s%s%s",
                self.max_workers,
                timeout_msg,
                spawn_msg,
                oom_msg,
            )

        progress_bar: Optional[CustomTqdm] = None
        if show_progress:
            progress_bar = CustomTqdm(total=total, desc=desc)

        slot_counter = 0

        def _apply_oom_backoff(item_id: Any) -> None:
            nonlocal max_slots
            if not self.oom_backoff:
                return
            new_max = max(1, max_slots - self.oom_reduce_workers_by)
            if new_max < max_slots and logger:
                logger.warning(
                    "Fatal memory error for item %s; reducing parallel workers "
                    "%s -> %s",
                    item_id,
                    max_slots,
                    new_max,
                )
            max_slots = new_max

        def _fill_slots() -> None:
            nonlocal slot_counter
            while len(active) < max_slots and pending_items and not is_job_cancelled():
                gpu_slot_index = slot_counter % max_slots
                slot_counter += 1
                active.append(
                    self._begin_slot_async(
                        func=func,
                        item=pending_items.pop(0),
                        log_queue=log_queue,
                        log_file_path=log_file_path,
                        log_level=log_level,
                        gpu_slot_index=gpu_slot_index,
                    )
                )

        def _finish_slot(
            slot: _ActiveSlot,
            proc_result: Optional[ProcessingResult],
            *,
            timed_out: bool,
            spawn_timed_out: bool = False,
        ) -> None:
            if spawn_timed_out:
                failed_items.append(slot.item_id)
                if logger:
                    logger.error(
                        "Spawn startup timeout (>%ss) for item %s; "
                        "releasing worker slot.",
                        self.spawn_startup_timeout_sec,
                        slot.item_id,
                    )
                if on_item_done is not None:
                    on_item_done(
                        ProcessingResult(
                            item_id=slot.item_id,
                            error=TimeoutError(
                                f"Item {slot.item_id} spawn startup exceeded "
                                f"{self.spawn_startup_timeout_sec}s"
                            ),
                        )
                    )
            elif timed_out:
                failed_items.append(slot.item_id)
                if logger:
                    logger.error(
                        "Timeout (>%ss) for item %s; terminating child process.",
                        self.per_item_timeout_sec,
                        slot.item_id,
                    )
                if on_item_done is not None:
                    on_item_done(
                        ProcessingResult(
                            item_id=slot.item_id,
                            error=TimeoutError(
                                f"Item {slot.item_id} exceeded "
                                f"{self.per_item_timeout_sec}s"
                            ),
                        )
                    )
            elif proc_result is None:
                failed_items.append(slot.item_id)
                exit_code = slot.proc.exitcode if slot.proc is not None else "unknown"
                if logger:
                    logger.error(
                        "Child exited without a queue result for item %s "
                        "(exit code %s).",
                        slot.item_id,
                        exit_code,
                    )
                if on_item_done is not None:
                    on_item_done(
                        ProcessingResult(
                            item_id=slot.item_id,
                            error=RuntimeError(
                                f"Child exited without queue result "
                                f"(exit code {exit_code})"
                            ),
                        )
                    )
            elif proc_result.success:
                successful_results.append(proc_result)
                if on_item_done is not None:
                    on_item_done(proc_result)
            else:
                failed_items.append(proc_result.item_id)
                if logger:
                    if is_fatal_memory_error(proc_result.error):
                        logger.error(
                            "Fatal memory error for item %s: %s",
                            proc_result.item_id,
                            proc_result.error,
                        )
                    else:
                        logger.error(
                            "Error processing %s: %s",
                            proc_result.item_id,
                            proc_result.error,
                        )
                if is_fatal_memory_error(proc_result.error):
                    _apply_oom_backoff(proc_result.item_id)
                if on_item_done is not None:
                    on_item_done(proc_result)
            if progress_bar is not None:
                progress_bar.update(1)

        _fill_slots()
        cancelled = False
        while active or pending_items:
            if is_job_cancelled():
                cancelled = True
                pending_items.clear()

            if cancelled and not active:
                if progress_bar is not None:
                    progress_bar.close()
                if logger:
                    logger.warning("Job cancelled; stopping after completed items.")
                raise JobCancelledError("Job cancelled by user")

            now = time.monotonic()
            still_active: List[_ActiveSlot] = []

            for slot in active:
                if not slot.spawn_complete:
                    spawn_elapsed = now - slot.spawn_started_at
                    spawn_limit = self.spawn_startup_timeout_sec
                    if spawn_limit is not None and spawn_elapsed > spawn_limit:
                        if slot.proc is not None and slot.proc.is_alive():
                            self._terminate_process(slot.proc, logger=logger)
                        if slot.reader is not None:
                            slot.reader.join(timeout=2.0)
                        _finish_slot(slot, None, timed_out=False, spawn_timed_out=True)
                        continue
                    still_active.append(slot)
                    continue

                if slot.spawn_error is not None:
                    if slot.reader is not None:
                        slot.reader.join(timeout=2.0)
                    _finish_slot(
                        slot,
                        ProcessingResult(item_id=slot.item_id, error=slot.spawn_error),
                        timed_out=False,
                    )
                    continue

                if slot.proc is None or slot.recv_bucket is None:
                    if slot.reader is not None:
                        slot.reader.join(timeout=2.0)
                    _finish_slot(slot, None, timed_out=False)
                    continue

                proc_result = slot.recv_bucket[0]
                if proc_result is not None:
                    if slot.proc is not None and slot.proc.is_alive():
                        self._release_child_after_queue_result(
                            slot.proc,
                            proc_result,
                            logger=logger,
                        )
                    if slot.reader is not None:
                        slot.reader.join(timeout=2.0)
                    _finish_slot(slot, proc_result, timed_out=False)
                    continue

                work_started_at = slot.started_at or slot.spawn_started_at
                elapsed = now - work_started_at
                timed_out = (
                    self.per_item_timeout_sec is not None
                    and elapsed > self.per_item_timeout_sec
                )

                if timed_out and slot.proc.is_alive():
                    self._terminate_process(slot.proc, logger=logger)
                    if slot.reader is not None:
                        slot.reader.join(timeout=2.0)
                    _finish_slot(slot, None, timed_out=True)
                    continue

                if slot.proc.is_alive():
                    still_active.append(slot)
                    continue

                if slot.proc_exited_at is None:
                    slot.proc_exited_at = now

                if proc_result is None and slot.reader is not None and slot.reader.is_alive():
                    slot.reader.join(timeout=self.poll_interval_sec)
                    proc_result = slot.recv_bucket[0]

                if proc_result is not None:
                    _finish_slot(slot, proc_result, timed_out=False)
                    continue

                grace_elapsed = now - (slot.proc_exited_at or now)
                if (
                    grace_elapsed <= QUEUE_RESULT_GRACE_SEC
                    and slot.reader is not None
                    and slot.reader.is_alive()
                ):
                    still_active.append(slot)
                    continue

                _finish_slot(slot, None, timed_out=False)

            active = still_active
            _fill_slots()

            if active or pending_items:
                _wait_for_any_result(active, self.poll_interval_sec)

        if logger and failed_items:
            logger.warning("Failed or timed out %s item(s)", len(failed_items))

        return successful_results, failed_items

    def _begin_slot_async(
        self,
        func: Callable[[T], R],
        item: T,
        log_queue: Any,
        log_file_path: Optional[Path],
        log_level: int,
        gpu_slot_index: int,
    ) -> _ActiveSlot:
        """
        Begin spawning a child process without blocking the poll loop.

        Args:
            func: User callable for the work item.
            item: Work item passed to ``func``.
            log_queue: Optional multiprocessing queue for ordered logging.
            log_file_path: Optional legacy log file restored in the child.
            log_level: Logging level for the child.
            gpu_slot_index: Zero-based worker slot mapped to ``gpuSlotIndex`` in children.

        Returns:
            Slot handle tracked by :meth:`map_items`.
        """
        worker_args = (
            func,
            item,
            log_queue,
            log_file_path,
            log_level,
            gpu_slot_index,
        )
        item_id = _item_id_from_worker_args(worker_args)
        slot = _ActiveSlot(
            item_id=item_id,
            worker_args=worker_args,
            spawn_started_at=time.monotonic(),
        )

        def _spawn_worker() -> None:
            try:
                result_queue = self._ctx.Queue(maxsize=1)
                recv_bucket: List[Optional[ProcessingResult]] = [None]
                poll_interval_sec = self.poll_interval_sec
                proc_holder: List[Optional[multiprocessing.Process]] = [None]

                def _drain_worker_result_queue() -> None:
                    grace_deadline: Optional[float] = None
                    while recv_bucket[0] is None:
                        proc_ref = proc_holder[0]
                        if proc_ref is not None and not proc_ref.is_alive():
                            if grace_deadline is None:
                                grace_deadline = (
                                    time.monotonic() + QUEUE_RESULT_GRACE_SEC
                                )
                            elif time.monotonic() >= grace_deadline:
                                slot.result_ready_event.set()
                                return
                        try:
                            recv_bucket[0] = result_queue.get(
                                timeout=poll_interval_sec
                            )
                            if recv_bucket[0] is not None:
                                slot.result_ready_event.set()
                        except Empty:
                            continue

                reader = threading.Thread(
                    target=_drain_worker_result_queue,
                    name="habit-isolated-runner-queue-reader",
                    daemon=True,
                )
                proc = self._ctx.Process(
                    target=_put_worker_wrapper_result,
                    args=(worker_args, result_queue),
                )
                proc_holder[0] = proc
                reader.start()
                proc.start()
                with slot._spawn_lock:
                    slot.proc = proc
                    slot.reader = reader
                    slot.recv_bucket = recv_bucket
                    slot.started_at = time.monotonic()
                    slot.spawn_complete = True
            except BaseException as exc:
                with slot._spawn_lock:
                    slot.spawn_error = exc
                    slot.spawn_complete = True

        threading.Thread(
            target=_spawn_worker,
            name=f"habit-isolated-spawn-{item_id}",
            daemon=True,
        ).start()
        return slot

    def _release_child_after_queue_result(
        self,
        proc: multiprocessing.Process,
        proc_result: ProcessingResult,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Reclaim a child process after its queue result was delivered.

        Fatal memory errors use ``os._exit(2)`` in the child after queue flush.
        Calling ``terminate()`` here races that exit and often raises WinError 5
        on Windows, which would abort the entire ``map_items`` batch.
        """
        if not proc.is_alive():
            return
        if is_fatal_memory_error(proc_result.error):
            proc.join(timeout=self.graceful_shutdown_sec)
            if proc.is_alive() and logger is not None:
                logger.warning(
                    "Child for item %s still alive after fatal memory error; "
                    "waiting for self-exit instead of terminate()",
                    proc_result.item_id,
                )
            return
        self._terminate_process(proc, logger=logger)

    def _terminate_process(
        self,
        proc: multiprocessing.Process,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if not proc.is_alive():
            return
        try:
            proc.terminate()
        except OSError as exc:
            if logger is not None:
                logger.warning(
                    "terminate() failed for pid %s (process may already be exiting): %s",
                    proc.pid,
                    exc,
                )
            return
        proc.join(timeout=self.graceful_shutdown_sec)
        if proc.is_alive():
            try:
                proc.kill()
            except OSError as exc:
                if logger is not None:
                    logger.warning(
                        "kill() failed for pid %s (process may already be exiting): %s",
                        proc.pid,
                        exc,
                    )
                return
            proc.join(timeout=10.0)
