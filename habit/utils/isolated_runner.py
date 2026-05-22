"""
Bounded parallel execution with one OS process per work item (spawn).

Each item runs in a dedicated child process so timeouts terminate only that
process and failures do not break a shared ProcessPoolExecutor.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any, Callable, List, Optional, Tuple, TypeVar

from habit.utils.log_utils import restore_logging_in_subprocess
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
    return isinstance(exc, MemoryError)


def _worker_wrapper(args: Tuple[Callable, Any, Optional[Path], int]) -> ProcessingResult:
    """
    Run user ``func`` in a child process with logging restored.

    Args:
        args: ``(func, item, log_file_path, log_level)``.

    Returns:
        ``ProcessingResult`` for the item (success or caught exception).
    """
    func, item, log_file_path, log_level = args

    if log_file_path is not None:
        restore_logging_in_subprocess(log_file_path, log_level)

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


def _item_id_from_worker_args(worker_args: Tuple[Any, ...]) -> Any:
    """Extract item id from ``(func, item, log_file_path, log_level)``."""
    item = worker_args[1]
    if isinstance(item, (list, tuple)) and len(item) > 0:
        return item[0]
    return item


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
    if result.error is not None and is_fatal_memory_error(result.error):
        # Allow the queue pipe to flush before skipping Python teardown.
        time.sleep(0.05)
        os._exit(2)


DEFAULT_GRACEFUL_SHUTDOWN_SEC: float = 15.0
DEFAULT_POLL_INTERVAL_SEC: float = 0.25
QUEUE_RESULT_GRACE_SEC: float = 5.0


@dataclass
class _ActiveSlot:
    """One in-flight isolated child process."""

    item_id: Any
    proc: multiprocessing.Process
    reader: threading.Thread
    recv_bucket: List[Optional[ProcessingResult]]
    started_at: float
    proc_exited_at: Optional[float] = None


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
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if oom_reduce_workers_by < 1:
            raise ValueError("oom_reduce_workers_by must be >= 1")
        self.max_workers = max_workers
        self.per_item_timeout_sec = per_item_timeout_sec
        self.graceful_shutdown_sec = graceful_shutdown_sec
        self.poll_interval_sec = poll_interval_sec
        self.oom_backoff = oom_backoff
        self.oom_reduce_workers_by = oom_reduce_workers_by
        self._ctx = multiprocessing.get_context(mp_start_method)

    def map_items(
        self,
        func: Callable[[T], R],
        items_list: List[T],
        desc: str,
        logger: Optional[logging.Logger],
        show_progress: bool,
        log_file_path: Optional[Path],
        log_level: int,
    ) -> Tuple[List[ProcessingResult], List[Any]]:
        """
        Map ``func`` over ``items_list`` with bounded isolated processes.

        Args:
            func: Callable applied to each item in the child process.
            items_list: Work items (already materialised).
            desc: Progress bar description.
            logger: Optional logger for failures and timeouts.
            show_progress: Whether to show ``CustomTqdm``.
            log_file_path: Log file restored in each child.
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
            oom_msg = (
                f", oom_backoff=on (reduce by {self.oom_reduce_workers_by})"
                if self.oom_backoff
                else ", oom_backoff=off"
            )
            logger.info(
                "Using isolated spawn workers: max_workers=%s%s%s",
                self.max_workers,
                timeout_msg,
                oom_msg,
            )

        progress_bar: Optional[CustomTqdm] = None
        if show_progress:
            progress_bar = CustomTqdm(total=total, desc=desc)

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
            while len(active) < max_slots and pending_items:
                active.append(
                    self._start_slot(
                        func=func,
                        item=pending_items.pop(0),
                        log_file_path=log_file_path,
                        log_level=log_level,
                    )
                )

        def _finish_slot(
            slot: _ActiveSlot,
            proc_result: Optional[ProcessingResult],
            *,
            timed_out: bool,
        ) -> None:
            if timed_out:
                failed_items.append(slot.item_id)
                if logger:
                    logger.error(
                        "Timeout (>%ss) for item %s; terminating child process.",
                        self.per_item_timeout_sec,
                        slot.item_id,
                    )
            elif proc_result is None:
                failed_items.append(slot.item_id)
                if logger:
                    logger.error(
                        "Child exited without a queue result for item %s "
                        "(exit code %s).",
                        slot.item_id,
                        slot.proc.exitcode,
                    )
            elif proc_result.success:
                successful_results.append(proc_result)
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
            if progress_bar is not None:
                progress_bar.update(1)

        _fill_slots()
        while active:
            now = time.monotonic()
            still_active: List[_ActiveSlot] = []

            for slot in active:
                proc_result = slot.recv_bucket[0]
                if proc_result is not None:
                    if slot.proc.is_alive():
                        self._terminate_process(slot.proc)
                    slot.reader.join(timeout=2.0)
                    _finish_slot(slot, proc_result, timed_out=False)
                    continue

                elapsed = now - slot.started_at
                timed_out = (
                    self.per_item_timeout_sec is not None
                    and elapsed > self.per_item_timeout_sec
                )

                if timed_out and slot.proc.is_alive():
                    self._terminate_process(slot.proc)
                    slot.reader.join(timeout=2.0)
                    _finish_slot(slot, None, timed_out=True)
                    continue

                if slot.proc.is_alive():
                    still_active.append(slot)
                    continue

                if slot.proc_exited_at is None:
                    slot.proc_exited_at = now

                # Child exited: poll the queue reader briefly instead of one long join.
                if proc_result is None and slot.reader.is_alive():
                    slot.reader.join(timeout=self.poll_interval_sec)
                    proc_result = slot.recv_bucket[0]

                if proc_result is not None:
                    _finish_slot(slot, proc_result, timed_out=False)
                    continue

                grace_elapsed = now - (slot.proc_exited_at or now)
                if grace_elapsed <= QUEUE_RESULT_GRACE_SEC and slot.reader.is_alive():
                    still_active.append(slot)
                    continue

                _finish_slot(slot, None, timed_out=False)

            active = still_active
            _fill_slots()

            if active:
                time.sleep(self.poll_interval_sec)

        if logger and failed_items:
            logger.warning("Failed or timed out %s item(s)", len(failed_items))

        return successful_results, failed_items

    def _start_slot(
        self,
        func: Callable[[T], R],
        item: T,
        log_file_path: Optional[Path],
        log_level: int,
    ) -> _ActiveSlot:
        worker_args = (func, item, log_file_path, log_level)
        item_id = _item_id_from_worker_args(worker_args)
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
                        grace_deadline = time.monotonic() + QUEUE_RESULT_GRACE_SEC
                    elif time.monotonic() >= grace_deadline:
                        return
                try:
                    recv_bucket[0] = result_queue.get(timeout=poll_interval_sec)
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
        return _ActiveSlot(
            item_id=item_id,
            proc=proc,
            reader=reader,
            recv_bucket=recv_bucket,
            started_at=time.monotonic(),
        )

    def _terminate_process(self, proc: multiprocessing.Process) -> None:
        if not proc.is_alive():
            return
        proc.terminate()
        proc.join(timeout=self.graceful_shutdown_sec)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=10.0)
