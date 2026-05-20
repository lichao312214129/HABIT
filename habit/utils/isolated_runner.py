"""
Bounded parallel execution with one OS process per work item (spawn).

Each item runs in a dedicated child process so timeouts terminate only that
process and failures do not break a shared ProcessPoolExecutor.
"""

from __future__ import annotations

import logging
import multiprocessing
import threading
import time
from dataclasses import dataclass
from pathlib import Path
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
        return ProcessingResult(item_id=item, error=exc)


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
    """Top-level child entry: run wrapper and put result on the queue (spawn-safe)."""
    result_queue.put(_worker_wrapper(worker_args))

DEFAULT_GRACEFUL_SHUTDOWN_SEC: float = 15.0
DEFAULT_POLL_INTERVAL_SEC: float = 0.25
DEFAULT_READER_JOIN_AFTER_PROC_SEC: float = 30.0


@dataclass
class _ActiveSlot:
    """One in-flight isolated child process."""

    item_id: Any
    proc: multiprocessing.Process
    reader: threading.Thread
    recv_bucket: List[Optional[ProcessingResult]]
    started_at: float


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
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self.max_workers = max_workers
        self.per_item_timeout_sec = per_item_timeout_sec
        self.graceful_shutdown_sec = graceful_shutdown_sec
        self.poll_interval_sec = poll_interval_sec
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
            logger.info(
                "Using isolated spawn workers: max_workers=%s%s",
                self.max_workers,
                timeout_msg,
            )

        progress_bar: Optional[CustomTqdm] = None
        if show_progress:
            progress_bar = CustomTqdm(total=total, desc=desc)

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
                    logger.error(
                        "Error processing %s: %s",
                        proc_result.item_id,
                        proc_result.error,
                    )
            if progress_bar is not None:
                progress_bar.update(1)

        _fill_slots()
        while active:
            now = time.monotonic()
            still_active: List[_ActiveSlot] = []

            for slot in active:
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

                reader_timeout = self._reader_join_timeout_sec()
                slot.reader.join(timeout=reader_timeout)
                proc_result = slot.recv_bucket[0]
                _finish_slot(slot, proc_result, timed_out=False)

            active = still_active
            _fill_slots()

            if active:
                time.sleep(self.poll_interval_sec)

        if logger and failed_items:
            logger.warning("Failed or timed out %s item(s)", len(failed_items))

        return successful_results, failed_items

    def _reader_join_timeout_sec(self) -> float:
        if self.per_item_timeout_sec is not None:
            return max(
                DEFAULT_READER_JOIN_AFTER_PROC_SEC,
                float(self.per_item_timeout_sec) + 10.0,
            )
        return DEFAULT_READER_JOIN_AFTER_PROC_SEC

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

        def _drain_worker_result_queue() -> None:
            recv_bucket[0] = result_queue.get()

        reader = threading.Thread(
            target=_drain_worker_result_queue,
            name="habit-isolated-runner-queue-reader",
            daemon=True,
        )
        proc = self._ctx.Process(
            target=_put_worker_wrapper_result,
            args=(worker_args, result_queue),
        )
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
