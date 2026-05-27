"""
Spawn entry point for long-lived parallel workers.

Must remain importable as a top-level function for Windows spawn pickling.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional

from habit.utils.isolated_runner import ProcessingResult, is_fatal_memory_error
from habit.utils.log_utils import restore_logging_in_subprocess
from habit.utils.parallel_gpu_utils import HABIT_GPU_SLOT_INDEX_ENV
from habit.utils.persistent_worker_protocol import (
    WorkerExitReply,
    WorkerReadyReply,
    WorkerResultReply,
    WorkerRunCommand,
    WorkerStopCommand,
)


def _execute_item(
    func: Callable[[Any], Any],
    item: Any,
) -> ProcessingResult:
    """
    Run ``func`` on one queue item and normalize the return value.

    Args:
        func: User callable (same contract as isolated worker wrapper).
        item: Work item passed to ``func``.

    Returns:
        ProcessingResult: Success or captured exception for the item.
    """
    if hasattr(item, "__getitem__") and len(item) > 0:
        item_id = item[0] if isinstance(item, (list, tuple)) else item
    else:
        item_id = item

    try:
        raw_result = func(item)
        if isinstance(raw_result, tuple) and len(raw_result) == 2:
            parsed_id, actual_result = raw_result
            if isinstance(actual_result, Exception):
                return ProcessingResult(item_id=parsed_id, error=actual_result)
            return ProcessingResult(item_id=parsed_id, result=actual_result)
        return ProcessingResult(item_id=item_id, result=raw_result)
    except Exception as exc:
        return ProcessingResult(item_id=item_id, error=exc)


def _maybe_empty_cuda_cache(logger: Optional[logging.Logger]) -> None:
    """Release cached GPU allocations after a task when torch CUDA is available."""
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
    except Exception as exc:
        if logger is not None:
            logger.warning("torch.cuda.empty_cache() failed after task: %s", exc)


def persistent_worker_main(
    worker_slot: int,
    log_file_path: Optional[str],
    log_level: int,
    task_queue: Any,
    result_queue: Any,
    func: Callable[[Any], Any],
    *,
    recycle_after_tasks: int = 0,
) -> None:
    """
    Long-lived worker loop: receive RUN commands until STOP.

    Args:
        worker_slot: Zero-based slot index (maps to ``HABIT_GPU_SLOT_INDEX``).
        log_file_path: Optional log file restored in this child.
        log_level: Logging level for this child.
        task_queue: Dedicated task queue for this worker.
        result_queue: Shared parent result queue.
        func: Pickled user callable executed for each RUN command.
        recycle_after_tasks: Exit cleanly after this many successful tasks (0=never).
    """
    logger = logging.getLogger(__name__)
    if log_file_path is not None:
        restore_logging_in_subprocess(Path(log_file_path), log_level)

    previous_slot = os.environ.get(HABIT_GPU_SLOT_INDEX_ENV)
    os.environ[HABIT_GPU_SLOT_INDEX_ENV] = str(worker_slot)
    successful_tasks = 0

    try:
        result_queue.put(WorkerReadyReply(worker_slot=worker_slot))

        while True:
            message = task_queue.get()
            if isinstance(message, WorkerStopCommand):
                break
            if not isinstance(message, WorkerRunCommand):
                continue

            proc_result = _execute_item(func, message.item)
            result_queue.put(
                WorkerResultReply(
                    worker_slot=worker_slot,
                    proc_result=proc_result,
                )
            )

            if proc_result.success:
                successful_tasks += 1
            _maybe_empty_cuda_cache(logger)

            if (
                recycle_after_tasks > 0
                and successful_tasks >= recycle_after_tasks
            ):
                break

            if proc_result.error is not None and is_fatal_memory_error(proc_result.error):
                break
    finally:
        if previous_slot is None:
            os.environ.pop(HABIT_GPU_SLOT_INDEX_ENV, None)
        else:
            os.environ[HABIT_GPU_SLOT_INDEX_ENV] = previous_slot
        result_queue.put(WorkerExitReply(worker_slot=worker_slot))
