"""
Parallel processing utilities for HABIT project.

Heavy work uses :class:`~habit.utils.isolated_runner.IsolatedTaskRunner`:
one spawn child process per item, bounded concurrency, and real per-item timeouts.
"""

from __future__ import annotations

import logging
from typing import (
    TypeVar,
    Callable,
    Iterable,
    List,
    Tuple,
    Optional,
    Any,
    Generator,
)
from pathlib import Path

from habit.utils.log_utils import LoggerManager
from habit.utils.isolated_runner import (
    DEFAULT_GRACEFUL_SHUTDOWN_SEC,
    IsolatedTaskRunner,
    ProcessingResult,
    _item_id_from_worker_args,
    _put_worker_wrapper_result,
    _worker_wrapper,
)

# Re-export for backward compatibility
__all__ = [
    "ProcessingResult",
    "parallel_map",
    "parallel_map_simple",
    "ParallelProcessor",
    "_worker_wrapper",
    "_put_worker_wrapper_result",
    "_item_id_from_worker_args",
]

T = TypeVar("T")
R = TypeVar("R")


def _run_inprocess_sequential(
    func: Callable[[T], R],
    items_list: List[T],
    desc: str,
    logger: Optional[logging.Logger],
    show_progress: bool,
) -> Tuple[List[ProcessingResult], List[Any]]:
    """Sequential in-process path (no spawn overhead)."""
    from habit.utils.progress_utils import CustomTqdm

    successful_results: List[ProcessingResult] = []
    failed_items: List[Any] = []
    total = len(items_list)

    progress_bar = CustomTqdm(total=total, desc=desc) if show_progress else None

    for item in items_list:
        try:
            raw_result = func(item)
            if isinstance(raw_result, tuple) and len(raw_result) == 2:
                item_id, actual_result = raw_result
                if isinstance(actual_result, Exception):
                    failed_items.append(item_id)
                    if logger:
                        logger.error("Error processing %s: %s", item_id, actual_result)
                else:
                    successful_results.append(
                        ProcessingResult(item_id=item_id, result=actual_result)
                    )
            else:
                successful_results.append(
                    ProcessingResult(item_id=item, result=raw_result)
                )
        except Exception as exc:
            failed_items.append(item)
            if logger:
                logger.error("Error processing %s: %s", item, exc)

        if progress_bar is not None:
            progress_bar.update(1)

    if logger and failed_items:
        logger.warning("Failed to process %s item(s)", len(failed_items))

    return successful_results, failed_items


def parallel_map(
    func: Callable[[T], R],
    items: Iterable[T],
    n_processes: int = 1,
    desc: str = "Processing",
    logger: Optional[logging.Logger] = None,
    show_progress: bool = True,
    log_file_path: Optional[Path] = None,
    log_level: int = logging.INFO,
    per_item_timeout_sec: Optional[float] = None,
    graceful_shutdown_sec: float = DEFAULT_GRACEFUL_SHUTDOWN_SEC,
) -> Tuple[List[ProcessingResult], List[Any]]:
    """
    Apply a function to items with bounded isolated subprocesses.

    When ``n_processes > 1`` or a positive ``per_item_timeout_sec`` is set, each item
    runs in its own ``spawn`` child process (at most ``n_processes`` at once).
    Timeouts call ``terminate`` then ``kill`` on that child only.

    When ``n_processes == 1`` and timeout is disabled, runs in-process sequentially
    without spawn overhead.

    Args:
        func: Function applied per item; may return ``result`` or ``(item_id, result)``.
        items: Iterable of items to process.
        n_processes: Maximum concurrent child processes (>= 1).
        desc: Progress bar description.
        logger: Optional logger.
        show_progress: Whether to show a progress bar.
        log_file_path: Log file path restored in each child.
        log_level: Logging level for children.
        per_item_timeout_sec: Wall-clock limit per item from child ``start()``; ``None``
            disables timeout.
        graceful_shutdown_sec: Seconds to wait after ``terminate`` before ``kill``.

    Returns:
        ``(successful_results, failed_item_ids)``.
    """
    items_list = list(items)
    total = len(items_list)

    if total == 0:
        return [], []

    use_isolated = (n_processes > 1 and total > 1) or (
        per_item_timeout_sec is not None and per_item_timeout_sec > 0
    )

    if not use_isolated:
        return _run_inprocess_sequential(
            func=func,
            items_list=items_list,
            desc=desc,
            logger=logger,
            show_progress=show_progress,
        )

    if log_file_path is None:
        manager = LoggerManager()
        log_file_path = manager.get_log_file()

    runner = IsolatedTaskRunner(
        max_workers=max(1, n_processes),
        per_item_timeout_sec=(
            per_item_timeout_sec
            if per_item_timeout_sec is not None and per_item_timeout_sec > 0
            else None
        ),
        graceful_shutdown_sec=graceful_shutdown_sec,
    )
    return runner.map_items(
        func=func,
        items_list=items_list,
        desc=desc,
        logger=logger,
        show_progress=show_progress,
        log_file_path=log_file_path,
        log_level=log_level,
    )


def parallel_map_simple(
    func: Callable[[T], R],
    items: Iterable[T],
    n_processes: int = 1,
    desc: str = "Processing",
    show_progress: bool = True,
    per_item_timeout_sec: Optional[float] = None,
) -> Generator[R, None, None]:
    """
    Simplified parallel map that yields raw results (or exceptions) per item.

    Uses the same isolated runner as :func:`parallel_map` when subprocesses are required.
    """
    items_list = list(items)
    if not items_list:
        return

    use_isolated = (n_processes > 1 and len(items_list) > 1) or (
        per_item_timeout_sec is not None and per_item_timeout_sec > 0
    )

    if not use_isolated:
        from habit.utils.progress_utils import CustomTqdm

        progress_bar = CustomTqdm(total=len(items_list), desc=desc) if show_progress else None
        for item in items_list:
            try:
                yield func(item)
            except Exception as exc:
                yield exc  # type: ignore[misc]
            if progress_bar is not None:
                progress_bar.update(1)
        return

    successful, _failed = parallel_map(
        func=func,
        items=items_list,
        n_processes=n_processes,
        desc=desc,
        show_progress=show_progress,
        per_item_timeout_sec=per_item_timeout_sec,
    )
    for proc_result in successful:
        if proc_result.result is not None:
            yield proc_result.result


class ParallelProcessor:
    """
    Thin wrapper around :func:`parallel_map` with logging configuration.
    """

    def __init__(
        self,
        n_processes: int = 1,
        logger: Optional[logging.Logger] = None,
        graceful_shutdown_sec: float = DEFAULT_GRACEFUL_SHUTDOWN_SEC,
    ) -> None:
        self.n_processes = n_processes
        self.logger = logger
        self.graceful_shutdown_sec = graceful_shutdown_sec

        manager = LoggerManager()
        self._log_file_path = manager.get_log_file()
        self._log_level = logging.INFO
        if manager._root_logger:
            self._log_level = manager._root_logger.getEffectiveLevel()

    def __enter__(self) -> ParallelProcessor:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False

    def map(
        self,
        func: Callable[[T], R],
        items: Iterable[T],
        desc: str = "Processing",
        show_progress: bool = True,
        per_item_timeout_sec: Optional[float] = None,
    ) -> Tuple[List[ProcessingResult], List[Any]]:
        return parallel_map(
            func=func,
            items=items,
            n_processes=self.n_processes,
            desc=desc,
            logger=self.logger,
            show_progress=show_progress,
            log_file_path=self._log_file_path,
            log_level=self._log_level,
            per_item_timeout_sec=per_item_timeout_sec,
            graceful_shutdown_sec=self.graceful_shutdown_sec,
        )
