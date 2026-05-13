"""
Parallel processing utilities for HABIT project.

This module provides a unified interface for parallel and sequential processing,
eliminating code duplication across different modules that need multiprocessing.
"""

import logging
import multiprocessing
import queue
import time
from typing import (
    TypeVar, Callable, Iterable, List, Tuple,
    Optional, Any, Union, Generator, Dict
)
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

from habit.utils.progress_utils import CustomTqdm
from habit.utils.log_utils import restore_logging_in_subprocess, LoggerManager

# Type variable for generic processing
T = TypeVar('T')  # Input type
R = TypeVar('R')  # Result type


@dataclass
class ProcessingResult:
    """
    Container for processing result with error handling.
    
    Attributes:
        item_id: Identifier for the processed item
        result: The processing result (None if failed)
        error: Exception if processing failed (None if successful)
        success: Whether processing was successful
    """
    item_id: Any
    result: Optional[Any] = None
    error: Optional[Exception] = None
    
    @property
    def success(self) -> bool:
        """Check if processing was successful."""
        return self.error is None
    
    def unwrap(self) -> Any:
        """
        Get the result, raising the error if processing failed.
        
        Returns:
            The processing result
            
        Raises:
            Exception: The original exception if processing failed
        """
        if self.error is not None:
            raise self.error
        return self.result


def _worker_wrapper(args: Tuple[Callable, Any, Optional[Path], int]) -> ProcessingResult:
    """
    Internal wrapper for worker function that handles logging restoration
    and exception catching in child processes.
    
    Args:
        args: Tuple of (function, item, log_file_path, log_level)
        
    Returns:
        ProcessingResult: Container with result or error
    """
    func, item, log_file_path, log_level = args
    
    # Restore logging in subprocess (for Windows spawn mode)
    if log_file_path is not None:
        restore_logging_in_subprocess(log_file_path, log_level)
    
    try:
        # Get item_id from item if it has one, otherwise use item itself
        if hasattr(item, '__getitem__') and len(item) > 0:
            item_id = item[0] if isinstance(item, (list, tuple)) else item
        else:
            item_id = item
            
        result = func(item)
        
        # Handle tuple returns where first element is ID
        if isinstance(result, tuple) and len(result) == 2:
            item_id, actual_result = result
            if isinstance(actual_result, Exception):
                return ProcessingResult(item_id=item_id, error=actual_result)
            return ProcessingResult(item_id=item_id, result=actual_result)
        
        return ProcessingResult(item_id=item_id, result=result)
        
    except Exception as e:
        return ProcessingResult(item_id=item, error=e)


def _item_id_from_worker_args(worker_args: Tuple[Any, ...]) -> Any:
    """
    Extract a stable item id from parallel_map worker argument tuple.

    worker_args layout: (func, item, log_file_path, log_level). For habitat
    pipelines, item is (subject_id, subject_data).

    Args:
        worker_args: Arguments tuple passed to _worker_wrapper.

    Returns:
        Identifier used in logs and failed_items (typically subject_id).
    """
    item = worker_args[1]
    if isinstance(item, (list, tuple)) and len(item) > 0:
        return item[0]
    return item


def _put_worker_wrapper_result(
    worker_args: Tuple[Any, ...],
    result_queue: Any,
) -> None:
    """
    Child-process entry point (must be top-level for Windows spawn).

    Runs _worker_wrapper once and pushes ProcessingResult to result_queue.

    Args:
        worker_args: Tuple (func, item, log_file_path, log_level).
        result_queue: multiprocessing.Queue with maxsize compatible with one put.
    """
    result_queue.put(_worker_wrapper(worker_args))


def _parallel_map_with_item_timeout(
    func: Callable[[T], R],
    items_list: List[T],
    n_processes: int,
    total: int,
    desc: str,
    logger: Optional[logging.Logger],
    show_progress: bool,
    log_file_path: Optional[Path],
    log_level: int,
    per_item_timeout_sec: float,
) -> Tuple[List[ProcessingResult], List[Any]]:
    """
    Parallel path: ProcessPoolExecutor + per-future wall-clock timeout.

    Timed-out tasks are counted as failed without waiting for worker completion;
    executor.shutdown(wait=False) does not block on stuck workers. Orphan
    processes may continue until they exit naturally.

    Args:
        func: User function (same contract as parallel_map).
        items_list: Materialised items.
        n_processes: Worker count.
        total: len(items_list).
        desc: Progress description.
        logger: Optional logger.
        show_progress: tqdm flag.
        log_file_path: Subprocess logging path.
        log_level: Subprocess log level.
        per_item_timeout_sec: Seconds from submit before treating as failed.

    Returns:
        (successful_results, failed_items) same as parallel_map.
    """
    successful_results: List[ProcessingResult] = []
    failed_items: List[Any] = []

    worker_args = [
        (func, item, log_file_path, log_level)
        for item in items_list
    ]

    executor = ProcessPoolExecutor(max_workers=n_processes)
    future_to_args: Dict[Any, Tuple[Any, ...]] = {}
    submit_times: Dict[Any, float] = {}
    try:
        for wa in worker_args:
            fut = executor.submit(_worker_wrapper, wa)
            future_to_args[fut] = wa
            submit_times[fut] = time.monotonic()

        pending = set(future_to_args.keys())
        if show_progress:
            progress_bar = CustomTqdm(total=total, desc=desc)

        while pending:
            done, still_pending = wait(
                pending,
                timeout=0.5,
                return_when=FIRST_COMPLETED,
            )
            now = time.monotonic()

            for future in done:
                pending.discard(future)
                item_id = _item_id_from_worker_args(future_to_args[future])
                try:
                    proc_result = future.result()
                except Exception as exc:  # e.g. broken process pool
                    failed_items.append(item_id)
                    if logger:
                        logger.error(
                            "Error retrieving result for %s: %s",
                            item_id,
                            exc,
                            exc_info=True,
                        )
                else:
                    if proc_result.success:
                        successful_results.append(proc_result)
                    else:
                        failed_items.append(proc_result.item_id)
                        if logger:
                            logger.error(
                                "Error processing %s: %s",
                                proc_result.item_id,
                                proc_result.error,
                            )
                if show_progress:
                    progress_bar.update(1)

            timed_out = [
                fut for fut in still_pending
                if now - submit_times[fut] > per_item_timeout_sec
            ]
            for future in timed_out:
                pending.discard(future)
                item_id = _item_id_from_worker_args(future_to_args[future])
                failed_items.append(item_id)
                if logger:
                    logger.error(
                        "Timeout (>%ss) for item %s; skipping (worker may still run).",
                        per_item_timeout_sec,
                        item_id,
                    )
                if show_progress:
                    progress_bar.update(1)
    finally:
        # Timed-out futures may still run; do not block shutdown on them.
        executor.shutdown(wait=False)

    if logger and failed_items:
        logger.warning("Failed or timed out %s item(s)", len(failed_items))

    return successful_results, failed_items


def _sequential_map_with_item_timeout(
    func: Callable[[T], R],
    items_list: List[T],
    total: int,
    desc: str,
    logger: Optional[logging.Logger],
    show_progress: bool,
    log_file_path: Optional[Path],
    log_level: int,
    per_item_timeout_sec: float,
) -> Tuple[List[ProcessingResult], List[Any]]:
    """
    Sequential path with per-item timeout using one spawn child per item.

    After timeout the child is terminated (best-effort skip).

    Args:
        func: User function.
        items_list: Items to process.
        total: Number of items.
        desc: Progress description.
        logger: Optional logger.
        show_progress: tqdm flag.
        log_file_path: Subprocess logging path.
        log_level: Subprocess log level.
        per_item_timeout_sec: Per-item wall-clock limit.

    Returns:
        (successful_results, failed_items) same as parallel_map.
    """
    successful_results: List[ProcessingResult] = []
    failed_items: List[Any] = []
    ctx = multiprocessing.get_context("spawn")

    if show_progress:
        progress_bar = CustomTqdm(total=total, desc=desc)

    for item in items_list:
        worker_args = (func, item, log_file_path, log_level)
        item_id = _item_id_from_worker_args(worker_args)
        result_queue = ctx.Queue(maxsize=1)
        proc = ctx.Process(
            target=_put_worker_wrapper_result,
            args=(worker_args, result_queue),
        )
        proc.start()
        proc.join(timeout=per_item_timeout_sec)
        if proc.is_alive():
            if logger:
                logger.error(
                    "Timeout (>%ss) for item %s; terminating child process.",
                    per_item_timeout_sec,
                    item_id,
                )
            proc.terminate()
            proc.join(timeout=10)
            failed_items.append(item_id)
            if show_progress:
                progress_bar.update(1)
            continue

        try:
            proc_result = result_queue.get(timeout=2)
        except queue.Empty:
            failed_items.append(item_id)
            if logger:
                logger.error(
                    "Child exited without result for item %s (exit code %s).",
                    item_id,
                    proc.exitcode,
                )
            if show_progress:
                progress_bar.update(1)
            continue

        if proc_result.success:
            successful_results.append(proc_result)
        else:
            failed_items.append(proc_result.item_id)
            if logger:
                logger.error(
                    "Error processing %s: %s",
                    proc_result.item_id,
                    proc_result.error,
                )

        if show_progress:
            progress_bar.update(1)

    if logger and failed_items:
        logger.warning("Failed or timed out %s item(s)", len(failed_items))

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
) -> Tuple[List[ProcessingResult], List[Any]]:
    """
    Apply a function to items in parallel or sequentially with unified interface.
    
    This function provides:
    - Automatic switching between parallel and sequential processing
    - Progress bar display
    - Error collection without stopping processing
    - Logging restoration in child processes (Windows compatibility)
    
    Args:
        func: Function to apply to each item. Should return (item_id, result) tuple
              or just result. If processing fails, can return (item_id, Exception).
        items: Iterable of items to process
        n_processes: Number of parallel processes (1 = sequential)
        desc: Description for progress bar
        logger: Logger for status messages
        show_progress: Whether to show progress bar
        log_file_path: Path to log file for child process logging restoration
        log_level: Logging level for child processes
        per_item_timeout_sec: If set to a positive value, items exceeding this wall
            time since submit (parallel) or since child start (sequential) are
            recorded as failed and skipped. Parallel mode does not terminate stuck
            workers; sequential mode calls terminate(). None keeps legacy behaviour.

    Returns:
        Tuple[List[ProcessingResult], List[Any]]: 
            - List of successful ProcessingResult objects
            - List of failed item IDs
            
    Example:
        >>> def process_subject(subject_id):
        ...     # Do processing
        ...     return subject_id, processed_data
        >>> 
        >>> results, failed = parallel_map(
        ...     process_subject,
        ...     subject_list,
        ...     n_processes=4,
        ...     desc="Processing subjects"
        ... )
    """
    items_list = list(items)
    total = len(items_list)

    if total == 0:
        return [], []

    if per_item_timeout_sec is not None and per_item_timeout_sec > 0:
        if log_file_path is None:
            manager = LoggerManager()
            log_file_path = manager.get_log_file()
        if n_processes > 1 and total > 1:
            return _parallel_map_with_item_timeout(
                func=func,
                items_list=items_list,
                n_processes=n_processes,
                total=total,
                desc=desc,
                logger=logger,
                show_progress=show_progress,
                log_file_path=log_file_path,
                log_level=log_level,
                per_item_timeout_sec=per_item_timeout_sec,
            )
        return _sequential_map_with_item_timeout(
            func=func,
            items_list=items_list,
            total=total,
            desc=desc,
            logger=logger,
            show_progress=show_progress,
            log_file_path=log_file_path,
            log_level=log_level,
            per_item_timeout_sec=per_item_timeout_sec,
        )

    successful_results: List[ProcessingResult] = []
    failed_items: List[Any] = []
    
    # Get logging configuration if not provided
    if log_file_path is None:
        manager = LoggerManager()
        log_file_path = manager.get_log_file()
    
    # Use parallel processing
    if n_processes > 1 and total > 1:
        if logger:
            logger.info(f"Using {n_processes} processes for parallel processing...")
        
        # Prepare arguments with logging info
        worker_args = [
            (func, item, log_file_path, log_level) 
            for item in items_list
        ]
        
        with multiprocessing.Pool(processes=n_processes) as pool:
            results_iter = pool.imap_unordered(_worker_wrapper, worker_args)
            
            if show_progress:
                progress_bar = CustomTqdm(total=total, desc=desc)
            
            for result in results_iter:
                if result.success:
                    successful_results.append(result)
                else:
                    failed_items.append(result.item_id)
                    if logger:
                        logger.error(
                            f"Error processing {result.item_id}: {result.error}"
                        )
                
                if show_progress:
                    progress_bar.update(1)
    
    # Use sequential processing
    else:
        if show_progress:
            progress_bar = CustomTqdm(total=total, desc=desc)
        
        for item in items_list:
            # Call function directly (no wrapper needed for sequential)
            try:
                raw_result = func(item)
                
                # Handle tuple returns
                if isinstance(raw_result, tuple) and len(raw_result) == 2:
                    item_id, actual_result = raw_result
                    if isinstance(actual_result, Exception):
                        failed_items.append(item_id)
                        if logger:
                            logger.error(f"Error processing {item_id}: {actual_result}")
                    else:
                        successful_results.append(
                            ProcessingResult(item_id=item_id, result=actual_result)
                        )
                else:
                    successful_results.append(
                        ProcessingResult(item_id=item, result=raw_result)
                    )
                    
            except Exception as e:
                failed_items.append(item)
                if logger:
                    logger.error(f"Error processing {item}: {e}")
            
            if show_progress:
                progress_bar.update(1)
    
    # Log summary
    if logger and failed_items:
        logger.warning(f"Failed to process {len(failed_items)} item(s)")
    
    return successful_results, failed_items


def parallel_map_simple(
    func: Callable[[T], R],
    items: Iterable[T],
    n_processes: int = 1,
    desc: str = "Processing",
    show_progress: bool = True,
) -> Generator[R, None, None]:
    """
    Simplified parallel map that yields results directly.
    
    This is a simpler alternative to parallel_map when you don't need
    detailed error tracking. Results are yielded as they complete.
    
    Args:
        func: Function to apply to each item
        items: Iterable of items to process
        n_processes: Number of parallel processes (1 = sequential)
        desc: Description for progress bar
        show_progress: Whether to show progress bar
        
    Yields:
        Results from the function (may include exceptions)
        
    Example:
        >>> for result in parallel_map_simple(process_fn, items, n_processes=4):
        ...     if isinstance(result, Exception):
        ...         handle_error(result)
        ...     else:
        ...         handle_success(result)
    """
    items_list = list(items)
    total = len(items_list)
    
    if total == 0:
        return
    
    if show_progress:
        progress_bar = CustomTqdm(total=total, desc=desc)
    
    if n_processes > 1 and total > 1:
        with multiprocessing.Pool(processes=n_processes) as pool:
            for result in pool.imap_unordered(func, items_list):
                yield result
                if show_progress:
                    progress_bar.update(1)
    else:
        for item in items_list:
            try:
                yield func(item)
            except Exception as e:
                yield e
            if show_progress:
                progress_bar.update(1)


class ParallelProcessor:
    """
    Context manager for parallel processing with automatic resource management.
    
    This class provides a cleaner interface for batch parallel processing
    with proper resource cleanup and logging configuration.
    
    Example:
        >>> with ParallelProcessor(n_processes=4) as processor:
        ...     results = processor.map(process_fn, items, desc="Processing")
    """
    
    def __init__(
        self,
        n_processes: int = 1,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize parallel processor.
        
        Args:
            n_processes: Number of parallel processes
            logger: Logger for status messages
        """
        self.n_processes = n_processes
        self.logger = logger
        self._pool: Optional[multiprocessing.Pool] = None
        
        # Get logging configuration
        manager = LoggerManager()
        self._log_file_path = manager.get_log_file()
        self._log_level = logging.INFO
        if manager._root_logger:
            self._log_level = manager._root_logger.getEffectiveLevel()
    
    def __enter__(self) -> 'ParallelProcessor':
        """Enter context manager."""
        if self.n_processes > 1:
            self._pool = multiprocessing.Pool(processes=self.n_processes)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and cleanup resources."""
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None
        return False
    
    def map(
        self,
        func: Callable[[T], R],
        items: Iterable[T],
        desc: str = "Processing",
        show_progress: bool = True,
        per_item_timeout_sec: Optional[float] = None,
    ) -> Tuple[List[ProcessingResult], List[Any]]:
        """
        Map function over items using the processor's configuration.

        Args:
            func: Function to apply to each item
            items: Iterable of items to process
            desc: Description for progress bar
            show_progress: Whether to show progress bar
            per_item_timeout_sec: Optional per-item wall-clock timeout (see parallel_map).

        Returns:
            Tuple of (successful_results, failed_items)
        """
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
        )
