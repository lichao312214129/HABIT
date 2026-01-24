"""
Parallel processing utilities for HABIT project.

This module provides a unified interface for parallel and sequential processing,
eliminating code duplication across different modules that need multiprocessing.
"""

import logging
import multiprocessing
from typing import (
    TypeVar, Callable, Iterable, List, Tuple, 
    Optional, Any, Union, Generator
)
from dataclasses import dataclass
from pathlib import Path

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


def parallel_map(
    func: Callable[[T], R],
    items: Iterable[T],
    n_processes: int = 1,
    desc: str = "Processing",
    logger: Optional[logging.Logger] = None,
    show_progress: bool = True,
    log_file_path: Optional[Path] = None,
    log_level: int = logging.INFO,
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
    ) -> Tuple[List[ProcessingResult], List[Any]]:
        """
        Map function over items using the processor's configuration.
        
        Args:
            func: Function to apply to each item
            items: Iterable of items to process
            desc: Description for progress bar
            show_progress: Whether to show progress bar
            
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
        )
