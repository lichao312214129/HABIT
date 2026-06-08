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
Centralized logging utility module for HABIT project.

This module provides a unified logging system with the following features:
- Hierarchical logger management
- Single log file per run (no duplicate logs folders)
- Console and file output with different formats
- Thread-safe logger initialization
- Multiprocess-safe logging via QueueHandler + QueueListener (Option C)
- Clear separation between main logs and module logs

Design principles:
1. One log file per application/script run
2. All logs stored in {output_dir}/processing.log (no logs/ subfolder)
3. Hierarchical logger names (habit.preprocessing, habit.habitat, etc.)
4. Console output: simple format for readability
5. File output: detailed format with file location and line numbers
6. Child processes enqueue records; the main process QueueListener writes in order
"""

from __future__ import annotations

import atexit
import logging
import multiprocessing
import sys
from contextlib import contextmanager
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path
from typing import Any, Iterator, Optional
import threading

# Global lock for thread-safe logger initialization
_logger_lock = threading.Lock()
_initialized_loggers = set()


def _console_formatter() -> logging.Formatter:
    """Simple formatter for console output."""
    return logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


def _file_formatter() -> logging.Formatter:
    """Detailed formatter for file output."""
    return logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


class LoggerManager:
    """
    Centralized logger manager for the HABIT project.

    This class ensures consistent logging across all modules with:
    - Single point of configuration
    - No duplicate handlers
    - Hierarchical logger structure
    - Multiprocess queue routing when file logging is enabled in the main process
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern to ensure only one LoggerManager instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the LoggerManager."""
        if not self._initialized:
            self._log_file: Optional[Path] = None
            self._root_logger: Optional[logging.Logger] = None
            self._log_level: int = logging.INFO
            self._log_queue: Any = None
            self._queue_listener: Optional[QueueListener] = None
            self._mp_context: Optional[multiprocessing.context.BaseContext] = None
            self._listener_atexit_registered: bool = False
            self._initialized = True

    def setup_root_logger(
        self,
        log_file: Optional[Path] = None,
        level: int = logging.INFO,
        console_level: Optional[int] = None,
        append_mode: bool = False,
    ) -> logging.Logger:
        """
        Setup the root logger for HABIT project.

        This should be called once at the start of each application/script.
        All subsequent module loggers will inherit from this configuration.

        When ``append_mode`` is False and ``log_file`` is set, the main process
        routes all ``habit`` logs through a multiprocessing queue so child
        processes and the main process share one ordered writer.

        Args:
            log_file: Path to the log file. If None, only console logging is enabled.
            level: Logging level for file output (default: INFO)
            console_level: Logging level for console output. If None, uses same as level.
            append_mode: Legacy child-process direct append when no queue is available.

        Returns:
            logging.Logger: The root logger for HABIT project
        """
        with _logger_lock:
            root_logger = logging.getLogger('habit')
            self.stop_queue_listener()

            root_logger.handlers.clear()
            root_logger.setLevel(logging.DEBUG)
            root_logger.propagate = False

            resolved_console_level = (
                console_level if console_level is not None else level
            )

            if log_file and not append_mode:
                log_file = Path(log_file)
                log_file.parent.mkdir(parents=True, exist_ok=True)

                self._mp_context = multiprocessing.get_context('spawn')
                self._log_queue = self._mp_context.Queue(-1)
                self._log_file = log_file
                self._log_level = level

                file_handler = logging.FileHandler(
                    str(log_file),
                    mode='w',
                    encoding='utf-8',
                )
                file_handler.setLevel(level)
                file_handler.setFormatter(_file_formatter())

                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(resolved_console_level)
                console_handler.setFormatter(_console_formatter())

                self._queue_listener = QueueListener(
                    self._log_queue,
                    file_handler,
                    console_handler,
                    respect_handler_level=True,
                )
                self._queue_listener.start()
                if not self._listener_atexit_registered:
                    atexit.register(self.stop_queue_listener)
                    self._listener_atexit_registered = True

                queue_handler = QueueHandler(self._log_queue)
                queue_handler.setLevel(logging.DEBUG)
                root_logger.addHandler(queue_handler)

                root_logger.info("Log file initialized: %s", log_file)
            elif log_file and append_mode:
                log_file = Path(log_file)
                log_file.parent.mkdir(parents=True, exist_ok=True)

                file_handler = logging.FileHandler(
                    str(log_file),
                    mode='a',
                    encoding='utf-8',
                )
                file_handler.setLevel(level)
                file_handler.setFormatter(_file_formatter())
                root_logger.addHandler(file_handler)

                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(resolved_console_level)
                console_handler.setFormatter(_console_formatter())
                root_logger.addHandler(console_handler)

                self._log_file = log_file
                self._log_level = level
                self._log_queue = None
            else:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(resolved_console_level)
                console_handler.setFormatter(_console_formatter())
                root_logger.addHandler(console_handler)

                self._log_file = None
                self._log_queue = None
                self._log_level = level

            self._root_logger = root_logger
            return root_logger

    def stop_queue_listener(self) -> None:
        """
        Stop the multiprocessing log listener and flush pending records.

        Safe to call when queue logging was never started or already stopped.
        """
        listener = self._queue_listener
        if listener is None:
            return
        try:
            listener.stop()
        finally:
            self._queue_listener = None

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger with the specified name under the HABIT hierarchy.

        Args:
            name: Logger name (will be prefixed with 'habit.' if not already)

        Returns:
            logging.Logger: Logger instance
        """
        if not name.startswith('habit'):
            name = f'habit.{name}'

        logger = logging.getLogger(name)
        logger.propagate = True
        return logger

    def get_log_file(self) -> Optional[Path]:
        """
        Get the current log file path.

        Returns:
            Optional[Path]: Path to log file, or None if file logging not enabled
        """
        return self._log_file

    def get_log_queue(self) -> Any:
        """
        Return the multiprocessing log queue for child processes.

        Returns:
            Optional[multiprocessing.Queue]: Queue when file logging uses queue mode;
            otherwise None.
        """
        return self._log_queue


def setup_logger(
    name: str,
    output_dir: Optional[Path] = None,
    log_filename: str = "processing.log",
    level: int = logging.INFO,
    console_level: Optional[int] = None,
) -> logging.Logger:
    """
    Setup a logger for a HABIT module or script.

    This is the main entry point for setting up logging in HABIT applications.

    Args:
        name: Name of the module/script (e.g., 'preprocessing', 'habitat')
        output_dir: Directory where log file will be created. If None, only console logging.
        log_filename: Name of the log file (default: 'processing.log')
        level: Logging level for file output (default: INFO)
        console_level: Logging level for console. If None, uses same as level.

    Returns:
        logging.Logger: Configured logger instance
    """
    manager = LoggerManager()

    if output_dir:
        log_file = Path(output_dir) / log_filename
        manager.setup_root_logger(
            log_file=log_file,
            level=level,
            console_level=console_level,
        )
    else:
        manager.setup_root_logger(level=level, console_level=console_level)

    return manager.get_logger(name)


def get_module_logger(module_name: str) -> logging.Logger:
    """
    Get a logger for a module.

    Args:
        module_name: The __name__ of the module

    Returns:
        logging.Logger: Logger instance for the module
    """
    if module_name.startswith('habit.'):
        logger_name = module_name
    elif module_name == '__main__':
        logger_name = 'habit.main'
    else:
        logger_name = f'habit.{module_name}'

    return logging.getLogger(logger_name)


def disable_external_loggers() -> None:
    """
    Disable verbose logging from external libraries.
    """
    external_loggers = [
        'SimpleITK',
        'matplotlib',
        'PIL',
        'sklearn',
        'numba',
        'radiomics',
    ]

    for logger_name in external_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


RADIOMICS_LOGGER_NAME = "radiomics"


def resolve_radiomics_logging_level(debug: bool = False) -> int:
    """
    Resolve PyRadiomics logger level for feature extraction.

    When ``debug`` is True, PyRadiomics emits DEBUG messages such as
    ``Calculating voxel batch no. X/Y`` during voxel-based extraction.

    Args:
        debug: Whether habitat debug mode is enabled.

    Returns:
        int: ``logging.DEBUG`` when debug is True, otherwise ``logging.INFO``.
    """
    return logging.DEBUG if debug else logging.INFO


@contextmanager
def radiomics_feature_class_logging(level: int = logging.INFO) -> Iterator[None]:
    """
    Temporarily raise the PyRadiomics logger level during voxel feature extraction.

    Pass :func:`resolve_radiomics_logging_level` when wiring habitat ``debug``
    so batch progress lines (DEBUG) are visible in ``habitat_analysis.log``.

    Args:
        level: Logging level to apply to the radiomics logger (default: INFO).

    Yields:
        None
    """
    radiomics_logger = logging.getLogger(RADIOMICS_LOGGER_NAME)
    previous_level = radiomics_logger.level
    radiomics_logger.setLevel(level)

    forward_handler: Optional[logging.Handler] = None
    if level <= logging.DEBUG:
        habit_logger = logging.getLogger("habit")

        class _RadiomicsForwardHandler(logging.Handler):
            """Forward PyRadiomics records into the habit logging tree."""

            def emit(self, record: logging.LogRecord) -> None:
                if habit_logger.isEnabledFor(record.levelno):
                    habit_logger.handle(record)

        forward_handler = _RadiomicsForwardHandler()
        forward_handler.setLevel(logging.DEBUG)
        radiomics_logger.addHandler(forward_handler)

    try:
        yield
    finally:
        if forward_handler is not None:
            radiomics_logger.removeHandler(forward_handler)
        radiomics_logger.setLevel(previous_level)


def restore_logging_in_subprocess(
    log_file_path: Optional[Path] = None,
    log_level: int = logging.INFO,
    log_queue: Any = None,
) -> None:
    """
    Restore logging configuration in a child process.

    Prefer ``log_queue`` from the parent process so records are written in order
    by the main-process QueueListener. When no queue is available, fall back to
    direct append-mode file logging (legacy path).

    Args:
        log_file_path: Path to the log file (legacy fallback)
        log_level: Logging level for the child root logger
        log_queue: Multiprocessing queue shared with the main process
    """
    root_logger = logging.getLogger('habit')
    if root_logger.handlers:
        return

    if log_queue is not None:
        root_logger.setLevel(logging.DEBUG)
        root_logger.propagate = False
        queue_handler = QueueHandler(log_queue)
        queue_handler.setLevel(log_level)
        root_logger.addHandler(queue_handler)
        return

    manager = LoggerManager()
    if manager.get_log_file() is None and log_file_path:
        manager.setup_root_logger(
            log_file=log_file_path,
            level=log_level,
            append_mode=True,
        )


def shutdown_subprocess_logging() -> None:
    """
    Flush and close logging handlers in a child process before exit.

    Ensures queued records reach the main-process listener before the worker exits.
    """
    logging.shutdown()


def stop_queue_listener() -> None:
    """Stop the main-process queue listener and flush pending log records."""
    LoggerManager().stop_queue_listener()


# Convenience function for backward compatibility
def setup_output_logger(
    output_dir: Path,
    name: str,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Legacy function for backward compatibility.

    Args:
        output_dir: Directory where log file will be created
        name: Name of the logger
        level: Logging level

    Returns:
        logging.Logger: Configured logger instance
    """
    return setup_logger(name=name, output_dir=output_dir, level=level)
