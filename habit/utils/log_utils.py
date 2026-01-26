"""
Centralized logging utility module for HABIT project.

This module provides a unified logging system with the following features:
- Hierarchical logger management
- Single log file per run (no duplicate logs folders)
- Console and file output with different formats
- Thread-safe logger initialization
- Clear separation between main logs and module logs

Design principles:
1. One log file per application/script run
2. All logs stored in {output_dir}/processing.log (no logs/ subfolder)
3. Hierarchical logger names (habit.preprocessing, habit.habitat, etc.)
4. Console output: simple format for readability
5. File output: detailed format with file location and line numbers
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import threading

# Global lock for thread-safe logger initialization
_logger_lock = threading.Lock()
_initialized_loggers = set()


class LoggerManager:
    """
    Centralized logger manager for the HABIT project.
    
    This class ensures consistent logging across all modules with:
    - Single point of configuration
    - No duplicate handlers
    - Hierarchical logger structure
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
            self._log_file = None
            self._root_logger = None
            self._initialized = True
    
    def setup_root_logger(
        self, 
        log_file: Optional[Path] = None,
        level: int = logging.INFO,
        console_level: Optional[int] = None,
        append_mode: bool = False
    ) -> logging.Logger:
        """
        Setup the root logger for HABIT project.
        
        This should be called once at the start of each application/script.
        All subsequent module loggers will inherit from this configuration.
        
        Args:
            log_file: Path to the log file. If None, only console logging is enabled.
            level: Logging level for file output (default: INFO)
            console_level: Logging level for console output. If None, uses same as level.
            append_mode: If True, append to existing log file instead of overwriting.
                         Used by child processes in multiprocessing to avoid overwriting.
            
        Returns:
            logging.Logger: The root logger for HABIT project
        """
        with _logger_lock:
            # Get or create root logger
            root_logger = logging.getLogger('habit')
            
            # Clear existing handlers to avoid duplicates
            root_logger.handlers.clear()
            root_logger.setLevel(logging.DEBUG)  # Set to DEBUG to allow all messages through
            
            # Prevent propagation to avoid duplicate logs
            root_logger.propagate = False
            
            # Console handler with simple format
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(console_level if console_level is not None else level)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)
            
            # File handler with detailed format
            if log_file:
                log_file = Path(log_file)
                log_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Use append mode for child processes, overwrite for main process
                file_mode = 'a' if append_mode else 'w'
                
                file_handler = logging.FileHandler(
                    str(log_file), 
                    mode=file_mode,
                    encoding='utf-8'
                )
                file_handler.setLevel(level)
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                file_handler.setFormatter(file_formatter)
                root_logger.addHandler(file_handler)
                
                self._log_file = log_file
                self._log_level = level
                if not append_mode:
                    root_logger.info(f"Log file initialized: {log_file}")
            
            self._root_logger = root_logger
            return root_logger
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger with the specified name under the HABIT hierarchy.
        
        Args:
            name: Logger name (will be prefixed with 'habit.' if not already)
            
        Returns:
            logging.Logger: Logger instance
            
        Example:
            logger = LoggerManager().get_logger('preprocessing')
            # Creates logger 'habit.preprocessing'
        """
        # Ensure name is under habit hierarchy
        if not name.startswith('habit'):
            name = f'habit.{name}'
        
        logger = logging.getLogger(name)
        
        # Module loggers should not add their own handlers
        # They inherit from the root 'habit' logger
        logger.propagate = True
        
        return logger
    
    def get_log_file(self) -> Optional[Path]:
        """
        Get the current log file path.
        
        Returns:
            Optional[Path]: Path to log file, or None if file logging not enabled
        """
        return self._log_file


def setup_logger(
    name: str,
    output_dir: Optional[Path] = None,
    log_filename: str = "processing.log",
    level: int = logging.INFO,
    console_level: Optional[int] = None
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
        
    Example:
        # In a script or CLI command:
        logger = setup_logger('preprocessing', output_dir=Path('/output'), level=logging.INFO)
        logger.info('Processing started')
        
        # In a module:
        logger = get_module_logger(__name__)
        logger.debug('Debug information')
    """
    manager = LoggerManager()
    
    # Setup root logger if output_dir is specified
    if output_dir:
        log_file = Path(output_dir) / log_filename
        manager.setup_root_logger(log_file=log_file, level=level, console_level=console_level)
    else:
        # Console-only logging
        manager.setup_root_logger(level=level, console_level=console_level)
    
    # Return module logger
    return manager.get_logger(name)


def get_module_logger(module_name: str) -> logging.Logger:
    """
    Get a logger for a module.
    
    This should be called in modules that don't initialize logging themselves.
    The module will use the logging configuration set by the main script/CLI command.
    
    Args:
        module_name: The __name__ of the module
        
    Returns:
        logging.Logger: Logger instance for the module
        
    Example:
        # At the top of a module file:
        from habit.utils.log_utils import get_module_logger
        logger = get_module_logger(__name__)
    """
    manager = LoggerManager()
    
    # Extract meaningful name from module path
    # e.g., 'habit.core.preprocessing.resample' -> 'habit.core.preprocessing.resample'
    if module_name.startswith('habit.'):
        logger_name = module_name
    elif module_name == '__main__':
        logger_name = 'habit.main'
    else:
        logger_name = f'habit.{module_name}'
    
    return logging.getLogger(logger_name)


def disable_external_loggers():
    """
    Disable verbose logging from external libraries.
    
    Many libraries (like SimpleITK, scikit-learn) can be very verbose.
    This function sets them to WARNING level to reduce noise.
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


def restore_logging_in_subprocess(
    log_file_path: Optional[Path] = None,
    log_level: int = logging.INFO
) -> None:
    """
    Restore logging configuration in a child process.
    
    In Windows spawn mode (and forkserver), child processes don't inherit
    the parent's logging configuration. This function should be called at
    the beginning of any function that runs in a child process.
    
    Args:
        log_file_path: Path to the log file (should be passed from parent process)
        log_level: Logging level (should be passed from parent process)
        
    Example:
        # In parent process, store the log config:
        self._log_file_path = LoggerManager().get_log_file()
        self._log_level = logging.INFO
        
        # In child process function:
        def process_in_child(self, data):
            restore_logging_in_subprocess(self._log_file_path, self._log_level)
            # ... rest of processing
    """
    manager = LoggerManager()
    
    # Only restore if not already configured (we're in a child process)
    if manager.get_log_file() is None and log_file_path:
        manager.setup_root_logger(
            log_file=log_file_path,
            level=log_level,
            append_mode=True  # Append to existing log file
        )


# Convenience function for backward compatibility
def setup_output_logger(
    output_dir: Path, 
    name: str, 
    level: int = logging.INFO
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
