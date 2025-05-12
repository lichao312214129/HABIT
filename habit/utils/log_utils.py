"""
Logging utility module for HABIT
"""

import os
import logging
from datetime import datetime
from typing import Optional

class HABITLogger:
    """
    Custom logger for HABIT project
    """
    def __init__(self, out_dir: str, name: str = "HABIT"):
        """
        Initialize logger

        Args:
            out_dir (str): Output directory for log files
            name (str): Logger name
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Create logs directory if it doesn't exist
        log_dir = os.path.join(out_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        # Create file handler with UTF-8 encoding
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"habit_{timestamp}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)

    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)

def get_logger(out_dir: str, name: Optional[str] = None) -> HABITLogger:
    """
    Get or create a logger instance

    Args:
        out_dir (str): Output directory for log files
        name (str, optional): Logger name. If None, uses default name "HABIT"

    Returns:
        HABITLogger: Logger instance
    """
    return HABITLogger(out_dir, name or "HABIT") 