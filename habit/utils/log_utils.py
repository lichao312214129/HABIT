"""
Logging utility module for HABIT project.
This module provides centralized logging configuration and utilities.
"""

import logging
import os
from datetime import datetime
from pathlib import Path

def setup_logger(name: str, log_dir: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Set up and configure a logger instance.
    
    Args:
        name (str): Name of the logger
        log_dir (str, optional): Directory to store log files. If None, logs will only be printed to console
        level (int, optional): Logging level. Defaults to logging.INFO
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_dir is provided)
    if log_dir:
        # Create log directory if it doesn't exist
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_path / f"{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger 

def setup_output_logger(output_dir: str, name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up and configure a logger instance for output results.
    This logger will save log files to a 'logs' subdirectory in the specified output directory.
    
    Args:
        output_dir (str): Directory where results are being saved
        name (str): Name of the logger
        level (int, optional): Logging level. Defaults to logging.INFO
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory within output directory
    logs_dir = os.path.join(output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Use the existing setup_logger function with the logs directory
    return setup_logger(name, log_dir=logs_dir, level=level) 