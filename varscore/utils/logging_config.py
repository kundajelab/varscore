"""
Centralized logging configuration for varscore.

This module provides a standardized logging setup with datetime formatting
and consistent log levels across the entire varscore package.
"""

import logging
import sys
from datetime import datetime
from typing import Optional


def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    include_datetime: bool = True,
    log_to_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration for varscore.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string. If None, uses default with datetime
        include_datetime: Whether to include datetime in log messages
        log_to_file: Optional file path to write logs to
        
    Returns:
        Configured logger instance
    """
    # Default format with datetime
    if format_string is None:
        if include_datetime:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        else:
            format_string = "%(name)s - %(levelname)s - %(message)s"
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[]
    )
    
    # Get root logger
    logger = logging.getLogger()
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_to_file:
        file_handler = logging.FileHandler(log_to_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (typically __name__ of the calling module)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_function_call(logger: logging.Logger, func_name: str, **kwargs):
    """
    Log a function call with its parameters.
    
    Args:
        logger: Logger instance
        func_name: Name of the function being called
        **kwargs: Function parameters to log
    """
    params_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
    logger.info(f"Calling {func_name}({params_str})")


def log_timing(logger: logging.Logger, operation: str, start_time: datetime, end_time: Optional[datetime] = None):
    """
    Log timing information for operations.
    
    Args:
        logger: Logger instance
        operation: Description of the operation
        start_time: When the operation started
        end_time: When the operation ended (if None, uses current time)
    """
    if end_time is None:
        end_time = datetime.now()
    
    duration = (end_time - start_time).total_seconds()
    logger.info(f"{operation} completed in {duration:.2f} seconds")


def log_memory_usage(logger: logging.Logger, operation: str, memory_mb: float):
    """
    Log memory usage information.
    
    Args:
        logger: Logger instance
        operation: Description of the operation
        memory_mb: Memory usage in MB
    """
    logger.info(f"{operation} - Memory usage: {memory_mb:.2f} MB")


def log_progress(logger: logging.Logger, current: int, total: int, operation: str = "Processing"):
    """
    Log progress information.
    
    Args:
        logger: Logger instance
        current: Current progress count
        total: Total items to process
        operation: Description of the operation
    """
    percentage = (current / total) * 100 if total > 0 else 0
    logger.info(f"{operation}: {current}/{total} ({percentage:.1f}%)")


# Initialize default logger for the package
default_logger = get_logger("varscore")
