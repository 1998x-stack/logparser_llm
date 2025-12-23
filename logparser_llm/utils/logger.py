"""
Logging utilities for LogParser-LLM.
"""
import sys
import logging
from pathlib import Path
from typing import Optional
from loguru import logger as loguru_logger


def setup_logger(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "100 MB",
    retention: str = "10 days",
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logger with file and console output.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        rotation: Log rotation size/time
        retention: Log retention period
        format_string: Custom format string
        
    Returns:
        Configured logger
    """
    # Remove default handler
    loguru_logger.remove()
    
    # Default format
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    # Console handler
    loguru_logger.add(
        sys.stderr,
        format=format_string,
        level=log_level,
        colorize=True
    )
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        loguru_logger.add(
            log_file,
            format=format_string,
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression="zip"
        )
    
    return loguru_logger


def get_logger(name: str = "logparser_llm") -> logging.Logger:
    """
    Get logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return loguru_logger.bind(name=name)


class LoggerContext:
    """Context manager for temporary log level changes."""
    
    def __init__(self, level: str):
        """
        Initialize context.
        
        Args:
            level: Temporary log level
        """
        self.level = level
        self.original_level = None
    
    def __enter__(self):
        """Enter context."""
        # Store current level and set new one
        self.original_level = loguru_logger._core.min_level
        loguru_logger.level(self.level)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore original level."""
        if self.original_level is not None:
            loguru_logger.level(self.original_level)


# Default logger instance
logger = setup_logger()


# Example usage
if __name__ == "__main__":
    # Setup logger
    logger = setup_logger(
        log_level="DEBUG",
        log_file="logs/logparser.log",
        rotation="10 MB"
    )
    
    # Test logging
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test context manager
    with LoggerContext("ERROR"):
        logger.debug("This won't be shown")
        logger.error("This will be shown")
    
    logger.info("Back to normal level")