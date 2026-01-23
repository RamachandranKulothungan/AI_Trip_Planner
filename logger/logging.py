"""Centralized logging configuration."""
import logging
import sys
from typing import Optional
from pathlib import Path


class LoggerConfig:
    """Singleton logger configuration."""
    
    _instance: Optional['LoggerConfig'] = None
    _configured = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def configure(
        self,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        format_string: Optional[str] = None
    ) -> None:
        """
        Configure logging for the application.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional path to log file
            format_string: Custom format string for log messages
        """
        if self._configured:
            return
        
        level = getattr(logging, log_level.upper(), logging.INFO)
        format_str = format_string or (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        # Configure root logger
        logging.basicConfig(
            level=level,
            format=format_str,
            handlers=self._create_handlers(log_file)
        )
        
        # Set third-party logger levels
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        
        self._configured = True
    
    def _create_handlers(self, log_file: Optional[str]) -> list:
        """Create log handlers."""
        handlers = [logging.StreamHandler(sys.stdout)]
        
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_file))
        
        return handlers


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    config = LoggerConfig()
    if not config._configured:
        config.configure()
    return logging.getLogger(name)
