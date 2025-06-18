"""
Professional logging module for Thinker AI Auxiliary Window
Provides centralized logging functionality with different levels and formatters
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

from src.config.config import get_config


class ThinkerLogger:
    """Professional logging class with multiple handlers and formatters"""
    
    def __init__(self, name: str = "ThinkerAI"):
        self.config = get_config()
        self.logger_name = name
        self.logger = None
        self._setup_logger()
    
    def _setup_logger(self) -> None:
        """Setup logger with file and console handlers"""
        # Ensure logs directory exists
        self.config.ensure_directories()
        
        # Create logger
        self.logger = logging.getLogger(self.logger_name)
        self.logger.setLevel(getattr(logging, self.config.LOG_LEVEL))
        
        # Clear existing handlers to avoid duplication
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '%(levelname)-8s | %(message)s'
        )
        
        # File handler with rotation
        log_file_path = self.config.LOGS_DIR / self.config.LOG_FILE
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Log startup message
        self.logger.info(f"Logger initialized for {self.logger_name}")
        self.logger.debug(f"Log file: {log_file_path}")
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message"""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message"""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Log error message with optional exception info"""
        self.logger.error(message, exc_info=exc_info, **kwargs)
    
    def critical(self, message: str, exc_info: bool = True, **kwargs) -> None:
        """Log critical message with exception info by default"""
        self.logger.critical(message, exc_info=exc_info, **kwargs)
    
    def log_function_call(self, func_name: str, args: tuple = None, kwargs: dict = None) -> None:
        """Log function call with parameters"""
        args_str = f"args={args}" if args else ""
        kwargs_str = f"kwargs={kwargs}" if kwargs else ""
        params = " | ".join(filter(None, [args_str, kwargs_str]))
        
        self.debug(f"Function called: {func_name} | {params}")
    
    def log_performance(self, operation: str, duration: float, details: str = "") -> None:
        """Log performance metrics"""
        message = f"Performance | {operation} | Duration: {duration:.4f}s"
        if details:
            message += f" | {details}"
        self.info(message)
    
    def log_exception(self, exception: Exception, context: str = "") -> None:
        """Log exception with context"""
        context_str = f" | Context: {context}" if context else ""
        self.error(f"Exception occurred: {type(exception).__name__}: {exception}{context_str}", exc_info=True)
    
    def log_user_action(self, action: str, details: str = "") -> None:
        """Log user actions for audit trail"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        message = f"User Action | {timestamp} | {action}"
        if details:
            message += f" | {details}"
        self.info(message)
    
    def log_ai_operation(self, operation: str, model: str = "", tokens: int = 0, duration: float = 0) -> None:
        """Log AI/ML operations with metrics"""
        metrics = []
        if model:
            metrics.append(f"Model: {model}")
        if tokens > 0:
            metrics.append(f"Tokens: {tokens}")
        if duration > 0:
            metrics.append(f"Duration: {duration:.3f}s")
        
        metrics_str = " | ".join(metrics)
        self.info(f"AI Operation | {operation} | {metrics_str}")
    
    def log_security_event(self, event_type: str, severity: str, details: str) -> None:
        """Log security-related events"""
        severity_upper = severity.upper()
        message = f"SECURITY | {severity_upper} | {event_type} | {details}"
        
        if severity_upper in ["HIGH", "CRITICAL"]:
            self.error(message)
        elif severity_upper == "MEDIUM":
            self.warning(message)
        else:
            self.info(message)


# Global logger instance
_global_logger: Optional[ThinkerLogger] = None


def get_logger(name: str = "ThinkerAI") -> ThinkerLogger:
    """Get or create global logger instance"""
    global _global_logger
    
    if _global_logger is None or _global_logger.logger_name != name:
        _global_logger = ThinkerLogger(name)
    
    return _global_logger


# Convenience functions for quick logging
def log_info(message: str) -> None:
    """Quick info logging"""
    get_logger().info(message)


def log_error(message: str, exc_info: bool = False) -> None:
    """Quick error logging"""
    get_logger().error(message, exc_info=exc_info)


def log_debug(message: str) -> None:
    """Quick debug logging"""
    get_logger().debug(message)


def log_warning(message: str) -> None:
    """Quick warning logging"""
    get_logger().warning(message) 