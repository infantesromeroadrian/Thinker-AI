"""
Custom exception hierarchy for Thinker AI Auxiliary Window.

This module defines a comprehensive exception hierarchy following the
error handling best practices specified in general_rules.mdc.
"""

from typing import Optional, Dict, Any


class ThinkerAIException(Exception):
    """
    Base exception for all Thinker AI Auxiliary Window exceptions.
    
    All custom exceptions should inherit from this base class to maintain
    a consistent exception hierarchy.
    """
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize ThinkerAI exception.
        
        Args:
            message: Human-readable error message
            error_code: Optional error code for programmatic handling
            context: Optional context information for debugging
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}


# Configuration-related exceptions
class ConfigurationError(ThinkerAIException):
    """Raised when configuration issues occur."""
    pass


class EnvironmentError(ConfigurationError):
    """Raised when environment setup fails."""
    pass


class InvalidConfigurationError(ConfigurationError):
    """Raised when configuration values are invalid."""
    pass


# Application lifecycle exceptions
class InitializationError(ThinkerAIException):
    """Raised when application initialization fails."""
    pass


class DependencyError(InitializationError):
    """Raised when required dependencies are missing or invalid."""
    pass


class ModuleInitializationError(InitializationError):
    """Raised when specific modules fail to initialize."""
    pass


# Data-related exceptions
class DataError(ThinkerAIException):
    """Base class for data-related errors."""
    pass


class DataValidationError(DataError):
    """Raised when data validation fails."""
    pass


class DataProcessingError(DataError):
    """Raised when data processing operations fail."""
    pass


class FileOperationError(DataError):
    """Raised when file operations fail."""
    pass


# AI/ML specific exceptions
class AIServiceError(ThinkerAIException):
    """Base class for AI service errors."""
    pass


class ModelError(AIServiceError):
    """Raised when AI model operations fail."""
    pass


class ModelNotAvailableError(ModelError):
    """Raised when required AI model is not available."""
    pass


class ModelTimeoutError(ModelError):
    """Raised when AI model operations timeout."""
    pass


class InferenceError(ModelError):
    """Raised when model inference fails."""
    pass


# Speech recognition exceptions
class SpeechServiceError(ThinkerAIException):
    """Base class for speech service errors."""
    pass


class MicrophoneError(SpeechServiceError):
    """Raised when microphone operations fail."""
    pass


class SpeechRecognitionError(SpeechServiceError):
    """Raised when speech recognition fails."""
    pass


class AudioProcessingError(SpeechServiceError):
    """Raised when audio processing fails."""
    pass


# Network and communication exceptions
class NetworkError(ThinkerAIException):
    """Base class for network-related errors."""
    pass


class ConnectionError(NetworkError):
    """Raised when network connections fail."""
    pass


class TimeoutError(NetworkError):
    """Raised when network operations timeout."""
    pass


class APIError(NetworkError):
    """Raised when API calls fail."""
    pass


class AuthenticationError(APIError):
    """Raised when API authentication fails."""
    pass


# UI and GUI exceptions
class UIError(ThinkerAIException):
    """Base class for UI-related errors."""
    pass


class WindowCreationError(UIError):
    """Raised when window creation fails."""
    pass


class UIComponentError(UIError):
    """Raised when UI component operations fail."""
    pass


class ThemeError(UIError):
    """Raised when theme operations fail."""
    pass


# Security-related exceptions
class SecurityError(ThinkerAIException):
    """Base class for security-related errors."""
    pass


class PermissionError(SecurityError):
    """Raised when permission checks fail."""
    pass


class ValidationError(SecurityError):
    """Raised when security validation fails."""
    pass


class CryptographyError(SecurityError):
    """Raised when cryptographic operations fail."""
    pass


# Performance and resource exceptions
class PerformanceError(ThinkerAIException):
    """Base class for performance-related errors."""
    pass


class ResourceExhaustedError(PerformanceError):
    """Raised when system resources are exhausted."""
    pass


class MemoryError(ResourceExhaustedError):
    """Raised when memory allocation fails."""
    pass


class ProcessingTimeoutError(PerformanceError):
    """Raised when operations exceed time limits."""
    pass


# Plugin and extension exceptions
class PluginError(ThinkerAIException):
    """Base class for plugin-related errors."""
    pass


class PluginLoadError(PluginError):
    """Raised when plugin loading fails."""
    pass


class PluginExecutionError(PluginError):
    """Raised when plugin execution fails."""
    pass


# Convenience functions for creating exceptions with context
def create_configuration_error(message: str, config_key: Optional[str] = None, 
                              config_value: Optional[Any] = None) -> ConfigurationError:
    """
    Create a ConfigurationError with structured context.
    
    Args:
        message: Error message
        config_key: Configuration key that caused the error
        config_value: Configuration value that was invalid
        
    Returns:
        ConfigurationError with structured context
    """
    context = {}
    if config_key:
        context['config_key'] = config_key
    if config_value is not None:
        context['config_value'] = str(config_value)
    
    return ConfigurationError(message, error_code="CONFIG_ERROR", context=context)


def create_model_error(message: str, model_name: Optional[str] = None, 
                      operation: Optional[str] = None) -> ModelError:
    """
    Create a ModelError with structured context.
    
    Args:
        message: Error message
        model_name: Name of the model that caused the error
        operation: Operation that was being performed
        
    Returns:
        ModelError with structured context
    """
    context = {}
    if model_name:
        context['model_name'] = model_name
    if operation:
        context['operation'] = operation
    
    return ModelError(message, error_code="MODEL_ERROR", context=context)


def create_speech_error(message: str, engine: Optional[str] = None, 
                       microphone_id: Optional[int] = None) -> SpeechServiceError:
    """
    Create a SpeechServiceError with structured context.
    
    Args:
        message: Error message
        engine: Speech recognition engine being used
        microphone_id: ID of microphone being used
        
    Returns:
        SpeechServiceError with structured context
    """
    context = {}
    if engine:
        context['engine'] = engine
    if microphone_id is not None:
        context['microphone_id'] = microphone_id
    
    return SpeechServiceError(message, error_code="SPEECH_ERROR", context=context) 