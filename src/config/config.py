"""
Configuration module for Thinker AI Auxiliary Window
This module centralizes all configuration settings for the tkinter application
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple, Union


class AppConfig:
    """Central configuration class for the Thinker AI Auxiliary application"""

    # Application Information
    APP_NAME = "Thinker AI - Auxiliary Window"
    APP_VERSION = "1.0.0"
    AUTHOR = "AI Assistant & Human Orchestrator"

    # Window Configuration
    WINDOW_WIDTH = 800
    WINDOW_HEIGHT = 600
    MIN_WINDOW_WIDTH = 600
    MIN_WINDOW_HEIGHT = 400
    WINDOW_RESIZABLE = True

    # UI Theme Configuration - Translucent Minimalist Theme
    PRIMARY_COLOR = "#1A1A1A"      # Deep dark background
    SECONDARY_COLOR = "#2A2A2A"    # Translucent dark overlay
    BACKGROUND_COLOR = "#0F0F0F"   # Ultra dark background
    TEXT_COLOR = "#E5E5E5"         # Soft white text
    SUCCESS_COLOR = "#00FF88"      # Bright neon green
    WARNING_COLOR = "#FFB347"      # Soft orange
    ERROR_COLOR = "#FF6B6B"        # Soft red
    ACCENT_COLOR = "#00D4FF"       # Electric blue
    BORDER_COLOR = "#404040"       # Subtle border
    BUTTON_COLOR = "transparent"   # Transparent buttons
    HOVER_COLOR = "#333333"        # Subtle hover

    # Font Configuration
    DEFAULT_FONT_FAMILY = "Segoe UI"
    DEFAULT_FONT_SIZE = 10
    HEADER_FONT_SIZE = 14
    TITLE_FONT_SIZE = 16

    # Layout Configuration
    PADDING_SMALL = 5
    PADDING_MEDIUM = 10
    PADDING_LARGE = 20
    BORDER_WIDTH = 1

    # File and Directory Configuration
    BASE_DIR = Path(__file__).parent.parent.parent
    SRC_DIR = BASE_DIR / "src"
    CONFIG_DIR = SRC_DIR / "config"
    LOGS_DIR = BASE_DIR / "logs"
    DATA_DIR = BASE_DIR / "data"

    # Logging Configuration
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = "thinker_aux.log"

    # AI Module Integration Configuration
    AI_MODULES_ENABLED = True
    CYBERSECURITY_MODULE = True
    ETHICAL_HACKING_MODULE = True

    # Qwen2.5-7B-Instruct-1M Model Configuration
    QWEN_BASE_URL = "http://192.168.1.45:1234"
    QWEN_MODEL_NAME = "qwen2.5-7b-instruct-1m"
    QWEN_TIMEOUT = 30  # Faster 7B model - normal timeout
    QWEN_MAX_RETRIES = 3  # Can afford more retries with faster model
    QWEN_TEMPERATURE = 0.7  # Higher temperature for more creative responses
    QWEN_MAX_TOKENS = 2000  # More tokens since model is faster

    # AI Assistant Configuration
    DEFAULT_SYSTEM_PROMPT = """Eres un asistente de IA inteligente y útil llamado Thinker AI, powered by Qwen2.5-7B-Instruct-1M. 
Ayudas con programación, ciberseguridad, análisis de código y tareas generales.
Siempre responde de manera clara, precisa y profesional en español.
Eres rápido, eficiente y puedes manejar conversaciones largas gracias a tu contexto extendido."""

    # Speech Recognition Configuration
    SPEECH_ENABLED = True
    SPEECH_ENGINE = "google"  # "google", "whisper", "azure"
    SPEECH_LANGUAGE = "es-ES"  # Spanish (Spain)
    SPEECH_ENERGY_THRESHOLD = 4000  # Higher for Windows (was 300)
    SPEECH_PAUSE_THRESHOLD = 3.0  # Seconds of silence to automatically send response
    SPEECH_TIMEOUT = 3.0  # Shorter timeout (was 5.0)
    SPEECH_PHRASE_TIME_LIMIT = 8.0  # Shorter phrase limit (was 10.0)

    # Azure Speech (if using Azure engine)
    AZURE_SPEECH_KEY = ""  # Set in environment or override
    AZURE_SPEECH_REGION = "westus2"  # Default region

    @classmethod
    def ensure_directories(cls) -> None:
        """
        Ensure all required directories exist.

        Creates the logs and data directories if they don't exist.
        """
        directories = [cls.LOGS_DIR, cls.DATA_DIR]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_font_config(cls, size_type: str = "default") -> Tuple[str, int]:
        """
        Get font configuration based on type.

        Args:
            size_type: Type of font size ("default", "header", "title")

        Returns:
            Tuple of (font_family, font_size)
        """
        size_map = {
            "default": cls.DEFAULT_FONT_SIZE,
            "header": cls.HEADER_FONT_SIZE,
            "title": cls.TITLE_FONT_SIZE
        }
        return (cls.DEFAULT_FONT_FAMILY, size_map.get(size_type, cls.DEFAULT_FONT_SIZE))

    @classmethod
    def get_color_scheme(cls) -> Dict[str, str]:
        """
        Get the complete color scheme as a dictionary.

        Returns:
            Dictionary mapping color names to hex color codes
        """
        return {
            "primary": cls.PRIMARY_COLOR,
            "secondary": cls.SECONDARY_COLOR,
            "background": cls.BACKGROUND_COLOR,
            "text": cls.TEXT_COLOR,
            "success": cls.SUCCESS_COLOR,
            "warning": cls.WARNING_COLOR,
            "error": cls.ERROR_COLOR,
            "accent": cls.ACCENT_COLOR,
            "border": cls.BORDER_COLOR,
            "button": cls.BUTTON_COLOR,
            "hover": cls.HOVER_COLOR
        }


class FeatureFlags:
    """Feature flags for enabling/disabling specific functionality"""

    # Core Features
    LOGGING_ENABLED = True
    AUTO_SAVE_ENABLED = True
    THEME_SWITCHING = True

    # AI Features
    AI_ASSISTANT_CHAT = True
    CODE_ANALYSIS = True
    SECURITY_SCANNER = True
    SPEECH_RECOGNITION = True

    # Advanced Features
    PLUGIN_SYSTEM = False
    NETWORK_FEATURES = True
    DATABASE_INTEGRATION = False

    # Development Features
    DEBUG_MODE = False
    PERFORMANCE_MONITORING = True
    CRASH_REPORTING = True


# Environment-specific configurations
class DevelopmentConfig(AppConfig):
    """Development environment configuration"""
    DEBUG_MODE = True
    LOG_LEVEL = "DEBUG"
    WINDOW_WIDTH = 1000
    WINDOW_HEIGHT = 700


class ProductionConfig(AppConfig):
    """Production environment configuration"""
    DEBUG_MODE = False
    LOG_LEVEL = "WARNING"
    CRASH_REPORTING = True


# Configuration factory
def get_config() -> Union[DevelopmentConfig, ProductionConfig]:
    """
    Get configuration based on environment.

    Returns:
        Configuration instance based on THINKER_ENV environment variable
    """
    env = os.getenv('THINKER_ENV', 'development').lower()

    if env == 'production':
        return ProductionConfig()
    else:
        return DevelopmentConfig() 
