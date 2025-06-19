"""
Test module for configuration functionality.

Tests the configuration classes and factory functions in src.config.config
following the project testing requirements.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from typing import Dict, Any
from pathlib import Path

from src.config.config import (
    AppConfig, 
    DevelopmentConfig, 
    ProductionConfig, 
    FeatureFlags,
    get_config
)


class TestAppConfig:
    """Test suite for AppConfig base class."""
    
    def test_app_config_constants(self) -> None:
        """Test that AppConfig has all required constants."""
        assert AppConfig.APP_NAME == "Thinker AI - Auxiliary Window"
        assert AppConfig.APP_VERSION == "1.0.0"
        assert AppConfig.AUTHOR == "AI Assistant & Human Orchestrator"
        
        # Test window configuration
        assert AppConfig.WINDOW_WIDTH == 800
        assert AppConfig.WINDOW_HEIGHT == 600
        assert AppConfig.MIN_WINDOW_WIDTH == 600
        assert AppConfig.MIN_WINDOW_HEIGHT == 400
        assert AppConfig.WINDOW_RESIZABLE is True
    
    def test_color_scheme(self) -> None:
        """Test color scheme configuration."""
        colors = AppConfig.get_color_scheme()
        
        # Test that color scheme is complete
        required_colors = [
            "primary", "secondary", "background", "text",
            "success", "warning", "error", "accent", 
            "border", "button", "hover"
        ]
        
        for color in required_colors:
            assert color in colors
            assert colors[color].startswith("#")  # Valid hex color
    
    def test_font_configuration(self) -> None:
        """Test font configuration methods."""
        default_font = AppConfig.get_font_config("default")
        header_font = AppConfig.get_font_config("header")
        title_font = AppConfig.get_font_config("title")
        
        assert isinstance(default_font, tuple)
        assert len(default_font) == 2
        assert isinstance(default_font[0], str)  # Font family
        assert isinstance(default_font[1], int)  # Font size
        
        # Test size hierarchy
        assert title_font[1] > header_font[1] > default_font[1]
    
    def test_directory_configuration(self) -> None:
        """Test directory path configuration."""
        assert isinstance(AppConfig.BASE_DIR, Path)
        assert isinstance(AppConfig.SRC_DIR, Path)
        assert isinstance(AppConfig.LOGS_DIR, Path)
        assert isinstance(AppConfig.DATA_DIR, Path)
        
        # Test directory relationships
        assert AppConfig.SRC_DIR.parent == AppConfig.BASE_DIR
        assert AppConfig.LOGS_DIR.parent == AppConfig.BASE_DIR
    
    def test_ensure_directories(self) -> None:
        """Test directory creation functionality."""
        # This should not raise any exceptions
        AppConfig.ensure_directories()
        
        # Verify directories exist
        assert AppConfig.LOGS_DIR.exists()
        assert AppConfig.DATA_DIR.exists()


class TestDevelopmentConfig:
    """Test suite for DevelopmentConfig."""
    
    def test_development_specific_settings(self) -> None:
        """Test development-specific configuration."""
        config = DevelopmentConfig()
        
        assert config.DEBUG_MODE is True
        assert config.LOG_LEVEL == "DEBUG"
        assert config.WINDOW_WIDTH == 1000
        assert config.WINDOW_HEIGHT == 700


class TestProductionConfig:
    """Test suite for ProductionConfig."""
    
    def test_production_specific_settings(self) -> None:
        """Test production-specific configuration."""
        config = ProductionConfig()
        
        assert config.DEBUG_MODE is False
        assert config.LOG_LEVEL == "WARNING"
        assert config.CRASH_REPORTING is True


class TestFeatureFlags:
    """Test suite for FeatureFlags."""
    
    def test_core_features_enabled(self) -> None:
        """Test that core features are enabled by default."""
        assert FeatureFlags.LOGGING_ENABLED is True
        assert FeatureFlags.AUTO_SAVE_ENABLED is True
        assert FeatureFlags.AI_ASSISTANT_CHAT is True
    
    def test_advanced_features_configuration(self) -> None:
        """Test advanced features configuration."""
        assert FeatureFlags.PLUGIN_SYSTEM is False  # Should be disabled by default
        assert FeatureFlags.DATABASE_INTEGRATION is False  # Should be disabled by default


class TestConfigFactory:
    """Test suite for configuration factory function."""
    
    def test_get_config_development(self, monkeypatch) -> None:
        """Test get_config returns DevelopmentConfig for development environment."""
        monkeypatch.setenv("THINKER_ENV", "development")
        config = get_config()
        assert isinstance(config, DevelopmentConfig)
        assert config.DEBUG_MODE is True
    
    def test_get_config_production(self, monkeypatch) -> None:
        """Test get_config returns ProductionConfig for production environment."""
        monkeypatch.setenv("THINKER_ENV", "production")
        config = get_config()
        assert isinstance(config, ProductionConfig)
        assert config.DEBUG_MODE is False
    
    def test_get_config_default(self, monkeypatch) -> None:
        """Test get_config defaults to development when environment not set."""
        monkeypatch.delenv("THINKER_ENV", raising=False)
        config = get_config()
        assert isinstance(config, DevelopmentConfig)
    
    def test_get_config_invalid_environment(self, monkeypatch) -> None:
        """Test get_config defaults to development for invalid environment."""
        monkeypatch.setenv("THINKER_ENV", "invalid_env")
        config = get_config()
        assert isinstance(config, DevelopmentConfig)


@pytest.mark.integration
class TestConfigIntegration:
    """Integration tests for configuration system."""
    
    def test_config_persistence_across_calls(self) -> None:
        """Test that configuration is consistent across multiple calls."""
        config1 = get_config()
        config2 = get_config()
        
        # Should return same type
        assert type(config1) == type(config2)
        assert config1.APP_NAME == config2.APP_NAME
    
    def test_qwen_configuration(self) -> None:
        """Test Qwen2.5-7B model configuration."""
        config = get_config()
        
        # Test Qwen2.5-7B specific settings
        assert "qwen2.5-7b-instruct-1m" in config.QWEN_MODEL_NAME
        assert config.QWEN_TIMEOUT == 30  # Optimized for 7B model
        assert config.QWEN_MAX_RETRIES == 3
        assert 0.0 <= config.QWEN_TEMPERATURE <= 1.0
        assert config.QWEN_MAX_TOKENS > 0
    
    def test_speech_configuration(self) -> None:
        """Test speech recognition configuration."""
        config = get_config()
        
        assert config.SPEECH_ENABLED is True
        assert config.SPEECH_ENGINE in ["google", "whisper", "azure"]
        assert config.SPEECH_LANGUAGE == "es-ES"
        assert config.SPEECH_ENERGY_THRESHOLD > 0
        assert config.SPEECH_TIMEOUT > 0
    
    def test_qwen_configuration_validation(self) -> None:
        """Test Qwen configuration validation."""
        config = get_config()
        
        # Test validation method
        validation = config.validate_qwen_configuration()
        
        assert isinstance(validation, dict)
        assert "valid" in validation
        assert "issues" in validation
        assert "suggestions" in validation
        assert isinstance(validation["valid"], bool)
        assert isinstance(validation["issues"], list)
        assert isinstance(validation["suggestions"], list)
    
    def test_network_alternatives_generation(self) -> None:
        """Test network alternatives generation."""
        config = get_config()
        
        # Test alternatives method
        alternatives = config.get_network_alternatives()
        
        assert isinstance(alternatives, list)
        assert len(alternatives) > 0
        
        # Check that alternatives are valid URLs
        for alt in alternatives:
            assert alt.startswith(('http://', 'https://'))
            assert ':' in alt  # Should have port
    
    def test_centralized_configuration_integration(self) -> None:
        """Test that QwenService uses centralized configuration."""
        from src.services.qwen_service import QwenService
        
        # Create service without parameters (should use config)
        service = QwenService()
        
        config = get_config()
        assert service.base_url == config.QWEN_BASE_URL
        assert service.model_name == config.QWEN_MODEL_NAME
        assert service.timeout == config.QWEN_TIMEOUT
        assert service.max_retries == config.QWEN_MAX_RETRIES
    
    def test_qwen_service_custom_configuration(self) -> None:
        """Test QwenService with custom configuration."""
        from src.services.qwen_service import create_qwen_service
        
        custom_url = "http://custom.example.com:8080"
        custom_model = "custom-model"
        
        service = create_qwen_service(base_url=custom_url, model_name=custom_model)
        
        assert service.base_url == custom_url
        assert service.model_name == custom_model 