"""Services module for Thinker AI Auxiliary Window"""

from .qwen_service import QwenService
from .speech_service import SpeechService, get_speech_service

__all__ = ['QwenService', 'SpeechService', 'get_speech_service'] 