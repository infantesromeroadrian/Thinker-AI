"""Services module for Thinker AI Auxiliary Window"""

from .qwen_service import QwenService

# Optional speech service - only import if dependencies are available
try:
    from .speech_service import SpeechService, get_speech_service
    SPEECH_AVAILABLE = True
    __all__ = ['QwenService', 'SpeechService', 'get_speech_service']
except ImportError:
    SPEECH_AVAILABLE = False
    __all__ = ['QwenService'] 