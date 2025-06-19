"""Services module - AI and Speech capabilities"""

# Core services are always available
from .qwen_service import QwenService, get_qwen_service

# Optional speech service - only import if dependencies are available
try:
    from .speech_service import SpeechService, get_speech_service
    SPEECH_AVAILABLE = True
    __all__ = ['QwenService', 'SpeechService', 'get_speech_service']
except ImportError:
    SPEECH_AVAILABLE = False
    __all__ = ['QwenService', 'get_qwen_service']

# Optional TTS service - only import if dependencies are available
try:
    from .tts_service import TTSService, get_tts_service
    TTS_AVAILABLE = True
    if 'TTSService' not in __all__:
        __all__.extend(['TTSService', 'get_tts_service'])
except ImportError:
    TTS_AVAILABLE = False 