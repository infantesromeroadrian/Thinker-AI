"""
Text-to-Speech Service for Thinker AI Auxiliary Window
Provides voice synthesis capabilities with multiple TTS engines
"""

import time
import threading
from typing import Optional, Dict, Any, Callable, List
from pathlib import Path

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    pyttsx3 = None

from src.config.config import get_config
from src.utils.logger import get_logger
from src.exceptions import *


class TTSService:
    """Professional text-to-speech service with multiple engine support"""
    
    def __init__(self):
        """Initialize TTS service with configuration"""
        self.config = get_config()
        self.logger = get_logger("TTSService")
        
        # TTS Engine setup
        self.engine = None
        self.is_available = TTS_AVAILABLE
        self.is_speaking = False
        self.speech_queue = []
        self.current_speech_thread = None
        
        # Configuration
        self.voice_id = self.config.TTS_VOICE_ID if hasattr(self.config, 'TTS_VOICE_ID') else 0
        self.speech_rate = self.config.TTS_SPEECH_RATE if hasattr(self.config, 'TTS_SPEECH_RATE') else 200
        self.speech_volume = self.config.TTS_SPEECH_VOLUME if hasattr(self.config, 'TTS_SPEECH_VOLUME') else 0.9
        
        # Callbacks
        self.on_speech_start = None
        self.on_speech_end = None
        self.on_speech_error = None
        
        # Initialize engine if available
        if self.is_available:
            self._initialize_engine()
        
        self.logger.info(f"TTSService initialized - Available: {self.is_available}")
    
    def _initialize_engine(self) -> bool:
        """Initialize the TTS engine"""
        try:
            self.engine = pyttsx3.init()
            
            # Configure engine properties
            self.engine.setProperty('rate', self.speech_rate)
            self.engine.setProperty('volume', self.speech_volume)
            
            # Set voice if specified
            voices = self.engine.getProperty('voices')
            if voices and len(voices) > self.voice_id:
                self.engine.setProperty('voice', voices[self.voice_id].id)
                self.logger.info(f"Using voice: {voices[self.voice_id].name}")
            
            # Set up engine callbacks
            self.engine.connect('started-utterance', self._on_start)
            self.engine.connect('finished-utterance', self._on_end)
            
            self.logger.info("🔊 TTS engine initialized successfully")
            return True
            
        except Exception as e:
            self.logger.log_exception(e, "TTS engine initialization")
            self.is_available = False
            return False
    
    def _on_start(self, name):
        """Called when TTS starts speaking"""
        self.is_speaking = True
        if self.on_speech_start:
            self.on_speech_start()
    
    def _on_end(self, name, completed):
        """Called when TTS finishes speaking"""
        self.is_speaking = False
        if self.on_speech_end:
            self.on_speech_end()
    
    def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get list of available voices"""
        if not self.is_available or not self.engine:
            return []
        
        try:
            voices = self.engine.getProperty('voices')
            voice_list = []
            
            for i, voice in enumerate(voices):
                voice_info = {
                    'id': i,
                    'name': voice.name,
                    'language': 'unknown',  # Default language
                    'gender': getattr(voice, 'gender', 'unknown')
                }
                
                # Safely get language if available
                if hasattr(voice, 'languages') and voice.languages:
                    try:
                        voice_info['language'] = voice.languages[0] if voice.languages else 'unknown'
                    except (IndexError, AttributeError):
                        voice_info['language'] = 'unknown'
                
                voice_list.append(voice_info)
            
            return voice_list
            
        except Exception as e:
            self.logger.log_exception(e, "Getting available voices")
            return []
    
    def set_voice(self, voice_id: int) -> bool:
        """Set the voice to use for TTS"""
        if not self.is_available or not self.engine:
            return False
        
        try:
            voices = self.engine.getProperty('voices')
            if voices and 0 <= voice_id < len(voices):
                self.engine.setProperty('voice', voices[voice_id].id)
                self.voice_id = voice_id
                self.logger.info(f"Voice changed to: {voices[voice_id].name}")
                return True
            else:
                self.logger.warning(f"Invalid voice ID: {voice_id}")
                return False
                
        except Exception as e:
            self.logger.log_exception(e, f"Setting voice {voice_id}")
            return False
    
    def set_speech_rate(self, rate: int) -> bool:
        """Set speech rate (words per minute)"""
        if not self.is_available or not self.engine:
            return False
        
        try:
            # Clamp rate between reasonable bounds
            rate = max(50, min(400, rate))
            self.engine.setProperty('rate', rate)
            self.speech_rate = rate
            self.logger.info(f"Speech rate set to: {rate} WPM")
            return True
            
        except Exception as e:
            self.logger.log_exception(e, f"Setting speech rate {rate}")
            return False
    
    def set_volume(self, volume: float) -> bool:
        """Set speech volume (0.0 to 1.0)"""
        if not self.is_available or not self.engine:
            return False
        
        try:
            # Clamp volume between 0.0 and 1.0
            volume = max(0.0, min(1.0, volume))
            self.engine.setProperty('volume', volume)
            self.speech_volume = volume
            self.logger.info(f"Speech volume set to: {volume}")
            return True
            
        except Exception as e:
            self.logger.log_exception(e, f"Setting volume {volume}")
            return False
    
    def speak_text(self, text: str, interrupt: bool = False) -> bool:
        """
        Speak the given text
        
        Args:
            text: Text to speak
            interrupt: Whether to interrupt current speech
            
        Returns:
            True if speech started successfully
        """
        self.logger.debug(f"🔍 speak_text called: available={self.is_available}, engine={self.engine is not None}")
        
        if not self.is_available or not self.engine:
            self.logger.warning(f"TTS not available: available={self.is_available}, engine={self.engine is not None}")
            return False
        
        if not text or not text.strip():
            self.logger.debug("🔍 speak_text: Empty text provided")
            return False
        
        try:
            self.logger.debug(f"🔍 speak_text: interrupt={interrupt}, is_speaking={self.is_speaking}")
            
            # Stop current speech if interrupting or if already speaking
            if (interrupt or self.is_speaking) and self.is_speaking:
                self.logger.debug("🔍 speak_text: Stopping current speech")
                self.stop_speech()
                # Give it a moment to stop
                time.sleep(0.1)
            
            # If still speaking after interrupt attempt, skip this speech
            if self.is_speaking and not interrupt:
                self.logger.debug("TTS busy, skipping speech request")
                return False
            
            # Clean text for speech
            clean_text = self._clean_text_for_speech(text)
            
            if not clean_text.strip():
                self.logger.debug("🔍 speak_text: Clean text is empty")
                return False
            
            self.logger.info(f"🗣️ Speaking: {clean_text[:50]}{'...' if len(clean_text) > 50 else ''}")
            
            # Speak in separate thread to avoid GUI blocking
            speech_thread = threading.Thread(
                target=self._speak_async,
                args=(clean_text,),
                daemon=True,
                name=f"TTS-{int(time.time() * 1000) % 10000}"  # Unique thread name
            )
            
            self.logger.debug(f"🔍 Starting TTS thread: {speech_thread.name}")
            speech_thread.start()
            self.current_speech_thread = speech_thread
            
            self.logger.debug("🔍 TTS thread started successfully")
            return True
            
        except Exception as e:
            self.logger.log_exception(e, "Speaking text")
            if self.on_speech_error:
                self.on_speech_error(str(e))
            return False
    
    def _speak_async(self, text: str) -> None:
        """Speak text asynchronously with thread safety"""
        self.logger.debug(f"🔍 _speak_async started in thread: {threading.current_thread().name}")
        self.logger.debug(f"🔍 _speak_async: Text length: {len(text)}")
        
        try:
            # Use a simple approach to avoid "run loop already started" error
            # Create a new engine instance for this thread if needed
            if threading.current_thread() != threading.main_thread():
                self.logger.debug("🔍 _speak_async: Running in non-main thread, using iterate approach")
                # For non-main threads, use a simpler approach
                try:
                    self.logger.debug("🔍 _speak_async: Calling engine.say()")
                    self.engine.say(text)
                    self.logger.debug("🔍 _speak_async: Starting iterate loop")
                    # Use iterate() instead of runAndWait() to avoid conflicts
                    while self.engine.isBusy():
                        self.engine.iterate()
                        time.sleep(0.01)  # Small delay to prevent CPU spinning
                    self.logger.debug("🔍 _speak_async: Iterate loop completed")
                except Exception as e:
                    # If that fails, try the standard approach
                    self.logger.debug(f"Iterate approach failed, trying runAndWait: {e}")
                    try:
                        self.logger.debug("🔍 _speak_async: Trying runAndWait as fallback")
                        self.engine.runAndWait()
                        self.logger.debug("🔍 _speak_async: runAndWait completed")
                    except RuntimeError as re:
                        if "run loop already started" in str(re):
                            self.logger.debug("Run loop conflict detected, skipping this speech")
                            return
                        else:
                            raise
            else:
                self.logger.debug("🔍 _speak_async: Running in main thread, using standard approach")
                # For main thread, use standard approach
                self.logger.debug("🔍 _speak_async: Calling engine.say() in main thread")
                self.engine.say(text)
                self.logger.debug("🔍 _speak_async: Calling runAndWait() in main thread")
                self.engine.runAndWait()
                self.logger.debug("🔍 _speak_async: runAndWait completed in main thread")
                
            self.logger.info("🗣️ Speech completed successfully")
                
        except Exception as e:
            self.logger.log_exception(e, "Async speech")
            if self.on_speech_error:
                self.on_speech_error(str(e))
    
    def speak_streaming(self, text_stream: str) -> bool:
        """
        Speak text as it arrives in streaming fashion
        
        Args:
            text_stream: New text chunk to add to speech queue
            
        Returns:
            True if added to speech queue
        """
        if not self.is_available:
            return False
        
        # Add to speech queue
        clean_chunk = self._clean_text_for_speech(text_stream)
        if clean_chunk:
            self.speech_queue.append(clean_chunk)
            
            # Start speaking if not already speaking
            if not self.is_speaking:
                self._process_speech_queue()
        
        return True
    
    def _process_speech_queue(self) -> None:
        """Process queued speech chunks"""
        if not self.speech_queue or self.is_speaking:
            return
        
        # Combine small chunks for better speech flow
        text_to_speak = ""
        chunks_to_remove = 0
        
        for chunk in self.speech_queue:
            text_to_speak += chunk + " "
            chunks_to_remove += 1
            
            # Speak when we have enough text or reach end of queue
            if len(text_to_speak) > 100 or chunks_to_remove == len(self.speech_queue):
                break
        
        # Remove processed chunks
        self.speech_queue = self.speech_queue[chunks_to_remove:]
        
        # Speak the combined text
        if text_to_speak.strip():
            self.speak_text(text_to_speak.strip())
    
    def _clean_text_for_speech(self, text: str) -> str:
        """Clean text for better speech synthesis"""
        if not text:
            return ""
        
        # Remove common markdown and special characters
        clean_text = text.replace("**", "").replace("*", "")
        clean_text = clean_text.replace("```", "").replace("`", "")
        clean_text = clean_text.replace("##", "").replace("#", "")
        clean_text = clean_text.replace("___", "").replace("__", "")
        clean_text = clean_text.replace("✅", "").replace("❌", "")
        clean_text = clean_text.replace("🔧", "").replace("📋", "")
        clean_text = clean_text.replace("🎯", "").replace("💡", "")
        
        # Replace URLs with "enlace"
        import re
        clean_text = re.sub(r'http[s]?://\S+', 'enlace', clean_text)
        clean_text = re.sub(r'www\.\S+', 'enlace', clean_text)
        
        # Clean up extra whitespace
        clean_text = ' '.join(clean_text.split())
        
        return clean_text
    
    def stop_speech(self) -> bool:
        """Stop current speech"""
        if not self.is_available or not self.engine:
            return False
        
        try:
            # Set flag first
            self.is_speaking = False
            
            # Stop engine
            self.engine.stop()
            
            # Clear speech queue
            self.speech_queue.clear()
            
            # Wait for current thread to finish (with timeout)
            if self.current_speech_thread and self.current_speech_thread.is_alive():
                self.current_speech_thread.join(timeout=1.0)
                if self.current_speech_thread.is_alive():
                    self.logger.warning("TTS thread did not stop within timeout")
            
            self.logger.info("🛑 Speech stopped")
            return True
            
        except Exception as e:
            self.logger.log_exception(e, "Stopping speech")
            return False
    
    def is_speech_available(self) -> bool:
        """Check if TTS is available"""
        return self.is_available and self.engine is not None
    
    def test_speech(self) -> Dict[str, Any]:
        """Test TTS functionality"""
        test_text = "Hola, soy el asistente de voz de Thinker AI. El sistema de síntesis de voz está funcionando correctamente."
        
        if not self.is_available:
            return {
                "status": "error",
                "message": "TTS not available - install pyttsx3 package"
            }
        
        try:
            # Test basic speech
            success = self.speak_text(test_text)
            
            if success:
                return {
                    "status": "success",
                    "message": "TTS test completed successfully",
                    "voice_info": {
                        "rate": self.speech_rate,
                        "volume": self.speech_volume,
                        "voice_id": self.voice_id
                    }
                }
            else:
                return {
                    "status": "error",
                    "message": "TTS test failed"
                }
                
        except Exception as e:
            self.logger.log_exception(e, "TTS test")
            return {
                "status": "error",
                "message": f"TTS test error: {str(e)}"
            }
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive TTS diagnostics"""
        diagnosis = {
            "tts_available": self.is_available,
            "engine_initialized": self.engine is not None,
            "currently_speaking": self.is_speaking,
            "speech_queue_length": len(self.speech_queue),
            "configuration": {
                "voice_id": self.voice_id,
                "speech_rate": self.speech_rate,
                "speech_volume": self.speech_volume
            },
            "available_voices": [],
            "pyttsx3_info": {}
        }
        
        try:
            # pyttsx3 information
            if TTS_AVAILABLE:
                import pyttsx3
                diagnosis["pyttsx3_info"]["version"] = getattr(pyttsx3, '__version__', 'unknown')
                diagnosis["available_voices"] = self.get_available_voices()
            
        except Exception as e:
            diagnosis["error"] = str(e)
            self.logger.log_exception(e, "TTS diagnostics")
        
        return diagnosis
    
    def close(self) -> None:
        """Clean up TTS resources"""
        try:
            if self.is_speaking:
                self.stop_speech()
            
            if self.current_speech_thread and self.current_speech_thread.is_alive():
                self.current_speech_thread.join(timeout=2.0)
            
            if self.engine:
                # pyttsx3 doesn't have explicit close method
                self.engine = None
            
            self.logger.info("🔊 TTS service closed")
            
        except Exception as e:
            self.logger.log_exception(e, "TTS cleanup")


# Global TTS service instance
_tts_service = None

def get_tts_service() -> Optional[TTSService]:
    """Get global TTS service instance"""
    global _tts_service
    
    if _tts_service is None:
        _tts_service = TTSService()
    
    return _tts_service if _tts_service.is_available else None 