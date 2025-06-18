"""
Speech-to-Text Service for Thinker AI Auxiliary Window
Provides voice dictation capabilities with multiple recognition engines
"""

import speech_recognition as sr
import pyaudio
import threading
import time
from typing import Optional, Callable, Dict, Any
from datetime import datetime

from src.utils.logger import get_logger
from src.config.config import get_config


class SpeechService:
    """Professional speech-to-text service with multiple engine support"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger("SpeechService")
        
        # Speech recognition setup
        self.recognizer = sr.Recognizer()
        self.microphone = None
        
        # State management
        self.is_listening = False
        self.is_recording = False
        self.last_recognition_time = None
        
        # Configuration
        self.language = self.config.SPEECH_LANGUAGE
        self.recognition_engine = self.config.SPEECH_ENGINE
        self.energy_threshold = self.config.SPEECH_ENERGY_THRESHOLD
        self.pause_threshold = self.config.SPEECH_PAUSE_THRESHOLD
        
        # Callbacks
        self.on_speech_start = None
        self.on_speech_end = None
        self.on_text_recognized = None
        self.on_error = None
        
        self.logger.info(f"SpeechService initialized with {self.recognition_engine} engine")
        self._initialize_microphone()
    
    def _initialize_microphone(self) -> bool:
        """Initialize microphone with optimal settings"""
        try:
            self.logger.info("🎤 Initializing microphone system...")
            
            # First, list all available microphones
            mic_list = sr.Microphone.list_microphone_names()
            self.logger.info(f"🔍 Found {len(mic_list)} audio devices:")
            for i, mic_name in enumerate(mic_list):
                self.logger.info(f"  [{i}] {mic_name}")
            
            if not mic_list:
                self.logger.error("❌ No audio devices found!")
                return False
            
            # Try to initialize with default device
            try:
                self.microphone = sr.Microphone()
                self.logger.info("🎤 Using default microphone device")
            except Exception as e:
                self.logger.warning(f"Default device failed: {e}, trying device index 0")
                try:
                    self.microphone = sr.Microphone(device_index=0)
                    self.logger.info("🎤 Using first available microphone device")
                except Exception as e2:
                    self.logger.error(f"All microphone initialization failed: {e2}")
                    return False
            
            # Configure recognition settings (more lenient)
            self.recognizer.energy_threshold = self.energy_threshold
            self.recognizer.pause_threshold = self.pause_threshold
            self.recognizer.dynamic_energy_threshold = True
            
            # Quick ambient noise calibration (shorter duration)
            try:
                with self.microphone as source:
                    self.logger.info("🔧 Quick ambient noise calibration...")
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
                    self.logger.info(f"✅ Energy threshold set to: {self.recognizer.energy_threshold}")
            except Exception as e:
                self.logger.warning(f"Ambient noise calibration failed: {e}, using default threshold")
            
            self.logger.info("✅ Microphone initialized successfully")
            return True
                
        except Exception as e:
            self.logger.log_exception(e, "Microphone initialization")
            return False
    
    def is_microphone_available(self) -> bool:
        """Check if microphone is available (simplified check)"""
        try:
            # Simple check: just list available microphones
            mic_list = sr.Microphone.list_microphone_names()
            self.logger.info(f"🎤 Found {len(mic_list)} audio devices")
            
            # Log first few microphones for debugging
            for i, mic_name in enumerate(mic_list[:3]):
                self.logger.info(f"  [{i}] {mic_name}")
            if len(mic_list) > 3:
                self.logger.info(f"  ... and {len(mic_list) - 3} more devices")
            
            # If we found any microphones, assume they'll work
            # (Don't do actual listening test to avoid permissions issues)
            if len(mic_list) > 0:
                self.logger.info("✅ Microphones detected - assuming available")
                return True
            else:
                self.logger.warning("❌ No audio input devices found")
                return False
            
        except Exception as e:
            self.logger.warning(f"Microphone check failed: {str(e)}")
            # Even if the check fails, allow user to try
            self.logger.info("🤷 Check failed but allowing user to try anyway")
            return True
    
    def start_listening(self, 
                       callback: Callable[[str], None],
                       error_callback: Optional[Callable[[Exception], None]] = None,
                       start_callback: Optional[Callable[[], None]] = None,
                       stop_callback: Optional[Callable[[], None]] = None) -> bool:
        """Start continuous speech recognition"""
        if self.is_listening:
            self.logger.warning("Speech recognition already active")
            return False
        
        if not self.microphone:
            self.logger.error("Microphone not initialized")
            return False
        
        # Set callbacks
        self.on_text_recognized = callback
        self.on_error = error_callback
        self.on_speech_start = start_callback
        self.on_speech_end = stop_callback
        
        # Start listening in background
        self.is_listening = True
        listening_thread = threading.Thread(
            target=self._continuous_listening_loop,
            daemon=True
        )
        listening_thread.start()
        
        self.logger.info("🎤 Started continuous speech recognition")
        return True
    
    def stop_listening(self) -> None:
        """Stop continuous speech recognition"""
        self.is_listening = False
        self.is_recording = False
        self.logger.info("⏹️ Stopped speech recognition")
    
    def record_once(self, 
                   callback: Callable[[str], None],
                   timeout: float = 3.0,
                   phrase_time_limit: float = 8.0) -> None:
        """Record a single phrase and recognize it"""
        if self.is_recording:
            self.logger.warning("Already recording")
            return
        
        def record_and_recognize():
            try:
                self.is_recording = True
                self.logger.info("🎙️ Starting single recording...")
                
                if self.on_speech_start:
                    self.on_speech_start()
                
                # Use configured timeout values
                timeout_val = self.config.SPEECH_TIMEOUT
                phrase_limit = self.config.SPEECH_PHRASE_TIME_LIMIT
                
                self.logger.info(f"🎤 Listening... (timeout: {timeout_val}s, phrase limit: {phrase_limit}s)")
                
                # Try to use microphone - create fresh instance to avoid issues
                try:
                    mic = sr.Microphone()
                    with mic as source:
                        # Quick calibration
                        self.recognizer.adjust_for_ambient_noise(source, duration=1)
                        self.logger.debug(f"🔧 Energy threshold: {self.recognizer.energy_threshold}")
                        
                        # Record audio with error handling
                        audio = self.recognizer.listen(
                            source,
                            timeout=timeout_val,
                            phrase_time_limit=phrase_limit
                        )
                        
                    self.logger.info("✅ Audio captured successfully")
                    
                except Exception as mic_error:
                    self.logger.error(f"❌ Microphone error: {mic_error}")
                    # Try alternative approach
                    if self.microphone:
                        self.logger.info("🔄 Trying with initialized microphone...")
                        with self.microphone as source:
                            audio = self.recognizer.listen(
                                source,
                                timeout=timeout_val,
                                phrase_time_limit=phrase_limit
                            )
                    else:
                        raise mic_error
                
                if self.on_speech_end:
                    self.on_speech_end()
                
                # Recognize speech
                self.logger.info("🧠 Processing audio...")
                text = self._recognize_audio(audio)
                
                if text and text.strip():
                    self.last_recognition_time = datetime.now()
                    callback(text.strip())
                    self.logger.info(f"✅ Recognized: '{text[:50]}{'...' if len(text) > 50 else ''}'")
                else:
                    self.logger.warning("⚠️ No speech recognized")
                    callback("")  # Send empty string to indicate no recognition
                
            except sr.WaitTimeoutError:
                self.logger.warning("⏰ Recording timeout - no speech detected")
                callback("")
            except sr.UnknownValueError:
                self.logger.warning("🤔 Could not understand audio")
                callback("")
            except Exception as e:
                self.logger.log_exception(e, "Single recording")
                if self.on_error:
                    self.on_error(e)
                callback("")  # Send empty string on error
            finally:
                self.is_recording = False
                self.logger.debug("🏁 Recording session ended")
        
        # Run in background thread
        record_thread = threading.Thread(target=record_and_recognize, daemon=True)
        record_thread.start()
    
    def _continuous_listening_loop(self) -> None:
        """Continuous listening loop for real-time recognition"""
        while self.is_listening:
            try:
                if not self.microphone:
                    self.logger.error("No microphone available for continuous listening")
                    break
                    
                with self.microphone as source:
                    self.logger.debug("👂 Listening for speech...")
                    
                    # Listen for speech
                    audio = self.recognizer.listen(
                        source,
                        timeout=1.0,
                        phrase_time_limit=8.0
                    )
                
                # Recognize in background to avoid blocking
                recognition_thread = threading.Thread(
                    target=self._process_audio_async,
                    args=(audio,),
                    daemon=True
                )
                recognition_thread.start()
                
            except sr.WaitTimeoutError:
                # Normal timeout, continue listening
                continue
            except Exception as e:
                self.logger.log_exception(e, "Continuous listening")
                if self.on_error:
                    self.on_error(e)
                time.sleep(1)  # Brief pause before retrying
    
    def _process_audio_async(self, audio) -> None:
        """Process audio recognition asynchronously"""
        try:
            if self.on_speech_start:
                self.on_speech_start()
            
            text = self._recognize_audio(audio)
            
            if self.on_speech_end:
                self.on_speech_end()
            
            if text and self.on_text_recognized:
                self.last_recognition_time = datetime.now()
                self.on_text_recognized(text)
                self.logger.info(f"✅ Continuous recognition: '{text[:50]}...'")
            
        except Exception as e:
            if self.on_speech_end:
                self.on_speech_end()
            self.logger.log_exception(e, "Async audio processing")
    
    def _recognize_audio(self, audio) -> Optional[str]:
        """Recognize audio using configured engine"""
        try:
            start_time = time.time()
            
            # Choose recognition engine
            if self.recognition_engine == "google":
                text = getattr(self.recognizer, 'recognize_google')(audio, language=self.language)
            elif self.recognition_engine == "whisper":
                text = getattr(self.recognizer, 'recognize_whisper')(audio, language=self.language)
            elif self.recognition_engine == "azure":
                text = getattr(self.recognizer, 'recognize_azure')(
                    audio, 
                    key=self.config.AZURE_SPEECH_KEY,
                    location=self.config.AZURE_SPEECH_REGION,
                    language=self.language
                )
            else:
                # Fallback to Google
                text = getattr(self.recognizer, 'recognize_google')(audio, language=self.language)
            
            recognition_time = time.time() - start_time
            self.logger.info(f"🧠 Speech recognized in {recognition_time:.2f}s")
            
            return text.strip() if text else None
            
        except sr.UnknownValueError:
            self.logger.debug("Could not understand audio")
            return None
        except sr.RequestError as e:
            self.logger.error(f"Recognition service error: {e}")
            raise
        except Exception as e:
            self.logger.log_exception(e, "Audio recognition")
            raise
    
    def get_microphone_list(self) -> list:
        """Get list of available microphones"""
        try:
            return sr.Microphone.list_microphone_names()
        except Exception as e:
            self.logger.log_exception(e, "Getting microphone list")
            return []
    
    def test_recognition(self) -> Dict[str, Any]:
        """Test speech recognition with a short recording"""
        try:
            self.logger.info("🧪 Starting speech recognition test...")
            
            if not self.is_microphone_available():
                return {
                    "status": "error",
                    "error": "Microphone not available"
                }
            
            test_result = {"status": "testing", "start_time": datetime.now().isoformat()}
            
            if not self.microphone:
                return {
                    "status": "error",
                    "error": "No microphone initialized"
                }
            
            with self.microphone as source:
                self.logger.info("📢 Please say something for testing...")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=3)
            
            text = self._recognize_audio(audio)
            
            if text:
                test_result.update({
                    "status": "success",
                    "recognized_text": text,
                    "engine": self.recognition_engine,
                    "language": self.language
                })
            else:
                test_result.update({
                    "status": "no_speech",
                    "message": "No speech detected during test"
                })
            
            test_result["end_time"] = datetime.now().isoformat()
            return test_result
            
        except Exception as e:
            self.logger.log_exception(e, "Speech recognition test")
            return {
                "status": "error",
                "error": str(e),
                "end_time": datetime.now().isoformat()
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current service status"""
        return {
            "is_listening": self.is_listening,
            "is_recording": self.is_recording,
            "microphone_available": self.is_microphone_available(),
            "recognition_engine": self.recognition_engine,
            "language": self.language,
            "last_recognition": self.last_recognition_time.isoformat() if self.last_recognition_time else None,
            "energy_threshold": self.recognizer.energy_threshold,
            "available_microphones": len(self.get_microphone_list())
        }
    
    def advanced_microphone_diagnosis(self) -> Dict[str, Any]:
        """Advanced microphone diagnosis for troubleshooting"""
        diagnosis = {
            "timestamp": datetime.now().isoformat(),
            "system_info": {},
            "microphone_info": {},
            "pyaudio_info": {},
            "permissions": {},
            "recommendations": []
        }
        
        try:
            import platform
            diagnosis["system_info"] = {
                "platform": platform.system(),
                "version": platform.version(),
                "python_version": platform.python_version()
            }
        except Exception as e:
            diagnosis["system_info"]["error"] = str(e)
        
        # PyAudio and microphone detection
        try:
            import pyaudio
            pa = pyaudio.PyAudio()
            
            diagnosis["pyaudio_info"]["version"] = getattr(pyaudio, '__version__', 'unknown')
            diagnosis["pyaudio_info"]["device_count"] = pa.get_device_count()
            
            # List all audio devices
            devices = []
            for i in range(pa.get_device_count()):
                try:
                    device_info = pa.get_device_info_by_index(i)
                    if int(device_info['maxInputChannels']) > 0:  # Input device
                        devices.append({
                            "index": i,
                            "name": device_info['name'],
                            "max_input_channels": device_info['maxInputChannels'],
                            "default_sample_rate": device_info['defaultSampleRate']
                        })
                except Exception:
                    continue
            
            diagnosis["microphone_info"]["input_devices"] = devices
            diagnosis["microphone_info"]["input_device_count"] = len(devices)
            
            # Try default input device
            try:
                default_input = pa.get_default_input_device_info()
                diagnosis["microphone_info"]["default_device"] = {
                    "name": default_input['name'],
                    "index": default_input['index'],
                    "channels": default_input['maxInputChannels']
                }
            except Exception as e:
                diagnosis["microphone_info"]["default_device_error"] = str(e)
                diagnosis["recommendations"].append(
                    "No default input device found. Check Windows Sound settings."
                )
            
            pa.terminate()
            
        except ImportError:
            diagnosis["pyaudio_info"]["error"] = "PyAudio not installed"
            diagnosis["recommendations"].append("Install PyAudio: pip install pyaudio")
        except Exception as e:
            diagnosis["pyaudio_info"]["error"] = str(e)
        
        # SpeechRecognition microphone detection
        try:
            mic_names = sr.Microphone.list_microphone_names()
            diagnosis["microphone_info"]["sr_microphones"] = mic_names
            diagnosis["microphone_info"]["sr_count"] = len(mic_names)
            
            if len(mic_names) == 0:
                diagnosis["recommendations"].append(
                    "SpeechRecognition can't detect microphones. Check PyAudio installation."
                )
            
        except Exception as e:
            diagnosis["microphone_info"]["sr_error"] = str(e)
        
        # Windows-specific checks
        if platform.system() == "Windows":
            diagnosis["recommendations"].extend([
                "Verify microphone permissions in Windows Privacy Settings",
                "Check that 'Allow apps to access your microphone' is enabled",
                "Ensure Python is in the allowed apps list for microphone access",
                "Try running as administrator if permission issues persist"
            ])
        
        # General recommendations based on findings
        if diagnosis["microphone_info"].get("input_device_count", 0) == 0:
            diagnosis["recommendations"].append(
                "No input devices detected. Check hardware connections and drivers."
            )
        elif diagnosis["microphone_info"].get("sr_count", 0) == 0:
            diagnosis["recommendations"].append(
                "Hardware detected but SpeechRecognition can't access it. Permission issue likely."
            )
        
        return diagnosis
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        self.stop_listening()
        self.microphone = None
        self.logger.info("SpeechService cleaned up")


# Global instance
_speech_service: Optional[SpeechService] = None


def get_speech_service() -> SpeechService:
    """Get or create global speech service instance"""
    global _speech_service
    
    if _speech_service is None:
        _speech_service = SpeechService()
    
    return _speech_service 