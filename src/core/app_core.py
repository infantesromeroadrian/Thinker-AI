"""
Thinker AI - Core Application Module

This module provides the core functionality for the Thinker AI application,
including AI services, security tools, and system orchestration.
"""

import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import uuid

from src.config.config import get_config, FeatureFlags
from src.utils.logger import get_logger
from src.utils.helpers import Performance, FileManager, ThreadingHelpers
from src.exceptions import *


class ThinkerCore:
    """Core application class managing business logic and state"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger("ThinkerCore")
        self.session_id = self._generate_session_id()
        self.is_running = False
        self.startup_time = datetime.now()
        
        # Application state
        self.state = {
            "session_id": self.session_id,
            "startup_time": self.startup_time.isoformat(),
            "current_user": None,
            "active_modules": [],
            "performance_metrics": {},
            "security_events": []
        }
        
        # Module registry
        self.modules = {}
        self.ai_modules = {}
        self.security_modules = {}
        
        # Event system
        self.event_handlers = {}
        
        self.logger.info(f"ThinkerCore initialized with session {self.session_id}")
        self._initialize_core_modules()
    
    def _generate_session_id(self) -> str:
        """Generate unique session identifier"""
        return str(uuid.uuid4())[:8]
    
    @Performance.time_function
    def _initialize_core_modules(self) -> None:
        """Initialize core application modules"""
        try:
            # Ensure required directories exist
            self.config.ensure_directories()
            
            # Initialize AI modules if enabled
            if FeatureFlags.AI_ASSISTANT_CHAT:
                self._initialize_ai_modules()
            
            # Initialize security modules if enabled
            if FeatureFlags.SECURITY_SCANNER:
                self._initialize_security_modules()
            
            # Initialize performance monitoring
            if FeatureFlags.PERFORMANCE_MONITORING:
                self._initialize_performance_monitoring()
            
            self.logger.info("Core modules initialized successfully")
            
        except Exception as e:
            self.logger.log_exception(e, "Core module initialization")
            raise
    
    def _initialize_ai_modules(self) -> None:
        """Initialize AI/ML related modules"""
        ai_modules = {
            "text_processor": TextProcessor(),
            "code_analyzer": CodeAnalyzer(),
            "assistant_chat": AssistantChat()
        }
        
        # Add speech recognition if enabled
        if FeatureFlags.SPEECH_RECOGNITION:
            ai_modules["speech_service"] = SpeechModule()
        
        for name, module in ai_modules.items():
            try:
                module.initialize()
                self.ai_modules[name] = module
                self.state["active_modules"].append(f"ai.{name}")
                self.logger.debug(f"AI module initialized: {name}")
                
            except Exception as e:
                self.logger.log_exception(e, f"AI module initialization: {name}")
    
    def _initialize_security_modules(self) -> None:
        """Initialize cybersecurity and ethical hacking modules"""
        security_modules = {
            "network_scanner": NetworkScanner(),
            "vulnerability_checker": VulnerabilityChecker(),
            "security_logger": SecurityLogger()
        }
        
        for name, module in security_modules.items():
            try:
                module.initialize()
                self.security_modules[name] = module
                self.state["active_modules"].append(f"security.{name}")
                self.logger.debug(f"Security module initialized: {name}")
                
            except Exception as e:
                self.logger.log_exception(e, f"Security module initialization: {name}")
    
    def _initialize_performance_monitoring(self) -> None:
        """Initialize performance monitoring system"""
        self.state["performance_metrics"] = {
            "startup_time": 0,
            "memory_usage": 0,
            "cpu_usage": 0,
            "active_threads": 0,
            "last_updated": datetime.now().isoformat()
        }
        
        # Start background performance monitoring
        if FeatureFlags.PERFORMANCE_MONITORING:
            ThreadingHelpers.run_in_background(
                self._monitor_performance_loop,
                error_callback=self._handle_monitoring_error
            )
    
    def _monitor_performance_loop(self) -> None:
        """Background loop for performance monitoring"""
        import psutil
        
        while self.is_running:
            try:
                # Update performance metrics
                process = psutil.Process()
                self.state["performance_metrics"].update({
                    "memory_usage": process.memory_info().rss / 1024 / 1024,  # MB
                    "cpu_usage": process.cpu_percent(),
                    "active_threads": threading.active_count(),
                    "last_updated": datetime.now().isoformat()
                })
                
                # Sleep for monitoring interval
                time.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                self.logger.log_exception(e, "Performance monitoring")
                time.sleep(60)  # Wait longer on error
    
    def _handle_monitoring_error(self, error: Exception) -> None:
        """Handle performance monitoring errors"""
        self.logger.log_exception(error, "Performance monitoring error")
    
    def start(self) -> bool:
        """Start the core application"""
        try:
            self.is_running = True
            startup_start = time.time()
            
            self.logger.info("Starting Thinker AI Core...")
            
            # Load saved state if exists
            self._load_application_state()
            
            # Start background services
            self._start_background_services()
            
            # Calculate startup time
            startup_duration = time.time() - startup_start
            self.state["performance_metrics"]["startup_time"] = startup_duration
            
            self.logger.info(f"Thinker AI Core started successfully in {startup_duration:.2f}s")
            return True
            
        except Exception as e:
            self.logger.log_exception(e, "Core startup")
            self.is_running = False
            return False
    
    def stop(self) -> bool:
        """Stop the core application"""
        try:
            self.logger.info("Stopping Thinker AI Core...")
            
            self.is_running = False
            
            # Save application state
            self._save_application_state()
            
            # Stop modules
            self._stop_modules()
            
            self.logger.info("Thinker AI Core stopped successfully")
            return True
            
        except Exception as e:
            self.logger.log_exception(e, "Core shutdown")
            return False
    
    def _load_application_state(self) -> None:
        """Load saved application state"""
        state_file = self.config.DATA_DIR / "app_state.json"
        saved_state = FileManager.safe_read_json(state_file)
        
        if saved_state:
            # Merge saved state with current state
            for key, value in saved_state.items():
                if key not in ["session_id", "startup_time"]:
                    self.state[key] = value
            
            self.logger.debug("Application state loaded")
    
    def _save_application_state(self) -> None:
        """Save current application state"""
        state_file = self.config.DATA_DIR / "app_state.json"
        
        # Prepare state for saving (remove sensitive data)
        safe_state = {
            key: value for key, value in self.state.items()
            if key not in ["session_id"]
        }
        
        FileManager.safe_write_json(state_file, safe_state)
        self.logger.debug("Application state saved")
    
    def _start_background_services(self) -> None:
        """Start background services"""
        # Auto-save service
        if FeatureFlags.AUTO_SAVE_ENABLED:
            ThreadingHelpers.delayed_execution(300, self._auto_save_loop)  # Save every 5 minutes
    
    def _auto_save_loop(self) -> None:
        """Auto-save loop for periodic state saving"""
        while self.is_running:
            try:
                self._save_application_state()
                time.sleep(300)  # Save every 5 minutes
            except Exception as e:
                self.logger.log_exception(e, "Auto-save")
                time.sleep(600)  # Wait longer on error
    
    def _stop_modules(self) -> None:
        """Stop all modules gracefully"""
        # Stop AI modules
        for name, module in self.ai_modules.items():
            try:
                module.shutdown()
                self.logger.debug(f"AI module stopped: {name}")
            except Exception as e:
                self.logger.log_exception(e, f"Stopping AI module: {name}")
        
        # Stop security modules
        for name, module in self.security_modules.items():
            try:
                module.shutdown()
                self.logger.debug(f"Security module stopped: {name}")
            except Exception as e:
                self.logger.log_exception(e, f"Stopping security module: {name}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "core_status": "running" if self.is_running else "stopped",
            "session_id": self.session_id,
            "uptime": str(datetime.now() - self.startup_time),
            "active_modules": self.state["active_modules"],
            "performance": self.state["performance_metrics"],
            "ai_modules": list(self.ai_modules.keys()),
            "security_modules": list(self.security_modules.keys()),
            "feature_flags": {
                name: getattr(FeatureFlags, name)
                for name in dir(FeatureFlags)
                if not name.startswith('_')
            }
        }
    
    def run_comprehensive_diagnostics(self) -> Dict[str, Any]:
        """Run comprehensive system diagnostics"""
        diagnostics = {
            "timestamp": datetime.now().isoformat(),
            "system_status": self.get_system_status(),
            "configuration_validation": {},
            "connectivity_tests": {},
            "module_health": {},
            "recommendations": []
        }
        
        try:
            # Configuration validation
            config_validation = get_config().validate_qwen_configuration()
            diagnostics["configuration_validation"] = config_validation
            
            if not config_validation["valid"]:
                diagnostics["recommendations"].extend([
                    f"🔧 Configuración: {issue}" for issue in config_validation["issues"]
                ])
            
            # Connectivity tests
            if "assistant_chat" in self.ai_modules:
                try:
                    assistant_chat = self.ai_modules["assistant_chat"]
                    
                    if hasattr(assistant_chat, 'qwen_service') and assistant_chat.qwen_service is not None:
                        qwen_service = assistant_chat.qwen_service
                        connectivity = qwen_service.get_connection_status()
                        diagnostics["connectivity_tests"]["qwen"] = connectivity
                        
                        if connectivity["status"] != "online":
                            diagnostics["recommendations"].append("🌐 Conectividad: Servidor Qwen no disponible")
                            
                            # Suggest alternatives
                            alternatives = get_config().get_network_alternatives()
                            if alternatives:
                                diagnostics["recommendations"].append(f"💡 Probar URLs alternativas: {alternatives[:3]}")
                    else:
                        diagnostics["connectivity_tests"]["qwen"] = {
                            "status": "unavailable", 
                            "error": "QwenService not initialized"
                        }
                        diagnostics["recommendations"].append("🔧 Servicio Qwen no está inicializado - verifique dependencias")
                            
                except Exception as e:
                    diagnostics["connectivity_tests"]["qwen"] = {"error": str(e)}
                    diagnostics["recommendations"].append(f"❌ Error de conectividad: {str(e)}")
            
            # Module health checks
            for module_name, module in self.ai_modules.items():
                try:
                    # Check if module has required attributes
                    health = {
                        "status": "healthy",
                        "has_logger": hasattr(module, 'logger'),
                        "initialized": True
                    }
                    
                    # Specific checks for assistant_chat
                    if module_name == "assistant_chat":
                        health["has_qwen_service"] = hasattr(module, 'qwen_service') and module.qwen_service is not None
                        if hasattr(module, 'qwen_service') and module.qwen_service is not None:
                            health["qwen_online"] = module.qwen_service.is_online
                        else:
                            health["qwen_online"] = False
                            health["qwen_service_status"] = "not_initialized"
                    
                    diagnostics["module_health"][module_name] = health
                    
                except Exception as e:
                    diagnostics["module_health"][module_name] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }
                    diagnostics["recommendations"].append(f"🔧 Módulo {module_name}: {str(e)}")
            
            # Generate summary recommendations
            if not diagnostics["recommendations"]:
                diagnostics["recommendations"].append("✅ Sistema funcionando correctamente")
            
        except Exception as e:
            diagnostics["error"] = str(e)
            diagnostics["recommendations"].append(f"❌ Error en diagnóstico: {str(e)}")
        
        return diagnostics
    
    def execute_ai_operation(self, operation: str, parameters: Dict[str, Any]) -> Any:
        """Execute AI operation with specified parameters"""
        try:
            self.logger.log_ai_operation(operation, parameters.get("model", ""), 
                                       parameters.get("tokens", 0))
            
            # Route to appropriate AI module
            if operation in ["analyze_text", "process_text"]:
                return self.ai_modules["text_processor"].process(parameters)
            elif operation in ["analyze_code", "review_code"]:
                return self.ai_modules["code_analyzer"].analyze(parameters)
            elif operation in ["chat", "assistant"]:
                return self.ai_modules["assistant_chat"].chat(parameters)
            else:
                raise ValueError(f"Unknown AI operation: {operation}")
                
        except Exception as e:
            self.logger.log_exception(e, f"AI operation: {operation}")
            raise
    
    def execute_security_operation(self, operation: str, parameters: Dict[str, Any]) -> Any:
        """Execute security operation with specified parameters"""
        try:
            self.logger.log_security_event(operation, "info", f"Parameters: {parameters}")
            
            # Route to appropriate security module
            if operation in ["scan_network", "network_scan"]:
                return self.security_modules["network_scanner"].scan(parameters)
            elif operation in ["check_vulnerabilities", "vuln_scan"]:
                return self.security_modules["vulnerability_checker"].check(parameters)
            else:
                raise ValueError(f"Unknown security operation: {operation}")
                
        except Exception as e:
            self.logger.log_exception(e, f"Security operation: {operation}")
            raise
    
    def execute_speech_operation(self, operation: str, parameters: Dict[str, Any]) -> Any:
        """Execute speech recognition operation with specified parameters"""
        try:
            self.logger.info(f"Speech operation: {operation}")
            
            # Check if speech module is available
            if "speech_service" not in self.ai_modules:
                return {
                    "status": "error",
                    "result": "Speech recognition not available. Install speech_recognition and pyaudio packages."
                }
            
            speech_module = self.ai_modules["speech_service"]
            if not hasattr(speech_module, 'speech_service') or not speech_module.speech_service:
                return {
                    "status": "error", 
                    "result": "Speech service not initialized properly."
                }
            
            speech_service = speech_module.speech_service
            
            # Route to appropriate speech operation
            if operation == "start_listening":
                callback = parameters.get("callback")
                return speech_service.start_listening(
                    callback=callback,
                    error_callback=parameters.get("error_callback"),
                    start_callback=parameters.get("start_callback"),
                    stop_callback=parameters.get("stop_callback")
                )
            elif operation == "stop_listening":
                speech_service.stop_listening()
                return {"status": "success", "result": "Stopped listening"}
            elif operation == "record_once":
                callback = parameters.get("callback")
                speech_service.record_once(
                    callback=callback,
                    timeout=parameters.get("timeout", 5.0),
                    phrase_time_limit=parameters.get("phrase_time_limit", 10.0)
                )
                return {"status": "success", "result": "Recording started"}
            elif operation == "test_recognition":
                return speech_service.test_recognition()
            elif operation == "get_status":
                return speech_service.get_status()
            elif operation == "get_microphones":
                return {
                    "status": "success",
                    "microphones": speech_service.get_microphone_list()
                }
            elif operation == "advanced_diagnosis":
                diagnosis = speech_service.advanced_microphone_diagnosis()
                return {
                    "status": "success",
                    "diagnosis": diagnosis
                }
            else:
                raise ValueError(f"Unknown speech operation: {operation}")
                
        except Exception as e:
            self.logger.log_exception(e, f"Speech operation: {operation}")
            return {
                "status": "error",
                "result": f"Speech operation failed: {str(e)}"
            }
    
    def execute_tts_operation(self, operation: str, parameters: Dict[str, Any] = None) -> Any:
        """Execute Text-to-Speech operations"""
        try:
            parameters = parameters or {}
            self.logger.info(f"TTS operation: {operation}")
            
            # Check if TTS is available
            if not TTS_AVAILABLE:
                return {
                    "status": "error",
                    "result": "TTS not available. Install pyttsx3 package."
                }
            
            # Get AssistantChat instance for TTS operations
            if "assistant_chat" not in self.ai_modules:
                return {
                    "status": "error",
                    "result": "Assistant chat not available for TTS operations."
                }
            
            assistant_chat = self.ai_modules["assistant_chat"]
            
            if operation == "speak_text":
                text = parameters.get("text", "")
                interrupt = parameters.get("interrupt", False)
                
                if not text:
                    return {"status": "error", "result": "No text provided to speak"}
                
                success = assistant_chat.speak_response(text, interrupt_current=interrupt)
                return {
                    "status": "success" if success else "error",
                    "result": "Speaking text..." if success else "Failed to start TTS"
                }
                
            elif operation == "stop_speech":
                success = assistant_chat.stop_speech()
                return {
                    "status": "success" if success else "error",
                    "result": "Speech stopped" if success else "Failed to stop speech"
                }
                
            elif operation == "test_tts":
                tts_service = get_tts_service()
                if not tts_service:
                    return {"status": "error", "result": "TTS service not available"}
                
                return tts_service.test_speech()
                
            elif operation == "get_voices":
                voices = assistant_chat.get_available_voices()
                return {
                    "status": "success",
                    "voices": voices,
                    "result": f"Found {len(voices)} available voices"
                }
                
            elif operation == "set_voice":
                voice_id = parameters.get("voice_id")
                if voice_id is None:
                    return {"status": "error", "result": "No voice_id provided"}
                
                success = assistant_chat.set_voice_settings(voice_id=voice_id)
                return {
                    "status": "success" if success else "error",
                    "result": f"Voice changed to ID {voice_id}" if success else "Failed to change voice"
                }
                
            elif operation == "set_rate":
                rate = parameters.get("rate")
                if rate is None:
                    return {"status": "error", "result": "No rate provided"}
                
                success = assistant_chat.set_voice_settings(rate=rate)
                return {
                    "status": "success" if success else "error",
                    "result": f"Speech rate set to {rate} WPM" if success else "Failed to set speech rate"
                }
                
            elif operation == "set_volume":
                volume = parameters.get("volume")
                if volume is None:
                    return {"status": "error", "result": "No volume provided"}
                
                success = assistant_chat.set_voice_settings(volume=volume)
                return {
                    "status": "success" if success else "error",
                    "result": f"Volume set to {volume}" if success else "Failed to set volume"
                }
                
            elif operation == "get_tts_diagnostics":
                tts_service = get_tts_service()
                if not tts_service:
                    return {"status": "error", "result": "TTS service not available"}
                
                diagnostics = tts_service.get_diagnostics()
                return {
                    "status": "success",
                    "diagnostics": diagnostics,
                    "result": "TTS diagnostics retrieved"
                }
                
            elif operation == "toggle_auto_speak":
                # Toggle auto-speak setting using local state
                assistant_chat = self.ai_modules["assistant_chat"]
                old_state = assistant_chat.auto_speak_enabled
                new_state = not assistant_chat.auto_speak_enabled
                assistant_chat.auto_speak_enabled = new_state
                
                self.logger.info(f"🔊 TTS Auto-speak toggled: {old_state} → {new_state}")
                
                return {
                    "status": "success",
                    "result": f"Auto-speak {'enabled' if new_state else 'disabled'}",
                    "auto_speak_enabled": new_state
                }
                
            elif operation == "get_auto_speak_status":
                assistant_chat = self.ai_modules["assistant_chat"]
                return {
                    "status": "success",
                    "auto_speak_enabled": assistant_chat.auto_speak_enabled,
                    "result": f"Auto-speak is {'enabled' if assistant_chat.auto_speak_enabled else 'disabled'}"
                }
                
            else:
                raise ValueError(f"Unknown TTS operation: {operation}")
                
        except Exception as e:
            self.logger.log_exception(e, f"TTS operation: {operation}")
            return {
                "status": "error",
                "result": f"TTS operation failed: {str(e)}"
            }
    
    def execute_conversation_operation(self, operation: str, parameters: Dict[str, Any] = None) -> Any:
        """Execute conversation management operations"""
        try:
            parameters = parameters or {}
            
            # Check if assistant chat is available
            if "assistant_chat" not in self.ai_modules:
                return {
                    "status": "error",
                    "result": "Assistant chat not available."
                }
            
            assistant_chat = self.ai_modules["assistant_chat"]
            
            if operation == "clear_conversation":
                assistant_chat.clear_conversation()
                return {
                    "status": "success",
                    "result": "🧹 Conversación limpiada. Nueva conversación iniciada."
                }
            elif operation == "get_conversation_summary":
                summary = assistant_chat.get_conversation_summary()
                return {
                    "status": "success",
                    "result": summary
                }
            elif operation == "get_conversation_history":
                return {
                    "status": "success",
                    "result": {
                        "history": assistant_chat.conversation_history,
                        "conversation_id": assistant_chat.conversation_id,
                        "message_count": len(assistant_chat.conversation_history)
                    }
                }
            elif operation == "save_conversation":
                assistant_chat._save_conversation_to_state()
                return {
                    "status": "success",
                    "result": "💾 Conversación guardada exitosamente."
                }
            else:
                raise ValueError(f"Unknown conversation operation: {operation}")
                
        except Exception as e:
            self.logger.log_exception(e, f"Conversation operation: {operation}")
            return {
                "status": "error",
                "result": f"Conversation operation failed: {str(e)}"
            }

    def get_auto_speak_enabled(self) -> bool:
        """Get current auto-speak state"""
        return self.auto_speak_enabled
    
    def set_auto_speak_enabled(self, enabled: bool):
        """Set auto-speak state"""
        self.auto_speak_enabled = enabled
        self.logger.info(f"🔊 Auto-speak state set to: {enabled}")

    def get_available_voices(self) -> list:
        """Get list of available TTS voices"""
        if not self.tts_service:
            return []
        
        try:
            return self.tts_service.get_available_voices()
        except Exception as e:
            self.logger.error(f"Error getting available voices: {e}")
            return []


# Import Qwen service
try:
    from src.services import QwenService, get_qwen_service
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False
    get_qwen_service = None

# Optional speech service for voice dictation
try:
    from src.services import SpeechService, get_speech_service
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False
    get_speech_service = None

# Optional TTS service for voice synthesis
try:
    from src.services import TTSService, get_tts_service
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    get_tts_service = None


# Real AI modules implementation
class TextProcessor:
    """AI text processing module"""
    def initialize(self): 
        self.logger = get_logger("TextProcessor")
        self.logger.info("TextProcessor initialized")
    
    def shutdown(self): 
        self.logger.info("TextProcessor shutdown")
    
    def process(self, parameters): 
        if QWEN_AVAILABLE:
            qwen = get_qwen_service()
            text = parameters.get("text", "")
            prompt = f"Analiza el siguiente texto y proporciona insights útiles:\n\n{text}"
            
            return qwen.chat(prompt, system_prompt="Eres un experto en análisis de texto.")
        else:
            return {"status": "error", "result": "Qwen service not available"}


class CodeAnalyzer:
    """AI code analysis module"""
    def initialize(self): 
        self.logger = get_logger("CodeAnalyzer")
        self.logger.info("CodeAnalyzer initialized")
    
    def shutdown(self): 
        self.logger.info("CodeAnalyzer shutdown")
    
    def analyze(self, parameters): 
        if QWEN_AVAILABLE:
            qwen = get_qwen_service()
            code = parameters.get("code", "")
            language = parameters.get("language", "python")
            
            prompt = f"Analiza el siguiente código {language} y proporciona un review detallado:\n\n```{language}\n{code}\n```"
            
            return qwen.chat(prompt, system_prompt="Eres un experto en análisis de código y arquitectura de software.")
        else:
            return {"status": "error", "result": "Qwen service not available"}


class AssistantChat:
    """AI assistant chat module using Qwen2.5-7B-Instruct-1M with conversation memory"""
    def __init__(self):
        """Initialize with conversation memory"""
        self.conversation_history = []  # Lista de mensajes para mantener contexto
        self.max_history_length = 20  # Máximo de mensajes a recordar
        self.conversation_start_time = None
        self.total_tokens_used = 0
        self.conversation_id = None
        
        # TTS state - local to this instance
        self.auto_speak_enabled = False  # Local auto-speak state
    
    def initialize(self): 
        self.logger = get_logger("AssistantChat")
        
        # Always initialize qwen_service, even if offline
        self.qwen_service = None
        self.tts_service = None
        
        if QWEN_AVAILABLE:
            try:
                self.qwen_service = get_qwen_service()
                self._start_new_conversation()
                self.logger.info(f"AssistantChat initialized with {self.qwen_service.model_name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize Qwen service: {str(e)}")
                self.qwen_service = None
        else:
            self.logger.error("AssistantChat failed to initialize - Qwen service unavailable")
        
        # Initialize TTS service if available
        if TTS_AVAILABLE:
            try:
                self.tts_service = get_tts_service()
                if self.tts_service:
                    self.logger.info("🔊 TTS service initialized for voice responses")
                    
                    # Verification tests
                    self.logger.info(f"🔍 TTS Verification - Available: {self.tts_service.is_speech_available()}")
                    self.logger.info(f"🔍 TTS Verification - Engine: {self.tts_service.engine is not None}")
                    
                    if hasattr(self.tts_service, 'get_available_voices'):
                        voices = self.tts_service.get_available_voices()
                        self.logger.info(f"🔍 TTS Verification - Voices count: {len(voices)}")
                        if voices:
                            self.logger.info(f"🔍 TTS Verification - Current voice: {voices[0] if voices else 'None'}")
                    
                    # Test configuration access
                    config = get_config()
                    self.logger.info(f"🔍 TTS Verification - Config TTS_ENABLED: {config.TTS_ENABLED}")
                    self.logger.info(f"🔍 TTS Verification - Config TTS_AUTO_SPEAK: {config.TTS_AUTO_SPEAK}")
                else:
                    self.logger.warning("⚠️ TTS service not available")
            except Exception as e:
                self.logger.error(f"Failed to initialize TTS service: {str(e)}")
                import traceback
                self.logger.error(f"TTS init traceback: {traceback.format_exc()}")
                self.tts_service = None
    
    def shutdown(self): 
        """Shutdown and save conversation"""
        try:
            if hasattr(self, 'conversation_history') and self.conversation_history:
                self._save_conversation_to_state()
            if hasattr(self, 'qwen_service') and self.qwen_service:
                self.qwen_service.close()
            if hasattr(self, 'tts_service') and self.tts_service:
                self.tts_service.close()
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
        finally:
            self.logger.info("AssistantChat shutdown")
    
    def _start_new_conversation(self):
        """Iniciar nueva conversación con ID único"""
        import uuid
        from datetime import datetime
        
        self.conversation_id = str(uuid.uuid4())[:8]
        self.conversation_start_time = datetime.now()
        self.conversation_history = []
        self.total_tokens_used = 0
        
        self.logger.info(f"🆕 Nueva conversación iniciada: {self.conversation_id}")
    
    def _add_to_history(self, role: str, content: str, metadata: dict = None):
        """Agregar mensaje al historial de conversación"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "conversation_id": self.conversation_id
        }
        
        if metadata:
            message["metadata"] = metadata
        
        self.conversation_history.append(message)
        
        # Mantener solo los últimos N mensajes para evitar overflow de contexto
        if len(self.conversation_history) > self.max_history_length:
            # Conservar el primer mensaje (system prompt) si existe
            if self.conversation_history[0].get("role") == "system":
                self.conversation_history = [self.conversation_history[0]] + self.conversation_history[-(self.max_history_length-1):]
            else:
                self.conversation_history = self.conversation_history[-self.max_history_length:]
        
        self.logger.debug(f"💬 Historial actualizado: {len(self.conversation_history)} mensajes")
    
    def _prepare_messages_for_qwen(self, new_message: str, system_prompt: str = None) -> list:
        """Preparar mensajes completos con historial para Qwen"""
        messages = []
        
        # Agregar system prompt si es diferente al último o es el primer mensaje
        if system_prompt and (
            not self.conversation_history or 
            not any(msg.get("role") == "system" for msg in self.conversation_history[-5:])
        ):
            messages.append({
                "role": "system", 
                "content": system_prompt
            })
        
        # Agregar historial existente (solo user y assistant)
        for msg in self.conversation_history:
            if msg["role"] in ["user", "assistant"]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Agregar nuevo mensaje del usuario
        messages.append({
            "role": "user",
            "content": new_message
        })
        
        return messages
    
    def clear_conversation(self):
        """Limpiar historial de conversación y empezar de nuevo"""
        old_id = self.conversation_id
        self._start_new_conversation()
        self.logger.info(f"🧹 Conversación {old_id} limpiada, nueva conversación: {self.conversation_id}")
    
    def get_conversation_summary(self) -> dict:
        """Obtener resumen de la conversación actual"""
        if not self.conversation_history:
            return {"status": "empty", "message": "No hay conversación activa"}
        
        user_messages = [msg for msg in self.conversation_history if msg["role"] == "user"]
        assistant_messages = [msg for msg in self.conversation_history if msg["role"] == "assistant"]
        
        return {
            "conversation_id": self.conversation_id,
            "start_time": self.conversation_start_time.isoformat() if self.conversation_start_time else None,
            "message_count": len(self.conversation_history),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "total_tokens_used": self.total_tokens_used,
            "duration_minutes": (datetime.now() - self.conversation_start_time).total_seconds() / 60 if self.conversation_start_time else 0,
            "last_message_time": self.conversation_history[-1]["timestamp"] if self.conversation_history else None
        }
    
    def _save_conversation_to_state(self):
        """Guardar conversación en el estado de aplicación"""
        try:
            from src.utils.helpers import FileManager
            config = get_config()
            
            conversation_file = config.DATA_DIR / f"conversation_{self.conversation_id}.json"
            conversation_data = {
                "conversation_id": self.conversation_id,
                "start_time": self.conversation_start_time.isoformat() if self.conversation_start_time else None,
                "end_time": datetime.now().isoformat(),
                "history": self.conversation_history,
                "summary": self.get_conversation_summary()
            }
            
            FileManager.safe_write_json(conversation_file, conversation_data)
            self.logger.debug(f"💾 Conversación guardada: {conversation_file}")
            
        except Exception as e:
            self.logger.error(f"Error guardando conversación: {e}")
    
    def speak_response(self, text: str, interrupt_current: bool = False) -> bool:
        """
        Speak AI response using TTS
        
        Args:
            text: Text to speak
            interrupt_current: Whether to interrupt current speech
        
        Returns:
            True if TTS started successfully
        """
        self.logger.debug(f"🔍 speak_response called with {len(text)} chars, interrupt: {interrupt_current}")
        
        if not self.tts_service:
            self.logger.debug("🔍 speak_response: No TTS service")
            return False
            
        if not self.tts_service.is_speech_available():
            self.logger.debug("🔍 speak_response: TTS not available")
            return False
        
        config = get_config()
        if not config.TTS_ENABLED:
            self.logger.debug("🔍 speak_response: TTS disabled in config")
            return False
        
        try:
            self.logger.info(f"🗣️ Calling TTS speak_text with: '{text[:100]}{'...' if len(text) > 100 else ''}'")
            result = self.tts_service.speak_text(text, interrupt=interrupt_current)
            self.logger.info(f"🗣️ TTS speak_text returned: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Error speaking response: {e}")
            import traceback
            self.logger.error(f"speak_response traceback: {traceback.format_exc()}")
            return False
    
    def speak_streaming_chunk(self, chunk: str) -> bool:
        """
        Add streaming text chunk to TTS queue - DISABLED to prevent delay issues
        
        Args:
            chunk: Text chunk from streaming response
        
        Returns:
            False - Streaming TTS disabled to prevent sync issues
        """
        # DISABLED: Streaming TTS causes delay/sync issues with pyttsx3
        # The TTS will speak the complete response after streaming finishes
        return False
    
    def stop_speech(self) -> bool:
        """Stop current TTS speech"""
        if not self.tts_service:
            return False
        
        try:
            return self.tts_service.stop_speech()
        except Exception as e:
            self.logger.error(f"Error stopping speech: {e}")
            return False
    
    def set_voice_settings(self, voice_id: int = None, rate: int = None, volume: float = None) -> bool:
        """
        Update TTS voice settings
        
        Args:
            voice_id: Voice index to use
            rate: Speech rate (words per minute)
            volume: Volume level (0.0-1.0)
        
        Returns:
            True if settings updated successfully
        """
        if not self.tts_service:
            return False
        
        try:
            success = True
            if voice_id is not None:
                success &= self.tts_service.set_voice(voice_id)
            if rate is not None:
                success &= self.tts_service.set_speech_rate(rate)
            if volume is not None:
                success &= self.tts_service.set_volume(volume)
            
            return success
        except Exception as e:
            self.logger.error(f"Error updating voice settings: {e}")
            return False
    
    def chat(self, parameters):
        if not QWEN_AVAILABLE or self.qwen_service is None:
            return {
                "status": "error", 
                "result": f"❌ Servicio {get_config().QWEN_MODEL_NAME} no disponible. Verifica que el servidor esté ejecutándose en {get_config().QWEN_BASE_URL}",
                "fallback_message": "El sistema de chat AI está temporalmente no disponible. Por favor, verifica la conexión con el servidor local."
            }
        
        message = parameters.get("message", "")
        system_prompt = parameters.get("system_prompt", get_config().DEFAULT_SYSTEM_PROMPT)
        temperature = parameters.get("temperature", get_config().QWEN_TEMPERATURE)
        max_tokens = parameters.get("max_tokens", get_config().QWEN_MAX_TOKENS)
        stream_callback = parameters.get("stream_callback", None)
        
        if not message.strip():
            return {
                "status": "error",
                "result": "⚠️ Por favor, ingresa un mensaje para el chat."
            }
        
        # Si no hay conversación activa, iniciar una nueva
        if not self.conversation_history:
            self._start_new_conversation()
        
        self.logger.info(f"Processing chat message: {len(message)} characters {'(streaming)' if stream_callback else ''}")
        
        try:
            # Check connection status first
            if not self.qwen_service.is_online:
                connection_status = self.qwen_service.get_connection_status()
                if connection_status["status"] != "online":
                    return {
                        "status": "error",
                        "result": f"❌ No se puede conectar con {self.qwen_service.model_name}",
                        "connection_status": connection_status,
                        "fallback_message": "El servidor de IA no está disponible. Verifica que esté ejecutándose y sea accesible."
                    }
            
            # Agregar mensaje del usuario al historial ANTES de enviar a Qwen
            self._add_to_history("user", message, {"temperature": temperature, "max_tokens": max_tokens})
            
            # Preparar mensajes con contexto completo
            messages_with_context = self._prepare_messages_for_qwen(message, system_prompt)
            
            self.logger.debug(f"🧠 Enviando {len(messages_with_context)} mensajes a Qwen (incluyendo contexto)")
            
            # Usar el método chat_with_history si está disponible, sino usar el método normal
            result = None
            if hasattr(self.qwen_service, 'chat_with_messages'):
                try:
                    self.logger.debug("🔄 Intentando chat_with_messages (con contexto)")
                    result = self.qwen_service.chat_with_messages(
                        messages=messages_with_context,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=stream_callback is not None,
                        stream_callback=stream_callback
                    )
                    self.logger.debug(f"✅ chat_with_messages exitoso: {result.get('status')}")
                except Exception as e:
                    self.logger.warning(f"⚠️ chat_with_messages falló: {str(e)}, usando fallback")
                    result = None  # Force fallback
            
            # Fallback: usar el método chat normal pero con contexto en el mensaje
            if result is None or result.get("status") != "success":
                self.logger.debug("🔄 Usando fallback: método chat regular con contexto")
                context_message = self._format_message_with_context(message)
                result = self.qwen_service.chat(
                    message=context_message,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream_callback is not None,
                    stream_callback=stream_callback
                )
            
            # Si la respuesta fue exitosa, agregar respuesta del asistente al historial
            if result.get("status") == "success":
                assistant_response = result.get("result", "")
                if assistant_response:
                    self._add_to_history("assistant", assistant_response, {
                        "model": self.qwen_service.model_name,
                        "temperature": temperature,
                        "tokens_used": result.get("tokens_used", 0)
                    })
                    
                    # Actualizar contador de tokens
                    self.total_tokens_used += result.get("tokens_used", 0)
                    
                    # NOTE: TTS auto-speak is now handled in GUI after streaming completes
                    # This prevents duplicate TTS calls and synchronization issues
                    
                    # Agregar información de contexto al resultado
                    result["conversation_context"] = {
                        "conversation_id": self.conversation_id,
                        "messages_in_context": len(messages_with_context),
                        "total_conversation_messages": len(self.conversation_history),
                        "total_tokens_used": self.total_tokens_used
                    }
            
            # Update online status based on result
            self.qwen_service.is_online = result.get("status") == "success"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Chat processing error: {str(e)}")
            if self.qwen_service:
                self.qwen_service.is_online = False
            return {
                "status": "error",
                "result": f"❌ Error en el chat: {str(e)}",
                "technical_details": str(e),
                "fallback_message": "Error de comunicación con el servidor de IA. Intenta nuevamente en unos momentos."
            }
    
    def _format_message_with_context(self, new_message: str) -> str:
        """Formatear mensaje con contexto para servicios que no soportan múltiples mensajes"""
        if not self.conversation_history:
            return new_message
        
        # Obtener últimos N mensajes para contexto
        recent_messages = self.conversation_history[-6:]  # Últimos 6 mensajes
        
        context_parts = ["=== CONTEXTO DE CONVERSACIÓN ==="]
        for msg in recent_messages:
            if msg["role"] == "user":
                context_parts.append(f"Usuario: {msg['content']}")
            elif msg["role"] == "assistant":
                context_parts.append(f"Asistente: {msg['content']}")
        
        context_parts.append("=== MENSAJE ACTUAL ===")
        context_parts.append(new_message)
        
        return "\n\n".join(context_parts)


class NetworkScanner:
    """Network scanning security module"""
    def initialize(self): pass
    def shutdown(self): pass
    def scan(self, parameters): return {"status": "scanned", "result": "network analysis"}


class VulnerabilityChecker:
    """Vulnerability checking security module"""
    def initialize(self): pass
    def shutdown(self): pass
    def check(self, parameters): return {"status": "checked", "result": "vulnerability report"}


class SecurityLogger:
    """Security event logging module"""
    def initialize(self): pass
    def shutdown(self): pass


class SpeechModule:
    """Speech recognition module wrapper"""
    def initialize(self): 
        self.logger = get_logger("SpeechModule")
        try:
            from src.services.speech_service import get_speech_service
            self.speech_service = get_speech_service()
            self.logger.info("SpeechModule initialized successfully")
        except ImportError as e:
            self.logger.warning(f"Speech recognition dependencies not available: {e}")
            self.speech_service = None
    
    def shutdown(self): 
        if hasattr(self, 'speech_service') and self.speech_service:
            self.speech_service.cleanup()
        self.logger.info("SpeechModule shutdown")


# Global core instance
_core_instance: Optional[ThinkerCore] = None


def get_core() -> ThinkerCore:
    """Get or create global core instance"""
    global _core_instance
    
    if _core_instance is None:
        _core_instance = ThinkerCore()
    
    return _core_instance 