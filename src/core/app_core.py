"""
Core application module for Thinker AI Auxiliary Window
Contains the main application logic and business rules
"""

import sys
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path

from src.config.config import get_config, FeatureFlags
from src.utils.logger import get_logger
from src.utils.helpers import Performance, FileManager, ThreadingHelpers


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
        import uuid
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


# Import Qwen service
try:
    from src.services.qwen_service import get_qwen_service
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False


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
    """AI assistant chat module using Qwen2.5-7B-Instruct-1M"""
    def initialize(self): 
        self.logger = get_logger("AssistantChat")
        if QWEN_AVAILABLE:
            self.qwen_service = get_qwen_service()
            self.logger.info("AssistantChat initialized with Qwen2.5-7B-Instruct-1M")
        else:
            self.logger.error("AssistantChat failed to initialize - Qwen service unavailable")
    
    def shutdown(self): 
        if hasattr(self, 'qwen_service'):
            self.qwen_service.close()
        self.logger.info("AssistantChat shutdown")
    
    def chat(self, parameters):
        if not QWEN_AVAILABLE:
            return {
                "status": "error", 
                "result": "❌ Servicio Qwen2.5-7B no disponible. Verifica que el servidor esté ejecutándose en http://172.29.208.1:1234"
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
        
        self.logger.info(f"Processing chat message: {len(message)} characters {'(streaming)' if stream_callback else ''}")
        
        try:
            return self.qwen_service.chat(
                message=message,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream_callback is not None,
                stream_callback=stream_callback
            )
        except Exception as e:
            self.logger.error(f"Chat processing error: {str(e)}")
            return {
                "status": "error",
                "result": f"❌ Error en el chat: {str(e)}"
            }


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