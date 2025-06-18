"""
Qwen2.5-7B-Instruct-1M AI Service
Connects to local Qwen2.5-7B-Instruct-1M model server for AI assistance
"""

import requests
import json
import time
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime

from src.utils.logger import get_logger


class QwenService:
    """Service to interact with Qwen2.5-7B-Instruct-1M model via local API"""
    
    def __init__(self, 
                 base_url: str = "http://192.168.1.45:1234",
                 model_name: str = "qwen2.5-7b-instruct-1m"):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.logger = get_logger("QwenService")
        
        # API endpoints
        self.chat_endpoint = f"{self.base_url}/v1/chat/completions"
        self.models_endpoint = f"{self.base_url}/v1/models"
        
        # Configuration
        self.timeout = 30  # 30 seconds timeout
        self.max_retries = 3
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        self.logger.info(f"QwenService initialized for {model_name} at {base_url}")
        
        # Test connection on initialization
        self._test_connection()
    
    def _test_connection(self) -> bool:
        """Test connection to the Qwen server"""
        try:
            response = self.session.get(self.models_endpoint, timeout=5)
            if response.status_code == 200:
                self.logger.info("✅ Successfully connected to Qwen server")
                return True
            else:
                self.logger.warning(f"⚠️ Qwen server responded with status {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"❌ Failed to connect to Qwen server: {str(e)}")
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models from the server"""
        try:
            response = self.session.get(self.models_endpoint, timeout=self.timeout)
            response.raise_for_status()
            
            models_data = response.json()
            model_names = [model.get('id', '') for model in models_data.get('data', [])]
            
            self.logger.debug(f"Available models: {model_names}")
            return model_names
            
        except Exception as e:
            self.logger.error(f"Failed to get available models: {str(e)}")
            return []
    
    def chat(self, message: str, 
             system_prompt: Optional[str] = None,
             temperature: float = 0.3,
             max_tokens: int = 1000,
             stream: bool = True,
             fast_mode: bool = True,
             stream_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """
        Send a chat message to Qwen2.5-7B-Instruct-1M and get response
        
        Args:
            message: User message
            system_prompt: Optional system prompt to set context
            temperature: Randomness (0.0 to 1.0)
            max_tokens: Maximum tokens in response
            stream: Whether to stream response (default: True)
            fast_mode: Use optimized settings for faster responses
            stream_callback: Function called with each streaming chunk
        
        Returns:
            Dict containing response and metadata
        """
        try:
            start_time = time.time()
            
            # Optimize settings for fast mode (7B model is already fast!)
            if fast_mode:
                temperature = min(temperature, 0.5)  # Moderate temperature for balanced speed/quality
                max_tokens = min(max_tokens, 1500)   # Allow longer responses since model is faster
                # Use optimized system prompt for 7B instruct model
                if not system_prompt:
                    system_prompt = "Responde de forma clara y útil en español. Sé conciso pero completo."
            
            # Prepare messages
            messages = []
            
            if system_prompt:
                messages.append({
                    "role": "system", 
                    "content": system_prompt
                })
            
            messages.append({
                "role": "user",
                "content": message
            })
            
            # Prepare request payload with optimizations
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream
            }
            
            # Add performance optimizations for large models
            if fast_mode:
                payload.update({
                    "top_p": 0.9,      # Nucleus sampling for faster generation
                    "top_k": 40,       # Limit vocabulary for speed
                    "repeat_penalty": 1.1,  # Prevent repetition
                })
            
            self.logger.debug(f"Sending request to Qwen (fast_mode={fast_mode}): {len(message)} chars, max_tokens={max_tokens}")
            
            # Show progress for long operations
            if len(message) > 100:
                self.logger.info("🔄 Processing long message, please wait...")
            
            # Send request with streaming support
            if stream and stream_callback:
                # Stream mode with callback
                response_text = self._send_streaming_request(payload, stream_callback)
            else:
                # Regular mode
                response = self._send_request_with_retries(payload)
                if response and 'choices' in response and len(response['choices']) > 0:
                    response_text = response['choices'][0]['message']['content']
                else:
                    raise Exception("Invalid response format from Qwen server")
            
            # Process response
            response_time = time.time() - start_time
            
            result = {
                "status": "success",
                "result": response_text,
                "response_time": round(response_time, 2),
                "model": self.model_name,
                "tokens_used": len(response_text.split()),  # Approximate token count
                "timestamp": datetime.now().isoformat(),
                "streaming": stream
            }
            
            self.logger.info(f"✅ Qwen response received in {response_time:.2f}s {'(streaming)' if stream else ''}")
            return result
                
        except Exception as e:
            self.logger.error(f"❌ Qwen chat error: {str(e)}")
            return {
                "status": "error",
                "result": f"Lo siento, hubo un error al conectar con el modelo Qwen3-32B: {str(e)}",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _send_request_with_retries(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send request with retry logic and dynamic timeout"""
        last_exception = None
        
        # Calculate dynamic timeout based on message complexity (optimized for 7B model)
        base_timeout = self.timeout
        message_length = len(payload["messages"][-1]["content"])
        max_tokens = payload.get("max_tokens", 1000)
        
        # Lighter timeout adjustments for the faster 7B model
        dynamic_timeout = base_timeout
        if message_length > 500:  # Only for very long messages
            dynamic_timeout += 15
        if max_tokens > 1500:     # Only for very long responses
            dynamic_timeout += 10
            
        self.logger.debug(f"Using timeout: {dynamic_timeout}s (base: {base_timeout}s)")
        
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"🔄 Enviando petición a Qwen (intento {attempt + 1}/{self.max_retries})...")
                
                response = self.session.post(
                    self.chat_endpoint,
                    json=payload,
                    timeout=dynamic_timeout
                )
                response.raise_for_status()
                
                self.logger.info("✅ Respuesta recibida de Qwen")
                return response.json()
                
            except requests.exceptions.Timeout:
                last_exception = f"⏰ Timeout después de {dynamic_timeout}s (intento {attempt + 1})"
                self.logger.warning(f"❌ {last_exception}")
                if attempt == 0:
                    self.logger.info("💡 El modelo Qwen2.5-7B puede estar ocupado. Reintentando...")
                
            except requests.exceptions.ConnectionError:
                last_exception = f"🔌 Error de conexión al servidor Qwen (intento {attempt + 1})"
                self.logger.warning(f"❌ {last_exception}")
                
            except requests.exceptions.HTTPError as e:
                last_exception = f"🌐 Error HTTP {e.response.status_code} del servidor (intento {attempt + 1})"
                self.logger.warning(f"❌ {last_exception}")
                
            except Exception as e:
                last_exception = f"🚨 Error inesperado: {str(e)} (intento {attempt + 1})"
                self.logger.warning(f"❌ {last_exception}")
            
            # Wait before retry (exponential backoff)
            if attempt < self.max_retries - 1:
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                time.sleep(wait_time)
        
        raise Exception(f"Failed after {self.max_retries} attempts. Last error: {last_exception}")
    
    def _send_streaming_request(self, payload: Dict[str, Any], stream_callback: Callable[[str], None]) -> str:
        """Send streaming request and call callback for each chunk"""
        full_response = ""
        
        try:
            self.logger.info("🌊 Iniciando respuesta en streaming...")
            
            response = self.session.post(
                self.chat_endpoint,
                json=payload,
                timeout=self.timeout,
                stream=True
            )
            response.raise_for_status()
            
            # Process streaming response
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    
                    # Skip data: prefix if present
                    if line_str.startswith('data: '):
                        line_str = line_str[6:]
                    
                    # Skip empty lines and [DONE] marker
                    if not line_str.strip() or line_str.strip() == '[DONE]':
                        continue
                    
                    try:
                        # Parse JSON chunk
                        chunk_data = json.loads(line_str)
                        
                        # Extract content from chunk
                        if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                            choice = chunk_data['choices'][0]
                            if 'delta' in choice and 'content' in choice['delta']:
                                chunk_content = choice['delta']['content']
                                if chunk_content:
                                    full_response += chunk_content
                                    # Call the callback with the new chunk
                                    stream_callback(chunk_content)
                            elif 'message' in choice and 'content' in choice['message']:
                                # Handle non-streaming format
                                chunk_content = choice['message']['content']
                                if chunk_content and chunk_content not in full_response:
                                    full_response = chunk_content
                                    stream_callback(chunk_content)
                                    break
                    
                    except json.JSONDecodeError:
                        # Some lines might not be JSON, skip them
                        continue
            
            self.logger.info("✅ Streaming completado")
            return full_response
            
        except Exception as e:
            self.logger.error(f"❌ Error en streaming: {str(e)}")
            # Fallback to regular request
            self.logger.info("🔄 Fallback a modo regular...")
            payload['stream'] = False
            response = self._send_request_with_retries(payload)
            if response and 'choices' in response and len(response['choices']) > 0:
                content = response['choices'][0]['message']['content']
                stream_callback(content)  # Send all at once as fallback
                return content
            else:
                raise Exception("Failed to get response in both streaming and regular modes")
    
    def get_server_status(self) -> Dict[str, Any]:
        """Get server status and health information"""
        try:
            start_time = time.time()
            response = self.session.get(self.models_endpoint, timeout=5)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                return {
                    "status": "online",
                    "response_time": round(response_time, 3),
                    "server_url": self.base_url,
                    "model": self.model_name,
                    "last_check": datetime.now().isoformat()
                }
            else:
                return {
                    "status": "error",
                    "error": f"HTTP {response.status_code}",
                    "server_url": self.base_url,
                    "last_check": datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                "status": "offline",
                "error": str(e),
                "server_url": self.base_url,
                "last_check": datetime.now().isoformat()
            }
    
    def set_system_prompt(self, prompt: str) -> None:
        """Set a default system prompt for all conversations"""
        self.default_system_prompt = prompt
        self.logger.info("Default system prompt updated")
    
    def close(self) -> None:
        """Close the session and cleanup"""
        if self.session:
            self.session.close()
            self.logger.debug("QwenService session closed")


# Global instance
_qwen_service: Optional[QwenService] = None


def get_qwen_service() -> QwenService:
    """Get or create global QwenService instance"""
    global _qwen_service
    
    if _qwen_service is None:
        _qwen_service = QwenService()
    
    return _qwen_service 