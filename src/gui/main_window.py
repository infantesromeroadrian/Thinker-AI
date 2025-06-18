"""
Main Window module for Thinker AI Auxiliary Application
Contains the primary GUI interface and main window components
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox, font
import _tkinter
import threading
from datetime import datetime
from typing import Optional, Dict, Any, Callable

from src.config.config import get_config
from src.core.app_core import get_core
from src.utils.logger import get_logger
from src.utils.helpers import UIHelpers, Performance, ThreadingHelpers
from src.exceptions import (
    UIError, WindowCreationError, UIComponentError, 
    InitializationError, ThinkerAIException
)

# Configure CustomTkinter settings
ctk.set_appearance_mode("dark")  # "dark", "light", "system"
ctk.set_default_color_theme("dark-blue")  # "blue", "green", "dark-blue"


class ThinkerMainWindow:
    """Main application window class"""
    
    def __init__(self):
        self.config = get_config()
        self.core = get_core()
        self.logger = get_logger("MainWindow")
        
        # Window components
        self.root = None
        self.style = None
        self.menu_bar = None
        self.status_bar = None
        self.main_frame = None
        
        # UI components
        self.notebook = None
        self.tabs = {}
        self.widgets = {}
        
        # Application state
        self.is_initialized = False
        self.current_theme = "light"
        self.placeholder_active = False
        
        # Streaming state
        self.current_streaming_message = ""
        self.streaming_sender = ""
        self.is_streaming = False
        
        # Voice recording state
        self.is_voice_recording = False
        self.speech_service_available = False
        
        # Shutdown state
        self.is_shutting_down = False
        
        self.logger.info("MainWindow initialized")
    
    @Performance.time_function
    def initialize(self) -> bool:
        """Initialize the main window and all components"""
        try:
            # Create main window
            self._create_main_window()
            
            # Setup styling
            self._setup_styling()
            
            # Create menu bar
            self._create_menu_bar()
            
            # Create main content area
            self._create_main_content()
            
            # Create status bar
            self._create_status_bar()
            
            # Setup event handlers
            self._setup_event_handlers()
            
            # Start core application
            if not self.core.start():
                raise Exception("Failed to start core application")
            
            # Initialize periodic updates
            self._start_periodic_updates()
            
            self.is_initialized = True
            self.logger.info("Main window initialized successfully")
            return True
            
        except Exception as e:
            self.logger.log_exception(e, "Main window initialization")
            return False
    
    def _create_main_window(self) -> None:
        """Create the main CustomTkinter window"""
        self.root = ctk.CTk()
        self.root.title(self.config.APP_NAME)
        self.root.minsize(self.config.MIN_WINDOW_WIDTH, self.config.MIN_WINDOW_HEIGHT)
        
        # Center the window
        UIHelpers.center_window(
            self.root, 
            self.config.WINDOW_WIDTH, 
            self.config.WINDOW_HEIGHT
        )
        
        # Configure window behavior
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self._on_window_close)
        
        # Set window icon (if available)
        try:
            # You can add an icon file here
            # self.root.iconbitmap("icon.ico")
            pass
        except Exception:
            pass
        
        self.logger.debug("Modern CustomTkinter window created")
    
    def _setup_styling(self) -> None:
        """Setup CustomTkinter styling and themes"""
        # CustomTkinter handles styling automatically, but we can customize
        colors = self.config.get_color_scheme()
        
        # Configure CustomTkinter appearance
        ctk.set_appearance_mode("dark")  # "dark", "light", "system"
        ctk.set_default_color_theme("dark-blue")  # "blue", "green", "dark-blue"
        
        # Set custom fonts globally
        ctk.FontManager.load_font("src/fonts/custom.ttf") if hasattr(ctk, 'FontManager') else None
        
        self.logger.debug("Modern CustomTkinter styling configured")
    
    def _create_menu_bar(self) -> None:
        """Create minimalist translucent menu bar"""
        self.menu_bar = tk.Menu(
            self.root,
            bg="#1A1A1A", 
            fg="#E5E5E5",
            activebackground="#333333",
            activeforeground="#00D4FF",
            borderwidth=0,
            relief="flat"
        )
        self.root.configure(menu=self.menu_bar)
        
        # Minimal Chat Menu
        chat_menu = tk.Menu(
            self.menu_bar, 
            tearoff=0,
            bg="#2A2A2A", 
            fg="#E5E5E5",
            activebackground="#333333",
            activeforeground="#00D4FF",
            borderwidth=0
        )
        self.menu_bar.add_cascade(label="💬", menu=chat_menu)
        chat_menu.add_command(label="🗑️ Limpiar", command=self._clear_chat)
        chat_menu.add_command(label="💾 Exportar", command=self._export_chat)
        chat_menu.add_separator()
        chat_menu.add_command(label="❌ Salir", command=self._on_window_close)
        
        # Minimal System Menu
        system_menu = tk.Menu(
            self.menu_bar, 
            tearoff=0,
            bg="#2A2A2A", 
            fg="#E5E5E5",
            activebackground="#333333",
            activeforeground="#00D4FF",
            borderwidth=0
        )
        self.menu_bar.add_cascade(label="⚙️", menu=system_menu)
        system_menu.add_command(label="📊 Estado", command=self._show_system_status)
        system_menu.add_command(label="🧪 Test", command=self._test_qwen_connection)
        system_menu.add_command(label="🎤 Test Voz", command=self._test_speech_recognition)
        system_menu.add_command(label="🔧 Diagnóstico Voz", command=self._advanced_microphone_diagnosis)
        
        # Minimal Help Menu
        help_menu = tk.Menu(
            self.menu_bar, 
            tearoff=0,
            bg="#2A2A2A", 
            fg="#E5E5E5",
            activebackground="#333333",
            activeforeground="#00D4FF",
            borderwidth=0
        )
        self.menu_bar.add_cascade(label="❓", menu=help_menu)
        help_menu.add_command(label="ℹ️ Info", command=self._show_about)
        
        self.logger.debug("Minimalist menu bar created")
    
    # Dummy methods for compatibility
    def _new_conversation(self):
        """Start new conversation (compatibility method)"""
        self._clear_chat()
    
    def _quit_app(self):
        """Quit application (compatibility method)"""
        self._on_window_close()
    
    def _increase_font_size(self):
        """Increase font size (placeholder)"""
        pass
    
    def _decrease_font_size(self):
        """Decrease font size (placeholder)"""
        pass
    
    def _open_preferences(self):
        """Open preferences (placeholder)"""
        UIHelpers.show_info(self.root, "Preferencias", "Funcionalidad en desarrollo...")
    
    def _create_main_content(self) -> None:
        """Create the main content area - Minimalist AI Chat Interface"""
        # Ultra-minimal main frame with transparency
        self.main_frame = ctk.CTkFrame(
            self.root, 
            corner_radius=0,
            fg_color="transparent",
            bg_color="transparent"
        )
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        
        # Create minimalist AI chat interface
        self._create_minimalist_chat_interface()
        
        self.logger.debug("Minimalist AI Chat interface created")
    
    def _create_minimalist_chat_interface(self) -> None:
        """Create ultra-minimalist translucent chat interface"""
        colors = self.config.get_color_scheme()
        
        # Header - Ultra minimal floating
        header_frame = ctk.CTkFrame(
            self.main_frame, 
            corner_radius=25,
            fg_color=("#2A2A2A", "#1A1A1A"),
            height=60
        )
        header_frame.pack(fill=tk.X, pady=(20, 0), padx=20)
        header_frame.pack_propagate(False)
        
        # Minimal title - just the essence
        title_label = ctk.CTkLabel(
            header_frame, 
            text="💭 Thinker AI",
            font=ctk.CTkFont(family=self.config.DEFAULT_FONT_FAMILY, size=16, weight="bold"),
            text_color=self.config.TEXT_COLOR
        )
        title_label.pack(side=tk.LEFT, padx=20, pady=15)
        
        # Minimal status indicator - floating dot
        self.widgets["status_indicator"] = ctk.CTkLabel(
            header_frame,
            text="●",
            text_color=self.config.SUCCESS_COLOR,
            font=ctk.CTkFont(size=20)
        )
        self.widgets["status_indicator"].pack(side=tk.RIGHT, padx=20, pady=15)
        
        # Chat area - Floating translucent container
        chat_container = ctk.CTkFrame(
            self.main_frame,
            corner_radius=25,
            fg_color=("#1A1A1A", "#0F0F0F"),
            bg_color="transparent"
        )
        chat_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Chat display - Visible pero limpio
        self.widgets["chat_display"] = ctk.CTkTextbox(
            chat_container, 
            wrap="word",
            font=ctk.CTkFont(family=self.config.DEFAULT_FONT_FAMILY, size=12),
            corner_radius=20,
            fg_color=("#0F0F0F", "#0A0A0A"),  # Muy sutil pero visible
            text_color=self.config.TEXT_COLOR,
            border_width=0  # Chat display sin borde está bien
        )
        self.widgets["chat_display"].pack(fill=tk.BOTH, expand=True, padx=25, pady=25)
        # Chat display: disabled (solo lectura del historial)
        self.widgets["chat_display"].configure(state="disabled")
        
        # Input area - Floating at bottom
        input_container = ctk.CTkFrame(
            self.main_frame,
            corner_radius=25,
            fg_color=("#2A2A2A", "#1A1A1A"),
            height=100  # Más alto para incluir etiqueta
        )
        input_container.pack(fill=tk.X, padx=20, pady=(0, 20))
        input_container.pack_propagate(False)
        
        # Etiqueta indicativa
        input_label = ctk.CTkLabel(
            input_container,
            text="✏️ Escribe aquí tu mensaje:",
            font=ctk.CTkFont(family=self.config.DEFAULT_FONT_FAMILY, size=10),
            text_color="#888888"
        )
        input_label.pack(pady=(8, 0), padx=20, anchor="w")
        
        # Input field - ULTRA VISIBLE con contraste máximo
        self.widgets["chat_input"] = ctk.CTkTextbox(
            input_container,
            height=55,  # MÁS ALTO - zona clickeable grande
            wrap="word",
            font=ctk.CTkFont(family="Consolas", size=13, weight="normal"),  # Fuente clara
            corner_radius=15,
            fg_color=("#FFFFFF", "#F8F8FF"),  # FONDO BLANCO - máximo contraste
            text_color=("#000000", "#1A1A1A"),  # TEXTO NEGRO - perfectamente visible
            border_width=3,  # Borde muy grueso
            border_color=("#00D4FF", "#00D4FF")  # Borde neón azul brillante
        )
        self.widgets["chat_input"].pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(20, 10), pady=(5, 15))
        
        # Placeholder contrastante para fondo blanco
        self.widgets["chat_input"].insert("1.0", "💬 Escribe tu mensaje aquí...")
        self.widgets["chat_input"].configure(text_color="#0066CC")  # Azul oscuro sobre fondo blanco
        self.placeholder_active = True
        
        # Voice button - Microphone for speech recognition
        self.widgets["voice_button"] = ctk.CTkButton(
            input_container,
            text="🎤",
            command=self._toggle_voice_recording,
            corner_radius=20,
            width=40,
            height=40,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#FF6B6B",  # Red for recording
            hover_color="#FF8E8E",
            border_width=0
        )
        self.widgets["voice_button"].pack(side=tk.RIGHT, padx=(0, 5), pady=(5, 15))
        
        # Send button - Minimal floating circle
        self.widgets["send_button"] = ctk.CTkButton(
            input_container, 
            text="→",
            command=self._send_chat_message,
            corner_radius=20,
            width=40,
            height=40,
            font=ctk.CTkFont(size=18, weight="bold"),
            fg_color=self.config.ACCENT_COLOR,
            hover_color=self.config.SUCCESS_COLOR,
            border_width=0
        )
        self.widgets["send_button"].pack(side=tk.RIGHT, padx=(0, 20), pady=(5, 15))
        
        # Event bindings MÍNIMOS - solo lo esencial
        self.widgets["chat_input"].bind("<Return>", self._on_enter_key)
        # TEMPORALMENTE quitamos otros bindings que pueden interferir
        # self.widgets["chat_input"].bind("<Control-Return>", self._on_ctrl_enter)
        # self.widgets["chat_input"].bind("<FocusIn>", self._on_input_focus_in)
        # self.widgets["chat_input"].bind("<FocusOut>", self._on_input_focus_out)
        # self.widgets["chat_input"].bind("<KeyPress>", self._on_key_press)
        
        # Welcome message - Minimalist with voice info
        welcome_msg = "🚀 Listo para conversar con streaming en tiempo real\n🎤 Usa el botón rojo para dictado de voz (Menu ⚙️ > 🔧 Diagnóstico Voz si hay problemas)"
        self._add_chat_message("💭", welcome_msg)
        
        # FORZAR focus en el campo de texto
        self.widgets["chat_input"].focus_force()  # Forzar focus
        
        # Debug: Información básica del widget (sin 'state' que no es compatible)
        print(f"🔍 DEBUG - Chat input creado:")
        print(f"   Widget: {self.widgets['chat_input']}")
        print(f"   Tipo: {type(self.widgets['chat_input'])}")
        
        # Test simple: Programar verificación después de que se muestre
        self.root.after(500, self._test_input_functionality)
    
    def _test_input_functionality(self):
        """Test para verificar funcionalidad del campo de entrada"""
        try:
            widget = self.widgets["chat_input"]
            print(f"\n🧪 TEST DEL CAMPO DE ENTRADA:")
            print(f"   Tipo de widget: {type(widget)}")
            print(f"   ¿Existe?: {widget is not None}")
            print(f"   Geometría: {widget.winfo_geometry()}")
            
            # Test: Insertar texto programáticamente
            current_content = widget.get("1.0", "end-1c")
            print(f"   Contenido actual: '{current_content}'")
            
            # Test: ¿Responde a configuración?
            widget.delete("1.0", tk.END)
            widget.insert("1.0", "✅ TEST: ¡CAMPO FUNCIONANDO! Haz click aquí ⬅️")
            widget.configure(text_color="#008000")  # Verde oscuro sobre fondo blanco
            
            print(f"   ✅ El widget responde a comandos programáticos")
            print(f"   👆 BUSCA EL TEXTO VERDE EN LA VENTANA Y HAZ CLICK")
            
        except Exception as e:
            print(f"   ❌ Error en test: {e}")
    
    def _on_enter_key(self, event):
        """Handle Enter key press"""
        self._send_chat_message()
        return "break"  # Prevent default behavior
    
    def _on_ctrl_enter(self, event):
        """Handle Ctrl+Enter for new line"""
        return None  # Allow default behavior (new line)
    
    def _on_key_press(self, event):
        """Handle key press - clear placeholder on typing"""
        self._clear_placeholder()
        
    def _on_input_focus_in(self, event):
        """Handle input focus in - clear placeholder"""
        self._clear_placeholder()
        
    def _on_input_focus_out(self, event):
        """Handle input focus out - restore placeholder if empty"""
        content = self.widgets["chat_input"].get("1.0", tk.END).strip()
        if not content:
            self._set_placeholder()
    
    def _clear_placeholder(self):
        """Clear placeholder text"""
        if hasattr(self, 'placeholder_active') and self.placeholder_active:
            current_text = self.widgets["chat_input"].get("1.0", tk.END).strip()
            if current_text == "💬 Escribe tu mensaje aquí...":
                self.widgets["chat_input"].delete("1.0", tk.END)
            self.widgets["chat_input"].configure(text_color="#000000")  # Negro sobre fondo blanco
            self.placeholder_active = False
    
    def _set_placeholder(self):
        """Set placeholder text"""
        if not hasattr(self, 'placeholder_active') or not self.placeholder_active:
            self.widgets["chat_input"].delete("1.0", tk.END)
            self.widgets["chat_input"].insert("1.0", "💬 Escribe tu mensaje aquí...")
            self.widgets["chat_input"].configure(text_color="#0066CC")  # Azul oscuro sobre fondo blanco
            self.placeholder_active = True
    

    
    def _clear_chat(self):
        """Clear the chat display"""
        if UIHelpers.ask_yes_no(self.root, "Confirmar", "¿Deseas limpiar toda la conversación?"):
            self.widgets["chat_display"].configure(state="normal")
            self.widgets["chat_display"].delete("1.0", tk.END)
            self.widgets["chat_display"].configure(state="disabled")
            
            # Add minimal welcome message again
            self._add_chat_message("💭", 
                                  "Chat limpio. Listo para streaming. ¿Qué necesitas?")
            
            self.logger.log_user_action("Chat Cleared")
    

    
    def _create_status_bar(self) -> None:
        """Create minimalist translucent status bar"""
        # Ultra-minimal floating status bar
        self.status_bar = ctk.CTkFrame(
            self.root, 
            corner_radius=15, 
            height=25,
            fg_color=("#2A2A2A", "#1A1A1A")
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=(0, 20))
        
        # Just time - minimal info
        self.widgets["time_label"] = ctk.CTkLabel(
            self.status_bar, 
            text="",
            font=ctk.CTkFont(family=self.config.DEFAULT_FONT_FAMILY, size=9),
            text_color="#888888"
        )
        self.widgets["time_label"].pack(side=tk.RIGHT, padx=15, pady=3)
        
        # Just session - minimal
        session_text = f"🔗 {self.core.session_id[:6]}"
        self.widgets["session_label"] = ctk.CTkLabel(
            self.status_bar, 
            text=session_text,
            font=ctk.CTkFont(family=self.config.DEFAULT_FONT_FAMILY, size=9),
            text_color="#666666"
        )
        self.widgets["session_label"].pack(side=tk.LEFT, padx=15, pady=3)
        
        self.logger.debug("Minimalist status bar created")
    
    def _setup_event_handlers(self) -> None:
        """Setup event handlers and bindings"""
        # Window close event
        self.root.protocol("WM_DELETE_WINDOW", self._on_window_close)
        
        # Keyboard shortcuts
        self.root.bind("<Control-q>", lambda e: self._on_window_close())
        self.root.bind("<F1>", lambda e: self._show_about())
        self.root.bind("<F5>", lambda e: self._show_system_status())
        self.root.bind("<Control-l>", lambda e: self._clear_chat())
        
        self.logger.debug("Event handlers setup")
    
    def _start_periodic_updates(self) -> None:
        """Start periodic UI updates"""
        self._update_time()
    
    def _update_time(self) -> None:
        """Update the time display in status bar"""
        if self.widgets.get("time_label"):
            current_time = datetime.now().strftime("%H:%M:%S")
            self.widgets["time_label"].configure(text=current_time)
        
        # Schedule next update
        if self.is_initialized:
            self.root.after(1000, self._update_time)
    
    # Event handler methods
    def _on_window_close(self) -> None:
        """Handle window close event"""
        try:
            if self.is_shutting_down:
                return  # Already shutting down, prevent duplicate calls
                
            if UIHelpers.ask_yes_no(self.root, "Confirmar Salida", "¿Estás seguro de que quieres salir?"):
                self.logger.log_user_action("Application close requested by user")
                self.shutdown()
        except (_tkinter.TclError, tk.TclError, RuntimeError) as e:
            # Handle specific Tkinter errors during close
            if "application has been destroyed" in str(e).lower():
                self.logger.info("Application already destroyed during close event")
            else:
                self.logger.log_exception(e, "Tkinter error during window close")
        except Exception as e:
            # Handle other unexpected errors
            self.logger.log_exception(e, "Window close handler error")
            # Force quit if there's an error in the close handler
            try:
                if self.root:
                    self.root.quit()
            except (_tkinter.TclError, tk.TclError):
                pass  # Ignore Tkinter errors during force quit
    
    def _send_chat_message(self, event=None) -> None:
        """Send chat message to AI assistant"""
        if not self.widgets.get("chat_input"):
            return
        
        # Get message from CTkTextbox
        message = self.widgets["chat_input"].get("1.0", tk.END).strip()
        
        # Skip si es placeholder o vacío
        if not message or message == "💬 Escribe tu mensaje aquí...":
            return
        
        # Clear input and restore placeholder
        self.widgets["chat_input"].delete("1.0", tk.END)
        self._set_placeholder()
        
        # Add user message to chat
        self._add_chat_message("👤 Tú", message)
        
        # Show processing status with better feedback
        self._show_processing_status(message)
        
        # Process message in background
        ThreadingHelpers.run_in_background(
            lambda: self._process_ai_message(message),
            callback=self._on_ai_response_received,
            error_callback=self._handle_ai_error
        )
    
    def _on_ai_response_received(self, result):
        """Called when AI response is received"""
        # Re-enable send button with minimalist styling
        self.root.after(0, lambda: self.widgets["send_button"].configure(
            state="normal", text="→"))
        
        # Restore status indicator
        if self.widgets.get("status_indicator"):
            self.root.after(0, lambda: self.widgets["status_indicator"].configure(
                text="●", 
                text_color=self.config.SUCCESS_COLOR
            ))
        
        # Focus back to input
        self.root.after(0, lambda: self.widgets["chat_input"].focus_set())
    
    def _process_ai_message(self, message: str) -> None:
        """Process AI message in background with streaming"""
        try:
            # Start streaming message
            self._start_streaming_message("🤖 AI Assistant")
            
            # Create stream callback
            def stream_callback(chunk: str):
                self.root.after(0, lambda: self._append_streaming_chunk(chunk))
            
            # Execute AI operation with streaming
            result = self.core.execute_ai_operation("chat", {
                "message": message,
                "stream_callback": stream_callback
            })
            
            # Finalize streaming
            self.root.after(0, lambda: self._finalize_streaming_message())
            
        except Exception as e:
            self.logger.log_exception(e, "AI message processing")
            error_msg = "Encontré un error procesando tu solicitud. Por favor, intenta de nuevo."
            self.root.after(0, lambda: self._add_chat_message("🚨 Sistema", error_msg))
    
    def _handle_ai_error(self, error: Exception) -> None:
        """Handle AI operation errors"""
        self.logger.log_exception(error, "AI operation error")
        error_msg = (
            "⚠️ El modelo Qwen2.5-7B no está disponible o hay problemas de conexión. "
            "Verifica que el servidor esté funcionando correctamente y reintenta."
        )
        self.root.after(0, lambda: self._add_chat_message("🚨 Sistema", error_msg))
        
        # Re-enable send button with minimalist styling
        self.root.after(0, lambda: self.widgets["send_button"].configure(
            state="normal", text="→"))
        
        # Update status indicator to show error
        if self.widgets.get("status_indicator"):
            self.root.after(0, lambda: self.widgets["status_indicator"].configure(
                text="⚠", 
                text_color=self.config.ERROR_COLOR
            ))
    
    def _show_processing_status(self, message: str) -> None:
        """Show minimalist processing status"""
        # Minimal status indication
        button_text = "…"
        status_text = "🌊 Streaming..."
        
        # Update button
        self.widgets["send_button"].configure(state="disabled", text=button_text)
        
        # Show minimal status in chat
        self._add_chat_message("💭", status_text)
        
        # Update status indicator
        if self.widgets.get("status_indicator"):
            self.widgets["status_indicator"].configure(
                text="◐", 
                text_color=self.config.ACCENT_COLOR
            )
    
    def _add_chat_message(self, sender: str, message: str) -> None:
        """Add message to chat display with modern styling"""
        if not self.widgets.get("chat_display"):
            return
        
        chat_display = self.widgets["chat_display"]
        chat_display.configure(state="normal")
        
        timestamp = datetime.now().strftime("%H:%M")
        formatted_message = f"[{timestamp}] {sender}: {message}\n\n"
        
        chat_display.insert(tk.END, formatted_message)
        chat_display.configure(state="disabled")
        
        # Auto-scroll to bottom
        chat_display.see(tk.END)
    
    def _start_streaming_message(self, sender: str) -> None:
        """Start a new streaming message"""
        self.is_streaming = True
        self.streaming_sender = sender
        self.current_streaming_message = ""
        
        # Add initial empty message
        chat_display = self.widgets["chat_display"]
        chat_display.configure(state="normal")
        
        timestamp = datetime.now().strftime("%H:%M")
        initial_message = f"[{timestamp}] {sender}: "
        
        chat_display.insert(tk.END, initial_message)
        chat_display.configure(state="disabled")
        chat_display.see(tk.END)
    
    def _append_streaming_chunk(self, chunk: str) -> None:
        """Append a chunk to the streaming message"""
        if not self.is_streaming:
            return
        
        self.current_streaming_message += chunk
        
        # Update the chat display
        chat_display = self.widgets["chat_display"]
        chat_display.configure(state="normal")
        
        # Insert the new chunk at the end
        chat_display.insert(tk.END, chunk)
        chat_display.configure(state="disabled")
        chat_display.see(tk.END)
    
    def _finalize_streaming_message(self) -> None:
        """Finalize the streaming message"""
        if not self.is_streaming:
            return
        
        self.is_streaming = False
        
        # Add completion indicator and final newlines
        chat_display = self.widgets["chat_display"]
        chat_display.configure(state="normal")
        chat_display.insert(tk.END, " ✅\n\n")  # Checkmark to indicate completion
        chat_display.configure(state="disabled")
        chat_display.see(tk.END)
        
        # Clear streaming state
        self.current_streaming_message = ""
        self.streaming_sender = ""
    
    # Voice recognition methods
    def _toggle_voice_recording(self) -> None:
        """Toggle voice recording on/off"""
        if not self._check_speech_availability():
            return
        
        if self.is_voice_recording:
            self._stop_voice_recording()
        else:
            self._start_voice_recording()
    
    def _check_speech_availability(self) -> bool:
        """Check if speech recognition is available"""
        try:
            result = self.core.execute_speech_operation("get_status", {})
            
            if result.get("status") == "error":
                # Show detailed error but allow user to try anyway
                response = UIHelpers.ask_yes_no(
                    self.root,
                    "Sistema de Voz con Problemas",
                    f"El sistema de dictado tiene problemas:\n\n{result.get('result', 'Error desconocido')}\n\n"
                    "¿Quieres intentar grabar de todas formas?\n\n"
                    "Nota: Ve a Menu ⚙️ > 🔧 Diagnóstico Voz para más detalles."
                )
                if response:
                    self.speech_service_available = True
                    return True
                return False
            
            # Check microphone but be more lenient
            mic_available = result.get("microphone_available", False)
            mic_count = result.get("available_microphones", 0)
            
            if not mic_available and mic_count == 0:
                response = UIHelpers.ask_yes_no(
                    self.root,
                    "Micrófono No Detectado",
                    "No se pudo detectar un micrófono automáticamente.\n\n"
                    "Esto puede deberse a:\n"
                    "• Permisos de Windows\n"
                    "• Configuración de PyAudio\n"
                    "• Drivers de audio\n\n"
                    "¿Quieres intentar grabar de todas formas?\n"
                    "(Tu micrófono puede funcionar aún así)\n\n"
                    "💡 Usa Menu ⚙️ > 🔧 Diagnóstico Voz para detalles"
                )
                if response:
                    self.speech_service_available = True
                    return True
                return False
            
            self.speech_service_available = True
            return True
            
        except Exception as e:
            self.logger.log_exception(e, "Speech availability check")
            response = UIHelpers.ask_yes_no(
                self.root,
                "Error de Verificación de Voz",
                f"Error verificando el sistema de dictado:\n\n{str(e)}\n\n"
                "¿Quieres intentar usar el dictado de todas formas?"
            )
            if response:
                self.speech_service_available = True
                return True
            return False
    
    def _start_voice_recording(self) -> None:
        """Start voice recording"""
        try:
            self.is_voice_recording = True
            
            # Update button appearance
            self.widgets["voice_button"].configure(
                text="⏹️",
                fg_color="#00FF88"  # Green while recording
            )
            
            # Show recording status
            self._add_chat_message("🎤 Sistema", "🔴 Grabando... Habla ahora")
            
            # Update status indicator
            if self.widgets.get("status_indicator"):
                self.widgets["status_indicator"].configure(
                    text="🎤",
                    text_color="#FF6B6B"
                )
            
            # Start recording
            result = self.core.execute_speech_operation("record_once", {
                "callback": self._on_voice_text_received,
                "timeout": 5.0,
                "phrase_time_limit": 10.0
            })
            
            if result.get("status") == "error":
                raise Exception(result.get("result", "Unknown error"))
                
        except Exception as e:
            self.logger.log_exception(e, "Start voice recording")
            self._stop_voice_recording()
            UIHelpers.show_error_dialog(
                self.root,
                "Error de Grabación",
                f"No se pudo iniciar la grabación:\n\n{str(e)}"
            )
    
    def _stop_voice_recording(self) -> None:
        """Stop voice recording"""
        try:
            self.is_voice_recording = False
            
            # Update button appearance
            self.widgets["voice_button"].configure(
                text="🎤",
                fg_color="#FF6B6B"  # Back to red
            )
            
            # Restore status indicator
            if self.widgets.get("status_indicator"):
                self.widgets["status_indicator"].configure(
                    text="●",
                    text_color=self.config.SUCCESS_COLOR
                )
            
            # Stop recording
            self.core.execute_speech_operation("stop_listening", {})
            
        except Exception as e:
            self.logger.log_exception(e, "Stop voice recording")
    
    def _on_voice_text_received(self, text: str) -> None:
        """Handle recognized voice text and send directly"""
        try:
            self._stop_voice_recording()
            
            if text and text.strip():
                # Log the voice input
                self.logger.log_user_action("Voice Message Sent", f"Text: {text.strip()}")
                
                # Send the voice text directly as a message
                self._process_ai_message(text.strip())
                
            else:
                # Only show error messages in chat, not recognition results
                self._add_chat_message("⚠️ Dictado", 
                    "No se detectó voz clara. Habla más cerca del micrófono.")
                
        except Exception as e:
            self.logger.log_exception(e, "Voice text processing")
            self._add_chat_message("❌ Dictado", f"Error procesando el texto: {str(e)}")
    
    def _test_speech_recognition(self) -> None:
        """Test speech recognition system"""
        self._add_chat_message("🧪 Sistema", "Iniciando test de reconocimiento de voz...")
        
        def run_test():
            try:
                result = self.core.execute_speech_operation("test_recognition", {})
                
                if result.get("status") == "success":
                    recognized_text = result.get("recognized_text", "")
                    engine = result.get("engine", "unknown")
                    language = result.get("language", "unknown")
                    
                    self.root.after(0, lambda: self._add_chat_message(
                        "✅ Test Exitoso",
                        f"Motor: {engine}\nIdioma: {language}\nTexto reconocido: \"{recognized_text}\""
                    ))
                elif result.get("status") == "no_speech":
                    self.root.after(0, lambda: self._add_chat_message(
                        "⚠️ Test Parcial",
                        "El micrófono funciona pero no se detectó voz durante el test."
                    ))
                else:
                    error = result.get("error", "Error desconocido")
                    self.root.after(0, lambda: self._add_chat_message(
                        "❌ Test Fallido",
                        f"Error en el test: {error}"
                    ))
                    
            except Exception as e:
                self.root.after(0, lambda: self._add_chat_message(
                    "❌ Test Fallido",
                    f"Error durante el test: {str(e)}"
                ))
        
        # Run test in background
        ThreadingHelpers.run_in_background(run_test)
    
    def _advanced_microphone_diagnosis(self) -> None:
        """Run advanced microphone diagnosis"""
        self._add_chat_message("🔧 Diagnóstico", "Iniciando diagnóstico avanzado del micrófono...")
        
        def run_diagnosis():
            try:
                result = self.core.execute_speech_operation("advanced_diagnosis", {})
                
                if result.get("status") == "success":
                    diagnosis = result.get("diagnosis", {})
                    
                    # Create diagnosis report
                    report = "🔧 DIAGNÓSTICO AVANZADO DEL MICRÓFONO\n"
                    report += "=" * 50 + "\n\n"
                    
                    # System info
                    sys_info = diagnosis.get("system_info", {})
                    report += f"💻 Sistema: {sys_info.get('platform', 'Unknown')} {sys_info.get('version', '')}\n"
                    report += f"🐍 Python: {sys_info.get('python_version', 'Unknown')}\n\n"
                    
                    # PyAudio info
                    pyaudio_info = diagnosis.get("pyaudio_info", {})
                    if "error" in pyaudio_info:
                        report += f"❌ PyAudio: {pyaudio_info['error']}\n\n"
                    else:
                        report += f"🔊 PyAudio: v{pyaudio_info.get('version', 'Unknown')}\n"
                        report += f"📱 Dispositivos totales: {pyaudio_info.get('device_count', 0)}\n\n"
                    
                    # Microphone info
                    mic_info = diagnosis.get("microphone_info", {})
                    input_devices = mic_info.get("input_devices", [])
                    
                    report += f"🎤 Dispositivos de entrada encontrados: {len(input_devices)}\n"
                    if input_devices:
                        for i, device in enumerate(input_devices[:5]):  # Show first 5
                            report += f"  [{device['index']}] {device['name']}\n"
                        if len(input_devices) > 5:
                            report += f"  ... y {len(input_devices) - 5} más\n"
                    else:
                        report += "  ❌ No se encontraron dispositivos de entrada\n"
                    
                    report += f"\n📢 SpeechRecognition detectó: {mic_info.get('sr_count', 0)} micrófonos\n"
                    
                    # Default device
                    default_device = mic_info.get("default_device")
                    if default_device:
                        report += f"🎯 Dispositivo por defecto: {default_device['name']}\n"
                    elif "default_device_error" in mic_info:
                        report += f"❌ Error dispositivo por defecto: {mic_info['default_device_error']}\n"
                    
                    report += "\n"
                    
                    # Recommendations
                    recommendations = diagnosis.get("recommendations", [])
                    if recommendations:
                        report += "💡 RECOMENDACIONES:\n"
                        for i, rec in enumerate(recommendations, 1):
                            report += f"  {i}. {rec}\n"
                    
                    # Show in chat
                    self.root.after(0, lambda: self._add_chat_message("📋 Diagnóstico Completo", report))
                    
                else:
                    error = result.get("result", "Error desconocido")
                    self.root.after(0, lambda: self._add_chat_message("❌ Diagnóstico Fallido", f"Error: {error}"))
                    
            except Exception as e:
                self.root.after(0, lambda: self._add_chat_message(
                    "❌ Diagnóstico Fallido", 
                    f"Error durante el diagnóstico: {str(e)}"
                ))
        
        # Run diagnosis in background
        ThreadingHelpers.run_in_background(run_diagnosis)

    
    # Menu action methods
    def _export_chat(self):
        """Export chat conversation to file"""
        from datetime import datetime
        
        filename = UIHelpers.save_file(
            self.root,
            title="Exportar Conversación",
            default_extension=".txt",
            filetypes=[("Archivos de texto", "*.txt"), ("Todos los archivos", "*.*")]
        )
        
        if filename:
            try:
                chat_content = self.widgets["chat_display"].get("1.0", tk.END)
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"Conversación Thinker AI - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(chat_content)
                
                UIHelpers.show_info_dialog(self.root, "Éxito", f"Conversación exportada a:\n{filename}")
                self.logger.log_user_action("Chat Exported", filename)
                
            except Exception as e:
                UIHelpers.show_error_dialog(self.root, "Error", f"Error al exportar: {str(e)}")
                self.logger.log_exception(e, "Chat export")
    
    def _open_preferences(self):
        """Open preferences dialog (placeholder)"""
        UIHelpers.show_info_dialog(self.root, "Preferencias", 
                                  "Las preferencias estarán disponibles en futuras versiones.")
        self.logger.log_user_action("Open Preferences")
    
    def _show_shortcuts(self):
        """Show keyboard shortcuts"""
        shortcuts_text = """
Atajos de Teclado:

Chat:
• Enter: Enviar mensaje
• Ctrl + Enter: Nueva línea
• Ctrl + L: Limpiar chat

General:
• Ctrl + Q: Salir
• F1: Acerca de
• F5: Actualizar estado
        """
        UIHelpers.show_info_dialog(self.root, "Atajos de Teclado", shortcuts_text)
    
    def _restart_core(self): 
        """Restart core system"""
        self.core.stop()
        self.core.start()
        self.logger.log_user_action("Core Restarted")
    
    def _show_system_status(self) -> None:
        """Show detailed system status including Qwen3-32B status"""
        status = self.core.get_system_status()
        status_text = f"🔧 System Status Report\n\n"
        status_text += f"Core Status: {status['core_status']}\n"
        status_text += f"Session ID: {status['session_id']}\n"
        status_text += f"Uptime: {status['uptime']}\n\n"
        
        # Check Qwen2.5-7B server status
        qwen_status = self._check_qwen_status()
        status_text += f"🤖 Qwen2.5-7B-Instruct-1M Model Status:\n"
        status_text += f"  Status: {qwen_status['status']}\n"
        status_text += f"  Server: {qwen_status['server_url']}\n"
        if qwen_status.get('response_time'):
            status_text += f"  Response Time: {qwen_status['response_time']}s\n"
        if qwen_status.get('error'):
            status_text += f"  Error: {qwen_status['error']}\n"
        status_text += f"  Last Check: {qwen_status['last_check']}\n\n"
        
        # Check Speech Recognition status
        speech_status = self._check_speech_status()
        status_text += f"🎤 Speech Recognition Status:\n"
        status_text += f"  Available: {speech_status.get('available', 'Unknown')}\n"
        status_text += f"  Engine: {speech_status.get('recognition_engine', 'N/A')}\n"
        status_text += f"  Language: {speech_status.get('language', 'N/A')}\n"
        status_text += f"  Microphone: {speech_status.get('microphone_available', 'Unknown')}\n"
        status_text += f"  Microphones: {speech_status.get('available_microphones', 0)}\n"
        if speech_status.get('last_recognition'):
            status_text += f"  Last Recognition: {speech_status['last_recognition']}\n"
        if speech_status.get('error'):
            status_text += f"  Error: {speech_status['error']}\n\n"
        
        status_text += f"📦 Active Modules ({len(status['active_modules'])}):\n"
        for module in status['active_modules']:
            status_text += f"  • {module}\n"
        
        UIHelpers.show_info_dialog(self.root, "System Status", status_text)
    
    def _check_qwen_status(self) -> Dict[str, Any]:
        """Check Qwen3-32B server status"""
        try:
            from src.services.qwen_service import get_qwen_service
            qwen_service = get_qwen_service()
            return qwen_service.get_server_status()
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "server_url": "http://172.29.208.1:1234",
                "last_check": datetime.now().isoformat()
            }
    
    def _check_speech_status(self) -> Dict[str, Any]:
        """Check speech recognition status"""
        try:
            result = self.core.execute_speech_operation("get_status", {})
            
            if result.get("status") == "error":
                return {
                    "available": False,
                    "error": result.get("result", "Speech service not available")
                }
            
            # Return the speech service status
            status = result.copy()
            status["available"] = True
            return status
            
        except Exception as e:
            return {
                "available": False,
                "error": str(e)
            }
    
    def _test_qwen_connection(self) -> None:
        """Test Qwen2.5-7B-Instruct-1M connection with a simple message"""
        self._add_chat_message("💻 Sistema", "🧪 Iniciando test de conexión con Qwen2.5-7B-Instruct-1M...")
        
        # Show processing
        self.widgets["send_button"].configure(state="disabled", text="🧪 Testing...")
        
        def test_connection():
            """Background test function"""
            try:
                from src.services.qwen_service import get_qwen_service
                qwen_service = get_qwen_service()
                
                # Simple test message without streaming for quick test
                result = qwen_service.chat(
                    message="Hola, ¿puedes responder brevemente que estás funcionando?",
                    fast_mode=True,
                    max_tokens=50,
                    stream=False  # Disable streaming for quick test
                )
                
                if result.get("status") == "success":
                    response = result.get("result", "Test exitoso")
                    response_time = result.get("response_time", 0)
                    self.root.after(0, lambda: self._add_chat_message(
                        "✅ Test Exitoso",
                        f"Qwen2.5-7B responde correctamente en {response_time}s:\n\n{response}"
                    ))
                else:
                    error = result.get("error", "Error desconocido")
                    self.root.after(0, lambda: self._add_chat_message(
                        "❌ Test Fallido",
                        f"Error en la conexión: {error}"
                    ))
                    
            except Exception as e:
                self.root.after(0, lambda: self._add_chat_message(
                    "❌ Test Fallido",
                    f"Error durante el test: {str(e)}"
                ))
            finally:
                # Re-enable button
                self.root.after(0, lambda: self.widgets["send_button"].configure(
                    state="normal", text="📤 Enviar Mensaje"
                ))
        
        # Run test in background
        ThreadingHelpers.run_in_background(test_connection)
    
    def _show_about(self) -> None:
        """Show about dialog"""
        about_text = (
            f"🤖 {self.config.APP_NAME}\n"
            f"Versión {self.config.APP_VERSION}\n\n"
            f"Desarrollado por: {self.config.AUTHOR}\n\n"
            "Ventana auxiliar profesional para conversación con IA\n"
            "construida con Python y tkinter.\n\n"
            "Características:\n"
            "• Interfaz de chat intuitiva\n"
            "• Tema oscuro profesional\n"
            "• Exportación de conversaciones\n"
            "• Atajos de teclado eficientes\n\n"
            "¡Disfruta conversando con tu asistente AI!"
        )
        UIHelpers.show_info_dialog(self.root, "Acerca de Thinker AI", about_text)
    
    def _show_documentation(self) -> None:
        """Show documentation"""
        doc_text = (
            "📖 Documentación de Thinker AI\n\n"
            "Uso Básico:\n"
            "• Escribe tu mensaje en el área de texto inferior\n"
            "• Presiona Enter para enviar\n"
            "• Usa Ctrl+Enter para salto de línea\n\n"
            "Atajos Útiles:\n"
            "• Ctrl+L: Limpiar conversación\n"
            "• Ctrl+Q: Salir de la aplicación\n"
            "• F1: Mostrar información\n\n"
            "Menú Chat:\n"
            "• Exportar conversación a archivo de texto\n"
            "• Limpiar todo el historial\n\n"
            "¿Necesitas ayuda? ¡Pregúntale directamente al AI!"
        )
        UIHelpers.show_info_dialog(self.root, "Documentación", doc_text)
    
    def run(self) -> None:
        """Start the main application loop"""
        if not self.is_initialized:
            if not self.initialize():
                self.logger.error("Failed to initialize main window")
                return
        
        self.logger.info("Starting main application loop")
        
        try:
            self.root.mainloop()
        except Exception as e:
            self.logger.log_exception(e, "Main loop error")
        finally:
            self.shutdown()
    
    def shutdown(self) -> None:
        """Shutdown the application with proper error handling"""
        # Prevent multiple shutdown calls
        if self.is_shutting_down:
            return
            
        self.is_shutting_down = True
        
        try:
            self.logger.info("Shutting down application")
            
            # Stop core services first
            if self.core:
                try:
                    self.core.stop()
                    self.logger.info("Core services stopped successfully")
                except Exception as e:
                    self.logger.log_exception(e, "Core shutdown error")
            
            # Handle GUI cleanup with specific exception handling
            if self.root:
                try:
                    # First try to quit the mainloop
                    self.root.quit()
                    self.logger.info("GUI mainloop quit successfully")
                except (_tkinter.TclError, tk.TclError, RuntimeError) as e:
                    # Handle specific GUI cleanup errors
                    if "application has been destroyed" in str(e).lower():
                        self.logger.info("GUI already destroyed - cleanup complete")
                    else:
                        raise UIError(f"GUI quit error: {str(e)}", "GUI_QUIT_FAILED")
                
                try:
                    # Then try to destroy the window
                    self.root.destroy()
                    self.logger.info("GUI window destroyed successfully")
                except (_tkinter.TclError, tk.TclError, RuntimeError) as e:
                    # Handle specific destruction errors
                    if "application has been destroyed" in str(e).lower():
                        self.logger.info("GUI window already destroyed - cleanup complete")
                    elif "can't invoke \"destroy\" command" in str(e).lower():
                        self.logger.info("GUI destroy command unavailable - window already cleaned up")
                    else:
                        raise UIError(f"GUI destroy error: {str(e)}", "GUI_DESTROY_FAILED")
            
            self.logger.info("Application shutdown completed successfully")
            
        except UIError as e:
            # Handle our custom UI errors with appropriate logging
            self.logger.error(f"UI shutdown error: {e.message} (Code: {e.error_code})")
        except ThinkerAIException as e:
            # Handle other custom exceptions
            self.logger.error(f"Application shutdown error: {e.message}")
        except Exception as e:
            # Handle any other unexpected errors
            self.logger.log_exception(e, "Unexpected shutdown error")
        finally:
            # Ensure shutdown flag remains set regardless of errors
            self.is_shutting_down = True 
