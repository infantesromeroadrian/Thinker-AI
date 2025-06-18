# 🧠 Thinker AI - Auxiliary Window

**Professional Auxiliary Window for AI Assistance with Qwen2.5-7B-Instruct-1M Integration, Cybersecurity Tools, and Ethical Hacking Education**

## 📋 Overview

Thinker AI Auxiliary Window is a comprehensive desktop application built with Python and CustomTkinter, designed to provide AI assistance using the efficient Qwen2.5-7B-Instruct-1M model, cybersecurity tools, and ethical hacking capabilities for educational and professional use. The application follows strict modular programming principles and PEP 8 standards.

## 🤖 AI Model Integration

### Qwen2.5-7B-Instruct-1M Local Server
- **Model**: qwen2.5-7b-instruct-1m (7 billion parameters with 1M context)
- **Server**: Local API server at `http://172.29.208.1:1234`
- **Features**: Fast conversational AI in Spanish, code analysis, technical assistance
- **Performance**: Ultra-fast local inference with excellent response quality and extended context
- **Streaming**: Real-time response generation for better user experience

## ✨ Features

### 🤖 AI Assistant (Qwen2.5-7B-Instruct-1M)
- **Interactive Chat Interface**: Real-time conversation with Qwen2.5-7B-Instruct-1M model
- **Streaming Responses**: Real-time text generation for immediate feedback
- **Spanish Conversation**: Native Spanish language support with fast responses
- **Voice Dictation**: 🎤 Speech-to-text with multiple recognition engines (Google, Whisper, Azure)
- **Code Analysis**: Automated code review and architectural suggestions
- **Technical Assistance**: Programming help, debugging, and best practices
- **Extended Context**: 1M token context for handling long conversations and documents

### 🔒 Cybersecurity Tools
- **Network Scanner**: Professional network scanning and discovery
- **Vulnerability Assessment**: Security vulnerability checking
- **Security Event Logging**: Comprehensive security event tracking

### 🛠️ System Tools
- **Performance Monitoring**: Real-time system performance metrics
- **System Status Dashboard**: Comprehensive system health overview
- **Utility Functions**: File management, data export, and system utilities

### 🎨 User Interface (CustomTkinter)
- **Modern Design**: Professional dark theme using CustomTkinter
- **Chat-Focused Interface**: Streamlined conversation experience
- **Responsive Layout**: Adaptive design with rounded corners and modern styling
- **Professional Aesthetics**: GitHub-style dark theme with green accents

## 🏗️ Architecture

The application follows a **strict modular architecture** with clear separation of concerns:

```
Thinker-AI-Aux/
├── src/
│   ├── config/          # Configuration management
│   │   ├── __init__.py
│   │   └── config.py    # Central configuration with Qwen settings
│   ├── core/            # Core business logic
│   │   ├── __init__.py
│   │   └── app_core.py  # Main application core with AI integration
│   ├── gui/             # User interface components (CustomTkinter)
│   │   ├── __init__.py
│   │   └── main_window.py # Modern chat interface
│   ├── services/        # External service integrations
│   │   ├── __init__.py
│   │   └── qwen_service.py # Qwen3-32B API service
│   └── utils/           # Utility functions
│       ├── __init__.py
│       ├── logger.py    # Professional logging
│       └── helpers.py   # Helper utilities
├── main.py              # Application entry point
├── requirements.txt     # Dependencies (includes requests, customtkinter)
└── README.md           # This file
```

### 🔧 Core Components

1. **Configuration Module** (`src/config/`)
   - Centralized configuration management
   - Environment-specific settings
   - Feature flags for enabling/disabling functionality

2. **Core Module** (`src/core/`)
   - Business logic and application state management
   - AI and security module orchestration
   - Performance monitoring and metrics

3. **GUI Module** (`src/gui/`)
   - Professional tkinter interface
   - Tabbed layout with dashboard, AI assistant, security tools, and utilities
   - Event handling and user interaction management

4. **Utils Module** (`src/utils/`)
   - Professional logging with rotation and multiple handlers
   - Helper functions for file management, UI operations, and security
   - Performance monitoring and system utilities

## ⚙️ Qwen2.5-7B-Instruct-1M Setup

### Prerequisites for AI Features

1. **LM Studio or Compatible Server**:
   - Install LM Studio or similar local AI server
   - Download the `qwen2.5-7b-instruct-1m` model
   - Configure server to run on `http://172.29.208.1:1234`

2. **Server Configuration**:
   ```bash
   # Start your AI server with Qwen2.5-7B-Instruct-1M model
   # Example with LM Studio:
   # 1. Open LM Studio
   # 2. Load qwen2.5-7b-instruct-1m model (much faster than 32B!)
   # 3. Start local server on port 1234
   # 4. Ensure server is accessible at 172.29.208.1:1234
   ```

3. **Network Configuration**:
   - Ensure your AI server is accessible from the application
   - Configure firewall if necessary
   - Test connection: `curl http://172.29.208.1:1234/v1/models`

### Custom Server Configuration

To use a different server or model, modify `src/config/config.py`:

```python
# Qwen2.5-7B-Instruct-1M Model Configuration
QWEN_BASE_URL = "http://your-server:port"
QWEN_MODEL_NAME = "qwen2.5-7b-instruct-1m"  # or your custom model
QWEN_TIMEOUT = 30
QWEN_MAX_RETRIES = 3
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** (recommended: Python 3.10+)
- **CustomTkinter support** (included in requirements)
- **Windows, macOS, or Linux**
- **Local AI Server** running Qwen3-32B model

### Installation

1. **Clone or download the project**:
   ```bash
   git clone <repository-url>
   cd Thinker-AI-Aux
   ```

2. **Install dependencies** (optional but recommended):
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python main.py
   ```

### Command Line Options

```bash
# Development mode (default)
python main.py

# Production mode
python main.py --production

# Skip startup banner
python main.py --no-banner

# Enable debug logging
python main.py --debug

# Show version information
python main.py --version

# Show help
python main.py --help
```

## 💻 Usage

### 🤖 Modern AI Chat Interface
- **Conversational AI**: Direct conversation with Qwen2.5-7B-Instruct-1M model
- **Real-time Streaming**: Watch responses appear as they're generated
- **Spanish Language**: Native Spanish conversation support with fast responses
- **Real-time Status**: Connection indicator for AI server
- **Message History**: Complete conversation history with timestamps
- **Character Counter**: Real-time character count for messages
- **Extended Context**: Handle long conversations up to 1M tokens
- **Keyboard Shortcuts**: 
  - `Enter`: Send message
  - `Ctrl+Enter`: New line
  - `Ctrl+L`: Clear chat

### 🎤 Voice Dictation System
- **One-Click Recording**: Red microphone button for instant voice capture
- **Multi-Engine Support**: Google Speech, OpenAI Whisper, Azure Speech Services
- **Spanish Recognition**: Optimized for Spanish language dictation (es-ES)
- **Visual Feedback**: Recording indicators and real-time status updates
- **Smart Integration**: Recognized text appears directly in chat input field
- **Test & Diagnostics**: Built-in microphone and recognition testing tools
- **Voice Control Menu**: System ⚙️ > 🎤 Test Voz for full diagnostics

### 📊 System Status Monitoring
- **AI Server Status**: Real-time Qwen2.5-7B server monitoring
- **Response Time**: Performance metrics for AI requests (typically 5-30s)
- **System Health**: CPU, memory, and application status
- **Session Information**: Unique session tracking

### 🗂️ Menu Features
- **Chat Menu**: Export conversations, clear chat, exit application
- **View Menu**: System status, Qwen2.5-7B connection test, preferences (coming soon)
- **Help Menu**: Documentation, keyboard shortcuts, about dialog

## 🔧 Configuration

The application uses a hierarchical configuration system:

### Environment Variables
- `THINKER_ENV`: Set to `production` or `development`

### Configuration Files
- Automatic directory creation for logs and data
- JSON-based state persistence
- Configurable logging levels and output formats

### Feature Flags
Enable or disable specific functionality:
- AI Assistant Chat
- Code Analysis
- Security Scanner
- Performance Monitoring
- And more...

## 📊 Logging and Monitoring

### Professional Logging
- **Rotating File Logs**: Automatic log rotation (10MB, 5 backups)
- **Multiple Log Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Structured Logging**: Detailed function calls, performance metrics, and security events
- **Console and File Output**: Dual output for development and production

### Performance Monitoring
- **Real-time Metrics**: CPU usage, memory consumption, active threads
- **Performance Timing**: Automatic function execution timing
- **System Health**: Comprehensive system status reporting

### Security Logging
- **Security Events**: Dedicated security event logging
- **Audit Trail**: User action tracking for compliance
- **Threat Detection**: Automated security event classification

## 🔐 Security Features

### Built-in Security
- **Input Validation**: Comprehensive input sanitization
- **Secure File Operations**: Safe file handling with error recovery
- **Session Management**: Unique session identification and tracking
- **Admin Privilege Detection**: Windows admin privilege checking

### Ethical Hacking Tools
- **Network Discovery**: Professional network scanning capabilities
- **Vulnerability Assessment**: Security weakness identification
- **Educational Focus**: Tools designed for learning and authorized testing

## 🧪 Development

### Code Quality
- **PEP 8 Compliance**: Strict adherence to Python style guidelines
- **Modular Design**: Clear separation of concerns and responsibilities
- **Type Hints**: Comprehensive type annotations for better code quality
- **Comprehensive Documentation**: Extensive inline documentation

### Testing
- **Error Handling**: Robust exception handling throughout the application
- **Logging Integration**: Detailed error logging for debugging
- **Graceful Degradation**: Fallback behavior for missing dependencies

## 📚 Educational Use

This application is designed for:
- **AI and Machine Learning Education**: Hands-on experience with AI tools
- **Cybersecurity Training**: Safe, controlled environment for security testing
- **Python Programming Learning**: Example of professional Python application development
- **Software Architecture Studies**: Demonstration of modular design principles

## ⚖️ Legal and Ethical Considerations

### Important Disclaimers
- **Educational Purpose**: This tool is designed for educational and authorized testing only
- **Legal Compliance**: Users must ensure compliance with local laws and regulations
- **Authorized Testing**: Only use security tools on networks and systems you own or have explicit permission to test
- **No Warranty**: This software is provided "as is" without warranty of any kind

### Responsible Use
- Always obtain proper authorization before using security scanning tools
- Respect privacy and confidentiality of any data encountered
- Use the tool in controlled, safe environments
- Report any security vulnerabilities responsibly

## 🤝 Contributing

Contributions are welcome! Please ensure:
- **Code Quality**: Follow PEP 8 and existing code patterns
- **Documentation**: Update documentation for any new features
- **Testing**: Test thoroughly before submitting changes
- **Security**: Consider security implications of any changes

## 📝 License

This project is created for educational and professional development purposes. Please ensure compliance with applicable laws and regulations when using security-related features.

## 📞 Support

For questions, issues, or contributions:
- Review the code documentation for implementation details
- Check the logging output for troubleshooting information
- Ensure all dependencies are properly installed
- **Voice Dictation**: See `VOICE_DICTATION_GUIDE.md` for complete setup and usage guide

## 🎯 Future Enhancements

Planned features include:
- **Enhanced AI Integration**: Additional AI/ML capabilities
- **Advanced Security Tools**: More cybersecurity utilities
- **Plugin System**: Extensible architecture for custom modules
- **Network Features**: Enhanced networking capabilities
- **Database Integration**: Data persistence and analysis

---

**Developed with ❤️ by AI Assistant & Human Orchestrator**

*Professional auxiliary window providing AI assistance, cybersecurity tools, and ethical hacking capabilities for educational and professional use.* 