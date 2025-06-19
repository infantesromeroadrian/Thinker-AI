#!/usr/bin/env python3
"""
Main entry point for Thinker AI Auxiliary Window.

Professional auxiliary window for AI assistance, cybersecurity tools, and ethical hacking education.
This module serves as the main entry point following clean architecture principles.

Usage:
    python -m src.main                      # Run in development mode
    python -m src.main --production         # Run in production mode
    python -m src.main --help               # Show help information

Authors:
    AI Assistant & Human Orchestrator

Version:
    1.0.0
"""

# Standard library imports
import argparse
import os
import sys
from pathlib import Path
from typing import NoReturn, Optional

# Add src directory to Python path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Local application imports
from src.config.config import get_config
from src.core.app_core import get_core
from src.gui.main_window import ThinkerMainWindow
from src.utils.logger import get_logger
from src.utils.helpers import Performance


class ApplicationError(Exception):
    """Base exception for application-level errors."""
    pass


class DependencyError(ApplicationError):
    """Raised when required dependencies are missing."""
    pass


class InitializationError(ApplicationError):
    """Raised when application initialization fails."""
    pass


def setup_environment(production_mode: bool = False) -> None:
    """
    Setup environment variables and configuration.
    
    Args:
        production_mode: Whether to run in production mode
        
    Raises:
        EnvironmentError: If environment setup fails
    """
    if production_mode:
        os.environ['THINKER_ENV'] = 'production'
    else:
        os.environ['THINKER_ENV'] = 'development'


def check_dependencies() -> bool:
    """
    Check if all required dependencies are available.
    
    Returns:
        True if all dependencies are available, False otherwise
        
    Raises:
        DependencyError: If critical dependencies are missing
    """
    required_modules = [
        'tkinter',
        'threading', 
        'pathlib',
        'datetime',
        'typing',
        'json',
        'logging'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        error_msg = f"Missing required modules: {', '.join(missing_modules)}"
        print(f"❌ {error_msg}")
        print("Please install the required dependencies.")
        raise DependencyError(error_msg)
    
    return True


def check_optional_dependencies() -> None:
    """Check optional dependencies and warn if missing."""
    optional_modules = {
        'psutil': 'Performance monitoring will be limited',
        'requests': 'Network features will be limited', 
        'cryptography': 'Enhanced security features will be unavailable'
    }
    
    for module, warning in optional_modules.items():
        try:
            __import__(module)
        except ImportError:
            print(f"⚠️  Optional module '{module}' not found: {warning}")


def print_banner() -> None:
    """Print application banner with configuration information."""
    config = get_config()
    banner = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          {config.APP_NAME}                        ║
║                               Version {config.APP_VERSION}                              ║
║                                                                              ║
║  Professional Auxiliary Window for AI Assistance & Cybersecurity Tools      ║
║  Developed by {config.AUTHOR}                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_system_info() -> None:
    """Print system information and configuration details."""
    config = get_config()
    
    print("🔧 System Information:")
    print(f"   • Python Version: {sys.version.split()[0]}")
    print(f"   • Platform: {sys.platform}")
    print(f"   • Environment: {os.getenv('THINKER_ENV', 'development')}")
    print(f"   • Config: {config.__class__.__name__}")
    print(f"   • Base Directory: {config.BASE_DIR}")
    print()


@Performance.time_function
def initialize_application(production_mode: bool = False) -> ThinkerMainWindow:
    """
    Initialize the main application with all required components.
    
    Args:
        production_mode: Whether to run in production mode
        
    Returns:
        Initialized main window instance
        
    Raises:
        InitializationError: If application initialization fails
    """
    try:
        # Initialize logger first
        logger = get_logger("Main")
        logger.info("Starting Thinker AI Auxiliary Window initialization")
        
        # Setup environment
        setup_environment(production_mode)
        logger.info(f"Environment configured: {'production' if production_mode else 'development'}")
        
        # Initialize core services
        core = get_core()
        if not core.start():
            raise InitializationError("Failed to start core services")
        logger.info("Core services initialized successfully")
        
        # Create main window
        main_window = ThinkerMainWindow()
        logger.info("Main window created successfully")
        
        return main_window
        
    except Exception as e:
        logger = get_logger("Main")
        logger.log_exception(e, "Application initialization")
        raise InitializationError(f"Application initialization failed: {e}") from e


def run_diagnostics():
    """Run comprehensive system diagnostics"""
    print("🔍 THINKER AI - DIAGNÓSTICO COMPLETO DEL SISTEMA")
    print("=" * 60)
    
    try:
        # Initialize core
        core = get_core()
        
        # Run diagnostics
        diagnostics = core.run_comprehensive_diagnostics()
        
        # Display results
        print(f"\n📊 RESULTADOS DEL DIAGNÓSTICO")
        print(f"Timestamp: {diagnostics['timestamp']}")
        
        # Configuration validation
        config_validation = diagnostics.get('configuration_validation', {})
        print(f"\n🔧 VALIDACIÓN DE CONFIGURACIÓN:")
        print(f"  Válida: {'✅' if config_validation.get('valid', False) else '❌'}")
        
        if config_validation.get('issues'):
            print(f"  Problemas:")
            for issue in config_validation['issues']:
                print(f"    - {issue}")
        
        if config_validation.get('suggestions'):
            print(f"  Sugerencias:")
            for suggestion in config_validation['suggestions']:
                print(f"    - {suggestion}")
        
        # Connectivity tests
        connectivity = diagnostics.get('connectivity_tests', {})
        print(f"\n🌐 PRUEBAS DE CONECTIVIDAD:")
        
        for service_name, status in connectivity.items():
            service_status = status.get('status', 'unknown')
            print(f"  {service_name}: {service_status}")
            
            if 'diagnostics' in status:
                diag = status['diagnostics']
                print(f"    DNS: {diag.get('dns_resolution', 'unknown')}")
                print(f"    TCP: {diag.get('tcp_connection', 'unknown')}")
                print(f"    HTTP: {diag.get('http_response', 'unknown')}")
                print(f"    API: {diag.get('api_compatibility', 'unknown')}")
        
        # Module health
        module_health = diagnostics.get('module_health', {})
        print(f"\n🔧 SALUD DE MÓDULOS:")
        
        for module_name, health in module_health.items():
            status = health.get('status', 'unknown')
            print(f"  {module_name}: {status}")
            
            if 'error' in health:
                print(f"    Error: {health['error']}")
        
        # Recommendations
        recommendations = diagnostics.get('recommendations', [])
        print(f"\n💡 RECOMENDACIONES:")
        
        for rec in recommendations:
            print(f"  {rec}")
        
        # Network alternatives
        config = get_config()
        alternatives = config.get_network_alternatives()
        
        if alternatives:
            print(f"\n🔄 ALTERNATIVAS DE RED DISPONIBLES:")
            for i, alt in enumerate(alternatives[:5], 1):
                print(f"  {i}. {alt}")
        
        print(f"\n{'=' * 60}")
        print(f"🏁 Diagnóstico completado.")
        
    except Exception as e:
        print(f"❌ Error durante el diagnóstico: {str(e)}")
        return 1
    
    return 0


def main() -> int:
    """
    Main application entry point with comprehensive error handling.
    
    Returns:
        Exit code (0 for success, non-zero for error)
        
    Raises:
        SystemExit: On application completion or critical error
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Thinker AI Auxiliary Window - Professional AI & Security Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m src.main                    # Run in development mode
    python -m src.main --production       # Run in production mode
    python -m src.main --version          # Show version information
    python -m src.main --diagnose         # Run system diagnostics
    python -m src.main --config-test      # Test configuration and exit

For more information, visit the project documentation.
        """
    )
    
    parser.add_argument(
        '--production', 
        action='store_true',
        help='Run in production mode with optimized settings'
    )
    
    parser.add_argument(
        '--version', 
        action='version',
        version='%(prog)s 1.0.0'
    )
    
    parser.add_argument(
        '--no-banner', 
        action='store_true',
        help='Skip the startup banner'
    )
    
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Enable debug logging'
    )
    
    parser.add_argument(
        '--diagnose', 
        action='store_true',
        help='Run system diagnostics instead of starting GUI'
    )
    
    parser.add_argument(
        '--config-test', 
        action='store_true',
        help='Test configuration and exit'
    )
    
    args = parser.parse_args()
    
    try:
        # Print banner unless disabled
        if not args.no_banner:
            print_banner()
        
        # Check dependencies
        print("🔍 Checking dependencies...")
        check_dependencies()
        
        # Check optional dependencies
        check_optional_dependencies()
        
        # Print system information
        print_system_info()
        
        if args.diagnose:
            return run_diagnostics()
        
        if args.config_test:
            print("🧪 TESTING CONFIGURATION...")
            try:
                config = get_config()
                validation = config.validate_qwen_configuration()
                
                if validation['valid']:
                    print("✅ Configuration is valid")
                else:
                    print("❌ Configuration has issues:")
                    for issue in validation['issues']:
                        print(f"  - {issue}")
                
                return 0 if validation['valid'] else 1
                
            except Exception as e:
                print(f"❌ Configuration test failed: {str(e)}")
                return 1
        
        # Initialize application
        print("🚀 Initializing Thinker AI Auxiliary Window...")
        main_window = initialize_application(args.production)
        
        print("✅ Initialization complete. Starting application...")
        logger = get_logger("Main")
        logger.info("Application loop starting")
        
        # Run the application
        main_window.run()
        
        # Cleanup
        core = get_core()
        core.stop()
        logger.info("Application shutdown completed")
        print("👋 Thank you for using Thinker AI Auxiliary Window!")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n🛑 Application interrupted by user")
        logger = get_logger("Main")
        logger.info("Application interrupted by user (Ctrl+C)")
        return 130  # Standard exit code for Ctrl+C
        
    except DependencyError as e:
        print(f"\n❌ Dependency error: {e}")
        return 1
        
    except InitializationError as e:
        print(f"\n❌ Initialization error: {e}")
        return 1
        
    except Exception as e:
        print(f"\n❌ Fatal error occurred: {e}")
        
        # Try to log the error if logger is available
        try:
            logger = get_logger("Main")
            logger.log_exception(e, "Fatal application error")
        except Exception:
            # If logging fails, print traceback
            import traceback
            traceback.print_exc()
        
        return 1


def run_application() -> NoReturn:
    """
    Entry point for running the application and exiting.
    
    Raises:
        SystemExit: Always exits with the appropriate code
    """
    exit_code = main()
    sys.exit(exit_code)


if __name__ == "__main__":
    """Entry point when script is run directly."""
    run_application() 