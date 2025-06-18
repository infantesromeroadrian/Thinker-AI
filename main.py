#!/usr/bin/env python3
"""
Legacy entry point for Thinker AI Auxiliary Window.

This file is kept for backward compatibility. The main application
has been moved to src/main.py following clean architecture principles.

Usage:
    python main.py                          # Run in development mode
    python main.py --production             # Run in production mode
    python main.py --help                   # Show help information

Note:
    Consider using: python -m src.main
"""

import sys
from pathlib import Path

# Add src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import and delegate to the actual main module
from src.main import run_application


if __name__ == "__main__":
    """Entry point when script is run directly - delegates to actual main."""
    run_application() 