"""
Test module for GUI shutdown functionality.

Tests the main window shutdown process to ensure it handles TclError correctly
and prevents multiple shutdown calls.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import Mock, patch, MagicMock
import _tkinter
import tkinter as tk
from typing import Dict, Any

from src.gui.main_window import ThinkerMainWindow
from src.exceptions import UIError, ThinkerAIException


class TestGUIShutdown:
    """Test suite for GUI shutdown functionality."""
    
    def test_shutdown_single_call(self) -> None:
        """Test that shutdown can be called only once."""
        window = ThinkerMainWindow()
        window.is_initialized = True
        window.is_shutting_down = False
        
        # Mock dependencies
        window.core = Mock()
        window.root = Mock()
        window.logger = Mock()
        
        # First call should execute shutdown
        window.shutdown()
        
        assert window.is_shutting_down is True
        window.core.stop.assert_called_once()
        
        # Second call should be ignored
        window.core.reset_mock()
        window.shutdown()
        
        # Core should not be called again
        window.core.stop.assert_not_called()
    
    def test_shutdown_handles_tcl_error_during_quit(self) -> None:
        """Test that shutdown handles TclError during root.quit()."""
        window = ThinkerMainWindow()
        window.is_initialized = True
        window.is_shutting_down = False
        
        # Mock dependencies
        window.core = Mock()
        window.logger = Mock()
        window.root = Mock()
        
        # Mock root.quit() to raise TclError
        window.root.quit.side_effect = _tkinter.TclError("application has been destroyed")
        
        # Should not raise exception
        window.shutdown()
        
        assert window.is_shutting_down is True
        window.logger.info.assert_any_call("GUI already destroyed - cleanup complete")
    
    def test_shutdown_handles_tcl_error_during_destroy(self) -> None:
        """Test that shutdown handles TclError during root.destroy()."""
        window = ThinkerMainWindow()
        window.is_initialized = True
        window.is_shutting_down = False
        
        # Mock dependencies
        window.core = Mock()
        window.logger = Mock()
        window.root = Mock()
        
        # Mock root.destroy() to raise TclError
        window.root.destroy.side_effect = _tkinter.TclError("can't invoke \"destroy\" command: application has been destroyed")
        
        # Should not raise exception
        window.shutdown()
        
        assert window.is_shutting_down is True
        window.logger.info.assert_any_call("GUI window already destroyed - cleanup complete")
    
    def test_shutdown_handles_core_error(self) -> None:
        """Test that shutdown handles errors during core.stop()."""
        window = ThinkerMainWindow()
        window.is_initialized = True
        window.is_shutting_down = False
        
        # Mock dependencies
        window.core = Mock()
        window.logger = Mock()
        window.root = Mock()
        
        # Mock core.stop() to raise exception
        window.core.stop.side_effect = Exception("Core shutdown error")
        
        # Should not raise exception and should continue with GUI cleanup
        window.shutdown()
        
        assert window.is_shutting_down is True
        window.logger.log_exception.assert_called()
        window.root.quit.assert_called_once()
    
    def test_on_window_close_prevents_multiple_calls(self) -> None:
        """Test that _on_window_close prevents multiple shutdown calls."""
        window = ThinkerMainWindow()
        window.is_initialized = True
        window.is_shutting_down = False
        window.logger = Mock()
        
        with patch('src.utils.helpers.UIHelpers.ask_yes_no', return_value=True):
            with patch.object(window, 'shutdown') as mock_shutdown:
                # First call should trigger shutdown
                window._on_window_close()
                mock_shutdown.assert_called_once()
                
                # Set flag to simulate shutdown in progress
                window.is_shutting_down = True
                mock_shutdown.reset_mock()
                
                # Second call should be ignored
                window._on_window_close()
                mock_shutdown.assert_not_called()
    
    def test_on_window_close_handles_tcl_error(self) -> None:
        """Test that _on_window_close handles TclError properly."""
        window = ThinkerMainWindow()
        window.is_initialized = True
        window.is_shutting_down = False
        window.logger = Mock()
        window.root = Mock()
        
        # Mock UIHelpers to raise TclError
        with patch('src.utils.helpers.UIHelpers.ask_yes_no', 
                  side_effect=_tkinter.TclError("application has been destroyed")):
            
            # Should not raise exception
            window._on_window_close()
            
            window.logger.info.assert_any_call("Application already destroyed during close event")
    
    def test_shutdown_with_no_root(self) -> None:
        """Test shutdown when root is None."""
        window = ThinkerMainWindow()
        window.is_initialized = True
        window.is_shutting_down = False
        window.logger = Mock()
        window.core = Mock()
        window.root = None
        
        # Should not raise exception
        window.shutdown()
        
        assert window.is_shutting_down is True
        window.core.stop.assert_called_once()
    
    def test_shutdown_ui_error_handling(self) -> None:
        """Test that UIError exceptions are handled properly."""
        window = ThinkerMainWindow()
        window.is_initialized = True
        window.is_shutting_down = False
        window.logger = Mock()
        window.core = Mock()
        window.root = Mock()
        
        # Mock root.quit() to raise non-standard TclError
        window.root.quit.side_effect = _tkinter.TclError("custom error")
        
        # Should handle as UIError
        window.shutdown()
        
        assert window.is_shutting_down is True
        window.logger.error.assert_called()
        # Check that error message contains UIError details
        error_calls = [call for call in window.logger.error.call_args_list 
                      if 'GUI_QUIT_FAILED' in str(call)]
        assert len(error_calls) > 0
    
    @pytest.mark.integration
    def test_full_shutdown_cycle(self) -> None:
        """Integration test for complete shutdown cycle."""
        # This test requires proper GUI environment
        # Skip if running in headless environment
        try:
            import tkinter
            root = tkinter.Tk()
            root.withdraw()  # Hide the window
            root.destroy()
        except Exception:
            pytest.skip("GUI environment not available")
        
        window = ThinkerMainWindow()
        
        # Initialize with minimal setup
        window.core = Mock()
        window.logger = Mock()
        
        # Mock successful initialization
        with patch.object(window, 'initialize', return_value=True):
            window.initialize()
        
        # Test shutdown
        window.shutdown()
        
        assert window.is_shutting_down is True 