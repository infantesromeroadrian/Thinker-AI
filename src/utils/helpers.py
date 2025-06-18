"""
Helper utilities module for Thinker AI Auxiliary Window
Contains common utility functions used across the application
"""

import json
import time
import threading
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union
import tkinter as tk
from tkinter import messagebox, filedialog

from src.utils.logger import get_logger


class Performance:
    """Performance monitoring and timing utilities"""
    
    @staticmethod
    def time_function(func: Callable) -> Callable:
        """Decorator to time function execution"""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            logger = get_logger()
            logger.log_performance(
                operation=func.__name__,
                duration=duration,
                details=f"Args: {len(args)}, Kwargs: {len(kwargs)}"
            )
            return result
        return wrapper
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get basic system information"""
        import platform
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "architecture": platform.architecture(),
            "processor": platform.processor(),
            "timestamp": datetime.now().isoformat()
        }


class FileManager:
    """File and directory management utilities"""
    
    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> Path:
        """Ensure directory exists, create if it doesn't"""
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        return path_obj
    
    @staticmethod
    def safe_write_json(file_path: Union[str, Path], data: Dict[str, Any]) -> bool:
        """Safely write JSON data to file with error handling"""
        try:
            path_obj = Path(file_path)
            FileManager.ensure_directory(path_obj.parent)
            
            with open(path_obj, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            get_logger().debug(f"JSON data written to {file_path}")
            return True
            
        except Exception as e:
            get_logger().log_exception(e, f"Writing JSON to {file_path}")
            return False
    
    @staticmethod
    def safe_read_json(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Safely read JSON data from file with error handling"""
        try:
            path_obj = Path(file_path)
            if not path_obj.exists():
                get_logger().warning(f"JSON file not found: {file_path}")
                return None
            
            with open(path_obj, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            get_logger().debug(f"JSON data read from {file_path}")
            return data
            
        except Exception as e:
            get_logger().log_exception(e, f"Reading JSON from {file_path}")
            return None
    
    @staticmethod
    def get_file_size_formatted(file_path: Union[str, Path]) -> str:
        """Get file size in human-readable format"""
        try:
            size_bytes = Path(file_path).stat().st_size
            
            if size_bytes == 0:
                return "0 B"
            
            size_names = ["B", "KB", "MB", "GB", "TB"]
            i = 0
            while size_bytes >= 1024 and i < len(size_names) - 1:
                size_bytes /= 1024.0
                i += 1
            
            return f"{size_bytes:.1f} {size_names[i]}"
            
        except Exception:
            return "Unknown"


class UIHelpers:
    """UI-related helper functions"""
    
    @staticmethod
    def center_window(window: tk.Tk, width: int, height: int) -> None:
        """Center a window on the screen"""
        # Get screen dimensions
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        
        # Calculate position
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        
        # Set window geometry
        window.geometry(f"{width}x{height}+{x}+{y}")
        
        get_logger().debug(f"Window centered: {width}x{height} at ({x}, {y})")
    
    @staticmethod
    def show_info_dialog(parent: tk.Widget, title: str, message: str) -> None:
        """Show information dialog"""
        messagebox.showinfo(title, message, parent=parent)
        get_logger().log_user_action("Info Dialog", f"Title: {title}")
    
    @staticmethod
    def show_error_dialog(parent: tk.Widget, title: str, message: str) -> None:
        """Show error dialog"""
        messagebox.showerror(title, message, parent=parent)
        get_logger().log_user_action("Error Dialog", f"Title: {title}")
    
    @staticmethod
    def show_warning_dialog(parent: tk.Widget, title: str, message: str) -> bool:
        """Show warning dialog with OK/Cancel options"""
        result = messagebox.askokcancel(title, message, parent=parent)
        get_logger().log_user_action("Warning Dialog", f"Title: {title}, Result: {result}")
        return result
    
    @staticmethod
    def ask_yes_no(parent: tk.Widget, title: str, question: str) -> bool:
        """Show yes/no question dialog"""
        result = messagebox.askyesno(title, question, parent=parent)
        get_logger().log_user_action("Yes/No Dialog", f"Title: {title}, Result: {result}")
        return result
    
    @staticmethod
    def select_file(parent: tk.Widget, title: str = "Select File", 
                   filetypes: List[tuple] = None) -> Optional[str]:
        """Show file selection dialog"""
        if filetypes is None:
            filetypes = [("All files", "*.*")]
        
        filename = filedialog.askopenfilename(
            parent=parent,
            title=title,
            filetypes=filetypes
        )
        
        if filename:
            get_logger().log_user_action("File Selected", f"File: {filename}")
        
        return filename if filename else None
    
    @staticmethod
    def select_directory(parent: tk.Widget, title: str = "Select Directory") -> Optional[str]:
        """Show directory selection dialog"""
        directory = filedialog.askdirectory(parent=parent, title=title)
        
        if directory:
            get_logger().log_user_action("Directory Selected", f"Directory: {directory}")
        
        return directory if directory else None
    
    @staticmethod
    def save_file(parent: tk.Widget, title: str = "Save File", 
                 default_extension: str = ".txt", 
                 filetypes: List[tuple] = None) -> Optional[str]:
        """Show save file dialog"""
        if filetypes is None:
            filetypes = [("Text files", "*.txt"), ("All files", "*.*")]
        
        filename = filedialog.asksaveasfilename(
            parent=parent,
            title=title,
            defaultextension=default_extension,
            filetypes=filetypes
        )
        
        if filename:
            get_logger().log_user_action("File Save Dialog", f"File: {filename}")
        
        return filename if filename else None


class ThreadingHelpers:
    """Threading and async operation utilities"""
    
    @staticmethod
    def run_in_background(func: Callable, callback: Optional[Callable] = None, 
                         error_callback: Optional[Callable] = None) -> threading.Thread:
        """Run function in background thread with optional callbacks"""
        def wrapper():
            try:
                result = func()
                if callback:
                    callback(result)
            except Exception as e:
                get_logger().log_exception(e, f"Background thread: {func.__name__}")
                if error_callback:
                    error_callback(e)
        
        thread = threading.Thread(target=wrapper, daemon=True)
        thread.start()
        get_logger().debug(f"Background thread started for {func.__name__}")
        return thread
    
    @staticmethod
    def delayed_execution(delay_seconds: float, func: Callable) -> threading.Timer:
        """Execute function after a delay"""
        def wrapper():
            try:
                func()
            except Exception as e:
                get_logger().log_exception(e, f"Delayed execution: {func.__name__}")
        
        timer = threading.Timer(delay_seconds, wrapper)
        timer.start()
        get_logger().debug(f"Delayed execution scheduled: {func.__name__} in {delay_seconds}s")
        return timer


class ValidationHelpers:
    """Data validation utilities"""
    
    @staticmethod
    def is_valid_email(email: str) -> bool:
        """Basic email validation"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def is_valid_ip(ip: str) -> bool:
        """Validate IP address format"""
        import ipaddress
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def is_valid_port(port: Union[str, int]) -> bool:
        """Validate port number"""
        try:
            port_int = int(port)
            return 1 <= port_int <= 65535
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe file operations"""
        import re
        # Remove or replace invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Remove leading/trailing whitespace and dots
        filename = filename.strip('. ')
        # Ensure it's not empty
        return filename if filename else "unnamed_file"


class SecurityHelpers:
    """Security-related utilities"""
    
    @staticmethod
    def hash_string(text: str, algorithm: str = "sha256") -> str:
        """Hash string using specified algorithm"""
        import hashlib
        
        hash_func = getattr(hashlib, algorithm, None)
        if not hash_func:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
        return hash_func(text.encode('utf-8')).hexdigest()
    
    @staticmethod
    def generate_session_id() -> str:
        """Generate unique session ID"""
        import uuid
        return str(uuid.uuid4())
    
    @staticmethod
    def check_admin_privileges() -> bool:
        """Check if running with admin privileges (Windows)"""
        try:
            import ctypes
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        except Exception:
            return False


# Convenience function for quick access to common operations
def quick_save_data(filename: str, data: Dict[str, Any]) -> bool:
    """Quick function to save data to JSON file"""
    return FileManager.safe_write_json(filename, data)


def quick_load_data(filename: str) -> Optional[Dict[str, Any]]:
    """Quick function to load data from JSON file"""
    return FileManager.safe_read_json(filename) 