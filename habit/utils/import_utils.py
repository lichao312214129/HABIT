"""
Import utilities for robust module loading and error handling.

This module provides utilities for safely importing modules and classes
with proper error handling and status tracking.
"""

import warnings
from typing import Dict, Any, Optional, List, Tuple
import importlib
import sys


class ImportManager:
    """
    A utility class for managing imports with error handling and status tracking.
    
    This class provides methods to safely import modules and classes,
    track import errors, and check availability of imported components.
    """
    
    def __init__(self):
        """Initialize the ImportManager with empty tracking dictionaries."""
        self._import_errors: Dict[str, str] = {}
        self._available_modules: Dict[str, Any] = {}
        self._import_warnings: List[str] = []
    
    def safe_import(self, module_path: str, class_name: Optional[str] = None, 
                   alias: Optional[str] = None) -> Optional[Any]:
        """
        Safely import a module or class with error handling.
        
        Args:
            module_path (str): The full path to the module (e.g., 'numpy.random')
            class_name (str, optional): Specific class to import from the module
            alias (str, optional): Alternative name for the imported object
            
        Returns:
            Any: The imported module or class, or None if import failed
        """
        try:
            if class_name:
                # Import specific class from module
                module = importlib.import_module(module_path)
                imported_object = getattr(module, class_name)
                key = alias or class_name
            else:
                # Import entire module
                imported_object = importlib.import_module(module_path)
                key = alias or module_path.split('.')[-1]
            
            self._available_modules[key] = imported_object
            return imported_object
            
        except ImportError as e:
            error_msg = f"Failed to import {module_path}"
            if class_name:
                error_msg += f".{class_name}"
            error_msg += f": {str(e)}"
            
            # Determine the key for error tracking
            error_key = alias or class_name or module_path.split('.')[-1]
            self._import_errors[error_key] = error_msg
            self._import_warnings.append(error_msg)
            return None
            
        except AttributeError as e:
            error_msg = f"Class '{class_name}' not found in module '{module_path}': {str(e)}"
            # Determine the key for error tracking
            error_key = alias or class_name or module_path.split('.')[-1]
            self._import_errors[error_key] = error_msg
            self._import_warnings.append(error_msg)
            return None
    
    def safe_import_multiple(self, imports: List[Tuple[str, Optional[str], Optional[str]]]) -> Dict[str, Any]:
        """
        Safely import multiple modules or classes at once.
        
        Args:
            imports (List[Tuple]): List of tuples containing (module_path, class_name, alias)
            
        Returns:
            Dict[str, Any]: Dictionary of successfully imported objects
        """
        results = {}
        for module_path, class_name, alias in imports:
            imported = self.safe_import(module_path, class_name, alias)
            if imported is not None:
                key = alias or (class_name or module_path.split('.')[-1])
                results[key] = imported
        return results
    
    def get_import_errors(self) -> Dict[str, str]:
        """
        Get dictionary of import errors that occurred.
        
        Returns:
            Dict[str, str]: Dictionary mapping module/class names to error messages
        """
        return self._import_errors.copy()
    
    def get_available_modules(self) -> Dict[str, Any]:
        """
        Get dictionary of successfully imported modules/classes.
        
        Returns:
            Dict[str, Any]: Dictionary mapping names to imported objects
        """
        return self._available_modules.copy()
    
    def is_available(self, name: str) -> bool:
        """
        Check if a specific module or class is available.
        
        Args:
            name (str): Name of the module or class to check
            
        Returns:
            bool: True if available, False otherwise
        """
        return name in self._available_modules
    
    def get_import_warnings(self) -> List[str]:
        """
        Get list of import warning messages.
        
        Returns:
            List[str]: List of warning messages
        """
        return self._import_warnings.copy()
    
    def print_import_status(self, verbose: bool = False) -> None:
        """
        Print the current import status.
        
        Args:
            verbose (bool): If True, print detailed information about all imports
        """
        print(f"Import Status:")
        print(f"  Available modules/classes: {len(self._available_modules)}")
        print(f"  Failed imports: {len(self._import_errors)}")
        
        if verbose:
            if self._available_modules:
                print("\nSuccessfully imported:")
                for name, obj in self._available_modules.items():
                    print(f"  - {name}: {type(obj).__name__}")
            
            if self._import_errors:
                print("\nFailed imports:")
                for name, error in self._import_errors.items():
                    print(f"  - {name}: {error}")
    
    def raise_warnings(self) -> None:
        """
        Raise warnings for all import errors that occurred.
        """
        if self._import_warnings:
            for warning in self._import_warnings:
                warnings.warn(warning, ImportWarning)


def safe_import_decorator(module_path: str, class_name: Optional[str] = None, 
                         alias: Optional[str] = None, default_value: Any = None):
    """
    Decorator for safely importing modules or classes.
    
    Args:
        module_path (str): The full path to the module
        class_name (str, optional): Specific class to import
        alias (str, optional): Alternative name for the imported object
        default_value (Any): Default value to return if import fails
        
    Returns:
        Any: The imported object or default_value
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                if class_name:
                    module = importlib.import_module(module_path)
                    imported_object = getattr(module, class_name)
                else:
                    imported_object = importlib.import_module(module_path)
                
                return func(imported_object, *args, **kwargs)
            except (ImportError, AttributeError) as e:
                warnings.warn(f"Import failed for {module_path}: {str(e)}", ImportWarning)
                return default_value
        return wrapper
    return decorator


def check_dependencies(required_modules: List[str], optional_modules: Optional[List[str]] = None) -> Dict[str, bool]:
    """
    Check availability of required and optional dependencies.
    
    Args:
        required_modules (List[str]): List of required module names
        optional_modules (List[str], optional): List of optional module names
        
    Returns:
        Dict[str, bool]: Dictionary mapping module names to availability status
    """
    if optional_modules is None:
        optional_modules = []
    
    status = {}
    
    # Check required modules
    for module in required_modules:
        try:
            importlib.import_module(module)
            status[module] = True
        except ImportError:
            status[module] = False
            warnings.warn(f"Required module '{module}' is not available", ImportWarning)
    
    # Check optional modules
    for module in optional_modules:
        try:
            importlib.import_module(module)
            status[module] = True
        except ImportError:
            status[module] = False
    
    return status


def get_module_info(module_name: str) -> Dict[str, Any]:
    """
    Get information about a module including version and availability.
    
    Args:
        module_name (str): Name of the module to check
        
    Returns:
        Dict[str, Any]: Dictionary containing module information
    """
    info = {
        'available': False,
        'version': None,
        'path': None,
        'error': None
    }
    
    try:
        module = importlib.import_module(module_name)
        info['available'] = True
        info['path'] = getattr(module, '__file__', None)
        
        # Try to get version
        try:
            info['version'] = getattr(module, '__version__', None)
        except:
            pass
            
    except ImportError as e:
        info['error'] = str(e)
    
    return info 