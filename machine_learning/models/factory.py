"""
Model Factory
Factory class for creating model instances
"""
from typing import Dict, Any, Optional, List, Union, Tuple
import importlib
import os
import sys
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

from .base import BaseModel

class ModelFactory:
    """Factory class for creating model instances"""
    
    _registry = {}
    
    @classmethod
    def register(cls, name: str):
        """
        Register a model class
        
        Args:
            name: Model name
            
        Returns:
            Decorator function
        """
        def decorator(model_class):
            cls._registry[name] = model_class
            return model_class
        return decorator
    
    @classmethod
    def create_model(cls, model_name: str, config: Dict[str, Any] = None) -> BaseModel:
        """
        Create a model instance
        
        Args:
            model_name: Name of model to create
            config: Configuration dictionary
            
        Returns:
            BaseModel: Model instance
            
        Raises:
            ValueError: If model name is not registered
        """
        if model_name not in cls._registry:
            # Try to import the model module
            try:
                # Discover and import all model modules
                cls._discover_models()
            except ImportError:
                pass
                
            # Check registry again after import attempt
            if model_name not in cls._registry:
                raise ValueError(f"Model '{model_name}' not registered. Available models: {list(cls._registry.keys())}")
        
        # Create instance with config or empty dict
        config = config or {}
        return cls._registry[model_name](config)
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """
        Get list of available model names
        
        Returns:
            List[str]: List of registered model names
        """
        # Try to import all model modules from models directory
        cls._discover_models()
        return list(cls._registry.keys())
    
    @classmethod
    def _discover_models(cls) -> None:
        """Dynamically discover and import all Python modules in the models directory"""
        # Get the models directory path
        models_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Import all .py files (excluding __init__.py and factory itself)
        for filename in os.listdir(models_dir):
            if filename.endswith('.py') and not filename.startswith('__') and filename != 'factory.py':
                # Remove .py extension
                module_name = filename[:-3]
                try:
                    importlib.import_module(f"machine_learning.models.{module_name}")
                    print(f"Successfully imported model module: {module_name}")
                except ImportError as e:
                    print(f"Warning: Failed to import {module_name}: {e}") 