# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text. Summary:
#
#   - Non-commercial use (academic, research, education, personal) is permitted
#     provided that copyright notices are retained and HABIT usage is
#     acknowledged in publications, reports, or documentation.
#   - Commercial use requires prior written consent from the copyright holder
#     (lichao19870617@163.com) and public acknowledgment of HABIT usage in
#     product documentation or user-facing materials.
#   - Unauthorized commercial use or removal of attribution is prohibited.
#
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
from habit.utils.log_utils import get_module_logger

LOGGER = get_module_logger("ml.model_factory")

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
                    # Use relative import
                    importlib.import_module(f".{module_name}", package="habit.core.machine_learning.models")
                    LOGGER.debug("Successfully imported model module: %s", module_name)
                except ImportError as e:
                    LOGGER.warning("Failed to import model module %s: %s", module_name, e)