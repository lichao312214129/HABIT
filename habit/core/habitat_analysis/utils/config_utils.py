"""
Configuration utilities for loading and saving configurations
"""

import os
import yaml
import json
from typing import Dict, List, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration file
    
    Args:
        config_path (str): Path to configuration file, supports YAML and JSON
    
    Returns:
        Dict[str, Any]: Configuration dictionary
        
    Raises:
        FileNotFoundError: If configuration file is not found
        ValueError: If file format is not supported
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Determine how to load based on file extension
    _, ext = os.path.splitext(config_path)
    
    if ext.lower() in ['.yaml', '.yml']:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    elif ext.lower() == '.json':
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported configuration file format: {ext}")
    
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to file
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        config_path (str): Path to save configuration file, supports YAML and JSON
        
    Raises:
        ValueError: If file format is not supported
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
    
    # Determine how to save based on file extension
    _, ext = os.path.splitext(config_path)
    
    if ext.lower() in ['.yaml', '.yml']:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    elif ext.lower() == '.json':
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
    else:
        raise ValueError(f"Unsupported configuration file format: {ext}")


def validate_config(config: Dict[str, Any], required_keys: Optional[List[str]] = None) -> bool:
    """
    Validate if configuration contains required keys
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        required_keys (Optional[List[str]]): List of required keys
    
    Returns:
        bool: Whether the configuration is valid
    
    Raises:
        ValueError: If required keys are missing
    """
    if required_keys is None:
        return True
    
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Configuration missing required keys: {missing_keys}")
    
    return True 