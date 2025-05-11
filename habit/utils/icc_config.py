"""
ICC configuration utilities for loading and validating ICC analysis config
"""

import os
from typing import Dict, Any, List, Union, Optional
import yaml
from habit.utils.config_utils import load_config, validate_config

def load_icc_config(config_path: str) -> Dict[str, Any]:
    """
    Load ICC analysis configuration from YAML file
    
    Args:
        config_path (str): Path to ICC configuration file (YAML)
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    config = load_config(config_path)
    validate_icc_config(config)
    return config

def validate_icc_config(config: Dict[str, Any]) -> bool:
    """
    Validate ICC configuration structure and required parameters
    
    Args:
        config (Dict[str, Any]): ICC configuration dictionary
    
    Returns:
        bool: Whether configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Define required top-level keys
    required_keys = ["input", "output"]
    validate_config(config, required_keys)
    
    # Validate input configuration
    if "type" not in config["input"]:
        raise ValueError("Input configuration must contain 'type' field ('files' or 'directories')")
    
    if config["input"]["type"] not in ["files", "directories"]:
        raise ValueError("Input type must be either 'files' or 'directories'")
    
    if config["input"]["type"] == "files" and "file_groups" not in config["input"]:
        raise ValueError("Input configuration with type 'files' must contain 'file_groups' field")
    
    if config["input"]["type"] == "directories" and "dir_list" not in config["input"]:
        raise ValueError("Input configuration with type 'directories' must contain 'dir_list' field")
    
    # Validate output configuration
    if "path" not in config["output"]:
        raise ValueError("Output configuration must contain 'path' field")
    
    return True

def parse_icc_config_files(config: Dict[str, Any]) -> List[List[str]]:
    """
    Parse file groups from ICC configuration
    
    Args:
        config (Dict[str, Any]): ICC configuration dictionary
    
    Returns:
        List[List[str]]: List of file groups
    """
    if config["input"]["type"] != "files":
        raise ValueError("Cannot parse file groups from non-file input type")
    
    file_groups = []
    for group in config["input"]["file_groups"]:
        if isinstance(group, list):
            file_groups.append(group)
        else:
            # If a single string is provided, treat it as a one-element group
            file_groups.append([group])
    
    return file_groups

def parse_icc_config_directories(config: Dict[str, Any]) -> List[str]:
    """
    Parse directory list from ICC configuration
    
    Args:
        config (Dict[str, Any]): ICC configuration dictionary
    
    Returns:
        List[str]: List of directories
    """
    if config["input"]["type"] != "directories":
        raise ValueError("Cannot parse directory list from non-directory input type")
    
    return config["input"]["dir_list"]

def get_icc_config_output_path(config: Dict[str, Any]) -> str:
    """
    Get output path from ICC configuration
    
    Args:
        config (Dict[str, Any]): ICC configuration dictionary
    
    Returns:
        str: Output file path
    """
    return config["output"]["path"]

def get_icc_config_processes(config: Dict[str, Any]) -> Optional[int]:
    """
    Get number of processes from ICC configuration
    
    Args:
        config (Dict[str, Any]): ICC configuration dictionary
    
    Returns:
        Optional[int]: Number of processes or None for default
    """
    return config.get("processes", None)

def create_default_icc_config() -> Dict[str, Any]:
    """
    Create default ICC configuration
    
    Returns:
        Dict[str, Any]: Default configuration dictionary
    """
    return {
        "input": {
            "type": "files",
            "file_groups": [
                ["path/to/file1.csv", "path/to/file2.csv", "path/to/file3.csv"],
                ["path/to/file4.csv", "path/to/file5.csv", "path/to/file6.csv"]
            ]
        },
        "output": {
            "path": "icc_results.json"
        },
        "processes": None,
        "debug": False
    }

def save_default_icc_config(output_path: str) -> None:
    """
    Save default ICC configuration to file
    
    Args:
        output_path (str): Output file path
    """
    default_config = create_default_icc_config()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(default_config, f, default_flow_style=False, sort_keys=False) 