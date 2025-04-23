"""
I/O utility functions for HABIT package.
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
from typing import Dict, Any, Union, Optional, List, Tuple


def load_config(config_file: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a config file (JSON or YAML format)
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Configuration dictionary containing all parameters
        
    Raises:
        FileNotFoundError: If the config file does not exist
        ValueError: If the file format is not supported (only JSON or YAML are supported)
    """
    config_file = str(config_file)
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
    # Determine parsing method based on file extension
    if config_file.endswith('.json'):
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
    elif config_file.endswith(('.yaml', '.yml')):
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_file}, only JSON or YAML are supported")
    
    return config


def save_config(config: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML or JSON file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save configuration file
        
    Raises:
        ValueError: If output file format is invalid
    """
    output_path = str(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if output_path.endswith(('.yaml', '.yml')):
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)
    elif output_path.endswith('.json'):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported output file format: {output_path}")


def setup_logging(out_dir: Union[str, Path], debug: bool = False) -> None:
    """
    Configure logging settings
    
    Args:
        out_dir: Output directory for log file
        debug: Whether to enable debug mode
    """
    import logging
    import time
    
    # Create output directory
    out_dir = str(out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Get timestamp
    data = time.time()
    timeArray = time.localtime(data)
    timestr = time.strftime('%Y_%m_%d_%H_%M_%S', timeArray)
    
    # Set log file
    log_file = os.path.join(out_dir, f'habit_{timestr}.log')
    
    # Configure log level
    log_level = logging.DEBUG if debug else logging.INFO
    
    # Configure log
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    
    logging.info(f"Log file will be saved to: {log_file}")


def save_results(results: Dict[str, Any], output_dir: Union[str, Path], prefix: str = "") -> None:
    """
    Save analysis results to disk.
    
    Args:
        results: Dictionary containing analysis results
        output_dir: Directory to save results
        prefix: Prefix for output files
    """
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save different types of data
    for key, value in results.items():
        # Numpy arrays
        if isinstance(value, np.ndarray):
            np.save(os.path.join(output_dir, f"{prefix}{key}.npy"), value)
        # DataFrames
        elif isinstance(value, pd.DataFrame):
            value.to_csv(os.path.join(output_dir, f"{prefix}{key}.csv"))
        # Dictionaries
        elif isinstance(value, dict):
            with open(os.path.join(output_dir, f"{prefix}{key}.json"), 'w') as f:
                json.dump(value, f, indent=2, cls=NumpyEncoder)
        # SimpleITK images
        elif isinstance(value, sitk.Image):
            sitk.WriteImage(value, os.path.join(output_dir, f"{prefix}{key}.nrrd"))
        # Other types
        else:
            try:
                with open(os.path.join(output_dir, f"{prefix}{key}.txt"), 'w') as f:
                    f.write(str(value))
            except:
                print(f"Warning: Could not save result '{key}' of type {type(value)}")


def get_image_and_mask_paths(
    root_dir: Union[str, Path],
    keyword_of_raw_folder: str = "images",
    keyword_of_mask_folder: str = "masks"
) -> Tuple[Dict[str, Union[Dict[str, str], List[str]]], Dict[str, Union[Dict[str, str], str]]]:
    """
    Get paths of images and masks.
    
    Args:
        root_dir: Root directory containing images and masks, or path to a txt file defining file locations
        keyword_of_raw_folder: Keyword for raw image folder
        keyword_of_mask_folder: Keyword for mask folder
        
    Returns:
        Tuple containing:
            - Dictionary mapping subject IDs to dictionary of modality-image path pairs
            - Dictionary mapping subject IDs to dictionary of modality-mask path pairs
    """
    import glob
    import os
    
    root_dir = str(root_dir)
    
    # Check if root_dir is a txt file
    if root_dir.endswith('.txt'):
        return _parse_paths_from_txt(root_dir)
    
    # Find image directory
    image_dir = os.path.join(root_dir, keyword_of_raw_folder)
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    # Find mask directory
    mask_dir = os.path.join(root_dir, keyword_of_mask_folder)
    if not os.path.exists(mask_dir):
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
    
    # Get image paths for each subject and modality
    images_paths = {}
    
    # 获取所有非隐藏的患者目录
    patient_dirs = [d for d in os.listdir(image_dir) 
                   if os.path.isdir(os.path.join(image_dir, d)) and not d.startswith('.')]
    
    for patient_id in patient_dirs:
        patient_path = os.path.join(image_dir, patient_id)
        if not os.path.isdir(patient_path):
            continue
            
        images_paths[patient_id] = {}
        
        # 获取所有非隐藏的模态/序列目录
        modality_dirs = [d for d in os.listdir(patient_path) 
                        if os.path.isdir(os.path.join(patient_path, d)) and not d.startswith('.')]
        
        for modality in modality_dirs:
            modality_path = os.path.join(patient_path, modality)
            # 获取目录中的所有非隐藏文件
            files = [f for f in os.listdir(modality_path) 
                    if os.path.isfile(os.path.join(modality_path, f)) and not f.startswith('.')]
            # 警告如果有多个文件
            if len(files) > 1:
                print(f"Warning: Multiple files found in {patient_id}/{modality}, using the first one") 
            if files:  # 如果目录中有文件
                images_paths[patient_id][modality] = os.path.join(modality_path, files[0])
    
    # Get mask path for each subject and modality
    mask_paths = {}
    # 获取所有非隐藏的患者目录
    patient_dirs = [d for d in os.listdir(mask_dir) 
                   if os.path.isdir(os.path.join(mask_dir, d)) and not d.startswith('.')]

    for patient_id in patient_dirs:
        patient_path = os.path.join(mask_dir, patient_id)
        if not os.path.isdir(patient_path):
            continue   
        mask_paths[patient_id] = {}
        # 获取所有非隐藏的模态/序列目录
        modality_dirs = [d for d in os.listdir(patient_path) 
                        if os.path.isdir(os.path.join(patient_path, d)) and not d.startswith('.')]
        for modality in modality_dirs:
            modality_path = os.path.join(patient_path, modality)
            # 获取目录中的所有非隐藏文件
            files = [f for f in os.listdir(modality_path) 
                    if os.path.isfile(os.path.join(modality_path, f)) and not f.startswith('.')]
            # 警告如果有多个文件
            if len(files) > 1:
                print(f"Warning: Multiple files found in {patient_id}/{modality} mask, using the first one")                
            if files:  # 如果目录中有文件
                mask_paths[patient_id][modality] = os.path.join(modality_path, files[0])
    
    return images_paths, mask_paths


def _parse_paths_from_txt(txt_path: str) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    """
    Parse image and mask paths from a txt file.
    
    Args:
        txt_path: Path to the txt file defining file locations
        
    Returns:
        Tuple containing:
            - Dictionary mapping subject IDs to dictionary of modality-image path pairs
            - Dictionary mapping subject IDs to dictionary of modality-mask path pairs
            
    The txt file format should be:
    Subject_ID, Modality, Image_Path, Mask_Path
    e.g.:
    001, T1, /path/to/001_T1.nii.gz, /path/to/001_T1_mask.nii.gz
    001, T2, /path/to/001_T2.nii.gz, /path/to/001_T2_mask.nii.gz
    002, T1, /path/to/002_T1.nii.gz, /path/to/002_T1_mask.nii.gz
    ...
    """
    images_paths = {}
    mask_paths = {}
    
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):  # Skip empty lines and comments
            continue
            
        parts = [part.strip() for part in line.split(',')]
        if len(parts) < 4:
            print(f"Warning: Invalid line format in {txt_path}: {line}")
            continue
            
        subject_id, modality, image_path, mask_path = parts[:4]
        
        # Initialize dictionaries for new subjects
        if subject_id not in images_paths:
            images_paths[subject_id] = {}
        if subject_id not in mask_paths:
            mask_paths[subject_id] = {}
            
        # Add paths
        if os.path.exists(image_path):
            images_paths[subject_id][modality] = image_path
        else:
            print(f"Warning: Image file does not exist: {image_path}")
            
        if os.path.exists(mask_path):
            mask_paths[subject_id][modality] = mask_path
        else:
            print(f"Warning: Mask file does not exist: {mask_path}")
    
    return images_paths, mask_paths


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types."""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj) 