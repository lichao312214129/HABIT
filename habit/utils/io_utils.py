"""
I/O utilities for habitat analysis
"""

import os
import json
import pandas as pd
import SimpleITK as sitk
import numpy as np
from typing import Dict, Any, Optional, List
import yaml
import logging
from datetime import datetime

def _scan_folder_for_paths(root_folder: str, keyword_of_raw_folder: str = "images", keyword_of_mask_folder: str = "masks") -> tuple:
    """
    Scan folder structure for image and mask paths (internal function)
    
    Args:
        root_folder (str): Root directory
        keyword_of_raw_folder (str, optional): Name of the images folder
        keyword_of_mask_folder (str, optional): Name of the masks folder
    
    Returns:
        tuple: Dictionary of image paths and dictionary of mask paths
    """
    # Get image paths
    images_paths = {}
    images_root = os.path.join(root_folder, keyword_of_raw_folder)
    # Filter out .DS_Store and other hidden files
    subjects = [f for f in os.listdir(images_root) if not f.startswith('.')]
    
    for subj in subjects:
        images_paths[subj] = {}
        subj_path = os.path.join(images_root, subj)
        # Filter out .DS_Store and other hidden files
        img_subfolders = [f for f in os.listdir(subj_path) if not f.startswith('.')]
        
        for img_subfolder in img_subfolders:
            img_subfolder_path = os.path.join(subj_path, img_subfolder)
            if os.path.isdir(img_subfolder_path):
                # Filter out .DS_Store and other hidden files
                img_files = [f for f in os.listdir(img_subfolder_path) if not f.startswith('.')]
                # Warning if multiple files
                if len(img_files) > 1:
                    print(f"Warning: Multiple image files in {subj}/{img_subfolder}")
                img_file = img_files[0]
                images_paths[subj][img_subfolder] = os.path.join(img_subfolder_path, img_file)
    
    # Get mask paths
    mask_paths = {}
    masks_root = os.path.join(root_folder, keyword_of_mask_folder)

    # if no masks folder, return empty mask_paths
    if not os.path.exists(masks_root):
        return images_paths, {}
    
    # Filter out .DS_Store and other hidden files
    subjects = [f for f in os.listdir(masks_root) if not f.startswith('.')]
    for subj in subjects:
        mask_paths[subj] = {}
        subj_path = os.path.join(masks_root, subj)
        # Filter out .DS_Store and other hidden files
        mask_subfolders = [f for f in os.listdir(subj_path) if not f.startswith('.')]
        
        for mask_subfolder in mask_subfolders:
            mask_subfolder_path = os.path.join(subj_path, mask_subfolder)
            if os.path.isdir(mask_subfolder_path):
                # Filter out .DS_Store and other hidden files
                mask_files = [f for f in os.listdir(mask_subfolder_path) if not f.startswith('.')]
                # Warning if multiple files
                if len(mask_files) > 1:
                    print(f"Warning: Multiple mask files in {subj}/{mask_subfolder}")
                mask_file = mask_files[0]
                mask_paths[subj][mask_subfolder] = os.path.join(mask_subfolder_path, mask_file)
    
    return images_paths, mask_paths

def get_image_and_mask_paths(root_folder: str, keyword_of_raw_folder: str = "images", keyword_of_mask_folder: str = "masks", auto_select_first_file: bool = True) -> tuple:
    """
    Get paths for all image and mask files
    
    Args:
        root_folder (str): Root directory or path to YAML configuration file
        keyword_of_raw_folder (str, optional): Name of the images folder (only used when root_folder is a directory)
        keyword_of_mask_folder (str, optional): Name of the masks folder (only used when root_folder is a directory)
        auto_select_first_file (bool, optional): If True, automatically select the first file when path is a directory.
                                                  If False, keep directory path as is. Defaults to True.
    
    Returns:
        tuple: Dictionary of image paths and dictionary of mask paths
        
    Note:
        If root_folder is a YAML file, it should contain the following structure:
        ```yaml
        images:
          subject1:
            image_type1: /path/to/image1
            image_type2: /path/to/image2
          subject2:
            image_type1: /path/to/image1
        masks:
          subject1:
            image_type1: /path/to/mask1
            image_type2: /path/to/mask2
          subject2:
            image_type1: /path/to/mask1
        
        # Optional: control whether to automatically select first file in directory
        auto_select_first_file: true  # or false
        ```
    """
    # Check if input is a YAML configuration file
    if os.path.isfile(root_folder) and root_folder.lower().endswith(('.yaml', '.yml')):
        # Load configuration from YAML file
        config = load_config(root_folder)
        
        # Check if auto_select_first_file is specified in config file
        # Config file takes precedence over function parameter
        if 'auto_select_first_file' in config:
            auto_select_first_file = config['auto_select_first_file']
        
        # Extract images and masks paths from config
        images_paths = config.get('images', {})
        mask_paths = config.get('masks', {})
        
        # Validate that all paths exist
        for subject, img_dict in images_paths.items():
            for img_type, img_path in img_dict.items():
                if not os.path.exists(img_path):
                    print(f"Warning: Image file not found: {img_path} for {subject}/{img_type}")
        
        for subject, mask_dict in mask_paths.items():
            for mask_type, mask_path in mask_dict.items():
                if not os.path.exists(mask_path):
                    print(f"Warning: Mask file not found: {mask_path} for {subject}/{mask_type}")

        # if is dir and auto_select_first_file is True, get the first file in the dir
        if auto_select_first_file:
            for subject, img_dict in images_paths.items():
                for img_type, img_path in img_dict.items():
                    if os.path.isdir(img_path):
                        files = [f for f in os.listdir(img_path) if not f.startswith('.')]
                        if files:
                            img_dict[img_type] = os.path.join(img_path, files[0])
            
            for subject, mask_dict in mask_paths.items():
                for mask_type, mask_path in mask_dict.items():
                    if os.path.isdir(mask_path):
                        files = [f for f in os.listdir(mask_path) if not f.startswith('.')]
                        if files:
                            mask_dict[mask_type] = os.path.join(mask_path, files[0])
        
        return images_paths, mask_paths
    
    # Use folder scanning logic
    return _scan_folder_for_paths(root_folder, keyword_of_raw_folder, keyword_of_mask_folder)

def load_timestamp(file_path: str, subjID_column: str = "Name") -> dict:
    """
    Load scan timestamps from Excel file
    
    Args:
        file_path (str): Path to the Excel file
        subjID_column (str, optional): Name of the subject ID column
    
    Returns:
        dict: Dictionary with subject names as keys and timestamp lists as values
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_excel(file_path, index_col=subjID_column)
    # convert index to string
    df.index = df.index.astype(str)
    return df

def save_results(out_folder: str, results: pd.DataFrame, config: dict = None, file_name: str = "habitats.csv") -> None:
    """
    Save clustering results
    
    Args:
        out_folder (str): Output directory
        results (DataFrame): Results DataFrame
        config (dict, optional): Configuration dictionary, saved as JSON if not None
        file_name (str, optional): Name of the CSV file to save
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    # Save configuration
    if config:
        with open(os.path.join(out_folder, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
    
    # Save CSV results
    results.to_csv(os.path.join(out_folder, file_name), index=True)
    print(f"Results saved to {os.path.join(out_folder, file_name)}")

def save_supervoxel_image(subject: str, supervoxel_labels: np.ndarray, mask_path: str, out_folder: str) -> str:
    """
    Save supervoxel image
    
    Args:
        subject (str): Subject name
        supervoxel_labels (ndarray): Supervoxel labels
        mask_path (str): Path to the mask file
        out_folder (str): Output directory
    
    Returns:
        str: Path to the saved file
    """
    # Load mask
    mask = sitk.ReadImage(mask_path)
    mask_array = sitk.GetArrayFromImage(mask)
    
    # Create supervoxel image
    supervoxel_map = np.zeros_like(mask_array)
    supervoxel_map[mask_array > 0] = supervoxel_labels
    
    # Convert to SimpleITK image and save
    supervoxel_img = sitk.GetImageFromArray(supervoxel_map)
    supervoxel_img.CopyInformation(mask)
    
    output_path = os.path.join(out_folder, f"{subject}_supervoxel.nrrd")
    sitk.WriteImage(supervoxel_img, output_path)
    
    return output_path

def save_habitat_image(subject: str, habitats_df: pd.DataFrame, supervoxel_path: str, out_folder: str) -> str:
    """
    Save habitat image
    
    Args:
        subject (str): Subject name
        habitats_df (DataFrame): Habitat DataFrame containing Supervoxel and Habitats columns
        supervoxel_path (str): Path to the supervoxel image
        out_folder (str): Output directory
    
    Returns:
        str: Path to the saved file

    TODO: 
    1. 某个团块的体素只有很少的几个，是否需要删除，或者归位其他相似的团块中去
    """
    # Load supervoxel image
    supervoxel = sitk.ReadImage(supervoxel_path)
    supervoxel_array = sitk.GetArrayFromImage(supervoxel)
    
    # Create habitat image
    habitats_array = np.zeros_like(supervoxel_array)
    habitats_subj = habitats_df.loc[subject]
    n_clusters_supervoxel = habitats_subj.shape[0]
    for i in range(n_clusters_supervoxel):
        # Assert that habitats_subj[habitats_subj['Supervoxel'] == i+1]['Habitats'] has exactly one value
        # assert habitats_subj[habitats_subj['Supervoxel'] == i+1].shape[0] == 1, f"Multiple rows for supervoxel {i+1} in subject {subject}, please check the data table"
        if (supervoxel_array == i+1).sum() > 0:
            habitats_array[supervoxel_array == i+1] = habitats_subj[habitats_subj['Supervoxel'] == i+1]['Habitats'].values[0]
    

    # Convert to SimpleITK image and save
    habitats_img = sitk.GetImageFromArray(habitats_array)
    habitats_img.CopyInformation(supervoxel)
    
    output_path = os.path.join(out_folder, f"{subject}_habitats.nrrd")
    sitk.WriteImage(habitats_img, output_path)
    
    return output_path

def detect_image_names(images_paths: dict) -> list:
    """
    Automatically detect image names
    
    Args:
        images_paths (dict): Dictionary of image paths
    
    Returns:
        list: List of all unique image names
    """
    # Collect all image names
    all_image_names = []
    for subj in images_paths:
        for img_name in images_paths[subj].keys():
            all_image_names.append(img_name)
    
    # Get unique image names and sort
    unique_image_names = sorted(list(set(all_image_names)))
    
    return unique_image_names

def check_data_structure(images_paths: dict, mask_paths: dict, image_names: list, time_dict: dict = None) -> bool:
    """
    Validate data structure
    
    Args:
        images_paths (dict): Dictionary of image paths
        mask_paths (dict): Dictionary of mask paths
        image_names (list): List of image names
        time_dict (dict, optional): Dictionary of timestamps, if None, not checked
    
    Raises:
        ValueError: If data structure is invalid
    
    Returns:
        bool: True if data structure is valid
    """
    # Check data structure for each subject
    for subj in images_paths.keys():
        img_names = images_paths[subj].keys()
        mask_names = mask_paths[subj].keys()
        
        # Check if image and mask names match
        diff_img_mask = set(img_names) - set(mask_names)
        if len(diff_img_mask) > 0:
            raise ValueError(f"Image and mask names don't match for {subj}, difference: {diff_img_mask}")
        
        # Check if required images exist
        diff_img = set(image_names) - set(img_names)
        if len(diff_img) > 0:
            raise ValueError(f"Image names don't match for {subj}, difference: {diff_img}")
    
    return True 

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

def setup_logging(out_dir: str, debug: bool = False) -> None:
    """
    Set up logging configuration
    
    Args:
        out_dir (str): Output directory for log files
        debug (bool, optional): Whether to enable debug mode. Defaults to False.
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up logging configuration
    log_level = logging.DEBUG if debug else logging.INFO
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"habit_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),  # save to file
            logging.StreamHandler()  # print to console
        ]
    ) 

def export_paths_to_yaml(root_folder: str, output_yaml_path: str, keyword_of_raw_folder: str = "images", keyword_of_mask_folder: str = "masks") -> None:
    """
    Export folder structure to YAML configuration file
    
    Args:
        root_folder (str): Root directory to scan
        output_yaml_path (str): Path to save the YAML configuration file
        keyword_of_raw_folder (str, optional): Name of the images folder
        keyword_of_mask_folder (str, optional): Name of the masks folder
    """
    # Get paths using the folder scanning method
    images_paths, mask_paths = _scan_folder_for_paths(root_folder, keyword_of_raw_folder, keyword_of_mask_folder)
    
    # Create configuration dictionary
    config = {
        'images': images_paths,
        'masks': mask_paths
    }
    
    # Save to YAML file
    save_config(config, output_yaml_path)
    print(f"Paths configuration exported to {output_yaml_path}") 