"""
I/O utilities for habitat analysis
"""

import os
import json
import pandas as pd
import SimpleITK as sitk
import numpy as np
from ....utils.io_utils import get_image_and_mask_paths


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