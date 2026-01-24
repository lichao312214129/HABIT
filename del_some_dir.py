import os
import shutil
from typing import List, Set
from pathlib import Path
import pandas as pd
import numpy as np


data = pd.DataFrame(np.random.randn

def check_and_delete_subjects(root_dir: str, required_dirs: List[str]) -> None:
    """
    Check each subject directory and delete it if it doesn't contain all required subdirectories.
    
    Args:
        root_dir (str): Root directory containing subject folders
        required_dirs (List[str]): List of required directory names that must exist in each subject folder
    """
    # Convert root_dir to Path object
    root_path = Path(root_dir)
    
    # Get all subject directories
    subject_dirs = [d for d in root_path.iterdir() if d.is_dir()]
    
    # Track statistics
    total_subjects = len(subject_dirs)
    deleted_subjects = 0
    
    print(f"Found {total_subjects} subject directories")
    print(f"Required directories: {required_dirs}")
    
    # Check each subject directory
    for subject_dir in subject_dirs:
        # Get all subdirectories in the subject folder
        subdirs = {d.name for d in subject_dir.iterdir() if d.is_dir()}
        
        # Check if all required directories exist
        if not all(req_dir in subdirs for req_dir in required_dirs):
            print(f"Deleting {subject_dir} - missing required directories")
            try:
                shutil.rmtree(subject_dir)
                deleted_subjects += 1
            except Exception as e:
                print(f"Error deleting {subject_dir}: {str(e)}")
    
    print(f"\nSummary:")
    print(f"Total subjects: {total_subjects}")
    print(f"Deleted subjects: {deleted_subjects}")
    print(f"Remaining subjects: {total_subjects - deleted_subjects}")

def delete_dirs_not_in_excel(root_dir: str, excel_path: str, subject_column: str) -> None:
    """
    Delete directories in root_dir that are not listed in the Excel file's subject column.
    
    Args:
        root_dir (str): Root directory containing subject folders
        excel_path (str): Path to the Excel file containing subject IDs
        subject_column (str): Name of the column containing subject IDs in the Excel file
    """
    # Read Excel file and get unique subject IDs
    df = pd.read_csv(excel_path)
    valid_subjects = set(df[subject_column].unique())
    
    # Convert root_dir to Path object
    root_path = Path(root_dir)
    
    # Get all subject directories
    subject_dirs = [d for d in root_path.iterdir() if d.is_dir()]
    
    # Track statistics
    total_subjects = len(subject_dirs)
    deleted_subjects = 0
    
    print(f"Found {total_subjects} subject directories")
    print(f"Valid subjects from Excel: {len(valid_subjects)}")
    
    # Check each subject directory
    for subject_dir in subject_dirs:
        # Check if the directory name is in the valid subjects list
        if subject_dir.name not in valid_subjects:
            print(f"Deleting {subject_dir} - not in Excel subject list")
            try:
                shutil.rmtree(subject_dir)
                deleted_subjects += 1
            except Exception as e:
                print(f"Error deleting {subject_dir}: {str(e)}")
    
    print(f"\nSummary:")
    print(f"Total subjects: {total_subjects}")
    print(f"Deleted subjects: {deleted_subjects}")
    print(f"Remaining subjects: {total_subjects - deleted_subjects}")

if __name__ == "__main__":
    # Example usage
    # root_directory = r"H:\Registration_ZSSY_1mm_structured\masks"  # Current directory
    # required_directories = ["pre_contrast", "LAP", "PVP", "delay_3min"]  # Example required directories
    
    # check_and_delete_subjects(root_directory, required_directories)
    
    # Example usage of new function
    root_directory = r"H:\Registration_ZSSY_1mm_structured\images"  # Current directory
    excel_path = "F:/work/research/radiomics_TLSs/data/results_365/habitats.csv"
    subject_column = "Subject"  # Replace with your actual column name
    delete_dirs_not_in_excel(root_directory, excel_path, subject_column) 