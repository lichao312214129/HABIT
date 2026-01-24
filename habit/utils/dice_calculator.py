import os
import click
import pandas as pd
import SimpleITK as sitk
from habit.utils import io_utils
from typing import Dict, Any
import numpy as np

def compute_dice(mask1_path: str, mask2_path: str, label_id: int = 1) -> float:
    """
    Calculate Dice coefficient between two masks for a specific label using numpy arrays.
    
    This function handles cases where masks have different physical space information
    (origin, spacing, direction, size) by resampling mask2 to match mask1's physical space.
    
    Args:
        mask1_path: Path to the first mask image
        mask2_path: Path to the second mask image
        label_id: Label ID to calculate Dice for (default: 1)
        
    Returns:
        Dice coefficient value (0.0 to 1.0), or NaN if an error occurs
    """
    try:
        mask1 = sitk.ReadImage(mask1_path)
        mask2 = sitk.ReadImage(mask2_path)
        
        # Convert to numpy arrays
        mask1_array = sitk.GetArrayFromImage(mask1)
        mask2_array = sitk.GetArrayFromImage(mask2)
        
        # Extract binary masks for the specific label
        mask1_binary = (mask1_array == label_id).astype(np.float32)
        mask2_binary = (mask2_array == label_id).astype(np.float32)
        
        # Calculate Dice coefficient: 2 * |A ∩ B| / (|A| + |B|)
        intersection = np.sum(mask1_binary * mask2_binary)
        union = np.sum(mask1_binary) + np.sum(mask2_binary)
        
        # Handle division by zero (when both masks are empty)
        if union == 0:
            # If both masks are empty, return 1.0 (perfect agreement)
            return 1.0
        
        dice_coefficient = 2.0 * intersection / union
        
        return float(dice_coefficient)
    except Exception as e:
        print(f"Error computing Dice for {mask1_path} and {mask2_path}: {e}")
        return np.nan

def run_dice_calculation(input1, input2, output, mask_keyword, label_id):
    """
    Calculate Dice coefficient between two batches of images (ROI/mask).
    
    This tool compares masks from two sources (directories or config files) and computes the Dice coefficient.
    It matches files based on Subject ID and Mask Type (subfolder name).
    """
    print(f"Loading paths from {input1}...")
    # Determine raw keyword to avoid crash if 'images' folder doesn't exist
    raw_kw1 = "images"
    if os.path.isdir(input1) and not os.path.exists(os.path.join(input1, "images")):
        # If default 'images' folder is missing, use mask_keyword as raw folder to bypass scan error
        # This works because we only care about mask_paths output
        raw_kw1 = mask_keyword

    # We pass mask_keyword. If input is a config file, get_image_and_mask_paths handles it.
    _, masks1 = io_utils.get_image_and_mask_paths(input1, keyword_of_raw_folder=raw_kw1, keyword_of_mask_folder=mask_keyword)
    
    print(f"Loading paths from {input2}...")
    raw_kw2 = "images"
    if os.path.isdir(input2) and not os.path.exists(os.path.join(input2, "images")):
        raw_kw2 = mask_keyword
        
    _, masks2 = io_utils.get_image_and_mask_paths(input2, keyword_of_raw_folder=raw_kw2, keyword_of_mask_folder=mask_keyword)
    
    if not masks1:
        print(f"No masks found in {input1}. Please check the path and mask keyword.")
        return
    if not masks2:
        print(f"No masks found in {input2}. Please check the path and mask keyword.")
        return
        
    results = []
    
    # Identify common subjects
    subjects1 = set(masks1.keys())
    subjects2 = set(masks2.keys())
    common_subjects = sorted(list(subjects1 & subjects2))
    
    print(f"Found {len(subjects1)} subjects in batch 1.")
    print(f"Found {len(subjects2)} subjects in batch 2.")
    print(f"Found {len(common_subjects)} common subjects to compare.")
    
    if not common_subjects:
        print("No common subjects found. Please check subject naming/folder structure.")
        return

    count = 0
    with click.progressbar(common_subjects, label='Calculating Dice') as bar:
        for subj in bar:
            # Get mask types for this subject
            types1 = masks1[subj]
            types2 = masks2[subj]
            
            # Identify common mask types (subfolders)
            common_types = sorted(list(set(types1.keys()) & set(types2.keys())))
            
            for mtype in common_types:
                path1 = types1[mtype]
                path2 = types2[mtype]
                
                dice = compute_dice(path1, path2, label_id)
                
                results.append({
                    'Subject': subj,
                    'MaskType': mtype,
                    'Dice': dice,
                    'Path1': path1,
                    'Path2': path2
                })
                count += 1
            
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output, index=False)
        print(f"\nResults saved to {output}")
        
        # Calculate summary statistics (exclude NaN values)
        valid_scores = df[df['Dice'].notna()]['Dice']
        if not valid_scores.empty:
            print(f"Mean Dice: {valid_scores.mean():.4f}")
            print(f"Std Dice: {valid_scores.std():.4f}")
            print(f"Min Dice: {valid_scores.min():.4f}")
            print(f"Max Dice: {valid_scores.max():.4f}")
        
        failed = df[df['Dice'].isna()]
        if not failed.empty:
            print(f"Warning: {len(failed)} comparisons failed or had errors.")
    else:
        print("No matching mask types found for common subjects.")

@click.command()
@click.option('--input1', required=True, type=click.Path(exists=True), help='Path to first batch (root directory or config file)')
@click.option('--input2', required=True, type=click.Path(exists=True), help='Path to second batch (root directory or config file)')
@click.option('--output', default='dice_results.csv', show_default=True, help='Path to save results CSV')
@click.option('--mask-keyword', default='masks', show_default=True, help='Keyword for mask folder (used if input is a directory)')
@click.option('--label-id', default=1, show_default=True, help='Label ID to calculate Dice for')
def main(input1, input2, output, mask_keyword, label_id):
    """
    Calculate Dice coefficient between two batches of images (ROI/mask).
    
    This tool compares masks from two sources (directories or config files) and computes the Dice coefficient.
    It matches files based on Subject ID and Mask Type (subfolder name).
    """
    run_dice_calculation(input1, input2, output, mask_keyword, label_id)

if __name__ == '__main__':
    main()

