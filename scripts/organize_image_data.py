"""
Reorganize files in a folder according to the expected data structure.
Input directory structure:

dataset/
|── subj001
|   ├── *.nii.gz
|   ├── *mask*.nii.gz
|── subj002
|   ├── *.nii.gz
|   ├── *mask*.nii.gz
|── MRI_Sequences_File_Names_Record_Part.xlsx

MRI_Sequences_File_Names_Record_Part.xlsx
| Name            | pre_contrast       | LAP                  | PVP                  | delay_3min           |
|-----------------|--------------------|----------------------|----------------------|----------------------|
| Patient_01      | Series_01          | Series_02            | Series_03            | Series_04            |
| Patient_02      | Series_05          | Series_06            | Series_07            | Series_08            |
| Patient_03      | Series_09          | Series_10            | Series_11            | Series_12            |

Output directory structure:
dataset/
├── images/
│   ├── subj001/
│   │   ├── img1
|   |   |   ├── img.nii.gz (OR img.nrrd)
│   │   ├── img2
|   |   |   ├── img.nii.gz (OR img.nrrd)
│   ├── subj002/
│   │   ├── img1
|   |   |   ├── img.nii.gz (OR img.nrrd)
│   │   ├── img2
|   |   |   ├── img.nii.gz (OR img.nrrd)
├── masks/
│   ├── subj001/
│   │   ├── img1
|   |   |   ├── mask.nii.gz (OR mask.nrrd)
│   │   ├── img2
|   |   |   ├── mask.nii.gz (OR mask.nrrd)
│   ├── subj002/
│   │   ├── img1
|   |   |   ├── mask.nii.gz (OR mask.nrrd)
│   │   ├── img2
|   |   |   ├── mask.nii.gz (OR mask.nrrd)
"""

import multiprocessing
from functools import partial
import shutil
import pandas as pd
from pathlib import Path
import argparse
import sys
import logging
from habit.utils.log_utils import setup_logger, get_module_logger
from habit.utils.progress_utils import CustomTqdm

_module_logger = get_module_logger(__name__)

def organize_dataset_subject_level(subj_dir, 
                                   series_mapping_table, 
                                   mask_keyword, 
                                   images_dir, 
                                   masks_dir):
    subj_id = subj_dir.name
        
    if subj_id not in series_mapping_table.index:
        _module_logger.warning(f"{subj_id} not found in mapping, skipping {subj_dir}")
        return
    
    series_map = series_mapping_table.loc[subj_id].to_dict()
    
    mask_files = list(subj_dir.glob(mask_keyword))
    if not mask_files:
        _module_logger.error(f"Mask file not found: {subj_dir}")
        return
    src_mask = mask_files[0]
    _module_logger.debug(f"Found mask file: {src_mask}")
    
    for img_idx, (phase, series_id) in enumerate(series_map.items(), start=1):
        img_pattern = f"*{series_id}*"
        img_files = list(subj_dir.glob(img_pattern))
        
        if not img_files:
            _module_logger.error(f"Image file not found: {img_pattern} in {subj_dir}")
            continue
            
        src_img = img_files[0]
        _module_logger.debug(f"Found image file: {src_img}")
        
        target_img_dir = images_dir / subj_id / f"{phase}"
        target_mask_dir = masks_dir / subj_id / f"{phase}"
        target_img_dir.mkdir(parents=True, exist_ok=True)
        target_mask_dir.mkdir(parents=True, exist_ok=True)
        
        shutil.copy(src_img, target_img_dir)
        shutil.copy(src_mask, target_mask_dir)
        
        _module_logger.info(f"Processed: {subj_id} {phase} → {phase}")

def organize_dataset(
                src_dir: str, 
                excel_path: str, 
                dst_dir: str,
                mask_keyword:str,
                n_processes=4,
                debug=False):
    """
    Reorganize MRI dataset structure (multiple sequences sharing the same mask)
    
    Parameters:
    src_dir: Original data root directory (containing subj001 subdirectories, etc.)
    excel_path: Path to the sequence mapping Excel file
    dst_dir: Output directory root path
    mask_keyword: Keyword for mask files
    n_processes: Number of parallel processing processes
    debug: Whether to enable debug mode
    """
    if debug:
        _module_logger.setLevel(logging.DEBUG)
        
    _module_logger.info(f"Starting dataset processing...")
    _module_logger.info(f"Source directory: {src_dir}")
    _module_logger.info(f"Target directory: {dst_dir}")
    _module_logger.info(f"Excel file: {excel_path}")
    
    try:
        series_mapping_table = pd.read_excel(excel_path).set_index('Name')
    except Exception as e:
        _module_logger.error(f"Failed to read Excel file: {e}")
        sys.exit(1)

    for name, row in series_mapping_table.iterrows():
        if row.isnull().any():
            _module_logger.warning(f"Empty value found for: {name}")
    
    images_dir = Path(dst_dir) / 'images'
    masks_dir = Path(dst_dir) / 'masks'
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    subj_dirs = list(Path(src_dir).glob('*'))
    if not subj_dirs:
        _module_logger.error(f"Source directory is empty: {src_dir}")
        sys.exit(1)

    process_func = partial(organize_dataset_subject_level,
                           series_mapping_table=series_mapping_table,
                           mask_keyword=mask_keyword,
                           images_dir=images_dir,
                           masks_dir=masks_dir
    )               

    with multiprocessing.Pool(processes=n_processes) as pool:
        for _ in CustomTqdm(pool.imap_unordered(process_func, subj_dirs), 
                            total=len(subj_dirs), desc="Organizing dataset"):
            pass
    
    _module_logger.info("Processing complete!")

def main():
    parser = argparse.ArgumentParser(description='MRI Dataset Structure Reorganization Tool')
    parser.add_argument('--src_dir', type=str, required=True, help='Source data directory path')
    parser.add_argument('--excel_path', type=str, required=True, help='Path to sequence mapping Excel file')
    parser.add_argument('--dst_dir', type=str, required=True, help='Target output directory path')
    parser.add_argument('--mask_keyword', type=str, default='*mask.nrrd', help='Mask file keyword (default: *mask.nrrd)')
    parser.add_argument('--n_processes', type=int, default=4, help='Number of parallel processes (default: 4)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    organize_dataset(
        src_dir=args.src_dir,
        excel_path=args.excel_path,
        dst_dir=args.dst_dir,
        mask_keyword=args.mask_keyword,
        n_processes=args.n_processes,
        debug=args.debug
    )

if __name__ == "__main__":
    import sys
    # Set command line arguments before parsing
    if len(sys.argv) == 1:  # If no command line arguments are provided
        sys.argv.extend([
            '--src_dir', 'H:\\Registration_ZSSY_1mm',
            '--excel_path', '../data/MRI_Sequences_File_Names_Record_Part.xlsx',
            '--dst_dir', 'H:\\Registration_ZSSY_1mm_structured',
            '--mask_keyword', '*mask*.nrrd',
            '--n_processes', '6',
            '--debug'
        ])
        
    if '--debug' in sys.argv:
        _module_logger.setLevel(logging.DEBUG)
        _module_logger.debug("Debug mode enabled")
    main()
