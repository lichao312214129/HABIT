"""
Image Dilation and Erosion Operations
"""
from typing import Tuple, List, Dict, Optional
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.ndimage import binary_dilation, binary_erosion
import SimpleITK as sitk
import os
import argparse
import multiprocessing
from functools import partial
import logging
import time
import sys
from habit.utils.io_utils import get_image_and_mask_paths


def read_medical_image(image_path: str) -> Tuple[np.ndarray, Dict]:
    """
    Read medical image (nrrd or nii.gz) and return mask with metadata
    
    Args:
        image_path: Path to the medical image file
        
    Returns:
        Tuple containing:
            - Binary mask as numpy array
            - Dictionary containing image metadata
    """
    # Read image using SimpleITK
    image = sitk.ReadImage(image_path)
    
    # Get image metadata
    metadata = {
        'spacing': image.GetSpacing(),
        'origin': image.GetOrigin(),
        'direction': image.GetDirection()
    }
    
    # Convert to numpy array and create binary mask
    mask = sitk.GetArrayFromImage(image) > 0
    
    return mask, metadata

def apply_morphological_operations(mask: np.ndarray, 
                                 voxel_size: int = 1,
                                 operation: str = 'dilation') -> np.ndarray:
    """
    Apply dilation or erosion operations to a binary mask and return the difference
    
    Args:
        mask: Binary mask to process
        voxel_size: Number of voxels to expand or shrink
        operation: Type of operation ('dilation' or 'erosion')
        
    Returns:
        Difference between processed and original mask
    """
    # 使用球形结构元素确保各向同性（对称）扩展
    # 首先创建一个立方体网格
    z, y, x = np.ogrid[-voxel_size:voxel_size+1, -voxel_size:voxel_size+1, -voxel_size:voxel_size+1]
    
    # 计算每个点到中心的距离（欧几里得距离）
    distance = np.sqrt(x*x + y*y + z*z)
    
    # 创建球形结构元素
    structure = distance <= voxel_size
    
    # 应用操作
    if operation == 'dilation':
        processed = binary_dilation(mask, structure=structure)
        # 获取差值（外环）
        result = processed & ~mask
    elif operation == 'erosion':
        processed = binary_erosion(mask, structure=structure)
        # 获取差值（内环）
        result = mask & ~processed
    else:
        raise ValueError(f"不支持的操作: {operation}")
    
    return result

def process_mask_file(args):
    """
    Process a single mask file
    
    Args:
        args: Tuple containing (subj_id, img_type, file_path, output_dir, voxel_size, operation)
        
    Returns:
        Tuple containing (subj_id, img_type, success)
    """
    subj_id, img_type, file_path, output_dir, voxel_size, operation = args
    
    try:
        # Read medical image
        mask, metadata = read_medical_image(file_path)
        
        # Apply morphological operation
        processed_mask = apply_morphological_operations(mask, voxel_size, operation)
        
        # Create output image
        output_image = sitk.GetImageFromArray(processed_mask.astype(np.uint8))
        
        # Set metadata
        output_image.SetSpacing(metadata['spacing'])
        output_image.SetOrigin(metadata['origin'])
        output_image.SetDirection(metadata['direction'])
        
        # Create output directory structure that matches the expected format
        # Path should be: output_dir/subj_id/img_type/mask.nii.gz
        output_subj_dir = os.path.join(output_dir, subj_id)
        output_img_dir = os.path.join(output_subj_dir, img_type)
        os.makedirs(output_img_dir, exist_ok=True)
        
        # Get original filename
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_img_dir, filename)
        
        # Save processed image
        sitk.WriteImage(output_image, output_path)
        
        logging.info(f"Processed {subj_id}/{img_type}: {file_path} -> {output_path}")
        return subj_id, img_type, True
        
    except Exception as e:
        logging.error(f"Error processing {subj_id}/{img_type}: {file_path}: {str(e)}")
        return subj_id, img_type, False

class ProgressBar:
    """简单的进度条实现"""
    
    def __init__(self, total, desc="Processing"):
        self.total = total
        self.desc = desc
        self.current = 0
        self.bar_length = 50
        self.start_time = time.time()
        
    def update(self, n=1):
        self.current += n
        percent = self.current / self.total
        filled_length = int(self.bar_length * percent)
        bar = '█' * filled_length + '-' * (self.bar_length - filled_length)
        
        # 计算剩余时间
        elapsed_time = time.time() - self.start_time
        if self.current > 0:
            remaining_time = (elapsed_time / self.current) * (self.total - self.current)
            time_str = f"ETA: {remaining_time:.1f}s"
        else:
            time_str = "ETA: --"
        
        print(f'\r{self.desc}: |{bar}| {self.current}/{self.total} {percent:.1%} {time_str}', end='')
        
        if self.current == self.total:
            print(f"\nTotal time: {elapsed_time:.1f}s")

def process_all_files(masks_root: str,
                     output_dir: str,
                     voxel_size: int,
                     operation: str,
                     n_processes: int = None) -> None:
    """
    Process all mask files in the root directory using multiple processes
    
    Args:
        masks_root: Root directory containing mask files
        output_dir: Directory to save processed files
        voxel_size: Number of voxels to expand or shrink
        operation: Type of operation ('dilation' or 'erosion')
        n_processes: Number of processes to use (default: CPU count - 2)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all mask files
    _, masks_paths = get_image_and_mask_paths(masks_root, keyword_of_raw_folder = "images", keyword_of_mask_folder = "masks")
    
    if not masks_paths:
        logging.warning(f"No mask files found in {masks_root}")
        return
    
    # Prepare task list
    tasks = []
    for subj_id in masks_paths:
        for img_type in masks_paths[subj_id]:
            file_path = masks_paths[subj_id][img_type]
            tasks.append((subj_id, img_type, file_path, output_dir, voxel_size, operation))
    
    # Set number of processes
    if n_processes is None:
        n_processes = max(1, multiprocessing.cpu_count() - 2)
    
    print(f"**************开始处理 {len(tasks)} 个文件，使用 {n_processes} 个进程**************")
    
    # Create progress bar
    progress_bar = ProgressBar(total=len(tasks), desc=f"Processing {operation}")
    
    # Create process pool
    with multiprocessing.Pool(processes=n_processes) as pool:
        # Process files in parallel
        for _ in pool.imap_unordered(process_mask_file, tasks):
            progress_bar.update(1)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process mask files with dilation or erosion')
    
    parser.add_argument('--masks_root', type=str, required=True,
                        help='Root directory containing mask files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save processed files')
    parser.add_argument('--voxel_size', type=int, default=5,
                        help='Number of voxels to expand or shrink')
    parser.add_argument('--operation', type=str, choices=['dilation', 'erosion'],
                        default='dilation', help='Type of operation to perform')
    parser.add_argument('--n_processes', type=int,
                        help='Number of processes to use (default: CPU count - 2)')
    
    return parser.parse_args()

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Process all files
    process_all_files(
        masks_root=args.masks_root,
        output_dir=args.output_dir,
        voxel_size=args.voxel_size,
        operation=args.operation,
        n_processes=args.n_processes
    )

if __name__ == "__main__":
    # 如果未提供命令行参数，使用默认值
    if len(sys.argv) == 1:
        sys.argv.extend([
            '--masks_root', '../demo_data/datasets',
            '--output_dir', '../demo_data/results/dilation',
            '--voxel_size', '5',
            '--operation', 'dilation',
            '--n_processes', '6'
        ])
    
    main()