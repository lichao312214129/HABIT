"""
Batch processing module for medical image preprocessing.

This module provides functionality for parallel batch processing of medical images
using multiple preprocessing steps defined in a configuration file.

Example:
    >>> from habit.core.preprocessing.image_processor_pipeline import BatchProcessor
    >>> processor = BatchProcessor(config_path="./config/config_kmeans.yaml")
    >>> processor.process_batch()
"""

from typing import Dict, List, Optional, Union, Callable
import logging
from pathlib import Path
import SimpleITK as sitk
import numpy as np
import os
from habit.core.preprocessing.preprocessor_factory import PreprocessorFactory
from habit.utils.io_utils import get_image_and_mask_paths
from habit.utils.progress_utils import CustomTqdm
from habit.utils.config_utils import load_config
import multiprocessing
import traceback
from habit.core.preprocessing.load_image import LoadImagePreprocessor


class BatchProcessor:
    """Batch processor for medical image preprocessing.
    
    This class handles parallel processing of medical images using multiple preprocessing
    steps defined in a configuration file. It supports processing multiple subjects and
    multiple time points for each subject.
    
    Attributes:
        config (Dict): Configuration dictionary containing preprocessing steps.
        num_workers (int): Number of worker processes for parallel processing.
    """
    
    def __init__(
        self,
        config_path: Union[str, Path],
        num_workers: Optional[int] = 1,
        log_level: str = "INFO",
        verbose: bool = True
    ):
        """Initialize the batch processor.
        
        Args:
            config_path (Union[str, Path]): Path to the configuration file.
            num_workers (Optional[int]): Number of worker processes. If None, uses
                number of CPU cores - 1 on Linux/Mac, or 0 on Windows.
            log_level (str): Logging level. Defaults to "INFO".
            verbose (bool): Whether to print verbose output. Defaults to True.
        """
        self.config = load_config(config_path)
        self.output_root = Path(self.config["out_dir"])
        self.output_root.mkdir(parents=True, exist_ok=True)
        # 不再创建全局预处理器，而是为每个样本单独创建
        self.verbose = verbose
        self._setup_logging(log_level)
        
        # Use config value or default to 0 if not specified or invalid
        try:
            self.num_workers = int(self.config.get("processes", num_workers))
            self.num_workers = min(self.num_workers, multiprocessing.cpu_count() - 2)
            # 确保最小为1个进程，即使设置为0也使用1个
            if self.num_workers <= 0:
                self.logger.warning(f"Setting num_workers to 1 (was {self.num_workers})")
                self.num_workers = 1
        except (ValueError, TypeError):
            self.logger.warning(f"Invalid 'processes' type in config, defaulting to 1.")
            self.num_workers = 1

        self.logger.info(f"Using {self.num_workers} worker processes.")
            
    def _setup_logging(self, log_level: str) -> None:
        """Setup logging configuration.
        
        Args:
            log_level (str): Logging level.
        """
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear any existing handlers
        self.logger.handlers = []
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Add file handler
        log_dir = self.output_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "processing.log"
        file_handler = logging.FileHandler(str(log_file), encoding='utf-8')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"Logging initialized. Log file: {log_file}")
        
    def _process_single_subject(self, subject_data):
        """处理单个样本的数据。
        
        Args:
            subject_data (Dict): 单个样本的数据字典，包含subject_id和输出目录信息

        Returns:
            tuple: (subject_id, 处理结果消息)
        """
        try:
            subject_id = subject_data['subj']
            # 创建单独的预处理管道实例
            transforms = []  

            # add load image
            load_keys = [self.config["Preprocessing"].get(k).get("images", {}) for k in self.config["Preprocessing"].keys()]
            load_keys = [item for sublist in load_keys for item in sublist]
            load_keys = list(set(load_keys))
            mask_keys = [f"mask_{mod}" for mod in load_keys]
            load_keys.extend(mask_keys)
            load_image = LoadImagePreprocessor(keys=load_keys)
            transforms.append(load_image)

            # 处理配置中定义的每个预处理步骤
            for step_name, params in self.config["Preprocessing"].items():
                # Extract modalities from params
                modalities = params.get("images", [])
                if not modalities:
                    continue
                
                # 仅处理存在于当前subject_data中的modalities
                modalities = [mod for mod in modalities if mod in subject_data]
                
                # 创建预处理器
                processor = PreprocessorFactory.create(
                    name=step_name,
                    keys=modalities,
                    **{k: v for k, v in params.items() if k != "images"}
                )
                transforms.append(processor)
            
            self.logger.info(f"Processing subject: {subject_data['subj']}")

            # Run pipeline
            for transform in transforms:
                transform(subject_data)

            return f"Success: {subject_id}", subject_data
        
        except Exception as e:
            return f"Error processing {subject_data.get('subj', 'unknown')}: {str(e)}\n{traceback.format_exc()}", None

    def save_processed_images(self, subject_data: Dict) -> None:
        """Save processed images to their respective output directories.
        
        Args:
            subject_data (Dict): Dictionary containing processed images and their output paths.
                Expected keys:
                - 'output_dirs': Dict mapping modality names to output directory paths
                - Other keys should be modality names containing the processed images
        """
        try:
            output_dirs = subject_data['output_dirs']
            
            # Save each modality image
            for key, value in subject_data.items():
                # Skip non-image keys
                if key in ['subj', 'output_dirs'] or key.startswith('mask_'):
                    continue
                    
                if key in output_dirs:
                    # Check if this image was already saved by dcm2niix
                    meta_key = f"{key}_meta_dict"
                    if meta_key in subject_data:
                        meta_dict = subject_data[meta_key]
                        if meta_dict.get("dcm2niix_converted", False) and "output_file_path" in meta_dict:
                            # File was already saved by dcm2niix, skip saving
                            self.logger.info(f"Skipping save for {key}: already saved by dcm2niix at {meta_dict['output_file_path']}")
                            continue
                    
                    output_path = Path(output_dirs[key])
                    output_path.mkdir(parents=True, exist_ok=True)
                    
                    # Get the image data
                    if isinstance(value, (np.ndarray, sitk.Image)):
                        if isinstance(value, np.ndarray):
                            # Convert numpy array to SimpleITK image if needed
                            image = sitk.GetImageFromArray(value)
                        else:
                            image = value
                            
                        # Save the image
                        sitk.WriteImage(image, os.path.join(output_path, f"{key}.nii.gz"))
                        self.logger.debug(f"Saved {key} image to {output_path}")
            
            # Save mask images if they exist
            for key, value in subject_data.items():
                if key.startswith('mask_') and key in output_dirs:
                    output_path = Path(output_dirs[key])
                    output_path.mkdir(parents=True, exist_ok=True)
                    
                    if isinstance(value, (np.ndarray, sitk.Image)):
                        if isinstance(value, np.ndarray):
                            mask = sitk.GetImageFromArray(value)
                        else:
                            mask = value
                            
                        # Save the mask
                        sitk.WriteImage(mask, os.path.join(output_path, f"{key}.nii.gz"))
                        self.logger.debug(f"Saved {key} mask to {output_path}")
                        
        except Exception as e:
            self.logger.error(f"Error saving processed images: {str(e)}")
            raise

    def process_batch(self) -> None:
        """处理所有样本数据。
        
        使用多进程方式进行批处理。
        """
        # 创建输出目录
        out_dir = self.output_root / "processed_images"
        images_dir = out_dir / "images"
        masks_dir = out_dir / "masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取所有图像和mask路径
        images_paths, mask_paths = get_image_and_mask_paths(self.config["data_dir"])

        # 由于是dcm2nii任务 需要获取倒数第二层文件夹（如果路径是文件则取其父目录）
        # Get parent directory (second to last level folder) for dcm2nii task
        processed_images = {}
        for subject, img_dict in images_paths.items():
            processed_img_dict = {}
            for img_type, img_path in img_dict.items():
                # If it's a file, get its parent directory; if it's already a directory, get its parent
                if os.path.isfile(img_path):
                    processed_img_dict[img_type] = os.path.dirname(img_path)
                elif os.path.isdir(img_path):
                    processed_img_dict[img_type] = os.path.dirname(img_path)
                else:
                    processed_img_dict[img_type] = img_path
            processed_images[subject] = processed_img_dict
        images_paths = processed_images
        
        processed_masks = {}
        for subject, mask_dict in mask_paths.items():
            processed_mask_dict = {}
            for mask_type, mask_path in mask_dict.items():
                # If it's a file, get its parent directory; if it's already a directory, get its parent
                if os.path.isfile(mask_path):
                    processed_mask_dict[mask_type] = os.path.dirname(mask_path)
                elif os.path.isdir(mask_path):
                    processed_mask_dict[mask_type] = os.path.dirname(mask_path)
                else:
                    processed_mask_dict[mask_type] = mask_path
            processed_masks[subject] = processed_mask_dict
        mask_paths = processed_masks
        
        subject_data_list = []
        
        for subject in images_paths.keys():
            data_entry = {"subj": subject}
            output_dirs = {}
            
            # 遍历当前科目的所有图像路径
            for scan_type, img_path in images_paths[subject].items():
                # 确保路径是字符串
                data_entry[scan_type] = str(img_path)
                
                # 创建该模态的输出目录
                output_dir = images_dir / subject / scan_type
                output_dir.mkdir(parents=True, exist_ok=True)
                output_dirs[scan_type] = str(output_dir)
                
                # 为每个模态添加对应的mask（如果存在）
                if subject in mask_paths and scan_type in mask_paths[subject]:
                    # 为该模态创建一个特定的mask键
                    mask_key = f"mask_{scan_type}"
                    mask_path = mask_paths[subject][scan_type]
                    # 确保路径是字符串
                    data_entry[mask_key] = str(mask_path)
                    
                    # 为该模态的mask创建输出目录
                    mask_output_dir = masks_dir / subject / scan_type
                    mask_output_dir.mkdir(parents=True, exist_ok=True)
                    output_dirs[mask_key] = str(mask_output_dir)
            
            # 将输出目录信息添加到数据条目中（确保所有路径都是字符串类型）
            data_entry["output_dirs"] = {k: str(v) for k, v in output_dirs.items()}
            
            # 检查是否至少有一个模态有对应的mask
            has_mask = any(key.startswith("mask_") for key in data_entry.keys())
            
            # 将生成的数据条目加入到列表中
            subject_data_list.append(data_entry)
            if has_mask:  # 确保至少有一个mask才添加
                self.logger.info(f"有mask的样本: {subject}")
            else:
                self.logger.warning(f"No mask found for subject {subject}, skipping")
            
        if not subject_data_list:
            self.logger.warning("No valid subjects found")
            return
        
        self.logger.info(f"将使用 {self.num_workers} 个进程进行处理")
        total_subjects = len(subject_data_list)
        self.logger.info(f"共有 {total_subjects} 个样本需要处理")
        
        # 创建进度条
        progress_bar = CustomTqdm(total=total_subjects, desc="Processing subjects")
        
        # 使用多进程处理
        self.logger.info("开始处理数据...")
        
        # 单进程模式
        if self.num_workers == 1:
            self.logger.info("使用单进程模式...")
            for subject_data in subject_data_list:
                subject_id, result = self._process_single_subject(subject_data)
                # Save processed images
                self.save_processed_images(result)

                progress_bar.update(1)
                if subject_id.startswith("Error"):
                    self.logger.error(subject_id)
                else:
                    self.logger.info(f"完成处理样本: {subject_id} ({progress_bar.n}/{total_subjects})")
        else:
            # 使用进程池并行处理
            try:
                with multiprocessing.Pool(processes=self.num_workers) as pool:
                    if self.verbose:
                        self.logger.info("开始并行处理所有样本...")
                    for subject_id, result in pool.imap(self._process_single_subject, subject_data_list):
                        # Save processed images
                        if not subject_id.startswith("Error"):  # 只保存成功处理的结果
                            self.save_processed_images(result)
                            self.logger.info(f"完成处理样本: {subject_id} ({progress_bar.n}/{total_subjects})")
                        else:
                            self.logger.error(subject_id)  # 记录错误信息
                        progress_bar.update(1)  # 无论成功或失败都更新进度条
            except Exception as e:
                self.logger.error(f"多进程处理发生错误: {str(e)}")
                self.logger.info("尝试降级到单进程模式...")
        
        self.logger.info("批处理完成") 