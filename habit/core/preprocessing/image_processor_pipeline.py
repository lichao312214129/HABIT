"""
Batch processing module for medical image preprocessing.

This module provides functionality for parallel batch processing of medical images
using multiple preprocessing steps defined in a configuration file.

Example:
    >>> from habit.core.preprocessing.image_processor_pipeline import BatchProcessor
    >>> processor = BatchProcessor(config_path="./config/config_kmeans.yaml")
    >>> processor.process_batch(data_root="H:\\Registration_ICC_structured_test")
"""

from typing import Dict, List, Optional, Union
import logging
from pathlib import Path
import numpy as np
import torch
import yaml
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd
from monai.data import Dataset, DataLoader
from habit.core.preprocessing.preprocessor_factory import PreprocessorFactory
from habit.utils.io_utils import get_image_and_mask_paths
from habit.utils.progress_utils import CustomTqdm
from monai.transforms.io.array import LoadImage
from monai.data.image_reader import ITKReader
import SimpleITK as sitk

# 定义全局函数用于替代lambda
def identity_collate(batch):
    """保持batch的原始结构
    
    Args:
        batch: 数据批次
    
    Returns:
        原始数据批次
    """
    return batch

class BatchProcessor:
    """Batch processor for medical image preprocessing.
    
    This class handles parallel processing of medical images using multiple preprocessing
    steps defined in a configuration file. It supports processing multiple subjects and
    multiple time points for each subject.
    
    Attributes:
        config (Dict): Configuration dictionary containing preprocessing steps.
        preprocessors (Compose): Preprocessing pipeline.
        num_workers (int): Number of worker processes for parallel processing.
    """
    
    def __init__(
        self,
        config_path: Union[str, Path],
        num_workers: Optional[int] = None,
        log_level: str = "INFO"
    ):
        """Initialize the batch processor.
        
        Args:
            config_path (Union[str, Path]): Path to the configuration file.
            num_workers (Optional[int]): Number of worker processes. If None, uses
                number of CPU cores.
            log_level (str): Logging level. Defaults to "INFO".
        """
        self.config = self._load_config(config_path)
        self._setup_logging(log_level)
        self.preprocessors = self._create_preprocessors()
        
    def _load_config(self, config_path: Union[str, Path]) -> Dict:
        """Load configuration from YAML file.
        
        Args:
            config_path (Union[str, Path]): Path to the configuration file.
            
        Returns:
            Dict: Configuration dictionary.
            
        Raises:
            FileNotFoundError: If config file does not exist.
            yaml.YAMLError: If config file is invalid.
        """
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML configuration: {e}")
            
    def _setup_logging(self, log_level: str) -> None:
        """Setup logging configuration.
        
        Args:
            log_level (str): Logging level.
        """
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def _create_preprocessors(self) -> Compose:
        """Create preprocessing pipelines from configuration.
        
        This method creates a single Compose object that handles all preprocessing steps
        for all modalities in an efficient way to avoid redundant processing
        and ensure correct execution order.
        
        Returns:
            Compose: A single preprocessing pipeline.
            
        Raises:
            ValueError: If preprocessing configuration is invalid.
        """
        if "Preprocessing" not in self.config:
            raise ValueError("No preprocessing configuration found")
            
        transforms = []
        
        # 首先添加图像加载和通道处理转换
        # 获取所有可能需要处理的图像键
        all_modalities = []
        for step_name, params in self.config["Preprocessing"].items():
            modalities = params.get("images", [])
            for mod in modalities:
                if mod not in all_modalities:
                    all_modalities.append(mod)
        
        # 如果存在mask，也添加到键列表中
        if "mask" not in all_modalities:
            all_modalities.append("mask")
            
        # 添加LoadImage转换，负责高效加载图像文件,keys中不包含subj    
        all_modalities_for_load = [mod for mod in all_modalities if mod != "subj"]
        # Use lowercase keys for LoadImage compatible with MONAI
        all_modalities_for_load = [mod.lower() for mod in all_modalities_for_load]
        
        # Create image loader with ITKReader for .nrrd files
        image_loader = LoadImaged(
            keys=all_modalities_for_load,
            reader=ITKReader(reverse_indexing=True),
            image_only=False
        )
        transforms.append(image_loader)
        
        # 添加EnsureChannelFirst转换，确保图像数据的通道维度在前，不包括subj
        ensure_channel = EnsureChannelFirstd(keys=all_modalities_for_load)
        transforms.append(ensure_channel)
        
        # 处理配置中定义的每个预处理步骤
        for step_name, params in self.config["Preprocessing"].items():
            # Extract modalities from params
            modalities = params.get("images", [])
            # Use lowercase keys for PreprocessorFactory compatible with MONAI FIXME
            modalities = [mod.lower() for mod in modalities]
            if not modalities:
                raise ValueError(f"No modalities specified for {step_name}")
            
            # add mask
            if "mask" not in modalities:
                modalities.append("mask")

            # Create a single preprocessor for all modalities
            processor = PreprocessorFactory.create(
                name=step_name,
                keys=modalities,
                **{k: v for k, v in params.items() if k != "images"}
            )
            transforms.append(processor)
            
        # Create a single Compose object with all transforms
        return Compose(transforms)
        
    def process_batch(self) -> None:
        """Process all subjects in parallel using MONAI's DataLoader.
        
        This method processes all subjects in parallel using MONAI's DataLoader.
        It first gets all image and mask paths, then creates a dataset and data loader
        for parallel processing.
        """
        output_root = Path(self.config["out_dir"])
        output_root.mkdir(parents=True, exist_ok=True)
        
        # Get all image and mask paths
        images_paths, mask_paths = get_image_and_mask_paths(self.config["data_dir"])
        
        subject_data_list = []
        for subject in images_paths.keys():
            data_entry = {"subj": subject}
            
            # 遍历当前科目的所有图像路径
            for scan_type, img_path in images_paths[subject].items():
                # 使用小写作为键名，与LoadImage的keys参数匹配
                data_entry[scan_type.lower()] = img_path
                
                # 只添加一个mask，使用第一个可用的scan_type的mask
                if "mask" not in data_entry and subject in mask_paths and scan_type in mask_paths[subject]:
                    mask_path = mask_paths[subject][scan_type]
                    data_entry["mask"] = mask_path
                    
            # 将生成的数据条目加入到列表中
            if "mask" in data_entry:  # 确保有mask才添加
                subject_data_list.append(data_entry)
            else:
                self.logger.warning(f"No mask found for subject {subject}, skipping")
            
        if not subject_data_list:
            self.logger.warning("No valid subjects found")
            return
            
        # Create dataset
        dataset = Dataset(
            data=subject_data_list,
            transform=self.preprocessors
        )
        
        # Create data loader - use single process mode on Windows to avoid errors
        dataloader = DataLoader(
            dataset,
            batch_size=1,  # 小批次以减少内存使用   FIXME
            num_workers=0,  # 设置为0使用主进程，避免Windows多进程问题 FIXME
            collate_fn=identity_collate,  # 使用命名函数而不是lambda函数
            pin_memory=False,  # 避免内存溢出
            persistent_workers=False,  # 避免工作进程持久化导致的内存累积
            shuffle=False  # 保持处理顺序一致
        )
        
        # Process batches with progress bar from utils
        progress_bar = CustomTqdm(total=len(subject_data_list), desc="Processing subjects")
        for batch_idx, batch in enumerate(dataloader):
            self.logger.info(f"Processing batch {batch_idx+1}")
            for subject in batch:
                try:
                    self._save_processed_batch(subject, output_root)
                    progress_bar.update(1)
                except Exception as e:
                    self.logger.error(f"Error processing subject {subject['subj']}: {e}")
                    
        progress_bar.close()
        self.logger.info("Batch processing completed")
        
    def _save_processed_batch(
        self,
        subject: Dict,
        output_root: Path
    ) -> None:
        """Save processed batch data following the structure:
        data_root/processed_images/
        ├── images/
        │   └── subject1/
        │       ├── pre_contrast/
        │       │   └── image.nii.gz
        │       └── ...
        └── masks/
            └── subject1/
                ├── pre_contrast/
                │   └── mask.nii.gz
                └── ...
        
        Args:
            subject (Dict): Processed subject data.
            output_root (Path): Root directory for output.
        """
        # Create base directories
        out_dir = output_root / "processed_images"
        images_dir = out_dir / "images"
        masks_dir = out_dir / "masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        
        subject_id = subject["subj"]
        
        # Get image modality keys - exclude metadata keys and special keys
        image_keys = [k for k in subject.keys() 
                    if k not in ["subj", "mask"] and not k.endswith("_meta_dict")]
        
        # Process each modality (scan type)
        for key in image_keys:
            if key in subject:
                # Get image data and metadata
                image_data = subject[key]
                # image data is 4D numpy array
                if image_data.ndim == 4:
                    # Get first 3D image
                    image_data = image_data[0, :, :, :]
                
                # Transpose the array to match SimpleITK's coordinate system (z,y,x)
                if isinstance(image_data, np.ndarray):
                    image_data = np.transpose(image_data, (2, 1, 0))
                elif isinstance(image_data, torch.Tensor):
                    image_data = image_data.permute(2, 1, 0).numpy()
                
                metadata_key = f"{key}_meta_dict"
                
                if metadata_key in subject:
                    metadata = subject[metadata_key]
                    
                    # Convert numpy array to sitk image
                    image = sitk.GetImageFromArray(image_data)
                        
                    # Copy metadata to the new image
                    for meta_key in metadata.keys():
                        try:
                            image.SetMetaData(meta_key, str(metadata[meta_key]))
                        except Exception as e:
                            self.logger.warning(f"Failed to set metadata {meta_key} for {key}: {e}")
                    
                    # Create scan type directory under images
                    scan_type_dir = images_dir / subject_id / key
                    scan_type_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save image
                    output_path = scan_type_dir / "image.nii.gz"
                    sitk.WriteImage(image, str(output_path))
                    
                    # If there's a corresponding mask, save it
                    if "mask" in subject:
                        mask_data = subject["mask"]
                        # mask data is 4D numpy array
                        if mask_data.ndim == 4:
                            mask_data = mask_data[0, :, :, :]
                            
                        # Transpose the mask to match SimpleITK's coordinate system
                        if isinstance(mask_data, np.ndarray):
                            mask_data = np.transpose(mask_data, (2, 1, 0))
                        elif isinstance(mask_data, torch.Tensor):
                            mask_data = mask_data.permute(2, 1, 0).numpy()
                            
                        mask_metadata_key = "mask_meta_dict"
                        
                        if mask_metadata_key in subject:
                            mask_metadata = subject[mask_metadata_key]
                            
                            # Convert mask data to sitk image
                            mask = sitk.GetImageFromArray(mask_data)
                            
                            # Copy metadata to mask
                            for meta_key in mask_metadata.keys():
                                try:
                                    mask.SetMetaData(meta_key, str(mask_metadata[meta_key]))
                                except Exception as e:
                                    self.logger.warning(f"Failed to set metadata {meta_key} for mask: {e}")
                            
                            # Create scan type directory under masks
                            mask_scan_type_dir = masks_dir / subject_id / key
                            mask_scan_type_dir.mkdir(parents=True, exist_ok=True)
                            
                            # Save mask
                            mask_output_path = mask_scan_type_dir / "mask.nii.gz"
                            sitk.WriteImage(mask, str(mask_output_path))
                else:
                    self.logger.warning(f"No metadata found for {key}, skipping save") 