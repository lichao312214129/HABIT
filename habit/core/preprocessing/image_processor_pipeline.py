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
import os
import platform
from multiprocessing import cpu_count
import multiprocessing
from functools import partial
import traceback
from multiprocessing import Pool

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
                number of CPU cores - 1 on Linux/Mac, or 0 on Windows.
            log_level (str): Logging level. Defaults to "INFO".
        """
        self.config = self._load_config(config_path)
        self._setup_logging(log_level)
        self.preprocessors = self._create_preprocessors()
        
        # --- 强制使用单进程进行测试 ---
        self.logger.warning("Forcing num_workers=0 for debugging purposes.")
        self.num_workers = 0 
        # --- 原来的逻辑 (注释掉) ---
        # # Set number of workers based on system
        # if num_workers is None:
        #     if platform.system() == 'Windows':
        #         self.num_workers = 0  # Use single process on Windows
        #     else:
        #         # Use config value or default to 0 if not specified or invalid
        #         try:
        #             self.num_workers = int(self.config.get("processes", 0))
        #             if self.num_workers < 0:
        #                 self.logger.warning(f"Invalid 'processes' value ({self.num_workers}) in config, defaulting to 0.")
        #                 self.num_workers = 0
        #         except (ValueError, TypeError):
        #             self.logger.warning(f"Invalid 'processes' type in config, defaulting to 0.")
        #             self.num_workers = 0 
        # else:
        #     self.num_workers = num_workers
        # self.logger.info(f"Using {self.num_workers} worker processes.")
        # --- 结束原来的逻辑 ---
        
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
        
    def _process_single_subject(self, subject_data, output_dir, progress_queue=None):
        """处理单个主题的数据。
        
        Args:
            subject_data (Dict): 单个主题的数据字典
            output_dir (Path): 输出目录
            progress_queue (Queue, optional): 用于报告进度的队列

        Returns:
            str: 处理结果消息
        """
        try:
            subject_id = subject_data['subj']
            # 创建单独的预处理管道实例
            transforms = []
            
            # 添加LoadImage转换
            all_modalities_for_load = [k for k in subject_data.keys() if k not in ["subj"]]
            
            image_loader = LoadImaged(
                keys=all_modalities_for_load,
                reader=ITKReader(reverse_indexing=True),
                image_only=False
            )
            transforms.append(image_loader)
            
            # 添加EnsureChannelFirst转换
            ensure_channel = EnsureChannelFirstd(keys=all_modalities_for_load)
            transforms.append(ensure_channel)
            
            # 处理配置中定义的每个预处理步骤
            for step_name, params in self.config["Preprocessing"].items():
                # Extract modalities from params
                modalities = params.get("images", [])
                modalities = [mod.lower() for mod in modalities]
                if not modalities:
                    continue
                
                # add mask
                if "mask" not in modalities:
                    modalities.append("mask")
                
                # 仅处理存在于当前subject_data中的modalities
                modalities = [mod for mod in modalities if mod in subject_data]
                
                # 创建预处理器
                processor = PreprocessorFactory.create(
                    name=step_name,
                    keys=modalities,
                    **{k: v for k, v in params.items() if k != "images"}
                )
                transforms.append(processor)
            
            # 创建组合转换
            pipeline = Compose(transforms)
            
            # 应用预处理
            processed_data = pipeline(subject_data)
            
            # 保存处理后的数据
            self._save_processed_data(processed_data, output_dir)
            
            # 报告进度
            if progress_queue is not None:
                progress_queue.put(subject_id)
            
            return f"Success: {subject_id}"
        except Exception as e:
            return f"Error processing {subject_data.get('subj', 'unknown')}: {str(e)}\n{traceback.format_exc()}"

    def _save_processed_data(self, processed_data, output_dir):
        """保存已处理的数据。
        
        Args:
            processed_data (Dict): 处理后的数据
            output_dir (Path): 输出根目录
        """
        if "subj" not in processed_data:
            self.logger.error("缺少subject ID，无法保存数据")
            return
        
        subject_id = processed_data["subj"]
        
        # 创建基本目录
        out_dir = Path(output_dir)
        images_dir = out_dir / "processed_images" / "images"
        masks_dir = out_dir / "processed_images" / "masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取图像模态键 - 排除元数据键和特殊键
        image_keys = [k for k in processed_data.keys() 
                    if k not in ["subj", "mask"] and not k.endswith("_meta_dict")]
        
        # 处理每个模态
        for key in image_keys:
            if key in processed_data:
                # 获取图像数据和元数据
                image_data = processed_data[key]
                # 图像数据是4D numpy数组
                if image_data.ndim == 4:
                    # 获取第一个3D图像
                    image_data = image_data[0, :, :, :]
                
                # 转置数组以匹配SimpleITK的坐标系统 (z,y,x)
                if isinstance(image_data, np.ndarray):
                    image_data = np.transpose(image_data, (2, 1, 0))
                elif isinstance(image_data, torch.Tensor):
                    image_data = image_data.permute(2, 1, 0).numpy()
                
                metadata_key = f"{key}_meta_dict"
                
                if metadata_key in processed_data:
                    metadata = processed_data[metadata_key]
                    
                    # 将numpy数组转换为sitk图像
                    image = sitk.GetImageFromArray(image_data)
                        
                    # 创建扫描类型目录
                    scan_type_dir = images_dir / subject_id / key
                    scan_type_dir.mkdir(parents=True, exist_ok=True)
                    
                    # 保存图像
                    output_path = scan_type_dir / "image.nii.gz"
                    sitk.WriteImage(image, str(output_path))
                    
                    # 如果有相应的mask，保存它
                    if "mask" in processed_data:
                        mask_data = processed_data["mask"]
                        # mask数据是4D numpy数组
                        if mask_data.ndim == 4:
                            mask_data = mask_data[0, :, :, :]
                            
                        # 转置mask以匹配SimpleITK的坐标系统
                        if isinstance(mask_data, np.ndarray):
                            mask_data = np.transpose(mask_data, (2, 1, 0))
                        elif isinstance(mask_data, torch.Tensor):
                            mask_data = mask_data.permute(2, 1, 0).numpy()
                            
                        mask_metadata_key = "mask_meta_dict"
                        
                        if mask_metadata_key in processed_data:
                            mask_metadata = processed_data[mask_metadata_key]
                            
                            # 将mask数据转换为sitk图像
                            mask = sitk.GetImageFromArray(mask_data)
                            
                            # 创建mask扫描类型目录
                            mask_scan_type_dir = masks_dir / subject_id / key
                            mask_scan_type_dir.mkdir(parents=True, exist_ok=True)
                            
                            # 保存mask
                            mask_output_path = mask_scan_type_dir / "mask.nii.gz"
                            sitk.WriteImage(mask, str(mask_output_path))

    def _update_progress_from_queue(self, progress_queue, total_subjects, progress_bar):
        """从队列更新进度信息。
        
        Args:
            progress_queue (Queue): 进度队列
            total_subjects (int): 总主题数
            progress_bar (CustomTqdm): 进度条对象
        """
        completed = 0
        while completed < total_subjects:
            subject_id = progress_queue.get()
            progress_bar.update(1)
            completed += 1
            self.logger.info(f"完成处理主题: {subject_id} ({completed}/{total_subjects})")

    def process_batch(self) -> None:
        """处理所有主题数据。
        
        使用自定义的并行处理替代MONAI的DataLoader，更好地处理Windows和多进程环境。
        """
        output_root = Path(self.config["out_dir"])
        output_root.mkdir(parents=True, exist_ok=True)
        
        # 获取所有图像和mask路径
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
            
        # 获取处理器数量
        if platform.system() == 'Windows':
            num_processes = 0  # Windows下默认使用单进程
        else:
            num_processes = self.config.get("processes", min(multiprocessing.cpu_count() - 1, 1))  # 默认使用CPU核心数-1
        
        self.logger.info(f"将使用 {num_processes} 个进程进行处理")
        total_subjects = len(subject_data_list)
        self.logger.info(f"共有 {total_subjects} 个主题需要处理")
        
        # 单进程处理
        if num_processes <= 0:
            self._process_batch_single_thread(subject_data_list, output_root, total_subjects)
        # 多进程处理
        else:
            self._process_batch_multi_thread(subject_data_list, output_root, total_subjects, num_processes)
        
        self.logger.info("批处理完成")

    def _process_batch_single_thread(self, subject_data_list, output_root, total_subjects):
        """使用单线程处理批量数据。
        
        Args:
            subject_data_list (List[Dict]): 主题数据列表
            output_root (Path): 输出目录
            total_subjects (int): 总主题数
        """
        self.logger.info("使用单进程模式处理")
        progress_bar = CustomTqdm(total=total_subjects, desc="Processing subjects")
        
        for subject_data in subject_data_list:
            try:
                self.logger.info(f"开始处理主题: {subject_data['subj']}")
                # 应用预处理转换
                processed_data = self.preprocessors(subject_data)
                # 保存处理后的数据
                self._save_processed_batch(processed_data, output_root)
                progress_bar.update(1)
                self.logger.info(f"完成处理主题: {subject_data['subj']} ({progress_bar.n}/{total_subjects})")
            except Exception as e:
                self.logger.error(f"处理主题 {subject_data['subj']} 时出错: {e}")
                self.logger.error(traceback.format_exc())
                
        progress_bar.close()

    def _process_batch_multi_thread(self, subject_data_list, output_root, total_subjects, num_processes):
        """使用多线程处理批量数据，使用与habitat_analysis模块相同的方式。
        
        Args:
            subject_data_list (List[Dict]): 主题数据列表
            output_root (Path): 输出目录
            total_subjects (int): 总主题数
            num_processes (int): 进程数
        """
        self.logger.info("使用多进程模式处理")
        
        # 创建进度队列
        manager = multiprocessing.Manager()
        progress_queue = manager.Queue()
        
        # 创建进度条
        progress_bar = CustomTqdm(total=total_subjects, desc="Processing subjects")
        
        # 启动进度更新线程
        import threading
        progress_thread = threading.Thread(
            target=self._update_progress_from_queue, 
            args=(progress_queue, total_subjects, progress_bar)
        )
        progress_thread.daemon = True
        progress_thread.start()
        
        # 为函数准备参数
        def process_wrapper(subject_data):
            return self._process_single_subject(subject_data, output_root, progress_queue)
        
        # 使用与habitat_analysis模块相同的方式处理
        with multiprocessing.Pool(processes=num_processes) as pool:
            self.logger.info("开始并行处理所有主题...")
            results_iter = pool.imap_unordered(process_wrapper, subject_data_list)
            
            # 处理结果
            for result in results_iter:
                if result.startswith("Error"):
                    self.logger.error(result)
        
        # 关闭进度条
        progress_bar.close()
        
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

            
            