"""
Batch processing module for medical image preprocessing.

This module provides functionality for parallel batch processing of medical images
using multiple preprocessing steps defined in a configuration file.

Example:
    >>> from habit.core.preprocessing.image_processor_pipeline import BatchProcessor
    >>> processor = BatchProcessor(config_path="./config/config_kmeans.yaml")
    >>> processor.process_batch()
"""

from typing import Dict, List, Optional, Union, Callable, Any
import logging
from pathlib import Path
import SimpleITK as sitk
import numpy as np
import os
from habit.core.preprocessing.preprocessor_factory import PreprocessorFactory
from habit.utils.io_utils import get_image_and_mask_paths
from habit.utils.progress_utils import CustomTqdm
from habit.core.common.config_loader import load_config
from habit.utils.log_utils import setup_logger, get_module_logger
import multiprocessing
import traceback
from habit.core.preprocessing.load_image import LoadImagePreprocessor
from habit.core.preprocessing.config_schemas import PreprocessingConfig


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
        # Load and validate config with schema to catch errors early
        raw_config = load_config(config_path)
        self.config_obj = PreprocessingConfig(**raw_config)
        
        self.output_root = Path(self.config_obj.out_dir)
        self.output_root.mkdir(parents=True, exist_ok=True)
        # 不再创建全局预处理器，而是为每个样本单独创建
        self.verbose = verbose
        self._setup_logging(log_level)
        
        # Parse save_options from config
        self._parse_save_options()
        
        # Use config value or default to 0 if not specified or invalid
        try:
            self.num_workers = int(self.config_obj.processes if self.config_obj.processes is not None else num_workers)
            self.num_workers = min(self.num_workers, multiprocessing.cpu_count() - 2)
            # 确保最小为1个进程，即使设置为0也使用1个
            if self.num_workers <= 0:
                self.logger.warning(f"Setting num_workers to 1 (was {self.num_workers})")
                self.num_workers = 1
        except (ValueError, TypeError):
            self.logger.warning(f"Invalid 'processes' type in config, defaulting to 1.")
            self.num_workers = 1

        self.logger.info(f"Using {self.num_workers} worker processes.")
    
    def _parse_save_options(self) -> None:
        """Parse save_options from configuration.
        
        Configures intermediate result saving behavior:
        - save_intermediate: Whether to save intermediate results (default: False)
        - intermediate_steps: List of step names to save, empty means save all steps
        """
        save_options = self.config_obj.save_options
        
        # Whether to save intermediate results (default: True to save all intermediate steps)
        self.save_intermediate = save_options.save_intermediate
        
        # Which steps to save (empty list means save all steps)
        self.intermediate_steps = save_options.intermediate_steps
        
        if self.save_intermediate:
            if self.intermediate_steps:
                self.logger.info(f"Intermediate results will be saved for steps: {self.intermediate_steps}")
            else:
                self.logger.info("Intermediate results will be saved for all steps")
    
    def _should_save_step(self, step_name: str) -> bool:
        """Check if intermediate results should be saved for a given step.
        
        Args:
            step_name (str): Name of the preprocessing step.
            
        Returns:
            bool: True if results should be saved for this step.
        """
        if not self.save_intermediate:
            return False
        
        # If intermediate_steps is empty, save all steps
        if not self.intermediate_steps:
            return True
        
        # Otherwise, only save specified steps
        return step_name in self.intermediate_steps
    
    def _save_step_results(
        self, 
        subject_data: Dict, 
        step_name: str, 
        step_index: int,
        modalities: List[str]
    ) -> None:
        """Save intermediate results after a preprocessing step.
        
        Output structure matches final output:
        step_name_01/
        ├── images/subject_id/modality/modality.nii.gz
        └── masks/subject_id/modality/mask_modality.nii.gz
        
        Args:
            subject_data (Dict): Subject data dictionary containing processed images.
            step_name (str): Name of the preprocessing step.
            step_index (int): Index of the step in the pipeline (1-based).
            modalities (List[str]): List of modality keys that were processed.
        """
        subject_id = subject_data.get('subj', 'unknown')
        
        # Create step output directory: step_name_01/
        step_dir_name = f"{step_name}_{step_index:02d}"
        step_dir = self.output_root / step_dir_name
        images_base = step_dir / "images" / subject_id
        masks_base = step_dir / "masks" / subject_id
        
        try:
            # Save processed images for this step
            for mod in modalities:
                # Create modality-specific directories (same structure as final output)
                images_dir = images_base / mod
                masks_dir = masks_base / mod
                images_dir.mkdir(parents=True, exist_ok=True)
                masks_dir.mkdir(parents=True, exist_ok=True)
                
                # Save image
                if mod in subject_data:
                    image = subject_data[mod]
                    if isinstance(image, sitk.Image):
                        output_path = images_dir / f"{mod}.nii.gz"
                        sitk.WriteImage(image, str(output_path))
                        self.logger.debug(f"[{subject_id}] Saved {mod} to {output_path}")
                    elif isinstance(image, np.ndarray):
                        sitk_image = sitk.GetImageFromArray(image)
                        output_path = images_dir / f"{mod}.nii.gz"
                        sitk.WriteImage(sitk_image, str(output_path))
                        self.logger.debug(f"[{subject_id}] Saved {mod} to {output_path}")
                
                # Save corresponding mask if exists
                mask_key = f"mask_{mod}"
                if mask_key in subject_data:
                    mask = subject_data[mask_key]
                    if isinstance(mask, sitk.Image):
                        mask_path = masks_dir / f"{mask_key}.nii.gz"
                        sitk.WriteImage(mask, str(mask_path))
                        self.logger.debug(f"[{subject_id}] Saved {mask_key} to {mask_path}")
                    elif isinstance(mask, np.ndarray):
                        sitk_mask = sitk.GetImageFromArray(mask)
                        mask_path = masks_dir / f"{mask_key}.nii.gz"
                        sitk.WriteImage(sitk_mask, str(mask_path))
                        self.logger.debug(f"[{subject_id}] Saved {mask_key} to {mask_path}")
            
            self.logger.info(f"[{subject_id}] Saved intermediate results for step: {step_name}")
            
        except Exception as e:
            self.logger.error(f"[{subject_id}] Error saving intermediate results for {step_name}: {e}")
            
    def _setup_logging(self, log_level: str) -> None:
        """Setup logging configuration using centralized log system.
        
        If logging has already been configured by the CLI entry point,
        this will simply get the existing logger. Otherwise, it will
        set up a new logger with file output.
        
        Also stores log configuration for child processes (multiprocessing support).
        
        Args:
            log_level (str): Logging level.
        """
        from habit.utils.log_utils import LoggerManager
        
        manager = LoggerManager()
        
        # Check if root logger already has handlers (configured by CLI)
        if manager.get_log_file() is not None:
            # Logging already configured by CLI, just get module logger
            self.logger = get_module_logger('preprocessing')
            self.logger.info("Using existing logging configuration from CLI entry point")
            
            # Store log configuration for child processes (Windows spawn mode)
            self._log_file_path = manager.get_log_file()
            self._log_level = manager._root_logger.getEffectiveLevel() if manager._root_logger else logging.INFO
        else:
            # Logging not configured yet (e.g., direct BatchProcessor usage)
            # Set up logging with file output
            level = getattr(logging, log_level.upper())
            self.logger = setup_logger(
                name='preprocessing',
                output_dir=self.output_root,
                log_filename='processing.log',
                level=level
            )
            
            # Store log configuration for child processes
            self._log_file_path = manager.get_log_file()
            self._log_level = level
    
    def _ensure_logging_in_subprocess(self) -> None:
        """Ensure logging is properly configured in child processes.
        
        In Windows spawn mode (and forkserver), child processes don't inherit
        the parent's logging configuration. This method restores it.
        """
        from habit.utils.log_utils import restore_logging_in_subprocess
        
        # Restore logging if we have stored config
        if hasattr(self, '_log_file_path') and self._log_file_path:
            restore_logging_in_subprocess(self._log_file_path, self._log_level)
        
    def _process_single_subject(self, subject_data):
        """Process a single subject's data through the preprocessing pipeline.
        
        Args:
            subject_data (Dict): Subject data dictionary containing subject_id and output directory info.

        Returns:
            tuple: (subject_id, processing result message)
        """
        # Restore logging configuration in child process (for multiprocessing)
        self._ensure_logging_in_subprocess()
        
        try:
            subject_id = subject_data['subj']
            self.logger.info(f"Processing subject: {subject_id}")
            
            # Step 1: Load images first
            load_keys = [step_config.images for step_config in self.config_obj.Preprocessing.values()]
            load_keys = [item for sublist in load_keys for item in sublist]
            load_keys = list(set(load_keys))
            mask_keys = [f"mask_{mod}" for mod in load_keys]
            load_keys.extend(mask_keys)
            load_image = LoadImagePreprocessor(keys=load_keys)
            load_image(subject_data)

            # Store original output_dirs for restoration after intermediate saves
            original_output_dirs = subject_data.get('output_dirs', {}).copy()
            
            # Step 2: Process each preprocessing step defined in config
            step_index = 0
            for step_name, step_config in self.config_obj.Preprocessing.items():
                step_index += 1
                
                # Extract modalities from params
                modalities = step_config.images
                if not modalities:
                    continue
                
                # Only process modalities that exist in current subject_data
                modalities = [mod for mod in modalities if mod in subject_data]
                if not modalities:
                    self.logger.warning(f"[{subject_id}] No valid modalities for step {step_name}, skipping")
                    continue
                
                # If saving intermediate results for this step, temporarily update output_dirs
                # This allows preprocessors like dcm2niix to save directly to the intermediate directory
                if self._should_save_step(step_name):
                    step_dir_name = f"{step_name}_{step_index:02d}"
                    step_output_dirs = {}
                    for mod in modalities:
                        # Create intermediate output directory path
                        step_images_dir = self.output_root / step_dir_name / "images" / subject_id / mod
                        step_images_dir.mkdir(parents=True, exist_ok=True)
                        step_output_dirs[mod] = str(step_images_dir)
                        
                        # Also set mask output directory
                        mask_key = f"mask_{mod}"
                        if mask_key in subject_data:
                            step_masks_dir = self.output_root / step_dir_name / "masks" / subject_id / mod
                            step_masks_dir.mkdir(parents=True, exist_ok=True)
                            step_output_dirs[mask_key] = str(step_masks_dir)
                    
                    subject_data['output_dirs'] = step_output_dirs
                    self.logger.debug(f"[{subject_id}] Set intermediate output_dirs for step {step_name}")
                
                # Create and execute preprocessor
                params = self._model_to_dict(step_config)
                processor = PreprocessorFactory.create(
                    name=step_name,
                    keys=modalities,
                    **{k: v for k, v in params.items() if k != "images"}
                )
                processor(subject_data)
                
                # Save intermediate results if configured (for preprocessors that don't save directly)
                # Skip _save_step_results for dcm2nii since it saves files directly to output_dirs
                if self._should_save_step(step_name):
                    if step_name != "dcm2nii":
                        self._save_step_results(subject_data, step_name, step_index, modalities)
                    else:
                        self.logger.debug(f"[{subject_id}] Skipping _save_step_results for dcm2nii (already saved directly)")
                    # Restore original output_dirs after saving
                    subject_data['output_dirs'] = original_output_dirs.copy()

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

                        # if meta_dict.get("dcm2niix_converted", False) and "output_file_path" in meta_dict:
                            # File was already saved by dcm2niix, but does not need to skip saving
                            # self.logger.info(f"Skipping save for {key}: already saved by dcm2niix at {meta_dict['output_file_path']}")
                            # pass
                    
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
            # 报告错误的被试名
            subject_id = subject_data.get('subj', 'unknown')
            self.logger.error(f"Error saving processed images for subject {subject_id}: {str(e)}")

            # raise

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
        # 如果配置文件中没有指定 auto_select_first_file，则使用默认值 True
        auto_select = self.config_obj.auto_select_first_file
        images_paths, mask_paths = get_image_and_mask_paths(self.config_obj.data_dir, auto_select_first_file=auto_select)
        
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

    def _model_to_dict(self, model: Any) -> Dict[str, Any]:
        """
        Convert pydantic model to a plain dict with v1/v2 compatibility.

        Args:
            model (Any): Pydantic model instance.

        Returns:
            Dict[str, Any]: Serialized model data.
        """
        if hasattr(model, "model_dump"):
            return model.model_dump()
        return model.dict()