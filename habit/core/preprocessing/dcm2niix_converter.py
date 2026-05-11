"""
DICOM to NIfTI converter using dcm2niix tool.

This module provides functionality to batch convert DICOM files to NIfTI format
using the dcm2niix tool, with integration into the HABIT preprocessing pipeline.
"""

import shutil
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import tempfile
import SimpleITK as sitk
import json

from .base_preprocessor import BasePreprocessor
from .dcm2niix_runner import Dcm2niixRunner
from .preprocessor_factory import PreprocessorFactory
from habit.utils.progress_utils import CustomTqdm
from habit.utils.file_system_utils import safe_mkdir


@PreprocessorFactory.register("dcm2nii")
class Dcm2niixConverter(BasePreprocessor):
    """
    Convert DICOM files to NIfTI format using dcm2niix tool.
    
    This preprocessor takes DICOM directories and converts them to NIfTI format
    using the dcm2niix command-line tool. It supports batch processing and
    integrates with the HABIT preprocessing pipeline.
    """
    
    def __init__(
        self,
        keys: Union[str, List[str]], # 默认不指定键
        dcm2niix_path: Optional[str] = None, # 默认不指定dcm2niix可执行文件路径
        filename_format: Optional[str] = None, # 默认不指定输出文件名格式
        adjacent_dicoms: bool = True, # 默认相邻DICOMs在同一文件夹
        compress: bool = True, # 默认压缩输出文件
        anonymize: bool = False, # 默认不匿名化文件名
        ignore_derived: bool = False, # 默认不忽略衍生图像
        crop_images: bool = False, # 默认裁剪图像
        generate_json: bool = False, # Default: no BIDS JSON sidecar (-b n); set True for -b y
        verbose: bool = False, # 默认不输出详细信息
        batch_mode: bool = True, # 默认启用批处理模式
        merge_slices: Optional[str] = "2", # 合并模式: "y"/"1"=默认, "2"=按序列, "n"/"0"=不合并, None=不指定
        single_file_mode: Optional[bool] = None, # 单文件模式: True=强制单文件(-s y), False=允许多文件(-s n), None=不指定(推荐)
        allow_missing_keys: bool = False, # 默认不允许缺失键
        **kwargs
    ):
        """
        Initialize the dcm2niix converter.
        
        Args:
            keys (Union[str, List[str]]): Keys containing DICOM directory paths to convert
            dcm2niix_path (Optional[str]): Full path to dcm2niix executable or directory containing it
            filename_format (Optional[str]): Output filename format (default: uses subject name)
            compress (bool): Compress output files (default: True)
            anonymize (bool): Anonymize filenames (default: False)
            ignore_derived (bool): Ignore derived images (default: False)
            crop_images (bool): Crop images (default: False)
            generate_json (bool): Generate BIDS JSON sidecar files (default: False)
            verbose (bool): Verbose output (default: False)
            batch_mode (bool): Enable batch mode (default: True)
            merge_slices (Optional[str]): Merge mode - "y"/"1"=default merge, "2"=merge by series, 
                                         "n"/"0"=no merge, None=don't specify (default: "2")
            single_file_mode (Optional[bool]): Single file mode - True=force single file (-s y), 
                                              False=allow multiple (-s n), None=don't specify/use default (default: None)
            allow_missing_keys (bool): Allow missing keys (default: False)
            **kwargs: Additional parameters
        """
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        
        # Setup logging first
        from habit.utils.log_utils import get_module_logger
        self.logger = get_module_logger(__name__)
        
        self.runner = Dcm2niixRunner(dcm2niix_path)
        
        self.filename_format = filename_format
        self.compress = compress
        self.adjacent_dicoms = adjacent_dicoms
        self.anonymize = anonymize
        self.ignore_derived = ignore_derived
        self.crop_images = crop_images
        self.generate_json = generate_json
        self.verbose = verbose
        self.batch_mode = batch_mode
        self.merge_slices = merge_slices
        self.single_file_mode = single_file_mode
    
    def _build_dcm2niix_command(
        self, 
        input_dir: str, 
        output_dir: str, 
        filename_format: Optional[str] = None
    ) -> List[str]:
        """
        Build dcm2niix command with specified parameters.
        
        Args:
            input_dir (str): Input DICOM directory
            output_dir (str): Output directory for NIfTI files
            filename_format (Optional[str]): Output filename format
            
        Returns:
            List[str]: Command components for subprocess execution
        """
        cmd = []
        
        # Add filename format if specified
        if filename_format:
            cmd.extend(["-f", filename_format])
        
        # -a  adjacent DICOMs (images from same series always in same folder) for faster conversion (n/y, default n)

        if self.adjacent_dicoms:
            cmd.extend(["-a", "y"])
        else:
            cmd.extend(["-a", "n"])

        # Control JSON generation (BIDS sidecar)
        if self.generate_json:
            cmd.extend(["-b", "y"])
        else:
            cmd.extend(["-b", "n"])
        
        # Add boolean options
        if self.ignore_derived:
            cmd.extend(["-i", "y"])
        
        if self.batch_mode:
            cmd.extend(["-l", "y"])
        
        # Merge 2D slices into 3D volumes
        # Note: -m parameter controls slice merging behavior
        # -m 0 or -m n: no merging
        # -m 1 or -m y: merge 2D slices (default)
        # -m 2: merge based on series (more aggressive)
        # None: don't specify, use dcm2niix default
        if self.merge_slices is not None:
            cmd.extend(["-m", str(self.merge_slices)])
        
        if self.anonymize:
            cmd.extend(["-p", "y"])
        
        # Single file mode
        # Note: -s y may force 4D structure, -s n splits volumes
        # For 3D output, it's often best to NOT specify this parameter
        if self.single_file_mode is not None:
            if self.single_file_mode:
                cmd.extend(["-s", "y"])
            else:
                cmd.extend(["-s", "n"])
        
        if self.crop_images:
            cmd.extend(["-x", "y"])
        
        if self.verbose:
            cmd.extend(["-v", "y"])
        
        if self.compress:
            cmd.extend(["-z", "y"])
        
        # Add output directory
        cmd.extend(["-o", output_dir])
        
        # Add input directory
        cmd.append(input_dir)
        
        return cmd
    
    def _convert_single_dicom_dir(
        self, 
        input_dir: str, 
        subject_id: str,
        sequence_name: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, sitk.Image]:
        """
        Convert a single DICOM directory to NIfTI format and return as SimpleITK Image objects.
        
        Args:
            input_dir (str): Input DICOM directory path
            subject_id (str): Subject identifier
            sequence_name (Optional[str]): Sequence name for filename formatting
            output_dir (Optional[str]): Output directory for converted files. If None, uses temporary directory
            
        Returns:
            Dict[str, sitk.Image]: Dictionary containing SimpleITK Image objects
            
        Raises:
            RuntimeError: If conversion fails
        """
        input_path = Path(input_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
        
        if not input_path.is_dir():
            raise ValueError(f"Input path is not a directory: {input_dir}")
        
        # Determine output directory
        if output_dir:
            # Use provided output directory
            subject_output_dir = Path(output_dir)
            safe_mkdir(str(subject_output_dir))
            use_temp_dir = False
            self.logger.info(f"[{subject_id}] Using output directory: {subject_output_dir}")
        else:
            # Create temporary directory for conversion
            temp_dir = tempfile.mkdtemp(prefix=f"dcm2niix_{subject_id}_")
            subject_output_dir = Path(temp_dir)
            use_temp_dir = True
            self.logger.debug(f"[{subject_id}] Using temporary directory: {subject_output_dir}")
        
        try:
            # Determine filename format
            filename_format = self.filename_format
            if not filename_format:
                if sequence_name:
                    filename_format = f"{subject_id}_{sequence_name}"
                else:
                    filename_format = subject_id
            
            # Build dcm2niix command as list
            cmd_list = self._build_dcm2niix_command(
                str(input_path), 
                str(subject_output_dir), 
                filename_format
            )
            
            self.logger.info(f"[{subject_id}] Converting DICOM directory: {input_dir}")
            self.runner.run_convert(cmd_list, subject_id=subject_id)
            
            # Find converted files and load as SimpleITK Image objects
            converted_images = {}
            for ext in ['.nii', '.nii.gz']:
                pattern = f"*{ext}"
                nifti_files = list(subject_output_dir.glob(pattern))
                
                for nifti_file in nifti_files:
                    # Extract sequence name from filename if possible
                    file_stem = nifti_file.stem.replace('.nii', '')
                    if sequence_name:
                        key = sequence_name
                    else:
                        # Try to extract sequence info from filename
                        key = file_stem.replace(subject_id, '').strip('_')
                        if not key:
                            key = 'image'
                    
                    # Load as SimpleITK Image object
                    try:
                        sitk_image = sitk.ReadImage(str(nifti_file))
                        converted_images[key] = sitk_image
                        self.logger.debug(f"[{subject_id}] Loaded {nifti_file} as SimpleITK Image")
                        
                        # Store the output file path
                        converted_images[f"{key}_output_path"] = str(nifti_file)
                        
                        # Find corresponding JSON file (supports .nii.gz -> .json)
                        if str(nifti_file).endswith('.nii.gz'):
                            json_file = Path(str(nifti_file)[:-7] + '.json')
                        else:
                            json_file = nifti_file.with_suffix('.json')
                        if json_file.exists():
                            with open(json_file, 'r') as f:
                                json_metadata = json.load(f)
                            
                            # Add JSON metadata to image metadata
                            meta_key = f"{key}_meta_dict"
                            if meta_key not in converted_images:
                                converted_images[meta_key] = {}
                            
                            converted_images[meta_key]['dcm2niix_json'] = json_metadata
                            self.logger.debug(f"[{subject_id}] Loaded JSON metadata for {key}")
                        
                    except Exception as e:
                        self.logger.error(f"[{subject_id}] Failed to load {nifti_file} as SimpleITK Image: {e}")
                        raise RuntimeError(f"Failed to load converted NIfTI file: {e}")
            
            if not converted_images:
                raise RuntimeError(f"No NIfTI files were created for {input_dir}")
            
            self.logger.debug(f"[{subject_id}] Conversion saved to {subject_output_dir}")
            return converted_images
            
        except RuntimeError:
            # Re-raise RuntimeError as is
            raise
        except Exception as e:
            error_msg = f"[{subject_id}] dcm2niix conversion failed for {input_dir}: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        finally:
            # Clean up temporary directory only if using temp directory
            if use_temp_dir:
                try:
                    shutil.rmtree(str(subject_output_dir))
                    self.logger.debug(f"[{subject_id}] Cleaned up temporary directory: {subject_output_dir}")
                except Exception as e:
                    self.logger.warning(f"[{subject_id}] Failed to clean up temporary directory {subject_output_dir}: {e}")
    
    def batch_convert_subjects(
        self, 
        subjects_data: Dict[str, Dict[str, str]]
    ) -> Dict[str, Dict[str, sitk.Image]]:
        """
        Batch convert DICOM directories for multiple subjects.
        
        Args:
            subjects_data (Dict[str, Dict[str, str]]): 
                Dictionary with subject IDs as keys and sequence dictionaries as values
                Format: {subject_id: {sequence_name: dicom_dir_path}}
        
        Returns:
            Dict[str, Dict[str, sitk.Image]]: Dictionary containing SimpleITK Image objects
                Format: {subject_id: {sequence_name: sitk_image}}
        """
        all_converted_files = {}
        
        # Calculate total number of conversions for progress bar
        total_conversions = sum(len(sequences) for sequences in subjects_data.values())
        progress_bar = CustomTqdm(total=total_conversions, desc="Converting DICOM to NIfTI")
        
        try:
            for subject_id, sequences in subjects_data.items():
                subject_converted = {}
                
                for sequence_name, dicom_dir in sequences.items():
                    try:
                        converted_images = self._convert_single_dicom_dir(
                            dicom_dir, 
                            subject_id, 
                            sequence_name
                        )
                        subject_converted.update(converted_images)
                        
                    except Exception as e:
                        self.logger.error(f"[{subject_id}] Failed to convert {sequence_name}: {e}")
                        if not self.allow_missing_keys:
                            raise
                    
                    progress_bar.update(1)
                
                if subject_converted:
                    all_converted_files[subject_id] = subject_converted
        
        finally:
            # Progress bar automatically handles completion display
            pass
        
        return all_converted_files
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input data to convert DICOM directories to SimpleITK Image objects.
        
        Args:
            data (Dict[str, Any]): Input data dictionary containing DICOM directory paths
            
        Returns:
            Dict[str, Any]: Data dictionary with SimpleITK Image objects added
        """
        self._check_keys(data)
        
        # Extract subject ID if available (try different common keys)
        subject_id = data.get('subj', data.get('subject_id', 'unknown_subject'))
        
        # Process each specified key
        for key in self.keys:
            if key not in data:
                if self.allow_missing_keys:
                    continue
                else:
                    raise KeyError(f"Key {key} not found in data dictionary")
            
            # Get the value from data[key]
            value = data[key]
            
            # Case 1: If already a SimpleITK Image, skip (already processed)
            if isinstance(value, sitk.Image):
                self.logger.info(f"[{subject_id}] Key {key} is already a SimpleITK Image, skipping dcm2niix conversion")
                continue
            
            # Case 2: If not a string, skip (invalid type)
            if not isinstance(value, str):
                self.logger.warning(f"[{subject_id}] Key {key} has invalid type {type(value)}, expected str or sitk.Image")
                continue
            
            # Case 3: value is a string path (file or directory)
            dicom_path = value
            
            try:
                # Get output directory from data if available
                # The pipeline will set output_dirs to intermediate directory if save_intermediate is enabled
                output_dir = None
                if 'output_dirs' in data and key in data['output_dirs']:
                    output_dir = data['output_dirs'][key]
                    self.logger.debug(f"[{subject_id}] Using output directory for {key}: {output_dir}")
                
                # Convert single DICOM directory
                converted_images = self._convert_single_dicom_dir(
                    dicom_path, 
                    subject_id, 
                    key,
                    output_dir=output_dir
                )
                
                # Update data with SimpleITK Image objects
                for seq_name, sitk_image in converted_images.items():
                    # Skip metadata entries and output path entries
                    if seq_name.endswith('_meta_dict') or seq_name.endswith('_output_path'):
                        continue
                    
                    # Use the original key name for consistency with other preprocessors
                    data[key] = sitk_image
                    self.logger.debug(f"[{subject_id}] Converted {key}")
                    
                    # Store output file path if available
                    output_path_key = f"{seq_name}_output_path"
                    if output_path_key in converted_images:
                        meta_key = f"{key}_meta_dict"
                        if meta_key not in data:
                            data[meta_key] = {}
                        data[meta_key]["output_file_path"] = converted_images[output_path_key]
                    
                    break  # Only use the first image if multiple found
                
                # Store conversion metadata
                meta_key = f"{key}_meta_dict"
                if meta_key not in data:
                    data[meta_key] = {}
                data[meta_key]["dcm2niix_converted"] = True
                data[meta_key]["original_dicom_dir"] = dicom_path
                data[meta_key]["original_path"] = dicom_path
                data[meta_key]["converted_files"] = len([k for k in converted_images.keys() 
                                                        if not k.endswith('_meta_dict') 
                                                        and not k.endswith('_output_path')])
                data[meta_key]["conversion_params"] = {
                    'compress': self.compress,
                    'anonymize': self.anonymize,
                    'ignore_derived': self.ignore_derived,
                    'crop_images': self.crop_images,
                    'generate_json': self.generate_json
                }
                
                # Merge JSON metadata if available
                json_meta_key = f"{key}_meta_dict"
                if json_meta_key in converted_images:
                    data[meta_key].update(converted_images[json_meta_key])
                
            except Exception as e:
                self.logger.error(f"[{subject_id}] Error converting DICOM directory for key {key}: {e}")
                if not self.allow_missing_keys:
                    raise
        
        return data


def batch_convert_dicom_directories(
    input_mapping: Dict[str, Dict[str, str]],
    dcm2niix_path: Optional[str] = None,
    **kwargs
) -> Dict[str, Dict[str, sitk.Image]]:
    """
    Utility function for batch DICOM to NIfTI conversion.
    
    Args:
        input_mapping (Dict[str, Dict[str, str]]): 
            Mapping of subjects to their DICOM directories
            Format: {subject_id: {sequence_name: dicom_dir_path}}
        dcm2niix_path (Optional[str]): Full path to dcm2niix executable or directory containing it
        **kwargs: Additional parameters for Dcm2niixConverter
    
    Returns:
        Dict[str, Dict[str, sitk.Image]]: Dictionary containing SimpleITK Image objects
    
    Example:
        >>> input_data = {
        ...     "subject_001": {
        ...         "T1": "/path/to/subject_001/T1_dicom",
        ...         "T2": "/path/to/subject_001/T2_dicom"
        ...     },
        ...     "subject_002": {
        ...         "T1": "/path/to/subject_002/T1_dicom"
        ...     }
        ... }
        >>> converted = batch_convert_dicom_directories(
        ...     input_data, 
        ...     dcm2niix_path="/path/to/dcm2niix/bin"
        ... )
    """
    converter = Dcm2niixConverter(
        keys=["dummy"],  # Not used in batch mode
        dcm2niix_path=dcm2niix_path,
        **kwargs
    )
    
    return converter.batch_convert_subjects(input_mapping)

