"""
DICOM to NIfTI converter using dcm2niix tool.

This module provides functionality to batch convert DICOM files to NIfTI format
using the dcm2niix tool, with integration into the HABIT preprocessing pipeline.
"""

import os
import subprocess
import shutil
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import tempfile
import SimpleITK as sitk
import json

from .base_preprocessor import BasePreprocessor
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
        generate_json: bool = False, # 默认不生成JSON文件
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
        
        # Setup dcm2niix path and environment
        self.dcm2niix_executable = self._setup_dcm2niix_environment(dcm2niix_path)
        
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
        
        # Verify dcm2niix is available
        self._verify_dcm2niix()
    
    def _setup_dcm2niix_environment(self, dcm2niix_path: Optional[str]) -> str:
        """
        Setup dcm2niix environment by adding the executable path to PATH if provided.
        
        Args:
            dcm2niix_path (Optional[str]): Path to dcm2niix executable or directory containing it
            
        Returns:
            str: Name of the dcm2niix executable to use
        """
        if not dcm2niix_path:
            # Use default executable name if no path provided
            return "dcm2niix"
            
        dcm2niix_path_obj = Path(dcm2niix_path)
        
        if dcm2niix_path_obj.is_file():
            # If path points to the executable itself
            executable_name = dcm2niix_path_obj.name
            dcm2niix_dir = dcm2niix_path_obj.parent
        elif dcm2niix_path_obj.is_dir():
            # If path points to directory containing the executable
            executable_name = "dcm2niix"
            dcm2niix_dir = dcm2niix_path_obj
        else:
            self.logger.warning(f"Specified dcm2niix path does not exist: {dcm2niix_path}")
            return "dcm2niix"
        
        # Add to PATH environment variable
        current_path = os.environ.get('PATH', '')
        dcm2niix_path_str = str(dcm2niix_dir)
        
        # Check if path is already in PATH
        if dcm2niix_path_str not in current_path:
            # Add to beginning of PATH
            new_path = f"{dcm2niix_path_str}{os.pathsep}{current_path}"
            os.environ['PATH'] = new_path
            self.logger.info(f"Added dcm2niix path to environment: {dcm2niix_path_str}")
        else:
            self.logger.info(f"dcm2niix path already in environment: {dcm2niix_path_str}")
        
        return executable_name
    
    def _verify_dcm2niix(self) -> None:
        """
        Verify that dcm2niix executable is available.
        
        Raises:
            RuntimeError: If dcm2niix is not found or not executable
        """
        try:
            result = subprocess.run(
                [self.dcm2niix_executable, "--help"],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode != 0 and "dcm2niix" not in result.stderr.lower():
                raise RuntimeError(self._get_dcm2niix_not_found_message())
            self.logger.info(f"dcm2niix executable verified: {self.dcm2niix_executable}")
        except FileNotFoundError:
            raise RuntimeError(self._get_dcm2niix_not_found_message())
    
    def _get_dcm2niix_not_found_message(self) -> str:
        """
        Generate helpful error message with installation instructions when dcm2niix is not found.
        
        Returns:
            str: Error message with installation instructions
        """
        return f"""dcm2niix executable not found: {self.dcm2niix_executable}

Please install dcm2niix using one of the following methods:

1. Using pip (recommended):
   python -m pip install dcm2niix

2. Using conda:
   conda install -c conda-forge dcm2niix

3. Using package manager:
   - Debian/Ubuntu: sudo apt-get install dcm2niix
   - MacOS Homebrew: brew install dcm2niix
   - MacOS MacPorts: sudo port install dcm2niix

4. Download pre-built binaries:
   - Linux:   curl -fLO https://github.com/rordenlab/dcm2niix/releases/latest/download/dcm2niix_lnx.zip
   - MacOS:   curl -fLO https://github.com/rordenlab/dcm2niix/releases/latest/download/macos_dcm2niix.pkg
   - Windows: curl -fLO https://github.com/rordenlab/dcm2niix/releases/latest/download/dcm2niix_win.zip

5. MRIcroGL includes dcm2niix:
   - NITRC: https://www.nitrc.org/projects/mricrogl
   - GitHub: https://github.com/rordenlab/MRIcroGL

For more information, visit: https://github.com/rordenlab/dcm2niix
"""
    
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
        cmd = [self.dcm2niix_executable]
        
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
            
            # Convert command list to string for os.system()
            # Need to quote paths that may contain spaces
            cmd_parts = []
            for i, part in enumerate(cmd_list):
                # Quote the executable path and directory paths
                if i == 0 or part in [str(input_path), str(subject_output_dir)]:
                    # Check if path contains spaces
                    if ' ' in part:
                        cmd_parts.append(f'"{part}"')
                    else:
                        cmd_parts.append(part)
                else:
                    cmd_parts.append(part)
            
            cmd_string = ' '.join(cmd_parts)
            
            self.logger.info(f"[{subject_id}] Converting DICOM directory: {input_dir}")
            self.logger.info(f"[{subject_id}] Executing command:")
            self.logger.info(f"    {cmd_string}")
            print(f"\n{'='*80}")
            print(f"[DEBUG] Executing dcm2niix command:")
            print(f"    {cmd_string}")
            print(f"{'='*80}\n")

            # Select execution method
            # Different methods may produce different results with dcm2niix
            # Try "os.system" if subprocess methods give unexpected 4D output
            execution_method = "subprocess.Popen"  # Options: "os.system", "subprocess.run", "subprocess.Popen"
            
            if execution_method == "os.system":
                # Method 1: os.system() - Most similar to terminal behavior
                # This is the closest to typing the command directly in terminal
                # Recommended for dcm2niix to avoid 3D/4D conversion issues
                print(f"[Execution Method: os.system]")
                exit_code = os.system(cmd_string)
                # os.system returns the exit status in platform-specific format
                # On Windows, it's the actual exit code; on Unix, it may be shifted
                if exit_code != 0:
                    raise RuntimeError(f"dcm2niix conversion failed with exit code {exit_code}")
            
            elif execution_method == "subprocess.run":
                # Method 2: subprocess.run() with shell=True
                print(f"[Execution Method: subprocess.run]")
                result = subprocess.run(
                    cmd_string,
                    shell=True,
                    capture_output=False,
                    text=True,
                    check=True
                )
                exit_code = result.returncode
                if exit_code != 0:
                    raise RuntimeError(f"dcm2niix conversion failed with exit code {exit_code}")
            
            elif execution_method == "subprocess.Popen":
                # Method 3: subprocess.Popen() - Real-time output
                print(f"[Execution Method: subprocess.Popen]")
                process = subprocess.Popen(
                    cmd_string,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1 # 实时输出
                )
                
                # Print output in real-time
                if process.stdout:
                    for line in process.stdout:
                        print(line, end='')
                
                process.wait()
                exit_code = process.returncode
                if exit_code != 0:
                    raise RuntimeError(f"dcm2niix conversion failed with exit code {exit_code}")
            
            else:
                raise ValueError(f"Unknown execution method: {execution_method}")
            
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
                        
                        # Find corresponding JSON file
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
            
            self.logger.info(f"[{subject_id}] Successfully converted and saved to {subject_output_dir}")
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
                    self.logger.info(f"[{subject_id}] Converted {key} to SimpleITK Image")
                    
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

