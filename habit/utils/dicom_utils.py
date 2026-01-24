"""
DICOM utilities for reading and extracting DICOM file information
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import logging
from habit.utils.log_utils import get_module_logger
from habit.utils.io_utils import get_image_and_mask_paths, load_config
from habit.utils.progress_utils import CustomTqdm

# Default number of worker threads for parallel operations
# Use min(32, cpu_count + 4) as recommended by Python docs for I/O bound tasks
DEFAULT_NUM_WORKERS = min(32, (os.cpu_count() or 1) + 4)

logger = get_module_logger(__name__)

try:
    import pydicom
    from pydicom.dataset import Dataset
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    logger.warning("pydicom is not installed. DICOM reading functionality will not be available.")


def get_dicom_files(input_path: Union[str, Path], recursive: bool = True) -> List[Path]:
    """
    Get all DICOM files from a directory or file path.
    Uses io_utils functionality to handle YAML config files or directory paths.
    
    Args:
        input_path: Path to DICOM directory, file, or YAML config file
        recursive: Whether to search recursively in subdirectories
        
    Returns:
        List of Path objects pointing to DICOM files
    """
    input_path = Path(input_path)
    dicom_files = []
    
    # Check if it's a YAML config file
    if input_path.is_file() and input_path.suffix.lower() in ['.yaml', '.yml']:
        try:
            config = load_config(str(input_path))
            # Extract all paths from images and masks
            all_paths = []
            if 'images' in config:
                for subject_dict in config['images'].values():
                    all_paths.extend(subject_dict.values())
            if 'masks' in config:
                for subject_dict in config['masks'].values():
                    all_paths.extend(subject_dict.values())
            
            # Collect DICOM files from these paths
            for path_str in all_paths:
                path = Path(path_str)
                if path.is_file() and path.suffix.lower() in ['.dcm', '.dicom']:
                    dicom_files.append(path)
                elif path.is_dir():
                    if recursive:
                        dicom_files.extend(path.rglob('*.dcm'))
                        dicom_files.extend(path.rglob('*.dicom'))
                    else:
                        dicom_files.extend(path.glob('*.dcm'))
                        dicom_files.extend(path.glob('*.dicom'))
        except Exception as e:
            logger.warning(f"Could not load config from {input_path}: {e}")
            # Fall through to treat as regular path
    
    # If it's a directory, search for DICOM files
    if input_path.is_dir():
        if recursive:
            dicom_files.extend(input_path.rglob('*.dcm'))  # search for DICOM files in the directory and subdirectories
            dicom_files.extend(input_path.rglob('*.dicom'))
        else:
            dicom_files.extend(input_path.glob('*.dcm'))
            dicom_files.extend(input_path.glob('*.dicom'))
    # If it's a single file
    elif input_path.is_file() and input_path.suffix.lower() in ['.dcm', '.dicom']:
        dicom_files.append(input_path)
    
    return list(set(dicom_files))  # Remove duplicates


def _is_dicom_file_fast(filepath: Path) -> bool:
    """
    Quickly check if a file is a DICOM file by reading the magic bytes.
    DICOM files have "DICM" at byte offset 128 (after 128-byte preamble).
    Some DICOM files may not have this prefix (Part 10 vs non-Part 10),
    so we also check for common DICOM tags at the beginning.
    
    Args:
        filepath: Path to the file to check
        
    Returns:
        True if the file appears to be a DICOM file, False otherwise
    """
    try:
        with open(filepath, 'rb') as f:
            # Check for DICM magic bytes at offset 128 (standard DICOM Part 10)
            f.seek(128)
            magic = f.read(4)
            if magic == b'DICM':
                return True
            
            # For non-Part 10 DICOM files, check for common group tags at start
            # DICOM tags start with group numbers like 0x0002, 0x0008, 0x0010
            f.seek(0)
            header = f.read(4)
            if len(header) >= 4:
                # Check for common DICOM group numbers (little-endian)
                # Group 0x0002 (File Meta), 0x0008 (Study/Series), 0x0010 (Patient)
                group = int.from_bytes(header[0:2], 'little')
                if group in (0x0002, 0x0008, 0x0010):
                    return True
            
            return False
    except Exception:
        return False


def _find_dicom_in_folder(
    folder_info: Tuple[str, List[str]],
    dicom_extensions: set,
    include_no_extension: bool
) -> Optional[Path]:
    """
    Find one DICOM file in a folder. Used as worker function for parallel processing.
    
    Args:
        folder_info: Tuple of (directory_path, list_of_filenames)
        dicom_extensions: Set of valid DICOM extensions
        include_no_extension: Whether to check files without extensions
        
    Returns:
        Path to a DICOM file, or None if not found
    """
    dirpath, filenames = folder_info
    current_dir = Path(dirpath)
    
    # Sort filenames to ensure consistent selection across runs
    sorted_filenames = sorted(filenames)
    
    # First pass: check files with known DICOM extensions (fastest, no I/O needed)
    for filename in sorted_filenames:
        ext = os.path.splitext(filename)[1].lower()
        if ext in dicom_extensions:
            return current_dir / filename
    
    # Second pass: if no extension-matched file found and include_no_extension is True,
    # check files without extensions using magic byte validation
    if include_no_extension:
        for filename in sorted_filenames:
            ext = os.path.splitext(filename)[1]
            # Only check files without extensions (or with unusual extensions)
            if ext == '' or ext.lower() not in dicom_extensions:
                filepath = current_dir / filename
                # Skip directories and very small files
                try:
                    if filepath.is_file() and filepath.stat().st_size > 132:
                        if _is_dicom_file_fast(filepath):
                            return filepath
                except OSError:
                    continue
    
    return None


def _get_folders_at_depth(root_path: str, target_depth: int) -> List[str]:
    """
    Quickly get all folders at a specific depth level using os.scandir().
    This is very fast because it only traverses to the target depth without
    scanning any deeper structure.
    
    Args:
        root_path: Root directory to start from
        target_depth: Target depth level (0 = root itself)
        
    Returns:
        List of folder paths at the target depth
    """
    root_path = os.path.normpath(root_path)
    
    if target_depth == 0:
        return [root_path]
    
    # Level-by-level traversal to target depth
    current_level = [root_path]
    
    for depth in range(target_depth):
        next_level = []
        for folder in current_level:
            try:
                with os.scandir(folder) as entries:
                    for entry in entries:
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                next_level.append(entry.path)
                        except (PermissionError, OSError):
                            continue
            except (PermissionError, OSError) as e:
                logger.debug(f"Cannot access directory {folder}: {e}")
                continue
        current_level = next_level
        
        if not current_level:
            break
    
    return current_level


def _find_first_dicom_in_tree(
    folder_path: str,
    dicom_extensions: set,
    include_no_extension: bool
) -> Optional[Path]:
    """
    Find the first DICOM file in a folder and its subfolders.
    Stops as soon as a DICOM file is found - does not scan entire tree.
    
    Uses depth-first search with early termination for maximum speed.
    
    Args:
        folder_path: Root folder to search
        dicom_extensions: Set of valid DICOM extensions
        include_no_extension: Whether to check files without extensions
        
    Returns:
        Path to first DICOM file found, or None if not found
    """
    folders_to_check = [folder_path]
    
    while folders_to_check:
        current_folder = folders_to_check.pop(0)
        
        try:
            entries = list(os.scandir(current_folder))
        except (PermissionError, OSError):
            continue
        
        # Separate files and subdirectories
        files = []
        subdirs = []
        
        for entry in entries:
            try:
                if entry.is_file(follow_symlinks=False):
                    files.append(entry)
                elif entry.is_dir(follow_symlinks=False):
                    subdirs.append(entry.path)
            except (PermissionError, OSError):
                continue
        
        # Sort files for consistent selection
        files.sort(key=lambda e: e.name)
        
        # First: check files with known DICOM extensions (fastest)
        for entry in files:
            ext = os.path.splitext(entry.name)[1].lower()
            if ext in dicom_extensions:
                return Path(entry.path)
        
        # Second: check files without extensions if enabled
        if include_no_extension:
            for entry in files:
                ext = os.path.splitext(entry.name)[1]
                if ext == '' or ext.lower() not in dicom_extensions:
                    try:
                        if entry.stat().st_size > 132:
                            if _is_dicom_file_fast(Path(entry.path)):
                                return Path(entry.path)
                    except (PermissionError, OSError):
                        continue
        
        # Add subdirectories to check (sorted for consistent order)
        subdirs.sort()
        folders_to_check.extend(subdirs)
    
    return None


def _walk_with_depth(root_path: str, max_depth: Optional[int] = None):
    """
    Generator that walks directory tree with optional depth limit.
    
    When max_depth is None, uses os.walk() for best performance on unlimited traversal.
    When max_depth is specified, uses os.scandir() with level-by-level traversal,
    which is more efficient because it only reads directories at the required depth
    without pre-scanning the entire tree.
    
    Args:
        root_path: Root directory to start walking from
        max_depth: Maximum depth to recurse. 
                  0 = only root directory
                  1 = root + immediate subdirectories
                  None = unlimited (same as os.walk)
                  
    Yields:
        Tuple of (dirpath, dirnames, filenames) like os.walk()
    """
    root_path = os.path.normpath(root_path)
    
    # When no depth limit, use os.walk for best performance
    if max_depth is None:
        yield from os.walk(root_path)
        return
    
    # When depth is limited, use level-by-level traversal with os.scandir()
    # Use a queue to track directories to process: (dir_path, current_depth)
    dirs_to_process = [(root_path, 0)]
    
    while dirs_to_process:
        current_dir, current_depth = dirs_to_process.pop(0)
        
        try:
            entries = list(os.scandir(current_dir))
        except (PermissionError, OSError) as e:
            logger.debug(f"Cannot access directory {current_dir}: {e}")
            continue
        
        # Separate files and directories
        dirnames = []
        filenames = []
        
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    dirnames.append(entry.name)
                elif entry.is_file(follow_symlinks=False):
                    filenames.append(entry.name)
            except (PermissionError, OSError):
                continue
        
        yield current_dir, dirnames, filenames
        
        # If we haven't reached max depth, add subdirectories to process queue
        if current_depth < max_depth:
            for dirname in sorted(dirnames):
                subdir_path = os.path.join(current_dir, dirname)
                dirs_to_process.append((subdir_path, current_depth + 1))


def get_one_dicom_per_folder(
    input_path: Union[str, Path],
    dicom_extensions: Optional[set] = None,
    include_no_extension: bool = False,
    num_workers: Optional[int] = None,
    max_depth: Optional[int] = None
) -> List[Path]:
    """
    Fast method to get one DICOM file per folder by traversing directories first.
    This is much faster than recursively finding all DICOM files when there are 
    hundreds of thousands of files.
    
    Uses multi-threading for parallel I/O operations to significantly speed up scanning.
    
    Two strategies based on max_depth:
    
    1. When max_depth is None (unlimited):
       - Traverse all directories and find one DICOM per folder
       
    2. When max_depth is specified (FAST MODE):
       - Quickly locate folders at the target depth (no deep scanning)
       - For each target folder, find ONE DICOM file (may be in subfolders)
       - Stop searching each folder as soon as a DICOM is found
       - This is extremely fast because it only reads the minimum required
    
    Args:
        input_path: Path to root directory to search
        dicom_extensions: Set of valid DICOM extensions (with dot, lowercase).
                         Default: {'.dcm', '.dicom'}
        include_no_extension: If True, also check files without extensions.
                             These will be validated by reading DICOM magic bytes.
                             Default: False
        num_workers: Number of worker threads for parallel processing.
                    Default: min(32, cpu_count + 4) for I/O bound tasks
        max_depth: Target depth for folder search.
                  When specified, finds folders at this depth and gets ONE DICOM
                  from each (searching into subfolders if needed, but stopping
                  as soon as one is found).
                  0 = only search in root directory
                  1 = root + immediate subdirectories  
                  2 = root + 2 levels of subdirectories
                  None = unlimited depth, one DICOM per folder (default)
                  Example: For structure root/patient/study/series/*.dcm,
                  use max_depth=2 to get one DICOM per study (faster than
                  scanning all series folders).
        
    Returns:
        List of Path objects, one DICOM file per target folder
    """
    input_path = Path(input_path)
    
    # Default DICOM extensions
    if dicom_extensions is None:
        dicom_extensions = {'.dcm', '.dicom'}
    else:
        # Ensure extensions are lowercase and have dots
        dicom_extensions = {ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                          for ext in dicom_extensions}
    
    if not input_path.is_dir():
        logger.warning(f"{input_path} is not a directory")
        return []
    
    # Set number of workers
    if num_workers is None:
        num_workers = DEFAULT_NUM_WORKERS
    
    result_files = []
    
    # ========== STRATEGY 1: max_depth specified - FAST targeted search ==========
    if max_depth is not None:
        logger.info(f"Fast mode: locating folders at depth {max_depth}, then finding one DICOM per folder")
        
        # Step 1: Quickly get all folders at target depth (very fast, no deep scan)
        target_folders = _get_folders_at_depth(str(input_path), max_depth)
        
        if not target_folders:
            logger.warning(f"No folders found at depth {max_depth}")
            return []
        
        logger.info(f"Found {len(target_folders)} folder(s) at depth {max_depth}")
        
        # Step 2: For each target folder, find ONE DICOM (with early termination)
        if len(target_folders) >= 10 and num_workers > 1:
            # Parallel search for large number of folders
            logger.debug(f"Using {num_workers} threads to search {len(target_folders)} folders")
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(
                        _find_first_dicom_in_tree, 
                        folder, 
                        dicom_extensions, 
                        include_no_extension
                    ): folder
                    for folder in target_folders
                }
                
                for future in as_completed(futures):
                    try:
                        dicom_file = future.result()
                        if dicom_file:
                            result_files.append(dicom_file)
                    except Exception as e:
                        folder = futures[future]
                        logger.warning(f"Error searching folder {folder}: {e}")
        else:
            # Sequential search for small number of folders
            for folder in target_folders:
                dicom_file = _find_first_dicom_in_tree(folder, dicom_extensions, include_no_extension)
                if dicom_file:
                    result_files.append(dicom_file)
        
        return result_files
    
    # ========== STRATEGY 2: No depth limit - scan all folders ==========
    # Collect all folders with their files using os.walk
    folder_list = []
    for dirpath, dirnames, filenames in os.walk(str(input_path)):
        if filenames:  # Only include folders that have files
            folder_list.append((dirpath, filenames))
    
    if not folder_list:
        return []
    
    # For small number of folders, single-threaded is often faster
    if len(folder_list) < 100 or num_workers <= 1:
        for folder_info in folder_list:
            dicom_file = _find_dicom_in_folder(folder_info, dicom_extensions, include_no_extension)
            if dicom_file:
                result_files.append(dicom_file)
    else:
        # Multi-threaded processing for large directories
        logger.debug(f"Using {num_workers} threads to scan {len(folder_list)} folders")
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_find_dicom_in_folder, folder_info, dicom_extensions, include_no_extension): folder_info
                for folder_info in folder_list
            }
            
            for future in as_completed(futures):
                try:
                    dicom_file = future.result()
                    if dicom_file:
                        result_files.append(dicom_file)
                except Exception as e:
                    folder_info = futures[future]
                    logger.warning(f"Error processing folder {folder_info[0]}: {e}")
    
    return result_files


def read_dicom_tags(dicom_path: Union[str, Path], 
                    tags: Optional[List[Union[str, int, tuple]]] = None,
                    force: bool = True) -> Dict[str, Any]:
    """
    Read specified DICOM tags from a DICOM file.
    
    Args:
        dicom_path: Path to DICOM file
        tags: List of tags to read. Can be:
            - Tag names (e.g., 'PatientName', 'StudyDate')
            - Tag numbers (e.g., 0x00100010)
            - Tag tuples (e.g., (0x0010, 0x0010))
            If None, reads all standard tags
        force: Whether to force reading even if file is not a valid DICOM
        
    Returns:
        Dictionary mapping tag names/numbers to values
    """
    if not PYDICOM_AVAILABLE:
        raise ImportError("pydicom is required for reading DICOM files. Install it with: pip install pydicom")
    
    dicom_path = Path(dicom_path)
    if not dicom_path.exists():
        raise FileNotFoundError(f"DICOM file not found: {dicom_path}")
    
    try:
        ds = pydicom.dcmread(str(dicom_path), force=force)
    except Exception as e:
        logger.error(f"Error reading DICOM file {dicom_path}: {e}")
        raise
    
    result = {}
    
    # Standard tags to read if tags is None
    if tags is None:
        standard_tags = [
            'PatientID', 'PatientName', 'PatientBirthDate', 'PatientSex', 'PatientAge',
            'StudyInstanceUID', 'StudyDate', 'StudyTime', 'StudyDescription',
            'SeriesInstanceUID', 'SeriesNumber', 'SeriesDescription', 'SeriesDate', 'SeriesTime',
            'Modality', 'Manufacturer', 'ManufacturerModelName',
            'MagneticFieldStrength',  # MRI field strength (e.g., 3.0T, 1.5T)
            'SliceThickness', 'SpacingBetweenSlices', 'PixelSpacing',
            'Rows', 'Columns', 'BitsAllocated', 'BitsStored', 'HighBit',
            'ImagePositionPatient', 'ImageOrientationPatient',
            'EchoTime', 'RepetitionTime', 'FlipAngle',
            'InstanceNumber', 'SliceLocation',
            'AcquisitionDate', 'AcquisitionTime',
            'ContrastBolusAgent', 'ContrastBolusVolume',
            'KVP', 'XRayTubeCurrent', 'ExposureTime',
            'WindowCenter', 'WindowWidth',
            'RescaleIntercept', 'RescaleSlope'
        ]
        tags = standard_tags
    
    # Read specified tags
    for tag in tags:
        try:
            if isinstance(tag, str):
                # Tag name
                if hasattr(ds, tag):
                    value = getattr(ds, tag, None)
                    result[tag] = str(value) if value is not None else None
                else:
                    result[tag] = None
            elif isinstance(tag, (int, tuple)):
                # Tag number or tuple
                if isinstance(tag, int):
                    tag_tuple = (tag >> 16, tag & 0xFFFF)
                else:
                    tag_tuple = tag
                
                if tag_tuple in ds:
                    element = ds[tag_tuple]
                    tag_name = element.keyword if hasattr(element, 'keyword') and element.keyword else str(tag_tuple)
                    result[tag_name] = str(element.value) if element.value is not None else None
                else:
                    tag_name = str(tag_tuple) if isinstance(tag, tuple) else f"0x{tag:08X}"
                    result[tag_name] = None
        except Exception as e:
            logger.warning(f"Error reading tag {tag} from {dicom_path}: {e}")
            result[str(tag)] = None
    
    # Add file path
    result['File_Path'] = str(dicom_path)
    result['File_Name'] = dicom_path.name
    
    return result


def _get_series_uid(dicom_file: Path) -> Optional[str]:
    """
    Quickly get SeriesInstanceUID from a DICOM file.
    
    Args:
        dicom_file: Path to DICOM file
        
    Returns:
        SeriesInstanceUID or None if not available
    """
    try:
        ds = pydicom.dcmread(str(dicom_file), force=True, stop_before_pixels=True)
        return getattr(ds, 'SeriesInstanceUID', None)
    except Exception:
        return None


def batch_read_dicom_info(input_path: Union[str, Path],
                          tags: Optional[List[Union[str, int, tuple]]] = None,
                          recursive: bool = True,
                          output_file: Optional[Union[str, Path]] = None,
                          output_format: str = 'csv',
                          group_by_series: bool = True,
                          one_file_per_folder: bool = False,
                          dicom_extensions: Optional[set] = None,
                          include_no_extension: bool = False,
                          num_workers: Optional[int] = None,
                          max_depth: Optional[int] = None) -> pd.DataFrame:
    """
    Batch read DICOM information from multiple files.
    Uses io_utils functionality to handle YAML config files or directory paths.
    Uses multi-threading for parallel I/O operations to significantly speed up scanning.
    
    Args:
        input_path: Path to DICOM directory, file, or YAML config file
        tags: List of tags to read. If None, reads standard tags
        recursive: Whether to search recursively in subdirectories (only used when one_file_per_folder=False)
        output_file: Optional path to save results. If None, results are not saved
        output_format: Format to save results ('csv', 'excel', 'json')
        group_by_series: If True, group files by SeriesInstanceUID and only read one file per series.
                        If False, read all files. Default is True.
        one_file_per_folder: If True, only take one DICOM file per folder to speed up scanning.
                            This uses a fast directory traversal method instead of listing all files,
                            which is much faster when there are hundreds of thousands of DICOM files.
                            Useful when each folder contains exactly one series.
                            Note: When enabled, --recursive is ignored (always recursive with depth control).
        dicom_extensions: Set of valid DICOM file extensions (e.g., {'.dcm', '.dicom', '.ima'}).
                         Only used when one_file_per_folder=True. Default: {'.dcm', '.dicom'}
        include_no_extension: If True, also check files without extensions by reading DICOM magic bytes.
                             Only used when one_file_per_folder=True. This is useful for some medical
                             devices that produce DICOM files without file extensions. Default: False
        num_workers: Number of worker threads for parallel processing.
                    Default: min(32, cpu_count + 4) for I/O bound tasks.
                    Set to 1 to disable parallel processing.
        max_depth: Maximum recursion depth for directory traversal.
                  Only used when one_file_per_folder=True.
                  0 = only search in root directory
                  1 = root + immediate subdirectories
                  None = unlimited depth (default)
                  Example: For typical DICOM structure (root/patient/study/series/),
                  set max_depth=3 to search up to the series level.
        
    Returns:
        DataFrame with DICOM information, one row per series (if group_by_series=True) or per file
    """
    if not PYDICOM_AVAILABLE:
        raise ImportError("pydicom is required for reading DICOM files. Install it with: pip install pydicom")
    
    input_path = Path(input_path)
    
    # Determine number of workers for parallel processing
    if num_workers is None:
        num_workers = DEFAULT_NUM_WORKERS
    
    # Get DICOM files using appropriate strategy
    if one_file_per_folder and input_path.is_dir():
        # Fast mode: traverse directories first, take one file per folder
        # This avoids listing all files which is very slow with hundreds of thousands of files
        logger.info("Speed mode enabled: traversing directories to find one DICOM file per folder...")
        if include_no_extension:
            logger.info("Including files without extensions (will validate DICOM magic bytes)")
        dicom_files = get_one_dicom_per_folder(
            input_path,
            dicom_extensions=dicom_extensions,
            include_no_extension=include_no_extension,
            num_workers=num_workers,
            max_depth=max_depth
        )
        logger.info(f"Speed mode: found {len(dicom_files)} folder(s) with DICOM files")
    else:
        # Standard mode: get all DICOM files
        dicom_files = get_dicom_files(input_path, recursive=recursive)
    
    if not dicom_files:
        logger.warning(f"No DICOM files found in {input_path}")
        return pd.DataFrame()
    
    logger.info(f"Found {len(dicom_files)} DICOM file(s)")
    
    # Determine if parallel processing should be used
    # Disable parallel if num_workers is 1 or too few files
    use_parallel = num_workers > 1 and len(dicom_files) >= 10
    
    # If group_by_series is True, group files by SeriesInstanceUID
    files_to_read = []
    series_groups = {}  # Initialize for use in both branches
    
    if group_by_series:
        # Group files by SeriesInstanceUID
        files_without_series = []
        
        if use_parallel:
            # Parallel processing for getting SeriesInstanceUID
            logger.debug(f"Using {num_workers} threads to group {len(dicom_files)} files by series")
            progress_bar = CustomTqdm(total=len(dicom_files), desc="Grouping files by series (parallel)")
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Submit all tasks
                future_to_file = {executor.submit(_get_series_uid, f): f for f in dicom_files}
                
                # Collect results as they complete
                for future in as_completed(future_to_file):
                    dicom_file = future_to_file[future]
                    try:
                        series_uid = future.result()
                        if series_uid:
                            if series_uid not in series_groups:
                                series_groups[series_uid] = []
                            series_groups[series_uid].append(dicom_file)
                        else:
                            files_without_series.append(dicom_file)
                    except Exception as e:
                        logger.warning(f"Error getting series UID from {dicom_file}: {e}")
                        files_without_series.append(dicom_file)
                    finally:
                        progress_bar.update(1)
        else:
            # Sequential processing for small number of files
            progress_bar = CustomTqdm(total=len(dicom_files), desc="Grouping files by series")
            
            for dicom_file in dicom_files:
                series_uid = _get_series_uid(dicom_file)
                if series_uid:
                    if series_uid not in series_groups:
                        series_groups[series_uid] = []
                    series_groups[series_uid].append(dicom_file)
                else:
                    files_without_series.append(dicom_file)
                progress_bar.update(1)
        
        # Select one representative file from each series (first file)
        for series_uid, files in series_groups.items():
            files_to_read.append(files[0])
            if len(files) > 1:
                logger.debug(f"Series {series_uid}: selecting first file from {len(files)} files")
        
        # Add files without series UID
        files_to_read.extend(files_without_series)
        
        logger.info(f"Grouped into {len(series_groups)} series, reading {len(files_to_read)} representative file(s)")
    else:
        # Read all files
        files_to_read = dicom_files
        logger.info(f"Reading all {len(files_to_read)} file(s) (not grouping by series)")
    
    # Read information from selected files
    all_info = []
    failed_files = []
    use_parallel_read = num_workers > 1 and len(files_to_read) >= 10
    
    if use_parallel_read:
        # Parallel reading of DICOM information
        logger.debug(f"Using {num_workers} threads to read {len(files_to_read)} DICOM files")
        progress_bar = CustomTqdm(total=len(files_to_read), desc="Reading DICOM information (parallel)")
        
        def _read_single_file(dicom_file: Path) -> Tuple[Optional[Dict], Optional[str]]:
            """Read a single DICOM file and return (info_dict, error_message)"""
            try:
                info = read_dicom_tags(dicom_file, tags=tags)
                # Add number of files in series if grouping was used
                if group_by_series:
                    series_uid = info.get('SeriesInstanceUID')
                    if series_uid and series_uid in series_groups:
                        info['Files_In_Series'] = len(series_groups[series_uid])
                    else:
                        info['Files_In_Series'] = 1
                return (info, None)
            except Exception as e:
                return (None, str(e))
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_file = {executor.submit(_read_single_file, f): f for f in files_to_read}
            
            for future in as_completed(future_to_file):
                dicom_file = future_to_file[future]
                try:
                    info, error = future.result()
                    if info:
                        all_info.append(info)
                    elif error:
                        logger.warning(f"Failed to read {dicom_file}: {error}")
                        failed_files.append(str(dicom_file))
                except Exception as e:
                    logger.warning(f"Failed to read {dicom_file}: {e}")
                    failed_files.append(str(dicom_file))
                finally:
                    progress_bar.update(1)
    else:
        # Sequential reading for small number of files
        progress_bar = CustomTqdm(total=len(files_to_read), desc="Reading DICOM information")
        
        for dicom_file in files_to_read:
            try:
                info = read_dicom_tags(dicom_file, tags=tags)
                # Add number of files in series if grouping was used
                if group_by_series:
                    series_uid = _get_series_uid(dicom_file)
                    if series_uid and series_uid in series_groups:
                        info['Files_In_Series'] = len(series_groups[series_uid])
                    else:
                        info['Files_In_Series'] = 1
                all_info.append(info)
            except Exception as e:
                logger.warning(f"Failed to read {dicom_file}: {e}")
                failed_files.append(str(dicom_file))
            finally:
                progress_bar.update(1)
    
    if not all_info:
        logger.error("No DICOM files could be read successfully")
        return pd.DataFrame()
    
    # Create DataFrame
    df = pd.DataFrame(all_info)
    
    # Log summary
    logger.info(f"Successfully read {len(all_info)} DICOM file(s)")
    if failed_files:
        logger.warning(f"Failed to read {len(failed_files)} file(s)")
    
    # Save if output file is specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_format.lower() == 'csv':
            df.to_csv(output_path, index=False)
        elif output_format.lower() == 'excel':
            df.to_excel(output_path, index=False)
        elif output_format.lower() == 'json':
            df.to_json(output_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        logger.info(f"Results saved to {output_path}")
    
    return df


def list_available_tags(dicom_path: Union[str, Path], 
                       num_samples: int = 1) -> List[str]:
    """
    List all available tags in DICOM file(s).
    
    Args:
        dicom_path: Path to DICOM file or directory
        num_samples: Number of files to sample (if directory)
        
    Returns:
        List of available tag names
    """
    if not PYDICOM_AVAILABLE:
        raise ImportError("pydicom is required for reading DICOM files. Install it with: pip install pydicom")
    
    dicom_files = get_dicom_files(dicom_path, recursive=True)
    
    if not dicom_files:
        return []
    
    # Sample files
    sample_files = dicom_files[:min(num_samples, len(dicom_files))]
    
    all_tags = set()
    for dicom_file in sample_files:
        try:
            ds = pydicom.dcmread(str(dicom_file), force=True)
            for element in ds:
                if hasattr(element, 'keyword') and element.keyword:
                    all_tags.add(element.keyword)
                else:
                    all_tags.add(str(element.tag))
        except Exception as e:
            logger.warning(f"Error reading {dicom_file}: {e}")
    
    return sorted(list(all_tags))

