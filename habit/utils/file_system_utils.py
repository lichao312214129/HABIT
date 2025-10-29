"""
File system utilities for safe file operations, especially in Windows environments
"""
import os
import time
import logging
import shutil
from pathlib import Path
from typing import Union, Optional
from habit.utils.log_utils import get_module_logger

logger = get_module_logger(__name__)

def safe_mkdir(path: Union[str, Path], exist_ok: bool = True) -> bool:
    """
    Safely create a directory, with retry mechanism for Windows environments
    
    Args:
        path (Union[str, Path]): Directory path to create
        exist_ok (bool): Whether it's okay if directory already exists
        
    Returns:
        bool: True if directory was created or already exists, False otherwise
    """
    path = Path(path) if isinstance(path, str) else path
    
    # If directory already exists and exist_ok is True, return True
    if path.exists() and exist_ok:
        return True
        
    # Try to create directory with retries
    max_retries = 3
    retry_delay = 0.5  # seconds
    
    for attempt in range(max_retries):
        try:
            path.mkdir(parents=True, exist_ok=exist_ok)
            return True
        except PermissionError as e:
            logger.warning(f"权限错误，无法创建目录 {path}：{str(e)}")
            return False
        except FileExistsError:
            if exist_ok:
                return True
            logger.warning(f"目录已存在: {path}")
            return False
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"创建目录 {path} 时发生错误 (尝试 {attempt+1}/{max_retries}): {str(e)}")
                time.sleep(retry_delay)
            else:
                logger.error(f"创建目录 {path} 失败: {str(e)}")
                return False
    
    return False

def safe_save_file(data: bytes, filepath: Union[str, Path], overwrite: bool = True) -> bool:
    """
    Safely save binary data to a file with retry mechanism
    
    Args:
        data (bytes): Binary data to save
        filepath (Union[str, Path]): Path to save the file
        overwrite (bool): Whether to overwrite an existing file
        
    Returns:
        bool: True if file was saved successfully, False otherwise
    """
    filepath = Path(filepath) if isinstance(filepath, str) else filepath
    
    # Check if file exists and overwrite is False
    if filepath.exists() and not overwrite:
        logger.warning(f"文件已存在且不允许覆盖: {filepath}")
        return False
    
    # Ensure directory exists
    if not safe_mkdir(filepath.parent):
        return False
    
    # Try to save file with retries
    max_retries = 3
    retry_delay = 0.5  # seconds
    
    for attempt in range(max_retries):
        try:
            with open(filepath, 'wb') as f:
                f.write(data)
            return True
        except PermissionError as e:
            logger.warning(f"权限错误，无法保存文件 {filepath}：{str(e)}")
            return False
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"保存文件 {filepath} 时发生错误 (尝试 {attempt+1}/{max_retries}): {str(e)}")
                time.sleep(retry_delay)
            else:
                logger.error(f"保存文件 {filepath} 失败: {str(e)}")
                return False
    
    return False

def safe_copy_file(src: Union[str, Path], dst: Union[str, Path], overwrite: bool = True) -> bool:
    """
    Safely copy file with retry mechanism
    
    Args:
        src (Union[str, Path]): Source file path
        dst (Union[str, Path]): Destination file path
        overwrite (bool): Whether to overwrite existing destination file
        
    Returns:
        bool: True if file was copied successfully, False otherwise
    """
    src = Path(src) if isinstance(src, str) else src
    dst = Path(dst) if isinstance(dst, str) else dst
    
    # Check if source file exists
    if not src.exists():
        logger.warning(f"源文件不存在: {src}")
        return False
    
    # Check if destination file exists and overwrite is False
    if dst.exists() and not overwrite:
        logger.warning(f"目标文件已存在且不允许覆盖: {dst}")
        return False
    
    # Ensure destination directory exists
    if not safe_mkdir(dst.parent):
        return False
    
    # Try to copy file with retries
    max_retries = 3
    retry_delay = 0.5  # seconds
    
    for attempt in range(max_retries):
        try:
            shutil.copy2(src, dst)
            return True
        except PermissionError as e:
            logger.warning(f"权限错误，无法复制文件 {src} 到 {dst}：{str(e)}")
            return False
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"复制文件 {src} 到 {dst} 时发生错误 (尝试 {attempt+1}/{max_retries}): {str(e)}")
                time.sleep(retry_delay)
            else:
                logger.error(f"复制文件 {src} 到 {dst} 失败: {str(e)}")
                return False
    
    return False 