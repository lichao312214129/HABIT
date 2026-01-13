"""
Configuration utilities for loading, saving, and resolving configurations.
This module combines configuration I/O and path resolution capabilities.
"""

import os
import re
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from copy import deepcopy


# -----------------------------------------------------------------------------
# Configuration I/O Functions
# -----------------------------------------------------------------------------

def load_config(config_path: str, resolve_paths: bool = True) -> Dict[str, Any]:
    """
    Load configuration file and optionally resolve relative paths.
    
    Args:
        config_path (str): Path to configuration file, supports YAML and JSON
        resolve_paths (bool): Whether to resolve relative paths to absolute paths.
                              Defaults to True.
    
    Returns:
        Dict[str, Any]: Configuration dictionary
        
    Raises:
        FileNotFoundError: If configuration file is not found
        ValueError: If file format is not supported
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Determine how to load based on file extension
    _, ext = os.path.splitext(config_path)
    
    if ext.lower() in ['.yaml', '.yml']:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    elif ext.lower() == '.json':
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported configuration file format: {ext}")
    
    # Resolve paths if requested
    if resolve_paths:
        config = resolve_config_paths(config, config_path)
        
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to file
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        config_path (str): Path to save configuration file, supports YAML and JSON
        
    Raises:
        ValueError: If file format is not supported
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
    
    # Determine how to save based on file extension
    _, ext = os.path.splitext(config_path)
    
    if ext.lower() in ['.yaml', '.yml']:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    elif ext.lower() == '.json':
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
    else:
        raise ValueError(f"Unsupported configuration file format: {ext}")


def validate_config(config: Dict[str, Any], required_keys: Optional[List[str]] = None) -> bool:
    """
    Validate if configuration contains required keys
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        required_keys (Optional[List[str]]): List of required keys
    
    Returns:
        bool: Whether the configuration is valid
    
    Raises:
        ValueError: If required keys are missing
    """
    if required_keys is None:
        return True
    
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Configuration missing required keys: {missing_keys}")
    
    return True


# -----------------------------------------------------------------------------
# Path Resolution Logic
# -----------------------------------------------------------------------------

# Default patterns for identifying path fields (case-insensitive matching)
DEFAULT_PATH_PATTERNS = {
    # Suffix patterns - fields ending with these are likely paths
    'suffixes': [
        '_path',
        '_dir', 
        '_file',
        '_folder',
        '_directory',
        '_root',
        '_config',
        '_location',
    ],
    # Exact match patterns - fields with these exact names are paths
    'exact': [
        'path',
        'dir',
        'file',
        'folder',
        'directory',
        'root',
        'data_dir',
        'out_dir',
        'output_dir',
        'input_dir',
        'config',
        'config_file',
        'mask_path',
        'image_path',
        'source',
        'destination',
        'target',
    ],
}

# Common file extensions that indicate a path value
COMMON_FILE_EXTENSIONS = {
    # Configuration files
    '.yaml', '.yml', '.json', '.xml', '.ini', '.cfg', '.conf', '.toml',
    # Medical imaging files
    '.nii', '.nii.gz', '.nrrd', '.dcm', '.mha', '.mhd', '.npy', '.npz',
    # Data files
    '.csv', '.xlsx', '.xls', '.txt', '.tsv', '.parquet',
    # Image files
    '.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif',
    # Model files
    '.pkl', '.pickle', '.joblib', '.h5', '.hdf5', '.pt', '.pth', '.onnx',
    # Archive files
    '.zip', '.tar', '.gz', '.bz2', '.7z',
    # Log files
    '.log',
}

# Regex pattern for relative path prefixes
RELATIVE_PATH_PREFIXES = re.compile(r'^\.{1,2}[/\\]')

# Regex pattern for path-like strings (contains path separators and looks like a path)
PATH_LIKE_PATTERN = re.compile(r'^[a-zA-Z]:[/\\]|^[/\\]|^\.{1,2}[/\\]|[/\\][^/\\]+\.[a-zA-Z0-9]+$')


class PathResolver:
    """
    A flexible path resolver for configuration files.
    
    Resolves relative paths in configuration dictionaries to absolute paths,
    using the configuration file's directory as the base.
    
    Attributes:
        base_dir (Path): Base directory for resolving relative paths
        patterns (Dict): Patterns for identifying path fields
        resolved_count (int): Number of paths resolved in last operation
        
    Example:
        >>> resolver = PathResolver('/path/to/config.yaml')
        >>> resolved_config = resolver.resolve(config_dict)
        >>> print(f"Resolved {resolver.resolved_count} paths")
    """
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        base_dir: Optional[Union[str, Path]] = None,
        extra_suffixes: Optional[List[str]] = None,
        extra_exact: Optional[List[str]] = None,
        custom_patterns: Optional[Dict[str, List[str]]] = None
    ):
        """
        Initialize the PathResolver.
        
        Args:
            config_path: Path to the configuration file (used to determine base_dir)
            base_dir: Explicit base directory for resolving paths (overrides config_path)
            extra_suffixes: Additional suffix patterns to match (e.g., ['_location'])
            extra_exact: Additional exact match patterns (e.g., ['my_path_field'])
            custom_patterns: Complete custom patterns dict to replace defaults
            
        Note:
            Either config_path or base_dir must be provided.
        """
        # Determine base directory
        if base_dir is not None:
            self.base_dir = Path(base_dir).absolute()
        elif config_path is not None:
            self.base_dir = Path(config_path).parent.absolute()
        else:
            self.base_dir = Path.cwd()
        
        # Setup patterns
        if custom_patterns is not None:
            self.patterns = custom_patterns
        else:
            self.patterns = deepcopy(DEFAULT_PATH_PATTERNS)
            
            # Add extra patterns if provided
            if extra_suffixes:
                self.patterns['suffixes'].extend(extra_suffixes)
            if extra_exact:
                self.patterns['exact'].extend(extra_exact)
        
        # Convert to sets for faster lookup
        self._suffix_set = set(s.lower() for s in self.patterns.get('suffixes', []))
        self._exact_set = set(e.lower() for e in self.patterns.get('exact', []))
        
        # Statistics
        self.resolved_count = 0
        self._resolved_fields: List[str] = []
    
    def is_path_field(self, key: str) -> bool:
        """
        Check if a field name represents a path (key-based detection).
        
        Args:
            key: The field name to check
            
        Returns:
            True if the field is likely a path field
        """
        key_lower = key.lower()
        
        # Check exact match
        if key_lower in self._exact_set:
            return True
        
        # Check suffix match
        for suffix in self._suffix_set:
            if key_lower.endswith(suffix):
                return True
        
        return False
    
    def is_path_value(self, value: str) -> bool:
        """
        Check if a string value looks like a path (value-based detection).
        
        Detection strategies:
        1. Starts with relative path prefix: ./ .\ ../ ..\
        2. Ends with common file extension: .yaml, .nii.gz, .csv, etc.
        3. Matches path-like pattern: contains path separators in meaningful way
        
        Args:
            value: The string value to check
            
        Returns:
            True if the value looks like a path
        """
        if not isinstance(value, str) or not value:
            return False
        
        # Skip URLs and special protocols
        if value.startswith(('http://', 'https://', 'ftp://', 's3://', 'gs://')):
            return False
        
        # Skip if it's just a simple word without any path indicators
        if ' ' in value and not ('/' in value or '\\' in value):
            return False
        
        # Strategy 1: Check for relative path prefixes (./ .\ ../ ..\)
        if RELATIVE_PATH_PREFIXES.match(value):
            return True
        
        # Strategy 2: Check for common file extensions
        value_lower = value.lower()
        for ext in COMMON_FILE_EXTENSIONS:
            if value_lower.endswith(ext):
                return True
        
        # Special case: .nii.gz (compound extension)
        if value_lower.endswith('.nii.gz'):
            return True
        
        # Strategy 3: Check if it looks like a path (has separators and file extension)
        if PATH_LIKE_PATTERN.match(value):
            return True
        
        # Strategy 4: Contains path separators and ends with something that looks like a filename
        if ('/' in value or '\\' in value):
            # Check if it ends with something that looks like a filename (word.ext)
            parts = value.replace('\\', '/').split('/')
            last_part = parts[-1] if parts else ''
            if '.' in last_part and len(last_part) > 1:
                return True
        
        return False
    
    def should_resolve(self, key: str, value: Any) -> bool:
        """
        Determine if a key-value pair should have its path resolved.
        
        Combines key-based and value-based detection strategies.
        
        Args:
            key: The field name
            value: The field value
            
        Returns:
            True if this field should be resolved as a path
        """
        if not isinstance(value, str):
            return False
        
        # Key-based detection (field name suggests it's a path)
        if self.is_path_field(key):
            return True
        
        # Value-based detection (value content suggests it's a path)
        if self.is_path_value(value):
            return True
        
        return False
    
    def resolve_path(self, path_value: str) -> str:
        """
        Resolve a single path value.
        
        Args:
            path_value: The path string to resolve
            
        Returns:
            Absolute path if the input was relative and exists, otherwise original path
        """
        if not isinstance(path_value, str):
            return path_value
        
        # Skip if already absolute
        if os.path.isabs(path_value):
            return path_value
        
        # Skip URLs and special paths
        if path_value.startswith(('http://', 'https://', 'ftp://', 's3://')):
            return path_value
        
        # Try to resolve relative to base_dir
        resolved = self.base_dir / path_value
        
        # Return resolved path if it exists, otherwise return original
        # (to avoid breaking paths that are meant to be created later)
        if resolved.exists():
            return str(resolved)
        
        # Even if doesn't exist, if it looks like a relative path that should be resolved
        # (starts with ./ or ../ or is a simple filename), resolve it
        if path_value.startswith(('./', '../')) or '/' in path_value or '\\' in path_value:
            return str(resolved)
        
        # For simple filenames without path separators, check if they exist in base_dir
        potential = self.base_dir / path_value
        if potential.exists():
            return str(potential)
        
        return path_value
    
    def resolve(
        self,
        config: Dict[str, Any],
        _path_prefix: str = ""
    ) -> Dict[str, Any]:
        """
        Resolve all path fields in a configuration dictionary.
        
        Args:
            config: Configuration dictionary to process
            _path_prefix: Internal use for tracking nested paths
            
        Returns:
            New dictionary with resolved paths (original dict is not modified)
        """
        if _path_prefix == "":
            # Reset statistics at the start of a new resolution
            self.resolved_count = 0
            self._resolved_fields = []
        
        result = {}
        
        for key, value in config.items():
            current_path = f"{_path_prefix}.{key}" if _path_prefix else key
            
            if isinstance(value, dict):
                # Recursively process nested dictionaries
                result[key] = self.resolve(value, current_path)
            elif isinstance(value, list):
                # Process lists (may contain paths or nested dicts)
                result[key] = self._resolve_list(value, key, current_path)
            elif self.should_resolve(key, value):
                # Resolve path field (detected by key name or value content)
                resolved = self.resolve_path(value)
                if resolved != value:
                    self.resolved_count += 1
                    self._resolved_fields.append(current_path)
                result[key] = resolved
            else:
                # Keep other values as-is
                result[key] = value
        
        return result
    
    def _resolve_list(
        self,
        items: List,
        key: str,
        path_prefix: str
    ) -> List:
        """
        Resolve paths within a list.
        
        Args:
            items: List to process
            key: The key name of this list
            path_prefix: Current path prefix for tracking
            
        Returns:
            New list with resolved paths
        """
        result = []
        
        # Check if the parent key suggests this is a list of paths
        key_suggests_paths = (
            self.is_path_field(key) or 
            key.lower() in ['files', 'paths', 'images', 'masks', 'file_groups', 'file_list']
        )
        
        for i, item in enumerate(items):
            item_path = f"{path_prefix}[{i}]"
            
            if isinstance(item, dict):
                # Nested dict in list
                result.append(self.resolve(item, item_path))
            elif isinstance(item, list):
                # Nested list (e.g., file_groups: [[file1, file2], [file3, file4]])
                result.append(self._resolve_list(item, key, item_path))
            elif isinstance(item, str):
                # Check if this string should be resolved as a path
                # Either the parent key suggests paths, or the value itself looks like a path
                if key_suggests_paths or self.is_path_value(item):
                    resolved = self.resolve_path(item)
                    if resolved != item:
                        self.resolved_count += 1
                        self._resolved_fields.append(item_path)
                    result.append(resolved)
                else:
                    result.append(item)
            else:
                result.append(item)
        
        return result
    
    def get_resolved_fields(self) -> List[str]:
        """
        Get list of field paths that were resolved.
        
        Returns:
            List of field path strings (e.g., ['data_dir', 'input.file_path'])
        """
        return self._resolved_fields.copy()


def resolve_config_paths(
    config: Dict[str, Any],
    config_path: Union[str, Path],
    extra_patterns: Optional[List[str]] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to resolve paths in a configuration dictionary.
    
    This is the recommended way to use path resolution in most cases.
    
    Args:
        config: Configuration dictionary to process
        config_path: Path to the configuration file (for determining base directory)
        extra_patterns: Additional exact-match patterns for path fields
        verbose: If True, print information about resolved paths
        
    Returns:
        New configuration dictionary with resolved paths
        
    Example:
        >>> config = load_config('demo_data/config.yaml')
        >>> config = resolve_config_paths(config, 'demo_data/config.yaml')
    """
    resolver = PathResolver(
        config_path=config_path,
        extra_exact=extra_patterns
    )
    
    resolved_config = resolver.resolve(config)
    
    if verbose and resolver.resolved_count > 0:
        print(f"Resolved {resolver.resolved_count} path(s):")
        for field in resolver.get_resolved_fields():
            print(f"  - {field}")
    
    return resolved_config


def load_config_with_paths(
    config_path: Union[str, Path],
    extra_patterns: Optional[List[str]] = None,
    resolve_paths: bool = True
) -> Dict[str, Any]:
    """
    Load a configuration file and optionally resolve relative paths.
    
    This is a convenience function that combines load_config and path resolution.
    
    Args:
        config_path: Path to the configuration file
        extra_patterns: Additional patterns for path field detection
        resolve_paths: Whether to resolve relative paths (default: True)
        
    Returns:
        Configuration dictionary with resolved paths
        
    Example:
        >>> config = load_config_with_paths('demo_data/config.yaml')
    """
    # Direct call since they are now in the same module
    config = load_config(config_path)
    
    if resolve_paths:
        config = resolve_config_paths(config, config_path, extra_patterns)
    
    return config
