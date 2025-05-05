from typing import Dict, Any, Tuple, Optional
import torch
import numpy as np
import SimpleITK as sitk
import ants

class ImageConverter:
    """Utility class for converting between different image formats."""
    
    @staticmethod
    def get_metadata(meta_dict: Dict[str, Any], ndim: int) -> Tuple[tuple, tuple, tuple]:
        """Extract and validate metadata from dictionary.
        
        Args:
            meta_dict (Dict[str, Any]): Metadata dictionary.
            ndim (int): Number of dimensions.
            
        Returns:
            Tuple[tuple, tuple, tuple]: Validated spacing, origin, and direction.
        """
        # Default values
        default_spacing = tuple([1.0] * ndim)
        default_origin = tuple([0.0] * ndim)
        default_direction = tuple([1.0 if i == j else 0.0 for i in range(ndim) for j in range(ndim)])
        
        # Get metadata with defaults
        spacing = meta_dict.get("spacing", default_spacing)
        origin = meta_dict.get("origin", default_origin)
        direction = meta_dict.get("direction", default_direction)
        
        # Convert to tuples if necessary
        if not isinstance(spacing, tuple):
            spacing = tuple(spacing[:ndim])
        if not isinstance(origin, tuple):
            origin = tuple(origin[:ndim])
        if not isinstance(direction, tuple):
            direction = tuple(direction)
            
        # Validate direction matrix size
        direction_size = ndim * ndim
        if len(direction) != direction_size:
            direction = default_direction
            
        return spacing, origin, direction
    
    @staticmethod
    def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """Convert torch tensor to numpy array.
        
        Args:
            tensor (torch.Tensor): Input tensor (MONAI format: [C,Z,Y,X] or [C,H,W]).
            
        Returns:
            np.ndarray: Numpy array with channel dimension removed if single channel.
        """
        array = tensor.cpu().numpy()
        if array.shape[0] == 1:  # If single channel
            array = array.squeeze(0)  # Remove channel dimension
        return array
    
    @staticmethod
    def numpy_to_tensor(array: np.ndarray, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None) -> torch.Tensor:
        """Convert numpy array to torch tensor.
        
        Args:
            array (np.ndarray): Input array in [Z,Y,X] format.
            dtype (Optional[torch.dtype]): Target tensor dtype.
            device (Optional[torch.device]): Target tensor device.
            
        Returns:
            torch.Tensor: Torch tensor with added channel dimension [1,Z,Y,X].
        """
        if array.ndim == 2:
            array = array[np.newaxis, ...]  # Add channel dim for 2D
        elif array.ndim == 3:
            array = array[np.newaxis, ...]  # Add channel dim for 3D
            
        tensor = torch.from_numpy(array)
        if dtype is not None or device is not None:
            tensor = tensor.to(dtype=dtype, device=device)
        return tensor 