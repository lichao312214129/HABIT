# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from typing import Dict, Any, Tuple, Optional
import numpy as np
import SimpleITK as sitk

from habit.exceptions import OptionalDependencyError

# Optional ANTs dependency: only the itk<->ants conversion helpers need it,
# so a missing antspyx must not break unrelated conversions (or importing
# this module). Used features raise OptionalDependencyError instead.
try:
    import ants
    ANTS_AVAILABLE = True
except ImportError:
    ANTS_AVAILABLE = False
    ants = None

# Optional torch import for tensor conversion methods (not used in core functionality)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

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
    def tensor_to_numpy(tensor) -> np.ndarray:
        """Convert torch tensor to numpy array.
        
        Args:
            tensor: Input tensor in format [C,Z,Y,X] or [C,H,W].
            
        Returns:
            np.ndarray: Numpy array with channel dimension removed if single channel.
            
        Raises:
            ImportError: If torch is not installed.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required for tensor_to_numpy. Install it with: pip install torch")
        array = tensor.cpu().numpy()
        if array.shape[0] == 1:  # If single channel
            array = array.squeeze(0)  # Remove channel dimension
        return array
    
    @staticmethod
    def numpy_to_tensor(array: np.ndarray, dtype=None, device=None):
        """Convert numpy array to torch tensor.
        
        Args:
            array (np.ndarray): Input array in [Z,Y,X] format.
            dtype: Target tensor dtype (requires torch).
            device: Target tensor device (requires torch).
            
        Returns:
            torch.Tensor: Torch tensor with added channel dimension [1,Z,Y,X].
            
        Raises:
            ImportError: If torch is not installed.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required for numpy_to_tensor. Install it with: pip install torch")
        if array.ndim == 2:
            array = array[np.newaxis, ...]  # Add channel dim for 2D
        elif array.ndim == 3:
            array = array[np.newaxis, ...]  # Add channel dim for 3D
            
        tensor = torch.from_numpy(array)
        if dtype is not None or device is not None:
            tensor = tensor.to(dtype=dtype, device=device)
        return tensor 
    
    @staticmethod
    def ants_2_itk(image):
        if not ANTS_AVAILABLE:
            raise OptionalDependencyError(
                "ANTs<->ITK conversion requires the optional antspyx dependency; "
                "install 'HABIT[registration]' to use it."
            )
        imageITK = sitk.GetImageFromArray(image.numpy().transpose(2, 1, 0))
        imageITK.SetOrigin(image.origin)
        imageITK.SetSpacing(image.spacing)
        imageITK.SetDirection(image.direction.reshape(9))
        return imageITK

    @staticmethod
    def itk_2_ants(image):
        if not ANTS_AVAILABLE:
            raise OptionalDependencyError(
                "ITK<->ANTs conversion requires the optional antspyx dependency; "
                "install 'HABIT[registration]' to use it."
            )
        image_ants = ants.from_numpy(sitk.GetArrayFromImage(image).transpose(2, 1, 0),
                                    origin=image.GetOrigin(),
                                    spacing=image.GetSpacing(),
                                    direction=np.array(image.GetDirection()).reshape(3, 3))
        return image_ants