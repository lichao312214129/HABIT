from typing import Dict, Any, Optional, Union, List, Tuple
import torch
import numpy as np
import ants
import SimpleITK as sitk
from .base_preprocessor import BasePreprocessor
from .preprocessor_factory import PreprocessorFactory

@PreprocessorFactory.register("registration")
class RegistrationPreprocessor(BasePreprocessor):
    """Image registration preprocessor using ANTs.
    
    This preprocessor performs image registration using ANTs, with support for mask registration.
    """
    
    def __init__(
        self,
        keys: Union[str, List[str]],
        fixed_image: str,
        moving_images: Optional[Union[str, List[str]]] = None,
        mask_key: Optional[str] = None,
        type_of_transform: str = "SyN",
        allow_missing_keys: bool = False,
        default_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        default_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        default_direction: Tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    ):
        """Initialize the registration preprocessor.
        
        Args:
            keys (Union[str, List[str]]): Keys of the moving images to be transformed.
            fixed_image (str): Key of the fixed image.
            moving_images (Optional[Union[str, List[str]]]): Keys of the moving images. If None, uses keys parameter.
            mask_key (Optional[str]): Key for the mask image. If None, no mask is used.
            type_of_transform (str): Type of transform to use. Defaults to "SyN".
            allow_missing_keys (bool): If True, allows missing keys in the input data.
            default_spacing (Tuple[float, float, float]): Default spacing to use if meta_dict is missing.
            default_origin (Tuple[float, float, float]): Default origin to use if meta_dict is missing.
            default_direction (Tuple[float, ...]): Default direction to use if meta_dict is missing.
        """
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.fixed_key = fixed_image  # Keep internal name as fixed_key for compatibility
        self.mask_key = mask_key
        self.type_of_transform = type_of_transform
        self.moving_images = moving_images if moving_images is not None else keys
        if isinstance(self.moving_images, str):
            self.moving_images = [self.moving_images]
        self.default_spacing = default_spacing
        self.default_origin = default_origin
        self.default_direction = default_direction
        
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform image registration.
        
        Args:
            data (Dict[str, Any]): Input data dictionary containing image and metadata.
            
        Returns:
            Dict[str, Any]: Data dictionary with registered images.
        """
        self._check_keys(data)
        
        # Get the fixed image
        if self.fixed_key not in data:
            raise KeyError(f"Fixed image key {self.fixed_key} not found")
        
        fixed_meta_key = f"{self.fixed_key}_meta_dict"
        
        # Handle SimpleITK images
        if isinstance(data[self.fixed_key], sitk.Image):
            fixed_image = self._sitk_to_ants(data[self.fixed_key])
        else:
            # Convert torch tensor to ANTs image
            if fixed_meta_key not in data:
                print(f"Warning: Metadata for fixed image {self.fixed_key} not found, using defaults")
                # Create metadata dictionary if not exists
                data[fixed_meta_key] = {
                    "spacing": self.default_spacing,
                    "origin": self.default_origin,
                    "direction": self.default_direction
                }
            fixed_image = self._torch_to_ants(data[self.fixed_key], data[fixed_meta_key])
        
        # Get the fixed mask if provided
        fixed_mask = None
        if self.mask_key is not None and self.mask_key in data:
            mask_meta_key = f"{self.mask_key}_meta_dict"
            
            # Handle SimpleITK images
            if isinstance(data[self.mask_key], sitk.Image):
                fixed_mask = self._sitk_to_ants(data[self.mask_key])
            else:
                # Convert torch tensor to ANTs image
                if mask_meta_key not in data:
                    print(f"Warning: Metadata for mask {self.mask_key} not found, using defaults")
                    # Create metadata dictionary if not exists
                    data[mask_meta_key] = {
                        "spacing": self.default_spacing,
                        "origin": self.default_origin,
                        "direction": self.default_direction
                    }
                fixed_mask = self._torch_to_ants(data[self.mask_key], data[mask_meta_key])
            
        for key in self.moving_images:
            if key not in data:
                continue
                
            # Get the moving image metadata key
            moving_meta_key = f"{key}_meta_dict"
            
            # Handle SimpleITK images
            if isinstance(data[key], sitk.Image):
                moving_image = self._sitk_to_ants(data[key])
                
                # Create metadata dictionary if not exists
                if moving_meta_key not in data:
                    data[moving_meta_key] = {
                        "spacing": data[key].GetSpacing(),
                        "origin": data[key].GetOrigin(),
                        "direction": data[key].GetDirection()
                    }
            else:
                # Convert torch tensor to ANTs image
                if moving_meta_key not in data:
                    print(f"Warning: Metadata for moving image {key} not found, using defaults")
                    # Create metadata dictionary if not exists
                    data[moving_meta_key] = {
                        "spacing": self.default_spacing,
                        "origin": self.default_origin,
                        "direction": self.default_direction
                    }
                moving_image = self._torch_to_ants(data[key], data[moving_meta_key])
            
            # Perform registration
            reg_result = ants.registration(
                fixed=fixed_image,
                moving=moving_image,
                type_of_transform=self.type_of_transform,
                mask=fixed_mask,
            )
            
            # Get the warped image
            warped_image = reg_result["warpedmovout"]
            
            # Convert back to appropriate format
            if isinstance(data[key], sitk.Image):
                warped_sitk = self._ants_to_sitk(warped_image)
                data[key] = warped_sitk
            else:
                # Convert to torch tensor
                warped_tensor = self._ants_to_torch(warped_image)
                data[key] = warped_tensor
            
            # Update metadata
            data[moving_meta_key]["affine"] = warped_image.direction
            data[moving_meta_key]["origin"] = warped_image.origin
            data[moving_meta_key]["registered"] = True
            data[moving_meta_key]["transform"] = reg_result["fwdtransforms"]
            
        return data
        
    def _torch_to_ants(self, tensor: torch.Tensor, meta_dict: Dict[str, Any]) -> ants.ANTsImage:
        """Convert torch tensor to ANTs image.
        
        Args:
            tensor (torch.Tensor): Input tensor.
            meta_dict (Dict[str, Any]): Metadata dictionary.
            
        Returns:
            ants.ANTsImage: ANTs image.
        """
        # Convert to numpy array
        array = tensor.squeeze().numpy()
        
        # Get metadata
        origin = meta_dict.get("origin", self.default_origin)
        spacing = meta_dict.get("spacing", self.default_spacing)
        direction = meta_dict.get("direction", self.default_direction)
        
        # Create ANTs image
        ants_image = ants.from_numpy(
            array,
            origin=origin,
            spacing=spacing,
            direction=direction
        )
        
        return ants_image
        
    def _ants_to_torch(self, ants_image: ants.ANTsImage) -> torch.Tensor:
        """Convert ANTs image to torch tensor.
        
        Args:
            ants_image (ants.ANTsImage): Input ANTs image.
            
        Returns:
            torch.Tensor: Output torch tensor.
        """
        # Convert to numpy array
        array = ants_image.numpy()
        
        # Convert to torch tensor and add channel dimension
        tensor = torch.from_numpy(array).unsqueeze(0)
        
        return tensor
        
    def _sitk_to_ants(self, sitk_image: sitk.Image) -> ants.ANTsImage:
        """Convert SimpleITK image to ANTs image.
        
        Args:
            sitk_image (sitk.Image): Input SimpleITK image.
            
        Returns:
            ants.ANTsImage: Output ANTs image.
        """
        # Convert to numpy array
        array = sitk.GetArrayFromImage(sitk_image)
        
        # Get metadata
        origin = sitk_image.GetOrigin()
        spacing = sitk_image.GetSpacing()
        direction = sitk_image.GetDirection()
        
        # Create ANTs image
        ants_image = ants.from_numpy(
            array,
            origin=origin,
            spacing=spacing,
            direction=direction
        )
        
        return ants_image
        
    def _ants_to_sitk(self, ants_image: ants.ANTsImage) -> sitk.Image:
        """Convert ANTs image to SimpleITK image.
        
        Args:
            ants_image (ants.ANTsImage): Input ANTs image.
            
        Returns:
            sitk.Image: Output SimpleITK image.
        """
        # Convert to numpy array
        array = ants_image.numpy()
        
        # Create SimpleITK image
        sitk_image = sitk.GetImageFromArray(array)
        
        # Set metadata
        sitk_image.SetOrigin(ants_image.origin)
        sitk_image.SetSpacing(ants_image.spacing)
        sitk_image.SetDirection(ants_image.direction)
        
        return sitk_image 