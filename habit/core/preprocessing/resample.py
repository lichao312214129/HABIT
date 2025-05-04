from typing import Dict, Any, Optional, Sequence, Union, List
import torch
import numpy as np
import SimpleITK as sitk
from monai.transforms import Resample
from .base_preprocessor import BasePreprocessor
from .preprocessor_factory import PreprocessorFactory

@PreprocessorFactory.register("resample")
class ResamplePreprocessor(BasePreprocessor):
    """Resample images to a target spacing.
    
    This preprocessor resamples images to a specified target spacing.
    """
    
    def __init__(
        self,
        keys: Union[str, List[str]],
        target_spacing: Sequence[float],
        mode: str = "bilinear",
        mask_keys: Optional[Union[str, List[str]]] = None,
        mask_mode: str = "nearest",
        padding_mode: str = "border",
        align_corners: bool = False,
        allow_missing_keys: bool = False,
        default_spacing: Sequence[float] = (1.0, 1.0, 1.0),
    ):
        """Initialize the resample preprocessor.
        
        Args:
            keys (Union[str, List[str]]): Keys of the corresponding items to be transformed.
            target_spacing (Sequence[float]): Target spacing to resample to.
            mode (str): Interpolation mode for images. Defaults to "bilinear".
            mask_keys (Optional[Union[str, List[str]]]): Keys of mask items that need different resampling.
            mask_mode (str): Interpolation mode for masks. Defaults to "nearest".
            padding_mode (str): Padding mode for out-of-bound values. Defaults to "border".
            align_corners (bool): Whether to align corners. Defaults to False.
            allow_missing_keys (bool): If True, allows missing keys in the input data.
            default_spacing (Sequence[float]): Default spacing to use if meta_dict is missing.
        """
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.target_spacing = target_spacing
        self.mode = mode
        self.mask_keys = mask_keys if mask_keys is not None else []
        if isinstance(self.mask_keys, str):
            self.mask_keys = [self.mask_keys]
        self.mask_mode = mask_mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.default_spacing = default_spacing
        
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Resample the images to the target spacing.
        
        Args:
            data (Dict[str, Any]): Input data dictionary containing image and metadata.
            
        Returns:
            Dict[str, Any]: Data dictionary with resampled images.
        """
        self._check_keys(data)
        
        # Process regular images
        for key in self.keys:
            if key not in data:
                continue
                
            # Check if this is a mask key
            if key in self.mask_keys:
                continue
                
            # Get the image
            image = data[key]
            
            # Handle path
            if isinstance(image, str):
                image = sitk.ReadImage(image)
                data[key] = self._resample_sitk_image(image, self.target_spacing, self.mode)
                continue
                
            # Handle SimpleITK images    
            if isinstance(image, sitk.Image):
                data[key] = self._resample_sitk_image(image, self.target_spacing, self.mode)
                continue
                
            meta_key = f"{key}_meta_dict"
            if meta_key not in data:
                print(f"Warning: Metadata for key {key} not found, using default spacing")
                # Create metadata dictionary with default values
                data[meta_key] = {
                    "spacing": self.default_spacing,
                    "origin": (0.0, 0.0, 0.0),
                    "direction": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
                }
                spacing = self.default_spacing
            else:
                spacing = data[meta_key].get("spacing", self.default_spacing)
            
            # Ensure image is a PyTorch tensor
            if not isinstance(image, torch.Tensor):
                if isinstance(image, np.ndarray):
                    image = torch.from_numpy(image).float()
                else:
                    print(f"Warning: Unsupported image type {type(image)} for key {key}")
                    continue
                    
            # Ensure image has batch dimension [B, C, H, W, D]
            if image.ndim == 3:  # [H, W, D]
                image = image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
            elif image.ndim == 4:  # [C, H, W, D]
                image = image.unsqueeze(0)  # Add batch dimension
            
            # Create Resample transform
            resampler = Resample(
                pixdim=self.target_spacing,
                mode=self.mode,
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
            )
            
            # Apply resampling
            resampled_image = resampler(
                img=image, 
                orig_size=image.shape[1:],  # Original spatial dimensions
                orig_pixdim=spacing,  # Original pixel dimensions
            )
            
            # Update the data dictionary
            data[key] = resampled_image
            
            # Update metadata
            data[meta_key]["spacing"] = self.target_spacing
            data[meta_key]["resampled"] = True
            
        # Process mask images with different resampling mode
        for key in self.mask_keys:
            if key not in data:
                continue
                
            # Get the mask
            mask = data[key]
            
            # Handle path
            if isinstance(mask, str):
                mask = sitk.ReadImage(mask)
                data[key] = self._resample_sitk_image(mask, self.target_spacing, self.mask_mode)
                continue
                
            # Handle SimpleITK images
            if isinstance(mask, sitk.Image):
                data[key] = self._resample_sitk_image(mask, self.target_spacing, self.mask_mode)
                continue
                
            # Handle torch tensors
            meta_key = f"{key}_meta_dict"
            if meta_key not in data:
                print(f"Warning: Metadata for key {key} not found, using default spacing")
                spacing = self.default_spacing
                # Create metadata dictionary if not exists
                data[meta_key] = {
                    "spacing": spacing,
                    "origin": (0.0, 0.0, 0.0),
                    "direction": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
                }
            else:
                spacing = data[meta_key].get("spacing", self.default_spacing)
            
            # Ensure mask is a PyTorch tensor
            if not isinstance(mask, torch.Tensor):
                if isinstance(mask, np.ndarray):
                    mask = torch.from_numpy(mask).float()
                else:
                    print(f"Warning: Unsupported mask type {type(mask)} for key {key}")
                    continue
                    
            # Ensure mask has batch dimension [B, C, H, W, D]
            if mask.ndim == 3:  # [H, W, D]
                mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
            elif mask.ndim == 4:  # [C, H, W, D]
                mask = mask.unsqueeze(0)  # Add batch dimension
            
            # Create Resample transform
            resampler = Resample(
                pixdim=self.target_spacing,
                mode=self.mask_mode,  # Use nearest for masks
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
            )
            
            # Apply resampling
            resampled_mask = resampler(
                img=mask,
                orig_size=mask.shape[1:],  # Original spatial dimensions
                orig_pixdim=spacing,  # Original pixel dimensions
            )
            
            # Update the data dictionary
            data[key] = resampled_mask
            
            # Update metadata
            data[meta_key]["spacing"] = self.target_spacing
            data[meta_key]["resampled"] = True
            
        return data
        
    def _resample_sitk_image(
        self, 
        image: sitk.Image, 
        target_spacing: Sequence[float],
        mode: str
    ) -> sitk.Image:
        """Resample a SimpleITK image to the target spacing.
        
        Args:
            image (sitk.Image): Input image.
            target_spacing (Sequence[float]): Target spacing.
            mode (str): Interpolation mode.
            
        Returns:
            sitk.Image: Resampled image.
        """
        # Get image properties
        size = image.GetSize()
        spacing = image.GetSpacing()
        
        # Calculate new size
        new_size = [
            int(round(size[i] * spacing[i] / target_spacing[i]))
            for i in range(len(size))
        ]
        
        # Create resampler
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(new_size)
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        
        # Set interpolation method
        if mode == "nearest":
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        elif mode == "linear" or mode == "bilinear":
            resampler.SetInterpolator(sitk.sitkLinear)
        elif mode == "bspline":
            resampler.SetInterpolator(sitk.sitkBSpline)
        else:
            resampler.SetInterpolator(sitk.sitkLinear)
        
        # Execute resampling
        resampled_image = resampler.Execute(image)
        
        return resampled_image 