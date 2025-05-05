from typing import Dict, Any, Optional, Union, List
import torch
import numpy as np
import SimpleITK as sitk
from .base_preprocessor import BasePreprocessor
from .preprocessor_factory import PreprocessorFactory
from ...utils.image_converter import ImageConverter

@PreprocessorFactory.register("N4_bias_correction")
class N4CorrectionPreprocessor(BasePreprocessor):
    """N4 bias field correction preprocessor.
    
    This preprocessor applies N4 bias field correction to images using SimpleITK.
    Compatible with MONAI's LoadImaged and EnsureChannelFirstd transforms.
    Automatically skips processing of mask/label keys.
    """
    
    def __init__(
        self,
        keys: Union[str, List[str]],
        shrink_factor: int = 8,
        convergence_threshold: float = 0.005,
        max_iterations: int = 5,
        num_fitting_levels: int = 2,
        bias_field_fwhm: float = 0.2,
        wiener_filter_noise: float = 0.02,
        num_histogram_bins: int = 100,
        allow_missing_keys: bool = False,
    ):
        """Initialize the N4 correction preprocessor.
        
        Args:
            keys (Union[str, List[str]]): Keys of the corresponding items to be transformed.
                Any keys containing 'mask' or 'label' will be automatically skipped.
            shrink_factor (int): Shrink factor for multi-resolution approach. Higher values speed up processing. Defaults to 8.
            convergence_threshold (float): Convergence threshold. Higher values speed up convergence. Defaults to 0.005.
            max_iterations (int): Maximum number of iterations per level. Lower values speed up processing. Defaults to 5.
            num_fitting_levels (int): Number of fitting levels. Lower values speed up processing. Defaults to 2.
            bias_field_fwhm (float): Bias field full width at half maximum. Defaults to 0.2.
            wiener_filter_noise (float): Wiener filter noise level. Defaults to 0.02.
            num_histogram_bins (int): Number of histogram bins. Lower values speed up processing. Defaults to 100.
            allow_missing_keys (bool): If True, allows missing keys in the input data.
        """
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.shrink_factor = shrink_factor
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.num_fitting_levels = num_fitting_levels
        self.bias_field_fwhm = bias_field_fwhm
        self.wiener_filter_noise = wiener_filter_noise
        self.num_histogram_bins = num_histogram_bins
        
    def _is_mask_or_label(self, key: str) -> bool:
        """Check if the key represents a mask or label.
        
        Args:
            key (str): The key to check.
            
        Returns:
            bool: True if the key represents a mask or label, False otherwise.
        """
        return 'mask' in key.lower() or 'label' in key.lower()
        
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply N4 bias field correction to the images.
        
        Args:
            data (Dict[str, Any]): Input data dictionary containing image and metadata from MONAI.
            
        Returns:
            Dict[str, Any]: Data dictionary with bias-corrected images.
        """
        self._check_keys(data)
        
        for key in self.keys:
            # Skip if key is missing and allowed to be missing
            if key not in data:
                if not self.allow_missing_keys:
                    raise KeyError(f"Key '{key}' is missing in the data dictionary.")
                continue
                
            # Skip processing if the key represents a mask or label
            if self._is_mask_or_label(key):
                continue
                
            # Get the image and metadata
            image = data[key]  # Should be torch.Tensor from MONAI's LoadImaged
            meta_dict = data[f"{key}_meta_dict"] if f"{key}_meta_dict" in data else None
            
            if meta_dict is None:
                raise ValueError(f"Metadata for key {key} not found. Ensure LoadImaged is used before N4CorrectionPreprocessor.")
            
            # Convert to SimpleITK for N4 correction
            array = ImageConverter.tensor_to_numpy(image)
            # transpose the array to [Z,Y,X] order
            array = np.transpose(array, (2, 1, 0))
            sitk_image = sitk.GetImageFromArray(array)
            print(f"sitk_image.shape: {sitk_image.GetSize()}")
            
            # Set metadata
            if "spacing" in meta_dict:
                sitk_image.SetSpacing(meta_dict["spacing"])
            if "original_affine" in meta_dict:
                affine = meta_dict["original_affine"]
                if affine is not None:
                    origin = affine[:3, 3].tolist()
                    direction = affine[:3, :3].flatten().tolist()
                    sitk_image.SetOrigin(origin)
                    sitk_image.SetDirection(direction)
                          
            # Apply N4 correction
            corrected_sitk_image = self._apply_n4_correction(sitk_image)
            
            # Convert back to numpy array
            corrected_array = sitk.GetArrayFromImage(corrected_sitk_image)
            # transpose the array to [Z,Y,X] order
            corrected_array = np.transpose(corrected_array, (2, 1, 0))
            
            # Convert to torch tensor with same dtype and device as input
            corrected_tensor = ImageConverter.numpy_to_tensor(corrected_array, dtype=image.dtype, device=image.device)
            
            # Update the data dictionary
            data[key] = corrected_tensor
            data[f"{key}_meta_dict"]["n4_corrected"] = True
            
        return data
        
    def _apply_n4_correction(
        self,
        image: sitk.Image
    ) -> sitk.Image:
        """Apply N4 bias field correction.
        
        Args:
            image (sitk.Image): Input image.
            
        Returns:
            sitk.Image: Bias-corrected image.
        """
        # Cast to float32 if needed
        if image.GetPixelID() != sitk.sitkFloat32:
            image = sitk.Cast(image, sitk.sitkFloat32)
            
        # Create N4 bias field correction filter
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        
        # Set parameters
        corrector.SetConvergenceThreshold(self.convergence_threshold)
        
        # Set number of fitting levels and iterations
        iterations = [self.max_iterations] * self.num_fitting_levels
        corrector.SetMaximumNumberOfIterations(iterations)
        
        # Set parameters to control the B-spline fitting
        corrector.SetBiasFieldFullWidthAtHalfMaximum(self.bias_field_fwhm)
        corrector.SetWienerFilterNoise(self.wiener_filter_noise)
        corrector.SetNumberOfHistogramBins(self.num_histogram_bins)
        
        try:
            # Apply correction without using mask
            corrected_image = corrector.Execute(image)
        except RuntimeError as e:
            print(f"N4 correction failed: {str(e)}")
            print(f"Image pixel type: {image.GetPixelIDTypeAsString()}")
            print(f"Image dimension: {image.GetDimension()}")
            # Return original image if correction fails
            return image
            
        return corrected_image 