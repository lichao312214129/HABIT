from typing import Dict, Any, Optional, Union, List
import torch
import numpy as np
import SimpleITK as sitk
from .base_preprocessor import BasePreprocessor
from .preprocessor_factory import PreprocessorFactory

@PreprocessorFactory.register("N4_bias_correction")
class N4CorrectionPreprocessor(BasePreprocessor):
    """N4 bias field correction preprocessor.
    
    This preprocessor applies N4 bias field correction to images using SimpleITK.
    """
    
    def __init__(
        self,
        keys: Union[str, List[str]],
        shrink_factor: int = 4,
        convergence_threshold: float = 0.001,
        max_iterations: int = 50,
        num_fitting_levels: int = 4,
        bias_field_fwhm: float = 0.15,
        wiener_filter_noise: float = 0.01,
        num_histogram_bins: int = 200,
        allow_missing_keys: bool = False,
    ):
        """Initialize the N4 correction preprocessor.
        
        Args:
            keys (Union[str, List[str]]): Keys of the corresponding items to be transformed.
            shrink_factor (int): Shrink factor for multi-resolution approach. Defaults to 4.
            convergence_threshold (float): Convergence threshold. Defaults to 0.001.
            max_iterations (int): Maximum number of iterations per level. Defaults to 50.
            num_fitting_levels (int): Number of fitting levels. Defaults to 4.
            bias_field_fwhm (float): Bias field full width at half maximum. Defaults to 0.15.
            wiener_filter_noise (float): Wiener filter noise level. Defaults to 0.01.
            num_histogram_bins (int): Number of histogram bins. Defaults to 200.
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
        
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply N4 bias field correction to the images.
        
        Args:
            data (Dict[str, Any]): Input data dictionary containing image and metadata.
            
        Returns:
            Dict[str, Any]: Data dictionary with bias-corrected SimpleITK images.
        """
        self._check_keys(data)
        
        for key in self.keys:
            if key not in data:
                continue
                
            # Get the image and its metadata
            image = data[key]
            meta_key = f"{key}_meta_dict"
            
            # Check if image is a string (file path)
            if isinstance(image, str):
                # Load image with SimpleITK
                sitk_image = sitk.ReadImage(image)
                
                # Create metadata dictionary if not exists
                if meta_key not in data:
                    data[meta_key] = {
                        "spacing": sitk_image.GetSpacing(),
                        "origin": sitk_image.GetOrigin(),
                        "direction": sitk_image.GetDirection()
                    }
            else:
                # Convert torch tensor to SimpleITK image
                sitk_image = self._torch_to_sitk(image, data[meta_key])
            
            # Apply N4 correction
            corrected_image = self._apply_n4_correction(sitk_image)
            
            # Update the data dictionary with SimpleITK image
            data[key] = corrected_image
            
            # Update metadata
            data[meta_key]["n4_corrected"] = True
            
        return data
        
    def _torch_to_sitk(self, tensor: torch.Tensor, meta_dict: Dict[str, Any]) -> sitk.Image:
        """Convert torch tensor to SimpleITK image.
        
        Args:
            tensor (torch.Tensor): Input tensor.
            meta_dict (Dict[str, Any]): Metadata dictionary.
            
        Returns:
            sitk.Image: SimpleITK image.
        """
        # Convert to numpy array
        array = tensor.squeeze().numpy()
        
        # Create SimpleITK image
        sitk_image = sitk.GetImageFromArray(array)
        
        # Set metadata
        sitk_image.SetSpacing(meta_dict["spacing"])
        sitk_image.SetOrigin(meta_dict["origin"])
        sitk_image.SetDirection(meta_dict["direction"])
        
        return sitk_image
        
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
        # Convert to float32 to ensure compatibility with N4 correction
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