from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
import SimpleITK as sitk
from .base_preprocessor import BasePreprocessor
from .preprocessor_factory import PreprocessorFactory
from ...utils.image_converter import ImageConverter
from ...utils.progress_utils import CustomTqdm

@PreprocessorFactory.register("histogram_standardization")
class HistogramStandardization(BasePreprocessor):
    """Apply histogram standardization to images using SimpleITK.
    
    This preprocessor matches the histogram of input images to a reference image
    to standardize intensity distributions across images.
    """
    
    def __init__(
        self,
        keys: Union[str, List[str]],
        reference_key: Optional[str] = None,
        reference_image: Optional[sitk.Image] = None,
        num_histogram_bins: int = 256,
        num_match_points: int = 100,
        threshold_mean_intensity: bool = True,
        allow_missing_keys: bool = False,
    ):
        """Initialize the histogram standardization preprocessor.
        
        Args:
            keys (Union[str, List[str]]): Keys of the images to be standardized.
            reference_key (Optional[str]): Key of the reference image in the data dictionary.
                If None, reference_image must be provided.
            reference_image (Optional[sitk.Image]): Reference SimpleITK image for standardization.
                If None, reference_key must be provided.
            num_histogram_bins (int): Number of histogram bins.
            num_match_points (int): Number of match points.
            threshold_mean_intensity (bool): If True, threshold at mean intensity.
            allow_missing_keys (bool): If True, allows missing keys in the input data.
        """
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        
        # Convert single key to list
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys
        
        # Set reference info
        self.reference_key = reference_key
        self.reference_image = reference_image
        
        # Check that at least one reference source is provided
        if reference_key is None and reference_image is None:
            raise ValueError("Either reference_key or reference_image must be provided")
        
        # Set histogram matching parameters
        self.num_histogram_bins = num_histogram_bins
        self.num_match_points = num_match_points
        self.threshold_mean_intensity = threshold_mean_intensity
        
    def _apply_histogram_matching(self, 
                                 input_image: sitk.Image,
                                 reference_image: sitk.Image) -> sitk.Image:
        """Apply histogram matching to a SimpleITK image.
        
        Args:
            input_image (sitk.Image): Input SimpleITK image to be standardized
            reference_image (sitk.Image): Reference SimpleITK image for standardization
            
        Returns:
            sitk.Image: Histogram-standardized SimpleITK image
        """
        # Cast images to float32
        input_image = sitk.Cast(input_image, sitk.sitkFloat32)
        reference_image = sitk.Cast(reference_image, sitk.sitkFloat32)
        
        # Create histogram matching filter
        matcher = sitk.HistogramMatchingImageFilter()
        matcher.SetNumberOfHistogramLevels(self.num_histogram_bins)
        matcher.SetNumberOfMatchPoints(self.num_match_points)
        matcher.SetThresholdAtMeanIntensity(self.threshold_mean_intensity)
        
        # Apply histogram matching
        matched_image = matcher.Execute(input_image, reference_image)
        
        return matched_image
        
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply histogram standardization to the specified images.
        
        Args:
            data (Dict[str, Any]): Input data dictionary containing SimpleITK image objects.
            
        Returns:
            Dict[str, Any]: Data dictionary with histogram-standardized images.
        """
        self._check_keys(data)
        
        # Get reference image
        reference_image = None
        if self.reference_image is not None:
            reference_image = self.reference_image
        elif self.reference_key is not None and self.reference_key in data:
            reference_image = data[self.reference_key]
            if not isinstance(reference_image, sitk.Image):
                raise TypeError(f"Reference key {self.reference_key} is not a SimpleITK Image object")
        else:
            if not self.allow_missing_keys:
                raise KeyError(f"Reference key {self.reference_key} not found in data dictionary")
            return data
        
        # Initialize progress bar
        progress_bar = CustomTqdm(total=len(self.keys), desc="Histogram Standardization")
        
        # Process each image
        for key in self.keys:
            # Skip reference image
            if key == self.reference_key:
                progress_bar.update()
                continue
                
            # Get SimpleITK image from data
            input_image = data[key]
            
            # Ensure we have a SimpleITK image object
            if not isinstance(input_image, sitk.Image):
                print(f"Warning: {key} is not a SimpleITK Image object. Skipping.")
                progress_bar.update()
                continue
            
            try:
                # Apply histogram matching
                matched_image = self._apply_histogram_matching(input_image, reference_image)
                
                # Store the matched image
                data[key] = matched_image
                
                # Update metadata
                meta_key = f"{key}_meta_dict"
                if meta_key not in data:
                    data[meta_key] = {}
                data[meta_key]["histogram_standardized"] = True
                data[meta_key]["reference_key"] = self.reference_key if self.reference_key else "external_reference"
                
            except Exception as e:
                print(f"Error applying histogram standardization to {key}: {e}")
                if not self.allow_missing_keys:
                    raise
            
            progress_bar.update()
        
        return data
