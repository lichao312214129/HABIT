from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
import SimpleITK as sitk
import logging
from .base_preprocessor import BasePreprocessor
from .preprocessor_factory import PreprocessorFactory
from ...utils.progress_utils import CustomTqdm
from ...utils.log_utils import get_module_logger

# Get module logger
logger = get_module_logger(__name__)

@PreprocessorFactory.register("histogram_standardization")
class HistogramStandardization(BasePreprocessor):
    """Apply histogram standardization to images using SimpleITK.
    
    This preprocessor matches the histogram of input images to a reference image
    to standardize intensity distributions across images.
    """
    
    def __init__(
        self,
        keys: Union[str, List[str]],
        reference_key: str,
        allow_missing_keys: bool = False,
    ):
        """Initialize the histogram standardization preprocessor.
        
        Args:
            keys (Union[str, List[str]]): Keys of the images to be standardized.
            reference_key (str): Key of the reference image in the data dictionary.
            allow_missing_keys (bool): If True, allows missing keys in the input data.
        """
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        
        # Convert single key to list
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys
        
        # Set reference key
        self.reference_key = reference_key
        
    def _apply_histogram_matching(self, 
                                 input_image: sitk.Image,
                                 reference_image: sitk.Image,
                                 subj: Optional[str] = None,
                                 key: Optional[str] = None) -> sitk.Image:
        """Apply histogram matching to a SimpleITK image.
        
        Args:
            input_image (sitk.Image): Input SimpleITK image to be standardized
            reference_image (sitk.Image): Reference SimpleITK image for standardization
            subj (Optional[str]): Subject identifier for logging
            key (Optional[str]): Image key for logging
            
        Returns:
            sitk.Image: Histogram-standardized SimpleITK image
        """
        subj_info = f"[{subj}] " if subj else ""
        key_info = f"({key}) " if key else ""
        
        # Cast images to float32
        input_image = sitk.Cast(input_image, sitk.sitkFloat32)
        reference_image = sitk.Cast(reference_image, sitk.sitkFloat32)
        
        logger.info(f"{subj_info}{key_info}Applying histogram matching")
        
        # Create histogram matching filter with default parameters
        matcher = sitk.HistogramMatchingImageFilter()
        matcher.SetNumberOfHistogramLevels(256)  # 默认值
        matcher.SetNumberOfMatchPoints(100)  # 默认值
        matcher.SetThresholdAtMeanIntensity(True)  # 默认值
        
        # Apply histogram matching
        matched_image = matcher.Execute(input_image, reference_image)
        
        logger.info(f"{subj_info}{key_info}Histogram matching completed")
        
        return matched_image
        
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply histogram standardization to the specified images.
        
        Args:
            data (Dict[str, Any]): Input data dictionary containing SimpleITK image objects.
            
        Returns:
            Dict[str, Any]: Data dictionary with histogram-standardized images.
        """
        self._check_keys(data)
        
        subj = data.get('subj', 'unknown')
        logger.info(f"Processing subject: {subj}")
        
        # Get reference image
        if self.reference_key not in data:
            if not self.allow_missing_keys:
                raise KeyError(f"[{subj}] Reference key {self.reference_key} not found in data dictionary")
            return data
            
        reference_image = data[self.reference_key]
        if not isinstance(reference_image, sitk.Image):
            raise TypeError(f"[{subj}] Reference key {self.reference_key} is not a SimpleITK Image object")
        
        logger.info(f"[{subj}] Using {self.reference_key} as reference image for histogram standardization")
        
        # Initialize progress bar
        progress_bar = CustomTqdm(total=len(self.keys), desc=f"[{subj}] 直方图标准化")
        
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
                logger.warning(f"[{subj}] Warning: {key} is not a SimpleITK Image object. Skipping.")
                progress_bar.update()
                continue
            
            try:
                # Apply histogram matching
                matched_image = self._apply_histogram_matching(input_image, reference_image, subj, key)
                
                # Store the matched image
                data[key] = matched_image
                
                # Update metadata
                meta_key = f"{key}_meta_dict"
                if meta_key not in data:
                    data[meta_key] = {}
                data[meta_key]["histogram_standardized"] = True
                data[meta_key]["reference_key"] = self.reference_key
                
                logger.info(f"[{subj}] Successfully standardized image {key}")
                
            except Exception as e:
                logger.error(f"[{subj}] Error applying histogram standardization to {key}: {e}")
                if not self.allow_missing_keys:
                    raise
            
            progress_bar.update()
        
        return data
